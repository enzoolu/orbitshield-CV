from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

try:
    import tensorflow as tf
except ImportError:  # pragma: no cover
    tf = None


ROOT_DIR = Path(__file__).resolve().parent
MODELS_DIR = ROOT_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "outputs"
SAMPLES_DIR = ROOT_DIR / "samples"
TRAIN_DIR = ROOT_DIR / "dataset" / "processed" / "train"
TRAINING_SUMMARY_PATH = OUTPUTS_DIR / "training_summary.json"
IMAGE_SIZE = (128, 128)
REFERENCE_ACCURACY = 0.88
FALLBACK_CLASSES = ["rocket", "satellite", "space_shuttle"]
LOCAL_EXECUTION_COMMANDS = """pip install -r requirements.txt
python src/collect_nasa_images.py
python src/audit_dataset.py
python src/clean_dataset.py
python src/generate_synthetic_images.py
python src/prepare_dataset.py
python src/train_models.py
streamlit run app.py"""
CLASS_EXPLANATIONS = {
    "satellite": "objeto orbital/satélite",
    "rocket": "foguete ou veículo lançador",
    "space_shuttle": "ônibus espacial/orbiter",
}


def pretty_label(label: str) -> str:
    return label.replace("_", " ").title()


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.6rem;
            padding-bottom: 2rem;
        }
        .orbitshield-panel {
            border: 1px solid rgba(40, 70, 110, 0.18);
            border-radius: 18px;
            padding: 1rem 1.1rem;
            background: linear-gradient(180deg, rgba(245,248,252,0.98), rgba(236,242,248,0.96));
            margin-bottom: 1rem;
        }
        .orbitshield-kicker {
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #4f6f8f;
            font-weight: 700;
        }
        .orbitshield-title {
            font-size: 1.1rem;
            font-weight: 700;
            color: #16324f;
            margin-bottom: 0.25rem;
        }
        .orbitshield-copy {
            font-size: 0.96rem;
            color: #24384f;
            margin-bottom: 0;
        }
        .orbitshield-badge {
            display: inline-block;
            padding: 0.35rem 0.6rem;
            border-radius: 999px;
            background: #123456;
            color: white;
            font-size: 0.84rem;
            font-weight: 700;
            margin-top: 0.4rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_panel(kicker: str, title: str, copy: str) -> None:
    st.markdown(
        f"""
        <div class="orbitshield-panel">
            <div class="orbitshield-kicker">{kicker}</div>
            <div class="orbitshield-title">{title}</div>
            <p class="orbitshield-copy">{copy}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_prediction_result(image: Image.Image, class_names: list[str], probabilities: np.ndarray, origin_label: str) -> None:
    prediction_idx = int(np.argmax(probabilities))
    predicted_label = class_names[prediction_idx]
    prediction_confidence = float(probabilities[prediction_idx])

    left_col, right_col = st.columns([1.05, 1], gap="large")

    with left_col:
        st.image(image, caption=origin_label, use_container_width=True)

    with right_col:
        metric_col_1, metric_col_2 = st.columns(2)
        metric_col_1.metric("Classe prevista", pretty_label(predicted_label))
        metric_col_2.metric("Confiança", f"{prediction_confidence:.2%}")

        probability_df = pd.DataFrame(
            {
                "classe": [pretty_label(name) for name in class_names],
                "probabilidade": probabilities,
            }
        ).sort_values("probabilidade", ascending=False)
        st.write("Probabilidades das classes")
        st.bar_chart(probability_df.set_index("classe"))
        st.dataframe(
            probability_df.assign(probabilidade=probability_df["probabilidade"].map(lambda value: f"{value:.2%}")),
            hide_index=True,
            use_container_width=True,
        )


@st.cache_data
def load_class_config() -> tuple[list[str], str]:
    if TRAINING_SUMMARY_PATH.exists():
        try:
            payload = json.loads(TRAINING_SUMMARY_PATH.read_text(encoding="utf-8"))
            class_names = payload.get("class_names", [])
            if isinstance(class_names, list) and class_names and all(isinstance(name, str) for name in class_names):
                return class_names, "metadata"
        except (OSError, json.JSONDecodeError):
            pass

    if TRAIN_DIR.exists():
        class_names = sorted([path.name for path in TRAIN_DIR.iterdir() if path.is_dir()])
        if class_names:
            return class_names, "dataset"

    return FALLBACK_CLASSES, "fallback"


@st.cache_data
def read_metrics() -> pd.DataFrame:
    path = OUTPUTS_DIR / "model_comparison.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def list_sample_paths() -> list[str]:
    if not SAMPLES_DIR.exists():
        return []
    return sorted(
        [
            path.name
            for path in SAMPLES_DIR.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
    )


def local_model_candidates() -> list[Path]:
    return [
        MODELS_DIR / "simple_cnn_inference.keras",
        MODELS_DIR / "simple_cnn.keras",
        MODELS_DIR / "deep_cnn.keras",
        MODELS_DIR / "deep_cnn_inference.keras",
    ]


def resolve_best_model_path() -> Path | None:
    for candidate_path in local_model_candidates():
        if candidate_path.exists():
            return candidate_path

    comparison = read_metrics()
    if not comparison.empty:
        comparison = comparison.sort_values(
            ["selected_best", "test_accuracy", "f1_macro"],
            ascending=[False, False, False],
        )
        csv_path = Path(str(comparison.iloc[0]["model_path"]))
        if csv_path.exists():
            return csv_path
    return None


@st.cache_resource
def load_model():
    if tf is None:
        return None
    model_path = resolve_best_model_path()
    if model_path is None or not model_path.exists():
        return None
    return tf.keras.models.load_model(model_path, compile=False)


def preprocess_image(image: Image.Image) -> np.ndarray:
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
    array = np.asarray(image, dtype=np.float32)
    array /= 255.0
    return np.expand_dims(array, axis=0)


def predict_probabilities(model, image: Image.Image) -> np.ndarray:
    return model.predict(preprocess_image(image), verbose=0)[0]


def render_explanations() -> None:
    st.write("Interpretação simplificada das classes")
    columns = st.columns(3, gap="medium")
    for column, class_name in zip(columns, FALLBACK_CLASSES):
        with column:
            render_panel(
                "Classe",
                pretty_label(class_name),
                CLASS_EXPLANATIONS[class_name],
            )


def render_local_execution_section() -> None:
    with st.expander("Execução local x Deploy online", expanded=True):
        st.write(
            "No ambiente local, é possível executar o pipeline completo do projeto: "
            "coleta de imagens reais da NASA, geração de imagens sintéticas, preparação do dataset, "
            "treinamento das CNNs, validação e execução do app."
        )
        st.write(
            "No Streamlit Cloud, por limitação de tamanho, o dataset completo não esta presente."
        )
        st.write(
            "Quando `dataset/processed/train` não existe no deploy, o app carrega as classes a partir da metadata do treinamento."
        )
        st.write(
            "A reprodução completa dos resultados deve ser feita localmente pelos comandos do README."
        )
        st.code(LOCAL_EXECUTION_COMMANDS, language="bash")


def render_prediction_tab(model, class_names: list[str]) -> None:
    st.subheader("Predição")
    render_panel(
        "Demonstração ACV",
        "Classificador orbital do OrbitShield Vision",
        "Envie uma imagem para o melhor modelo treinado estimar se o objeto pertence a rocket, satellite ou space shuttle.",
    )

    if tf is None:
        st.error("TensorFlow não está instalado no ambiente.")
        return
    if model is None:
        st.error("Modelo não encontrado no ambiente de deploy. Para executar a predição completa, rode localmente os comandos abaixo.")
        st.code(LOCAL_EXECUTION_COMMANDS, language="bash")
        return

    uploaded_file = st.file_uploader(
        "Envie uma imagem de satélite, foguete ou ônibus espacial",
        type=["jpg", "jpeg", "png"],
    )

    render_explanations()

    if uploaded_file is None:
        st.info("Envie uma imagem para visualizar a predição e as probabilidades.")
        return

    image = Image.open(uploaded_file)
    probabilities = predict_probabilities(model, image)
    render_prediction_result(image, class_names, probabilities, "Imagem enviada")


def render_examples_tab(model, class_names: list[str]) -> None:
    st.subheader("Exemplos")
    render_panel(
        "Amostras",
        "Teste rápido com exemplos do projeto",
        "Escolha uma imagem da pasta `samples/` para demonstrar a inferência sem precisar subir um arquivo manualmente.",
    )

    sample_names = list_sample_paths()
    if not sample_names:
        st.info("Nenhuma imagem encontrada em `samples/`. Execute `python src/prepare_dataset.py` para regenerar as amostras.")
        return

    selected_name = st.selectbox("Escolha uma imagem exemplo", sample_names, index=0)
    sample_path = SAMPLES_DIR / selected_name

    if not sample_path.exists():
        st.warning("A imagem exemplo selecionada não foi encontrada no disco.")
        return

    image = Image.open(sample_path)
    if tf is None or model is None:
        st.warning("Modelo indisponível para executar a predição neste momento.")
        return

    probabilities = predict_probabilities(model, image)
    render_prediction_result(image, class_names, probabilities, f"Exemplo: {selected_name}")

    st.write("Galeria de amostras")
    preview_cols = st.columns(3)
    for index, sample_name in enumerate(sample_names):
        with preview_cols[index % 3]:
            st.image(str(SAMPLES_DIR / sample_name), caption=sample_name, use_container_width=True)

    extra_output_images = ["correct_predictions.png", "wrong_predictions.png"]
    extra_cols = st.columns(2, gap="large")
    for column, image_name in zip(extra_cols, extra_output_images):
        with column:
            image_path = OUTPUTS_DIR / image_name
            if image_path.exists():
                st.image(str(image_path), caption=image_name, use_container_width=True)


def render_metrics_highlights(metrics_df: pd.DataFrame) -> None:
    simple_row = metrics_df.loc[metrics_df["model_name"] == "SimpleCNN"]
    deep_row = metrics_df.loc[metrics_df["model_name"] == "DeepCNN"]
    best_row = metrics_df.sort_values(["selected_best", "test_accuracy", "f1_macro"], ascending=[False, False, False]).iloc[0]

    col_1, col_2, col_3, col_4 = st.columns(4)
    col_1.metric("SimpleCNN", f"{float(simple_row.iloc[0]['test_accuracy']):.2%}" if not simple_row.empty else "N/D")
    col_2.metric("DeepCNN", f"{float(deep_row.iloc[0]['test_accuracy']):.2%}" if not deep_row.empty else "N/D")
    col_3.metric("Meta de referência", f"{REFERENCE_ACCURACY:.0%}")
    col_4.metric("Melhor modelo", str(best_row["model_name"]))


def render_metrics_tab() -> None:
    st.subheader("Métricas")
    render_panel(
        "Desempenho",
        "Comparação final das CNNs",
        "A aba reúne a comparação tabular, curvas de treinamento e matrizes de confusão usadas na demonstração da entrega ACV.",
    )

    metrics_df = read_metrics()
    if metrics_df.empty:
        st.warning("Arquivo `outputs/model_comparison.csv` não encontrado.")
        return

    render_metrics_highlights(metrics_df)
    st.dataframe(metrics_df, use_container_width=True)

    image_groups = [
        ("Matrizes de confusão", ["confusion_matrix_simple_cnn.png", "confusion_matrix_deep_cnn.png"]),
        ("Curvas de treinamento", ["training_curves_simple_cnn.png", "training_curves_deep_cnn.png"]),
    ]

    for group_title, image_names in image_groups:
        st.markdown(f"### {group_title}")
        cols = st.columns(2, gap="large")
        for column, image_name in zip(cols, image_names):
            with column:
                image_path = OUTPUTS_DIR / image_name
                if image_path.exists():
                    st.image(str(image_path), caption=image_name, use_container_width=True)
                else:
                    st.info(f"Arquivo ausente: `outputs/{image_name}`")


def render_about_tab() -> None:
    st.subheader("Sobre o projeto")
    col_1, col_2 = st.columns([1.15, 1], gap="large")

    with col_1:
        render_panel(
            "OrbitShield Vision",
            "Visão computacional aplicada a sistemas espaciais",
            "O app demonstra uma camada de classificação visual para o ecossistema OrbitShield, conectando monitoramento orbital, telemetria e centro de missão com reconhecimento de imagens.",
        )
        st.markdown(
            """
            - conexão direta com a Indústria Espacial;
            - uso de imagens reais da NASA combinadas com imagens sintéticas simuladas;
            - duas CNNs criadas do zero;
            - nenhum modelo pré-treinado ou transfer learning;
            - resultado final acima da referência ACV de 88%.
            """
        )

    with col_2:
        st.markdown(
            """
            <div class="orbitshield-panel">
                <div class="orbitshield-kicker">Resultado final</div>
                <div class="orbitshield-title">SimpleCNN como melhor modelo</div>
                <p class="orbitshield-copy">
                    Accuracy final de teste: <strong>89.96%</strong><br/>
                    Accuracy da DeepCNN: <strong>89.41%</strong><br/>
                    Meta de referência ACV: <strong>88%</strong>
                </p>
                <div class="orbitshield-badge">Sem pesos pré-treinados</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def main() -> None:
    st.set_page_config(page_title="OrbitShield Vision", layout="wide")
    inject_styles()
    st.title("OrbitShield Vision")
    st.caption(
        "Demonstração Streamlit da entrega ACV: classificação de sistemas espaciais com imagens reais da NASA e simulações de sensor orbital."
    )

    class_names, class_source = load_class_config()
    model = load_model()

    if not TRAIN_DIR.exists():
        if class_source == "metadata":
            st.warning(
                "Modo deploy: o dataset completo não foi incluído no ambiente online para reduzir o tamanho do repositório. "
                "As classes foram carregadas pela metadata do treinamento. Para reproduzir coleta, preparação, treino e "
                "validação completos, execute o projeto localmente conforme o README."
            )
        else:
            st.warning(
                "Modo deploy: o dataset completo não foi incluído no ambiente online para reduzir o tamanho do repositório. "
                "As classes foram carregadas por configuração fixa. Para reproduzir coleta, preparação, treino e "
                "validação completos, execute o projeto localmente conforme o README."
            )

    render_local_execution_section()

    tabs = st.tabs(["Predição", "Exemplos", "Métricas", "Sobre o projeto"])
    with tabs[0]:
        render_prediction_tab(model, class_names)
    with tabs[1]:
        render_examples_tab(model, class_names)
    with tabs[2]:
        render_metrics_tab()
    with tabs[3]:
        render_about_tab()


if __name__ == "__main__":
    main()
