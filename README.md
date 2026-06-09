# OrbitShield Vision

Entrega da disciplina **Applied Computer Vision (ACV)** da **Global Solution 2026 FIAP**, integrada ao projeto **OrbitShield**.

O OrbitShield Vision propõe uma solução de visão computacional para **classificação de imagens de sistemas espaciais**, simulando a camada visual de um sensor orbital usado em monitoramento de risco orbital, telemetria de nanossatélites e apoio a centro de missão.

## Objetivo da entrega

Construir e comparar **duas CNNs criadas do zero**, sem modelos pré-treinados, para classificar imagens das seguintes classes:

- `satellite`
- `rocket`
- `space_shuttle`

O foco da entrega foi manter aderência ao contexto da **Indústria Espacial**, usar imagens reais da NASA sempre que possível e documentar tecnicamente as decisões necessárias para atingir desempenho competitivo.

## Problema de Visão Computacional

- Entrada: imagem de um objeto ou sistema espacial
- Saída: classe prevista entre `satellite`, `rocket` e `space_shuttle`
- Tipo de tarefa: **classificação de imagens**

Aplicação no contexto OrbitShield:

- identificação automática de alvos visuais em cenários orbitais;
- apoio a monitoramento de ativos espaciais;
- simulação de sensor embarcado para triagem visual inicial.

## Evolução técnica do projeto

O projeto foi desenvolvido em três etapas:

1. dataset real-only com `satellite`, `rocket` e `space_station`;
2. dataset real-only com `satellite`, `rocket` e `space_shuttle`;
3. dataset **híbrido** com imagens reais da NASA + imagens sintéticas simuladas em Python.

A estratégia final foi adotar o dataset híbrido porque:

- o melhor resultado **real-only** ficou em `69.70%`;
- as imagens reais da NASA apresentaram alta heterogeneidade visual;
- o enunciado permite dataset **criado, coletado, adaptado ou simulado pela equipe**;
- as imagens sintéticas foram usadas como **simulações de sensor orbital**, sem remover a base real.

## Origem das imagens

### Imagens reais

Fonte principal: **NASA Image and Video Library API**

- endpoint de busca: `https://images-api.nasa.gov/search`
- endpoint de assets: `https://images-api.nasa.gov/asset/{nasa_id}`

A coleta automática é feita por `src/collect_nasa_images.py`, com:

- consulta por termos relacionados a cada classe;
- download apenas de imagens;
- validação com PIL;
- conversão para RGB;
- descarte de arquivos quebrados.

### Imagens sintéticas

As imagens sintéticas foram geradas por `src/generate_synthetic_images.py` e representam **simulações de sensor orbital**.

Padrões por classe:

- `satellite`: corpo central retangular com painéis solares laterais;
- `rocket`: fuselagem alongada com ponta cônica e base de propulsão;
- `space_shuttle`: fuselagem com asas triangulares e cauda.

Variações aplicadas:

- estrelas e fundo espacial;
- ruído;
- brilho e contraste;
- rotação;
- escala;
- translação;
- blur leve.

## Quantidade de imagens por classe

### Base real após coleta e limpeza

| Classe | Reais |
| --- | ---: |
| satellite | 287 |
| rocket | 299 |
| space_shuttle | 299 |

### Base sintética gerada

| Classe | Sintéticas |
| --- | ---: |
| satellite | 900 |
| rocket | 900 |
| space_shuttle | 900 |

### Dataset final utilizado no treino

| Classe | Reais | Sintéticas | Total |
| --- | ---: | ---: | ---: |
| satellite | 287 | 900 | 1187 |
| rocket | 299 | 900 | 1199 |
| space_shuttle | 299 | 900 | 1199 |

### Distribuição por split

| Split | rocket | satellite | space_shuttle |
| --- | ---: | ---: | ---: |
| train | 839 | 831 | 839 |
| val | 180 | 178 | 180 |
| test | 180 | 178 | 180 |

Documentação complementar:

- `docs/dataset_collection_report.md`
- `docs/dataset_audit_report.md`
- `docs/dataset_cleaning_report.md`
- `docs/dataset_split_report.md`

## Pré-processamento

Pipeline aplicado:

- auditoria de imagens quebradas, duplicatas e baixa variância;
- limpeza não destrutiva com registro das ações;
- conversão para RGB;
- redimensionamento para `128x128`;
- split estratificado em `70% treino`, `15% validação`, `15% teste`;
- preservação de balanceamento por classe;
- separação por `group_id` para evitar vazamento entre variantes sintéticas muito parecidas;
- normalização dos pixels no treino e na inferência;
- data augmentation leve apenas no treino.

## Arquiteturas implementadas

### 1. SimpleCNN

Arquitetura base:

- Conv2D
- MaxPooling2D
- Conv2D
- MaxPooling2D
- Flatten
- Dense
- Dropout
- Dense com softmax

### 2. DeepCNN

Arquitetura base:

- Conv2D
- BatchNormalization
- MaxPooling2D
- Conv2D
- BatchNormalization
- MaxPooling2D
- Conv2D
- BatchNormalization
- MaxPooling2D
- Dense
- Dropout
- Dense com softmax

Restrições respeitadas:

- sem `pretrained=True`
- sem `weights="imagenet"`
- sem ResNet
- sem VGG
- sem MobileNet
- sem EfficientNet
- sem YOLO
- sem transfer learning

## Treinamento

Treinamento realizado em `src/train_models.py` com:

- TensorFlow / Keras;
- normalização para intervalo `0-1`;
- data augmentation leve no conjunto de treino;
- `EarlyStopping` com `restore_best_weights=True`;
- `ModelCheckpoint`;
- comparação final por `test_accuracy`, `test_loss`, `precision_macro`, `recall_macro` e `f1_macro`.

## Resultados obtidos

### Baselines real-only anteriores

| Cenário | Melhor modelo | Accuracy |
| --- | --- | ---: |
| Real-only com `space_station` | SimpleCNN | 69.70% |
| Real-only com `space_shuttle` | SimpleCNN | 66.17% |

### Resultado final com dataset híbrido

| Modelo | Test Accuracy | Test Loss | Precision Macro | Recall Macro | F1 Macro | Melhor? |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| SimpleCNN | 89.96% | 0.2275 | 90.01% | 89.97% | 89.96% | Sim |
| DeepCNN | 89.41% | 0.3881 | 89.47% | 89.41% | 89.39% | Não |

Resumo final:

- melhor modelo: `SimpleCNN`
- accuracy final da `SimpleCNN`: `89.96%`
- accuracy final da `DeepCNN`: `89.41%`
- meta de referência ACV: `88%`
- resultado final: **meta atingida**
- ganho da melhor configuração em relação ao melhor real-only anterior: `69.70% -> 89.96%` (`+20.26` pontos percentuais)

## Análise técnica dos resultados

A troca de `space_station` por `space_shuttle` melhorou a separabilidade semântica da classe, mas só isso não foi suficiente para superar a meta. O ganho decisivo veio da composição **híbrida** do dataset, que permitiu:

- reforçar padrões morfológicos das classes;
- reduzir ambiguidade visual entre exemplos;
- manter o vínculo com imagens reais da NASA;
- treinar CNNs do zero com mais estabilidade.

A `SimpleCNN` terminou como melhor arquitetura da entrega, mesmo sendo mais simples, o que indica que a separação do problema ficou mais dependente de um dataset bem controlado do que de profundidade adicional da rede.

## Evidências geradas

Arquivos principais em `outputs/`:

- `model_comparison.csv`
- `training_summary.json`
- `confusion_matrix_simple_cnn.png`
- `confusion_matrix_deep_cnn.png`
- `training_curves_simple_cnn.png`
- `training_curves_deep_cnn.png`
- `correct_predictions.png`
- `wrong_predictions.png`

Também estão disponíveis:

- classificação detalhada por modelo;
- contact sheets por classe;
- resumos de auditoria e integridade do split.

## Aplicação Streamlit

O app `app.py` implementa uma demonstração funcional da entrega com:

- aba de **Predição** com upload de imagem;
- aba de **Exemplos** com imagens da pasta `samples/`;
- aba de **Métricas** com tabela e visualizações;
- aba **Sobre o projeto** com resumo da solução.

Predição exibida no app:

- classe prevista;
- confiança;
- distribuição de probabilidades entre as classes.

## Observação sobre o deploy

O deploy em ambientes como Streamlit Cloud pode não conter todo o diretório `dataset/processed/`, seja por limitação de tamanho, seja por restrições do ambiente de execução. Por isso:

- o app foi ajustado para carregar as classes pela metadata do treinamento sempre que disponível;
- se o dataset processado não estiver presente no deploy, a interface continua funcional para inferência com o modelo salvo;
- a validação completa do pipeline de preparação, treino e reprodução integral da entrega deve ser feita localmente.

Os artefatos abaixo comprovam o treinamento e os resultados finais da entrega:

- `outputs/model_comparison.csv`
- `outputs/training_summary.json`
- `outputs/confusion_matrix_simple_cnn.png`
- `outputs/confusion_matrix_deep_cnn.png`
- `notebooks/orbitshield_acv_training.ipynb`
- `docs/relatorio_tecnico.md`

### Execução local completa

```bash
pip install -r requirements.txt
python src/generate_synthetic_images.py
python src/prepare_dataset.py
python src/train_models.py
streamlit run app.py
```

## Como executar

### 1. Instalar dependências

```bash
pip install -r requirements.txt
```

### 2. Coletar imagens reais da NASA

```bash
python src/collect_nasa_images.py
```

### 3. Auditar o dataset

```bash
python src/audit_dataset.py
```

### 4. Limpar o dataset

```bash
python src/clean_dataset.py
```

### 5. Gerar imagens sintéticas

```bash
python src/generate_synthetic_images.py
```

### 6. Preparar o dataset final

```bash
python src/prepare_dataset.py
```

### 7. Treinar os modelos

```bash
python src/train_models.py
```

### 8. Executar predição por script

```bash
python src/predict.py
```

### 9. Abrir a demonstração em Streamlit

```bash
streamlit run app.py
```

## Notebook

O notebook principal está em `notebooks/orbitshield_acv_training.ipynb` e apresenta:

- contexto do problema;
- tentativa real-only;
- justificativa técnica para o dataset híbrido;
- origem e contagem das imagens;
- pré-processamento;
- arquiteturas das CNNs;
- treinamento;
- curvas de accuracy/loss;
- matrizes de confusão;
- comparação entre modelos;
- exemplos de acertos e erros;
- conclusão técnica.

## Estrutura de pastas

```text
orbitshield-acv/
|-- app.py
|-- README.md
|-- requirements.txt
|-- dataset/
|-- docs/
|-- models/
|-- notebooks/
|-- outputs/
|-- samples/
`-- src/
```

## Vídeo

- Vídeo da entrega: `COLOCAR_LINK_VIDEO_AQUI`

## Linkd do app
- https://orbitshield-cv.streamlit.app/

## Integrantes

| Nome | RM |
| --- | --- |
| João Pedro Cruz | `RM98650` |
| Tiago Paulino | `RM551169` |
| Victor Eid | `RM98668` |
| Enzo Luiz Goulart | `RM99666` |
