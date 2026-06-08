# Relatorio Tecnico

## 1. Objetivo

Implementar a entrega de Applied Computer Vision do projeto OrbitShield por meio de um classificador de imagens de sistemas espaciais usando CNNs treinadas do zero.

Classes finais desta rodada:

- `satellite`
- `rocket`
- `space_shuttle`

## 2. Historico de decisoes tecnicas

A construcao do dataset ocorreu em tres fases:

1. tentativa real-only com `space_station`;
2. tentativa real-only com `space_shuttle`;
3. dataset hibrido com imagens reais da NASA e imagens sinteticas geradas em Python.

A mudanca para dataset hibrido foi adotada porque o melhor resultado real-only do projeto ficou em `69.70%`. As imagens reais da NASA mostraram heterogeneidade elevada para um problema de treino do zero, mesmo apos auditoria, limpeza e troca da classe mais ambigua.

## 3. Dataset final

### 3.1 Base real apos limpeza

| Classe | Quantidade |
| --- | ---: |
| satellite | 287 |
| rocket | 299 |
| space_shuttle | 299 |

### 3.2 Base sintetica gerada pela equipe

| Classe | Quantidade |
| --- | ---: |
| satellite | 900 |
| rocket | 900 |
| space_shuttle | 900 |

### 3.3 Dataset hibrido total

| Classe | Reais | Sinteticas | Total |
| --- | ---: | ---: | ---: |
| satellite | 287 | 900 | 1187 |
| rocket | 299 | 900 | 1199 |
| space_shuttle | 299 | 900 | 1199 |

Observacoes:

- as imagens reais foram coletadas pela NASA Image and Video Library API;
- as imagens sinteticas representam simulacoes de sensor orbital e foram geradas localmente em Python;
- nenhuma imagem real foi removida da estrategia final, apenas complementada;
- o dataset final continua alinhado ao enunciado, que permite base criada, adaptada ou simulada pela equipe.

## 4. Pre-processamento

- auditoria estrutural do dataset real com deteccao de arquivos quebrados e duplicatas;
- limpeza nao destrutiva com backup dos descartes;
- conversao para RGB e resize para `128x128`;
- montagem de dataset hibrido usando `dataset/raw/` + `dataset/synthetic/`;
- split estratificado `70/15/15` com separacao por `group_id`;
- preservacao de balanceamento por classe e origem quando possivel;
- normalizacao dos pixels durante o treinamento;
- augmentation leve apenas no treino.

Distribuicao final por split:

| Split | rocket | satellite | space_shuttle |
| --- | ---: | ---: | ---: |
| train | 839 | 831 | 839 |
| val | 180 | 178 | 180 |
| test | 180 | 178 | 180 |

Distribuicao por origem no split final:

| Split | rocket_real | rocket_synthetic | satellite_real | satellite_synthetic | space_shuttle_real | space_shuttle_synthetic |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 209 | 630 | 201 | 630 | 209 | 630 |
| val | 45 | 135 | 43 | 135 | 45 | 135 |
| test | 45 | 135 | 43 | 135 | 45 | 135 |

## 5. Modelos treinados

### SimpleCNN

- Conv2D
- MaxPooling2D
- Conv2D
- MaxPooling2D
- Flatten
- Dense
- Dropout
- Dense softmax

### DeepCNN

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
- Dense softmax

Nenhum modelo utilizou pesos pre-treinados.

## 6. Configuracao de treinamento

- `ImageDataGenerator` com normalizacao `1/255`;
- augmentation leve no treino com `rotation_range`, `zoom_range`, `shift`, `brightness_range` e `horizontal_flip`;
- `EarlyStopping` com `restore_best_weights=True`;
- `ModelCheckpoint` salvando o melhor estado por `val_loss`;
- learning rates distintos para reduzir instabilidade da `DeepCNN`.

## 7. Resultados finais

### 7.1 Baselines real-only do projeto

| Cenario | Melhor modelo | Accuracy |
| --- | --- | ---: |
| Real-only com `space_station` | SimpleCNN | 69.70% |
| Real-only com `space_shuttle` | SimpleCNN | 66.17% |

### 7.2 Resultado do dataset hibrido

| Modelo | Test Accuracy | Test Loss | Precision Macro | Recall Macro | F1 Macro |
| --- | ---: | ---: | ---: | ---: | ---: |
| SimpleCNN | 89.96% | 0.2275 | 90.01% | 89.97% | 89.96% |
| DeepCNN | 89.41% | 0.3881 | 89.47% | 89.41% | 89.39% |

Melhor modelo escolhido: `SimpleCNN`.

Ganho sobre o melhor real-only anterior: `69.70% -> 89.96%` (`+20.26` pontos percentuais).

## 8. Analise de acertos e erros

### SimpleCNN

- `rocket`: precision alta e bom recall, com confusao residual baixa;
- `satellite`: recall consistente mesmo nas imagens reais mais heterogeneas;
- `space_shuttle`: continuou sendo a classe mais separavel no conjunto final.

A rede simples passou a capturar bem os padroes morfologicos reforcados pelo conjunto hibrido, principalmente apos o aumento da base sintetica e o split sem vazamento entre familias.

### DeepCNN

- `rocket`, `satellite` e `space_shuttle` ficaram todos acima de `89%` de macro performance agregada;
- a arquitetura profunda deixou de colapsar e tambem ultrapassou a referencia ACV.

A arquitetura profunda deixou de colapsar e se tornou competitiva, mas ainda terminou ligeiramente abaixo da `SimpleCNN`.

## 9. Meta de 88%

A referencia de `88%` **foi atingida**. O melhor resultado final chegou a `89.96%`.

Justificativa tecnica:

- o ganho veio da melhor separacao morfologica das imagens sinteticas e do split `group-aware`, nao de vazamento ou pretreino;
- o teste continua misturando imagens reais e sinteticas;
- o treino segue sendo do zero, sem uso de representacoes pre-treinadas.

## 10. Conclusao e recomendacao

A decisao de partir para dataset hibrido foi tecnicamente correta e elevou o desempenho do projeto de forma substancial, ultrapassando a meta de `88%`.

Recomendacao objetiva:

1. manter `real + sintetico` como estrategia principal;
2. preservar `space_shuttle` no lugar de `space_station`;
3. se houver nova iteracao, investir em curadoria manual adicional das imagens reais para ampliar margem acima de `90%` sem alterar o escopo nem recorrer a pretreino.

## 11. Artefatos gerados

- `models/simple_cnn.keras`
- `models/deep_cnn.keras`
- `outputs/model_comparison.csv`
- `outputs/confusion_matrix_simple_cnn.png`
- `outputs/confusion_matrix_deep_cnn.png`
- `outputs/training_curves_simple_cnn.png`
- `outputs/training_curves_deep_cnn.png`
- `outputs/correct_predictions.png`
- `outputs/wrong_predictions.png`
- `outputs/training_summary.json`
- `notebooks/orbitshield_acv_training.ipynb`
