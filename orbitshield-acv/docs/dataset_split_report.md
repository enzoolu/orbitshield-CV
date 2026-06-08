# Relatorio de Split do Dataset

## Dataset Hibrido Final

| Classe | Real | Sintetica | Total |
| --- | ---: | ---: | ---: |
| rocket | 299 | 900 | 1199 |
| satellite | 287 | 900 | 1187 |
| space_shuttle | 299 | 900 | 1199 |

## Distribuicao por Split

| Split | rocket | satellite | space_shuttle |
| --- | ---: | ---: | ---: |
| test | 180 | 178 | 180 |
| train | 839 | 831 | 839 |
| val | 180 | 178 | 180 |

## Distribuicao por Origem e Split

| Split | rocket_real | rocket_synthetic | satellite_real | satellite_synthetic | space_shuttle_real | space_shuttle_synthetic |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| test | 45 | 135 | 43 | 135 | 45 | 135 |
| train | 209 | 630 | 201 | 630 | 209 | 630 |
| val | 45 | 135 | 43 | 135 | 45 | 135 |

## Integridade do Split

- Vazamentos de caminho entre splits: 0
- Vazamentos de grupo entre splits: 0
- As imagens sinteticas foram divididas por `group_id` de familia para evitar que variantes muito parecidas caiam em splits diferentes.

## Observacoes

- O dataset processado combina imagens reais em `dataset/raw/` com imagens simuladas em `dataset/synthetic/`.
- O split foi gerado por classe e origem, preservando grupos sinteticos inteiros para evitar vazamento.
- A distribuicao por classe foi mantida balanceada entre treino, validacao e teste.
- Falhas de processamento de imagem durante o resize: 0
