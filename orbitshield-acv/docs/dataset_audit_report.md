# Relatorio de Auditoria do Dataset

## Resumo por Classe

| Classe | Arquivos encontrados | Imagens validas | Quebradas | Grupos de duplicata exata | Pares de duplicata aproximada | Muito pequenas | Baixa variancia |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rocket | 299 | 299 | 0 | 0 (0 excedentes) | 1 | 0 | 0 |
| satellite | 287 | 287 | 0 | 0 (0 excedentes) | 16 | 0 | 0 |
| space_shuttle | 299 | 299 | 0 | 0 (0 excedentes) | 0 | 0 | 0 |

## Regras Aplicadas

- Duplicata exata: hash SHA-256 identico.
- Duplicata aproximada: perceptual hash (dHash) com distancia de Hamming <= 4 e media de cor semelhante.
- Muito pequena: menor lado < 128 px.
- Baixa variancia: desvio padrao do canal em escala de cinza < 8.0.

## Observacoes por Classe

### rocket

- Contact sheet: `outputs/contact_sheet_rocket.png`
- Quebradas: 0
- Duplicatas exatas: 0 grupos
- Duplicatas aproximadas: 1 pares. Exemplo: `ksc_20200730_ph_tey01_0003.jpg` vs `ksc_20200730_ph_tey01_0004.jpg` (hamming=4, cor=2.03)
- Muito pequenas: 0 | Baixa variancia: 0

### satellite

- Contact sheet: `outputs/contact_sheet_satellite.png`
- Quebradas: 0
- Duplicatas exatas: 0 grupos
- Duplicatas aproximadas: 16 pares. Exemplo: `2014_1212.jpg` vs `2014_1213.jpg` (hamming=2, cor=2.54)
- Muito pequenas: 0 | Baixa variancia: 0

### space_shuttle

- Contact sheet: `outputs/contact_sheet_space_shuttle.png`
- Quebradas: 0
- Duplicatas exatas: 0 grupos
- Duplicatas aproximadas: 0 pares
- Muito pequenas: 0 | Baixa variancia: 0

