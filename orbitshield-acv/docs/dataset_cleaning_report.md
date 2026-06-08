# Relatorio de Limpeza do Dataset

- Backup das imagens removidas: `C:\Users\enzoo\PycharmProjects\orbitshield-CV\orbitshield-acv\dataset\backup\cleaned_20260607_204118`
- Registro detalhado das acoes: `outputs/clean_dataset_actions.csv`

## Resumo Geral

- Total de arquivos movidos: 0
- Removidos por quebra: 0
- Removidos por duplicata exata: 0
- Removidos por duplicata visual quase exata: 0
- Removidos por tamanho insuficiente: 0
- Removidos por baixa variancia: 0

## Resultado por Classe

| Classe | Antes | Depois | Removidos | Duplicatas exatas removidas | Duplicatas visuais removidas | Pequenas removidas | Baixa variancia removida |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rocket | 299 | 299 | 0 | 0 | 0 | 0 | 0 |
| satellite | 287 | 287 | 0 | 0 | 0 | 0 | 0 |
| space_shuttle | 299 | 299 | 0 | 0 | 0 | 0 | 0 |

## Observacoes

- A limpeza foi nao destrutiva: arquivos removidos foram movidos para backup.
- Duplicatas aproximadas foram removidas apenas quando a similaridade visual era praticamente total (hamming 0 e distancia de cor <= 1.0).
- Esta rodada manteve o dataset real-only e aplicou apenas filtros objetivos de qualidade estrutural.
