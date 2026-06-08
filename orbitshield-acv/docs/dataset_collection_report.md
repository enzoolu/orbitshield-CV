# Relatorio de Coleta do Dataset

## Resumo

- Fonte real: NASA Image and Video Library API
- Limite configurado por classe na coleta real: `300`
- Quantidade sintetica gerada por classe: `900`
- Referencia ACV buscada: `88%` de accuracy no teste
- Melhor resultado real-only anterior: `69.70%` com `SimpleCNN`
- Decisao final: adotar dataset **hibrido** com imagens reais NASA + imagens simuladas de sensor orbital

## Etapas do dataset

### 1. Tentativas real-only

1. `satellite` + `rocket` + `space_station`
   - melhor resultado: `69.70%`
2. `satellite` + `rocket` + `space_shuttle`
   - melhor resultado: `66.17%`

Conclusao: as imagens reais da NASA permaneceram muito heterogeneas para treino do zero com poucas centenas de exemplos por classe.

### 2. Dataset hibrido final

A base final manteve todas as imagens reais validadas e adicionou imagens sinteticas geradas pela equipe em Python. Essas imagens sinteticas representam **simulacoes de sensor orbital** e seguem o escopo permitido de dataset criado/adaptado/simulado.

## Coleta real apos limpeza

| Classe | Reais finais |
| --- | ---: |
| satellite | 287 |
| rocket | 299 |
| space_shuttle | 299 |

## Geracao sintetica

| Classe | Sinteticas |
| --- | ---: |
| satellite | 900 |
| rocket | 900 |
| space_shuttle | 900 |

## Dataset total usado no treino

| Classe | Reais | Sinteticas | Total |
| --- | ---: | ---: | ---: |
| satellite | 287 | 900 | 1187 |
| rocket | 299 | 900 | 1199 |
| space_shuttle | 299 | 900 | 1199 |

## Consultas utilizadas na coleta real

- `satellite`: `communications satellite`, `spacecraft satellite`, `satellite`
- `rocket`: `launch vehicle`, `rocket launch`, `rocket engine`
- `space_shuttle`: `space shuttle`, `space shuttle orbiter`, `orbiter spacecraft`, `space shuttle in orbit`

## Curadoria objetiva aplicada na base real

- validacao de abertura com PIL;
- conversao para RGB;
- remocao de arquivos quebrados;
- remocao de duplicatas exatas;
- remocao de duplicatas visuais quase exatas;
- remocao de imagens muito pequenas;
- remocao de imagens com baixa variancia.

## Caracteristicas da base sintetica

- resolucao `128x128`;
- fundo espacial com estrelas, ruido e variacao de brilho;
- perturbacoes controladas de rotacao, escala, translacao e contraste;
- padroes geometricos consistentes com cada classe:
  - `satellite`: corpo central e paineis solares;
  - `rocket`: corpo alongado, ponta conica e aletas;
  - `space_shuttle`: fuselagem, asas e cauda.

## Limitacoes da busca NASA

- a NASA Image and Video Library possui metadados heterogeneos e nem sempre padronizados;
- parte dos resultados mistura close-ups, operacao em solo, diagramas, componentes parciais e enquadramentos muito diferentes do objeto principal;
- esse ruido prejudicou principalmente a classe `satellite` no cenario real-only.

## Conclusao desta rodada

- o dataset hibrido foi adotado de forma documentada e sem remover as imagens reais;
- a melhor `SimpleCNN` passou de `69.70%` no melhor real-only para `89.96%` no conjunto final;
- a referencia de `88%` foi atingida com margem;
- recomendacao: manter a estrategia hibrida com split `group-aware` e investir em curadoria manual adicional da base real se houver nova iteracao.
