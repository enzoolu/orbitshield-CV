# Model Card

## Modelo

- Nome: `SimpleCNN`
- Arquivo: `models/simple_cnn.keras`
- Status: melhor modelo desta execucao final hibrida

## Finalidade de uso

Classificacao de imagens de sistemas espaciais em tres classes:

- `satellite`
- `rocket`
- `space_shuttle`

## Dataset

### Base real apos limpeza

- `rocket`: `299`
- `satellite`: `287`
- `space_shuttle`: `299`

### Base sintetica gerada

- `rocket`: `900`
- `satellite`: `900`
- `space_shuttle`: `900`

### Dataset total usado no treino

- `rocket`: `1199`
- `satellite`: `1187`
- `space_shuttle`: `1199`
- total global: `3585`

### Split final

- treino: `839 / 831 / 839`
- validacao: `180 / 178 / 180`
- teste: `180 / 178 / 180`

## Metricas principais

- Test Accuracy: `89.96%`
- Test Loss: `0.2275`
- Precision Macro: `90.01%`
- Recall Macro: `89.97%`
- F1 Macro: `89.96%`

## Baselines de referencia

- Melhor real-only anterior: `69.70%`
- Melhor real-only com `space_shuttle`: `66.17%`
- Melhor hibrido final: `89.96%`

## Limitacoes conhecidas

- parte dos erros restantes ainda ocorre em imagens reais da NASA com enquadramento ambiguo;
- o teste continua dependente de uma composicao hibrida entre dados reais e sinteticos;
- nao indicado para uso operacional real sem nova iteracao de curadoria e avaliacao externa.
