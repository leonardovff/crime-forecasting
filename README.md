# Predição de Crimes Violentos contra o Patrimônio com ConvLSTM

Pipeline de pré-processamento, treinamento e avaliação de um modelo ConvLSTM para predição espaço-temporal de crimes violentos contra o patrimônio nas cidades de Maceió e Arapiraca (AL), utilizando dados georreferenciados de 2012 a 2022.

## Estrutura do Projeto

### Notebooks

O pipeline é composto por 5 etapas sequenciais, replicadas para cada cidade:

| Etapa | Maceió | Arapiraca | Descrição |
|-------|--------|-----------|-----------|
| 1. Filtragem Espacial | `maceio-pre-processing-1.ipynb` | `arapiraca-pre-processing-1.ipynb` | Carrega o polígono municipal (OSM), aplica filtro espacial nas ocorrências, calcula o tamanho do grid (N) e gera a máscara de células válidas |
| 2. Criação do Grid | `maceio-pre-processing-2.ipynb` | `arapiraca-pre-processing-2.ipynb` | Atribui cada ocorrência a uma célula do grid N×N com base em latitude/longitude |
| 3. Séries Temporais | `maceio-pre-processing-3.ipynb` | `arapiraca-pre-processing-3.ipynb` | Gera matrizes de séries temporais (hora × célula) para cada tipo de crime |
| 4. Preparação dos Dados | `maceio-pre-trainning.ipynb` | `arapiraca-pre-trainning.ipynb` | Agrega por dia, binariza, extrai sub-grids 16×16, cria janelas deslizantes (lookback de 7 dias) e divide em treino/validação |
| 5. Treinamento e Avaliação | `maceio-trainning.ipynb` | `arapiraca-trainning.ipynb` | Treina o modelo ConvLSTM com BCEWithLogitsLoss (pos_weight), avalia com métricas tradicionais e com tolerância espacial (Chebyshev) |

Além destes, `pre-processing.ipynb` contém a etapa inicial de limpeza e padronização dos dados brutos (comum a ambas as cidades).

### Diretórios

| Diretório | Descrição |
|-----------|-----------|
| `input/` | Dados brutos de ocorrências criminais |
| `output/` | Dados intermediários e finais (CSVs filtrados, grids, séries temporais, datasets de treino/validação) |
| `predictions/` | Resultados das predições e métricas de avaliação |
| `scripts/` | Scripts auxiliares |
| `crime_deeping_learning/` | Projeto LaTeX do TCC |

### Parâmetros Principais

| Parâmetro | Maceió | Arapiraca |
|-----------|--------|-----------|
| Grid (N×N) | 73×73 | 57×57 |
| Células válidas | 3.116 (58,5%) | 1.896 (58,4%) |
| Ocorrências | 72.787 | 17.421 |
| Esparsidade | ~1,2% | ~0,8% |
| pos_weight | ~83,2 | ~121,7 |
| Sub-grid | 16×16 | 16×16 |
| Lookback | 7 dias | 7 dias |
| Granularidade | 24h | 24h |

## Troubleshooting

### GPU not available — `torch.cuda.is_available()` returns `False`

Symptoms:

```
UserWarning: CUDA initialization: CUDA unknown error ...
  return torch._C._cuda_getDeviceCount() > 0
Should return True if a GPU is available:  False
Number of GPUs available:  1
```

`device_count()` reads NVML (no runtime init) so it still reports `1`, while
`is_available()` fails during actual CUDA init with error `999`
(`CUDA_ERROR_UNKNOWN`). This is a driver-state issue, not a code bug —
typically after suspend/resume or a driver update leaves `nvidia_uvm` in a
broken state.

Fix: reload the UVM kernel module, then restart the notebook kernel.

```sh
sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm
```

If it still fails, reboot.
