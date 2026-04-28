# Predição de Crimes Violentos contra o Patrimônio com ConvLSTM

Pipeline de pré-processamento, treinamento e avaliação de um modelo ConvLSTM para predição espaço-temporal de crimes violentos contra o patrimônio nas cidades de Maceió e Arapiraca (AL), utilizando dados georreferenciados de 2012 a 2022 cedidos pela Polícia Militar de Alagoas. O trabalho replica metodologicamente, em contexto urbano brasileiro, o pipeline proposto por Albors Zumel, Tizzoni & Campedelli (2025) — *Deep Learning for Crime Forecasting: The Role of Mobility at Fine-grained Spatiotemporal Scales*, *Journal of Quantitative Criminology* — cujo PDF e código de referência estão em `original/`.

## Estrutura do Projeto

### Notebooks

O pipeline é composto por 5 etapas sequenciais, replicadas para cada cidade:

| Etapa | Maceió | Arapiraca | Descrição |
|-------|--------|-----------|-----------|
| 0. Limpeza inicial | `pre-processing.ipynb` (compartilhado) | | Lê os CSVs brutos em `./input/`, converte datas e coordenadas, filtra as 8 naturezas de roubo violento contra o patrimônio e separa por cidade |
| 1. Filtragem Espacial | `maceio-pre-processing-1.ipynb` | `arapiraca-pre-processing-1.ipynb` | Carrega o polígono municipal via OpenStreetMap (osmnx), calcula o tamanho do grid `N` para área-alvo de ~0,2 km² por célula, gera a máscara de células válidas e filtra ocorrências fora do polígono |
| 2. Indexação no Grid | `maceio-pre-processing-2.ipynb` | `arapiraca-pre-processing-2.ipynb` | Atribui cada ocorrência a uma célula do grid `N×N` por *point-in-polygon* (paralelizado com `pandarallel`) |
| 3. Séries Temporais | `maceio-pre-processing-3.ipynb` | `arapiraca-pre-processing-3.ipynb` | Constrói uma matriz `célula × hora_do_ano` para cada ano (2012–2022), agregando todas as naturezas selecionadas em um único indicador, e concatena em um CSV por cidade |
| 4. Preparação dos Dados | `maceio-pre-trainning.ipynb` | `arapiraca-pre-trainning.ipynb` | Agrega em janelas de 12h, binariza (≥1 crime → 1), aplica janela deslizante de 29 passos (28 de *lookback* = 14 dias + 1 alvo), extrai 5 sub-grids 16×16 por amostra (com ≥1 ocorrência no alvo) e divide cronologicamente 90/10 com *buffer* de 28 passos |
| 5. Treinamento e Avaliação | `maceio-trainning.ipynb` | `arapiraca-trainning.ipynb` | Treina a ConvLSTM (3 camadas, `hidden_chn=28`, kernel 3×3, dropout 0,8, BCEWithLogitsLoss com `pos_weight`), salva predições/alvos/perdas em `./output/{cidade}/predictions/` e varre 51 limiares de classificação calculando *precision*, *recall* e F1 nas variantes tradicional e com tolerância espacial (Chebyshev ≤ 1) |

### Diretórios

| Diretório | Descrição |
|-----------|-----------|
| `input/` | Dados brutos de ocorrências criminais (CSVs separados por `;`) |
| `output/maceio/`, `output/arapiraca/` | Dados intermediários e finais por cidade: CSVs filtrados, máscara, GeoJSON do polígono, matriz célula × hora, `*_chrono.npz` (datasets de treino/validação), `predictions/` (predições, alvos, perdas, índices de sub-grid) e `perf_thrs/` (métricas por limiar) |
| `crime_deeping_learning/` | Projeto LaTeX do TCC (capítulos, bibliografia e figuras) |
| `original/` | Material de referência: PDF do artigo de Albors Zumel et al. (2025), código original de pré-processamento e treinamento, e GeoJSONs das cidades estudadas no paper |
| `cache/`, `scripts/` | Cache da execução paralelizada e scripts auxiliares |

### Documentação complementar

| Arquivo | Conteúdo |
|---------|----------|
| `workflow.md` | Descrição passo a passo do pipeline (formato técnico, com formas de tensores e fluxo de dados) |
| `convLSTM.md` | Notas conceituais sobre a arquitetura ConvLSTM |
| `funcionamento.md` | Visão geral do funcionamento do projeto |

## Configuração do Pipeline

### Parâmetros temporais e espaciais

| Parâmetro | Valor | Observação |
|-----------|-------|------------|
| Granularidade temporal | 12h | 2 passos por dia (paper: 12h) |
| *Lookback* | 14 dias (28 passos) | `seq_length = 29` (28 + alvo) |
| Área-alvo por célula | ~0,2 km² | Equivalente a ~0,077 sq mi do paper |
| Sub-grid | 16×16 | Padronização entre cidades + viabilidade computacional |
| Sub-grids por amostra | 5 | Aleatórios, com ≥1 ocorrência no alvo |
| *Split* treino/validação | 90/10 cronológico | *Buffer* de 28 passos para evitar vazamento |

### Por cidade

| Parâmetro | Maceió | Arapiraca |
|-----------|--------|-----------|
| Grid (`N × N`) | 73 × 73 | 57 × 57 |
| Células válidas (intra-município) | 2.714 (50,9%) | 1.896 (58,4%) |
| Ocorrências (após filtragem espacial) | 72.649 | 17.356 |
| Amostras de treino (sub-grids) | 32.556 | 27.590 |
| Amostras de validação (sub-grids) | 3.238 | 2.484 |
| Esparsidade do conjunto de treino | ~0,80% | ~0,61% |
| `pos_weight` | ~123,4 | ~163,3 |

### Hiperparâmetros do treinamento

| Parâmetro | Valor |
|-----------|-------|
| Arquitetura | 3 camadas ConvLSTM, kernel 3×3, BatchNorm3d + ReLU entre camadas |
| Canais ocultos (`hidden_chn`) | 28 (= `seq_length - 1`) |
| Dropout | 0,8 |
| Otimizador | Adam, `lr = 1×10⁻⁵` |
| Escalonamento | `CosineAnnealingLR` com `T_max = 20` |
| Função de perda | `BCEWithLogitsLoss` ponderada por `pos_weight` |
| `batch_size` | 252 |
| Épocas | 200 |
| Semente | 42 |

## Como executar

Os notebooks usam dependências gerenciadas por [Pixi](https://pixi.sh) (ver `pixi.toml` e `pixi.lock`):

```sh
pixi install
pixi run jupyter lab
```

Ordem de execução para uma cidade (substitua `{cidade}` por `maceio` ou `arapiraca`):

1. `pre-processing.ipynb` (uma vez, gera `filtered_*.csv` para ambas as cidades)
2. `{cidade}-pre-processing-1.ipynb`
3. `{cidade}-pre-processing-2.ipynb`
4. `{cidade}-pre-processing-3.ipynb`
5. `{cidade}-pre-trainning.ipynb`
6. `{cidade}-trainning.ipynb`

## Troubleshooting

### GPU not available — `torch.cuda.is_available()` returns `False`

Sintomas:

```
UserWarning: CUDA initialization: CUDA unknown error ...
  return torch._C._cuda_getDeviceCount() > 0
Should return True if a GPU is available:  False
Number of GPUs available:  1
```

`device_count()` lê NVML (sem inicializar o runtime CUDA), então ainda reporta `1`, enquanto `is_available()` falha na inicialização do CUDA com erro `999` (`CUDA_ERROR_UNKNOWN`). Geralmente é estado quebrado do driver após *suspend/resume* ou atualização, e não bug do código.

Solução: recarregar o módulo de kernel UVM e reiniciar o kernel do notebook.

```sh
sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm
```

Se persistir, reinicie a máquina.

### CUDA out of memory durante o treinamento

A configuração atual (`hidden_chn = 28`, `batch_size = 252`, sub-grids 16×16) consome próximo do limite de uma RTX 4070 Laptop (8 GB). Caminhos de mitigação, em ordem de menor impacto:

1. Verificar se há sessões antigas de Jupyter retendo a GPU (`nvidia-smi` para checar PIDs e `kill <pid>` para limpar);
2. Habilitar segmentos expansíveis: `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` antes de iniciar o kernel;
3. Reduzir `batch_size` (ex.: 252 → 128 ou 64). Não muda o resultado final, mas aumenta o tempo de treino;
4. Reduzir `hidden_chn` (afasta a configuração da equivalência com o paper de referência).
