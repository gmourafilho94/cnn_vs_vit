# Arquitetura do projeto

## 1. Estrutura do repositório

```
cnn_vs_vit/
├── configs/
│   ├── base.yaml            dados, receita de treino e medição (comum aos dois modelos)
│   ├── models/cnn.yaml      hiperparâmetros só da arquitetura CNN
│   ├── models/vit.yaml      hiperparâmetros só da arquitetura ViT
│   └── energy.yaml          constantes de energia e carbono (usadas só na análise)
├── src/                     (implementação do usuário; nomes indicativos)
│   ├── data/                fetch (download), preprocess (divisão e redimensionamento), carregamento em GPU
│   ├── models/              cnn, vit, fábrica que escolhe pelo nome
│   ├── energy/              telemetria, metadados do nó, rastreador de fases, CCF, carbono ONS
│   └── train                ponto de entrada do treino
├── slurm/                   job de treino (array), job de análise, script que encadeia os dois, smoke test
├── scripts/                 preparação no nó de login (ambiente e dados)
├── analysis/                cálculo de energia a partir dos dados brutos, comparação e gráficos
├── data/                    (fora do git) raw/, processed/, ons/
├── results/                 (fora do git) um diretório por execução
└── logs/                    (fora do git) saídas do SLURM
```

**Princípio:** separar a coleta (job de treino) do cálculo (análise). O job grava apenas dados brutos: telemetria, eventos de fase e metadados do nó. Energia e carbono são calculados depois, a partir do `energy.yaml`. Assim, corrigir uma constante não exige treinar de novo.

## 2. Dados

### 2.1 Origem

- Repositório no Hugging Face: `updatebao/geonamebase_1`, um único split `train`, no formato *imagefolder* (uma pasta por classe).
- 90 classes, cerca de 1.000 imagens por classe, JPEG 1024×1024 RGB, cerca de 13,5 GB no total.
- O dataset não tem *dataset card* nem licença declarada. Isso deve ser registrado no trabalho.

### 2.2 Seleção (nó de login)

1. Listar as pastas de classe e os arquivos de cada pasta pela API do Hub, sem baixar tudo.
2. Para cada classe, ordenar os arquivos e sortear 100 com semente fixa (`selection_seed`).
3. Gravar a lista escolhida em `data/raw/selection.csv`. Esse arquivo é a fonte da verdade e deve ser versionado ou anexado aos resultados.
4. Baixar só os arquivos selecionados (cerca de 9.000 arquivos, aproximadamente 1 GB).

### 2.3 Pré-processamento (nó de login)

- Divisão estratificada por classe: **80 treino / 10 validação / 10 teste**, com semente fixa. Total: 7.200 / 900 / 900.
- Recorte central quadrado e redimensionamento bicúbico para **224×224**.
- Saída em tensores `uint8` (N, 3, 224, 224), um arquivo por split, mais os rótulos, um `manifest.csv` (caminho, classe, índice, split) e um `classes.json`.
- Tamanho aproximado: treino 1,08 GB, validação e teste 0,14 GB cada.

**Por que tensores prontos:** a decodificação de JPEGs de 1024 px na CPU dominaria o tempo e a energia do treino e mediria o *dataloader*, não o modelo. Com tensores prontos, o conjunto de treino inteiro cabe na memória da H100 (80 GB). Os dois modelos leem os mesmos bytes, e o custo de entrada é praticamente nulo e igual para ambos.

### 2.4 Aumento de dados (na GPU, dentro do treino)

- *Random resized crop* por amostra, com escala entre 0,35 e 1,0, devolvido em 224×224. Pode ser feito em lote com `roi_align` e uma caixa aleatória por imagem.
- *Horizontal flip* por amostra, com probabilidade 0,5.
- Normalização pela média e desvio padrão calculados no conjunto de treino.
- Validação e teste: sem aumento, só normalização.

## 3. Modelos

Os dois modelos são implementados do zero em PyTorch, sem pesos pré-treinados e sem bibliotecas de modelos prontos. A meta é que fiquem na mesma faixa de custo computacional por imagem.

### 3.1 CNN: residual, estilo ResNet-18

| Bloco | Detalhe | Saída (224 px) |
|---|---|---|
| Stem | Conv 7×7/2 (64) → BN → ReLU → MaxPool 3×3/2 | 64 × 56 × 56 |
| Estágio 1 | 2 blocos básicos, 64 canais | 64 × 56 × 56 |
| Estágio 2 | 2 blocos básicos, 128 canais, primeiro com stride 2 | 128 × 28 × 28 |
| Estágio 3 | 2 blocos básicos, 256 canais, stride 2 | 256 × 14 × 14 |
| Estágio 4 | 2 blocos básicos, 512 canais, stride 2 | 512 × 7 × 7 |
| Cabeça | Global average pooling → Linear(512, 90) | 90 |

Bloco básico: Conv 3×3 → BN → ReLU → Conv 3×3 → BN, somado ao atalho (projeção 1×1 + BN quando o número de canais ou o stride mudam) e seguido de ReLU.

Aproximadamente **11,2 M parâmetros** e **1,8 GFLOPs** (MACs) por imagem.

### 3.2 ViT: estilo ViT-Tiny/16

| Componente | Detalhe |
|---|---|
| Patch embedding | Conv 16×16 com stride 16 → 196 tokens de dimensão 192 |
| Token de classe + posição | Token `[CLS]` aprendido e embedding posicional aprendido (197 × 192) |
| Encoder | 12 blocos *pre-norm*: LN → MHSA (3 cabeças) → residual; LN → MLP (192→768→192, GELU) → residual |
| Cabeça | LN → Linear(192, 90) aplicado ao `[CLS]` |

Aproximadamente **5,5 M parâmetros** e **1,3 GFLOPs** (MACs) por imagem.

### 3.3 O que registrar sobre cada modelo

- Parâmetros contados no modelo instanciado (não os valores teóricos acima).
- FLOPs medidos por contador de operações numa passada *forward* de uma imagem.
- Throughput de treino (imagens/s) por época.

Essas medidas permitem normalizar a energia: J por imagem processada, J por GFLOP e energia até atingir certa acurácia.

## 4. Receita de treino (idêntica para os dois modelos)

| Item | Valor |
|---|---|
| Épocas | 100 (fixo) |
| Batch | 128 |
| Otimizador | AdamW, lr 1e-3, weight decay 0,05 |
| Agenda | Warmup linear de 5 épocas + cosseno até zero |
| Perda | Entropia cruzada com *label smoothing* 0,1 |
| Precisão | Mista bf16 (autocast) |
| Regularização extra | Nenhuma em nenhum dos dois (sem dropout, *drop path*, mixup) |
| Compilação | Sem `torch.compile` (o tempo de compilação adicionaria um custo energético que não é do modelo) |
| Seeds | 5 por modelo (0–4) |
| Threads de CPU | Fixo (`num_threads`), igual a `--cpus-per-task` |

**Ponto de atenção:** ViTs treinados do zero com 80 imagens por classe tendem a ter acurácia bem menor que CNNs, porque não têm o viés indutivo de localidade. Como a receita é igual para os dois, essa diferença é parte do resultado e deve ser discutida, não corrigida com receitas diferentes.

## 5. Fases de uma execução

Cada execução (um modelo, uma seed) passa por fases marcadas com carimbo de tempo. A telemetria anota em qual fase cada amostra foi coletada.

| Fase | Medida? | Descrição |
|---|---|---|
| `setup` | Não | Carregar tensores na GPU, construir o modelo, contar parâmetros e FLOPs |
| `idle_baseline` | Referência | 60 s com a GPU alocada e ociosa; dá a potência mínima da GPU e da CPU no nó |
| `train` | **Sim** | Uma por época: passadas *forward*/*backward* e passo do otimizador |
| `val` | Separada | Uma por época, logo após o treino; reportada à parte (pode entrar ou não no total) |
| `test` | Não | Uma vez ao final, só para a acurácia |

As fronteiras de fase só são registradas depois de `torch.cuda.synchronize()`. Sem isso, o tempo de uma fase "vaza" para a seguinte, porque a execução na GPU é assíncrona.

## 6. Pipeline no NPAD (SLURM)

### 6.1 No nó de login

1. Criar o ambiente Python no `$HOME` (venv ou conda), instalando o PyTorch com CUDA compatível com o driver dos nós H100. A versão do driver é verificada no smoke test.
2. Baixar o subconjunto (seção 2.2) e pré-processar (seção 2.3).
3. Baixar os arquivos mensais do ONS do período dos jobs (pode ficar para depois do treino; ver a metodologia).

### 6.2 Smoke test

Job curto na partição `gpu-8-h100`, com 1 GPU e 1 época por modelo. Verifica:
- `nvidia-smi` e a versão do driver;
- `torch.cuda.is_available()` e o nome da GPU;
- acesso ao NVML e leitura do contador de energia;
- leitura dos tensores e escrita em `results/`;
- se `sacct` devolve `ConsumedEnergy` (indica se a contabilização de energia do SLURM está ativa).

### 6.3 Job de treino (job array)

| Diretiva | Valor proposto | Motivo |
|---|---|---|
| `--partition` | `gpu-8-h100` | Decisão do projeto |
| `--gpus-per-node` | 1 | Os modelos são pequenos; 1 GPU por execução |
| `--cpus-per-task` | 8 | Fixo para as duas arquiteturas; entra no termo de CPU do CCF |
| `--mem` | 64G | O treino usa pouco; o valor alocado é registrado |
| `--time` | a definir após o smoke test | |
| `--array` | `0-9%1` | 10 tarefas (2 modelos × 5 seeds), **uma por vez** |
| `--output` | `logs/train_%A_%a.out` | `%A` = id do array, `%a` = índice |

**Mapeamento do índice:** `modelo = [cnn, vit][idx % 2]`, `seed = idx // 2`. Os modelos ficam intercalados (cnn s0, vit s0, cnn s1, …), de modo que variações ao longo do dia (temperatura da sala, outros usuários no nó, intensidade de carbono) afetam os dois por igual.

**Por que `%1` (sequencial):** com várias tarefas ao mesmo tempo no mesmo nó, elas disputariam CPU, memória e PCIe, e a medição de uma contaminaria a outra.

**Sem `--exclusive`:** só existe um nó H100 com 8 GPUs, e reservá-lo inteiro bloquearia outros usuários. A consequência é que métricas no nível do nó (CPU total, rede) podem incluir outros jobs. Por isso o CCF usa as métricas **do processo** e **da GPU alocada**, e as do nó são guardadas só como contexto (ver a metodologia).

### 6.4 Job de análise (dependência)

- Submetido com `--dependency=afterany:<id_do_array>`, roda quando todas as tarefas terminarem. Usa `afterany` em vez de `afterok` para analisar também execuções que falharam.
- Partição de CPU barata (por exemplo `amd-512`), sem GPU, tempo curto.
- Executa `sacct` para todas as tarefas do array e grava a saída por execução.
- Executa o cálculo de energia e carbono e a comparação final.

### 6.5 Encadeamento

Um script de submissão:
1. submete o array e captura o id (`sbatch --parsable`);
2. submete a análise com a dependência nesse id.

Assim, uma única chamada no nó de login dispara o pipeline inteiro.

## 7. Resultados por execução

```
results/<modelo>/seed<k>/<jobid>_<taskid>/
├── config.json        configuração efetiva (base + modelo), seed, commit do git
├── node.json          metadados do nó (seção 4 da metodologia)
├── telemetry.csv      amostras a cada 1 s (seção 3 da metodologia)
├── phases.jsonl       início e fim de cada fase, energia NVML acumulada nas fronteiras
├── epochs.jsonl       por época: loss, acc de treino e validação, lr, imagens/s, duração
├── model.json         parâmetros, FLOPs
├── sacct.txt          gravado pelo job de análise
└── energy.json        gravado pelo job de análise (seção 7 da metodologia)
```

E na raiz de `results/`: `summary.csv` (uma linha por execução) e os gráficos da comparação.

## 8. Saídas da comparação

- Energia de treino total (kWh) por modelo: média ± desvio padrão nas 5 seeds, decomposta em CPU, memória, GPU, armazenamento e rede.
- Energia por época, por imagem processada e por GFLOP de treino.
- Curva de acurácia de validação × energia acumulada; energia até atingir X% de acurácia.
- CO₂e operacional e embutido.
- Estimativa CCF da GPU × medição do NVML (erro do modelo do artigo quando aplicado a GPU).
- Acurácia final de teste.
