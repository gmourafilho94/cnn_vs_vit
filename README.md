# CNN vs ViT: comparativo energético de treino

## O que é o projeto

Este repositório compara o **custo energético de treinar do zero** duas famílias de redes para classificação de imagens:

- uma **rede convolucional (CNN)** residual, no estilo ResNet-18;
- um **Vision Transformer (ViT)**, no estilo ViT-Tiny/16.

As duas resolvem o mesmo problema, com os mesmos dados, a mesma receita de treino e o mesmo hardware. A única variável é a arquitetura. A pergunta central é: **quanta energia cada arquitetura consome para ser treinada, e qual a emissão de carbono correspondente**, e como isso se relaciona com a acurácia obtida.

O foco é energético. A acurácia é registrada para dar contexto ao custo (por exemplo, energia gasta até atingir certa acurácia), mas não é o objetivo de otimização.

| Item | Definição |
|---|---|
| Dataset | [`updatebao/geonamebase_1`](https://huggingface.co/datasets/updatebao/geonamebase_1): 99.927 JPEGs 1024×1024, 90 classes (`geo10_1` … `geo99_1`) |
| Subconjunto | 100 imagens por classe = 9.000 imagens (80 treino / 10 validação / 10 teste por classe) |
| Modelos | CNN residual × ViT, ambos implementados e treinados do zero, sem pesos pré-treinados |
| Fase medida | Somente o treino |
| Infraestrutura | [NPAD/UFRN](https://npad.ufrn.br/npad/hardware), partição `gpu-8-h100` (2× AMD EPYC 9654, 8× NVIDIA H100 80 GB SXM5), via SLURM |
| Ambiente virtual | Cada nó alocado a um job de treino |
| Método energético | Cloud Carbon Footprint (CCF), como em [Saavedra et al., arXiv:2510.26413](https://arxiv.org/abs/2510.26413), estendido com um termo de GPU |
| Região de carbono | South America: intensidade horária calculada com dados abertos do ONS, subsistema Nordeste (o NPAD fica em Natal/RN) |

## Como será desenvolvido

### Etapa 1: preparação (nó de login do NPAD, fora da medição)

1. **Ambiente:** ambiente Python no `$HOME` com PyTorch compatível com o driver CUDA dos nós H100.
2. **Seleção dos dados:** listar os arquivos de cada classe no Hugging Face, sortear 100 por classe com semente fixa e baixar só esses (cerca de 1 GB, em vez dos 13,5 GB). A lista escolhida fica salva e versionada.
3. **Pré-processamento:** dividir em treino, validação e teste de forma estratificada, reduzir para 224×224 e salvar como tensores `uint8`. Assim a decodificação de JPEG sai do laço de treino, e os dois modelos leem exatamente os mesmos bytes, direto da memória da GPU.

### Etapa 2: modelos e receita de treino

- **CNN:** stem convolucional + 4 estágios de blocos residuais básicos (64→128→256→512 canais) + pooling global. Cerca de 11,2 M parâmetros e 1,8 GFLOPs por imagem.
- **ViT:** patches 16×16 (196 tokens), dimensão 192, 12 blocos de encoder, 3 cabeças de atenção, token `[CLS]`. Cerca de 5,5 M parâmetros e 1,3 GFLOPs por imagem.
- **Receita idêntica para os dois:** 100 épocas, batch 128, AdamW (lr 1e-3, weight decay 0,05), warmup de 5 épocas + cosseno, label smoothing 0,1, precisão mista bf16 e o mesmo aumento de dados (recorte aleatório + flip, feitos na GPU).
- **Repetições:** 5 seeds por modelo, com resultados reportados como média ± desvio padrão.

### Etapa 3: execução no SLURM

```
nó de login                                nó gpu-8-h100
───────────                                ──────────────────────────────────────────────
submete o pipeline ──► job array 0-9%1 ──► cnn s0, vit s0, cnn s1, vit s1, … (uma por vez)
                         │                 cada tarefa: metadados do nó → 60 s ocioso
                         │                              → treino com telemetria → dados brutos
                         ▼
                       job de análise (--dependency=afterany) ──► sacct + CCF + ONS → relatório
```

- **Job array** com 10 tarefas (2 modelos × 5 seeds), 1 GPU e 8 CPUs cada. As tarefas rodam **uma por vez** (`%1`), para que nenhuma interfira na medição da outra, e os modelos são **intercalados**, para que variações ao longo do tempo afetem os dois igualmente.
- **Job de análise** encadeado por dependência: começa sozinho quando o array termina, coleta o `sacct` de cada tarefa e calcula energia e carbono.
- Uma única submissão no nó de login dispara o pipeline inteiro.

### Etapa 4: análise e comparação

Energia total e por componente, energia por época, por imagem e por GFLOP, curva de acurácia × energia acumulada, CO₂e e a comparação entre a estimativa do CCF e a medição real da GPU.

## Como será feita a medição

O escopo é **energia e carbono**. A pegada hídrica, que o artigo também calcula, está fora do escopo deste projeto.

### Princípio

A medição segue o método do artigo de referência, que **estima** a energia a partir de **telemetria de uso** do ambiente virtual com o modelo Cloud Carbon Footprint, em vez de medir a energia com instrumentos. Aqui, o "ambiente virtual" é o nó do NPAD alocado ao job.

A coleta e o cálculo são separados:

- **Durante o treino**, o job grava só dados brutos: telemetria, fronteiras de fase e metadados do nó.
- **Depois**, o job de análise aplica as fórmulas e as constantes de `configs/energy.yaml`. Corrigir uma constante não exige treinar de novo.

### O que é coletado em cada job

| Fonte | O quê | Quando |
|---|---|---|
| Metadados do nó | hostname, `lscpu`, memória, `nvidia-smi -q` da GPU alocada, variáveis `SLURM_*`, `scontrol show job`, versões de software | Início do job |
| Telemetria (a cada 1 s) | % de CPU e memória RSS do processo de treino; % de uso, memória, potência e energia acumulada da GPU alocada (NVML); CPU, memória e rede do nó (contexto) | Durante todo o job |
| Fronteiras de fase | Início e fim de cada fase (`idle_baseline`, `train`, `val`, `test`), registrados após sincronizar a GPU, com a leitura do contador de energia do NVML | Em cada fronteira |
| `sacct` | Elapsed, TotalCPU, MaxRSS, NodeList, ConsumedEnergy (se habilitado no NPAD) | Job de análise |

Só a fase `train` compõe o resultado principal. `val` é reportada à parte, e `idle_baseline` serve de referência de potência mínima.

### Fórmulas (CCF, como no artigo, mais o termo de GPU)

Para cada fase, com duração `t` em horas:

```
E_cpu     = (vCPU_min × N_vCPU + (vCPU_max − vCPU_min) × U_vCPU) × t
E_mem     = 0,000392 kWh/(GB·h) × memória média (GB) × t
E_gpu     = (P_min + (P_max − P_min) × U_gpu) × N_gpu × t          ← extensão (o artigo não tem GPU)
E_storage = 0,0012 kWh/(TB·h) × armazenamento reservado (TB) × t
E_net     = 0,001 kWh/GB × tráfego de rede (GB)

E_total   = (E_cpu + E_mem + E_gpu + E_storage + E_net) × PUE
```

- `N_vCPU` = CPUs alocadas pelo SLURM; `U_vCPU` = uso médio do processo em vCPUs (% médio × N_vCPU).
- `vCPU_min` / `vCPU_max` = 0,739 W / 2,282 W: coeficientes do CCF para **EPYC 4ª geração** (a do EPYC 9654 do nó H100). O artigo usou 0,434 W / 1,948 W, que são de EPYC 3ª geração.
- `P_min` da GPU = potência medida na fase ociosa; `P_max` = 700 W (TDP da H100 SXM5); `U_gpu` = uso médio da GPU pelo NVML.
- PUE = 1,17 (valor do artigo para as Américas, até o NPAD informar o real).

### Carbono

```
CO2_operacional = E_total × intensidade de carbono (hora de início do job)
CO2_embutido    = 1,61079 tCO2e × (t / 4 anos) × (vCPUs alocadas / vCPUs do nó)
```

A **intensidade de carbono** da hora é calculada com o dataset aberto do ONS *Geração por Usina em Base Horária*, filtrado para o subsistema **Nordeste**:

```
intensidade(h) = Σ (geração_fonte(h) × fator_fonte) / Σ geração_fonte(h)
```

Os fatores por fonte (gCO₂e/kWh) são as medianas de ciclo de vida do IPCC AR5, as mesmas usadas pelo Electricity Maps, que é a fonte do artigo: hidráulica 24, eólica 11, solar 48, gás 490, carvão 820, biomassa 230, óleo 650, nuclear 12.

### Validação

O CCF é um modelo de estimativa. Como o nó tem GPUs NVIDIA, o contador de energia do NVML dá a **energia real da GPU** em cada fase. Comparar as duas mede o erro do método do artigo quando estendido para GPU, e esse erro é um resultado do trabalho. Se o NPAD tiver contabilização de energia no SLURM, o `ConsumedEnergy` do `sacct` serve como segunda referência.

## Documentação detalhada

| Documento | Conteúdo |
|---|---|
| [docs/arquitetura.md](docs/arquitetura.md) | Estrutura do repositório, dados, modelos camada a camada, receita de treino, fases, diretivas SLURM, formato dos resultados |
| [docs/metodologia_energia.md](docs/metodologia_energia.md) | Fórmulas, constantes e suas fontes, telemetria, carbono via ONS, validação |
| [docs/pendencias.md](docs/pendencias.md) | Decisões em aberto, valores a confirmar e riscos conhecidos |

## Referências

- Saavedra, Mendes e Ferreira. *Environmental Impact of CI/CD Pipelines*. arXiv:2510.26413.
- Cloud Carbon Footprint: metodologia de estimativa de energia e carbono em nuvem.
- ONS: Dados Abertos, *Geração por Usina em Base Horária*.
- NPAD/UFRN: [hardware](https://npad.ufrn.br/npad/hardware), [comandos SLURM](https://npad.ufrn.br/tutorials/intermediate/slurm_commands/), [job array](https://npad.ufrn.br/tutorials/advanced/job_array/), [treino em GPU](https://npad.ufrn.br/tutorials/advanced/tch-rs/).
