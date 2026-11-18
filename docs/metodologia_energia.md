# Metodologia de medição energética

## 1. Referência

Saavedra, Mendes e Ferreira, *Environmental Impact of CI/CD Pipelines* ([arXiv:2510.26413](https://arxiv.org/abs/2510.26413)).

O que o artigo faz e o que este projeto herda:

| No artigo | Neste projeto |
|---|---|
| Ambiente: runner do GitHub Actions (VM do Azure, 4 vCPUs, AMD EPYC 7763) | Nó `gpu-8-h100` do NPAD alocado pelo SLURM |
| Energia **estimada** pelo modelo Cloud Carbon Footprint (CCF) a partir de telemetria | Igual |
| Telemetria de CPU, memória e rede "em intervalos fixos" (intervalo não informado) | Telemetria a cada 1 s |
| Componentes: vCPU, memória, armazenamento, rede; **sem GPU** | Os mesmos + **GPU** (extensão) |
| Carbono: intensidade horária do Electricity Maps na hora de início | Intensidade horária calculada com dados do **ONS, subsistema Nordeste**, na hora de início |
| Água: no datacenter (WUE) + na geração elétrica (EWIF) + embutida | **Fora do escopo** (só energia e carbono) |
| Cenários de uso: base, ×2, ×0,5 | Não se aplica (temos a telemetria real de cada execução) |

## 2. Unidade de medida e fases

- **Execução** = um modelo × uma seed = uma tarefa do job array.
- Cada execução é dividida em **fases** com carimbo de tempo: `setup`, `idle_baseline`, `train` (uma por época), `val` (uma por época), `test`.
- **Resultado principal:** soma de todas as fases `train` da execução. `val` é reportada à parte; `idle_baseline` serve só de referência; `setup` e `test` não são contabilizados.
- Cada fronteira de fase é registrada **depois de `torch.cuda.synchronize()`**, para que o trabalho assíncrono da GPU seja atribuído à fase certa.

## 3. Telemetria

Uma thread separada do treino coleta uma amostra a cada `sample_interval_s` (1 s) e grava em `telemetry.csv`.

| Coluna | Fonte | Escopo | Usada no CCF? |
|---|---|---|---|
| `t` | relógio do sistema (epoch, s) | — | — |
| `phase` | rótulo da fase corrente | — | Sim (agregação) |
| `proc_cpu_pct` | psutil: % de CPU do processo de treino e filhos (pode passar de 100) | Processo | **Sim** |
| `n_vcpus_alloc` | afinidade de CPU do processo (CPUs dadas pelo SLURM) | Job | **Sim** |
| `proc_rss_gb` | psutil: memória residente do processo e filhos | Processo | **Sim** |
| `gpu_util_pct` | NVML: utilização da GPU | GPU alocada | **Sim** |
| `gpu_mem_used_gb` | NVML | GPU alocada | Não (contexto) |
| `gpu_power_w` | NVML: potência instantânea | GPU alocada | Validação |
| `gpu_energy_mj` | NVML: contador de energia acumulada | GPU alocada | Validação |
| `gpu_temp_c`, `gpu_sm_clock_mhz` | NVML | GPU alocada | Não (contexto) |
| `node_cpu_pct`, `node_mem_used_gb` | psutil, sistema todo | Nó (compartilhado) | Não (contexto) |
| `node_net_sent_b`, `node_net_recv_b` | psutil, contadores do nó | Nó (compartilhado) | Sim, como limite superior |

**Escopo processo × nó:** o nó H100 é compartilhado (não usamos `--exclusive`). Métricas do nó inteiro incluiriam outros usuários, então o CCF usa as métricas do **processo** e da **GPU alocada**. As do nó ficam como contexto, para detectar interferência.

**Identificar a GPU alocada no NVML:** o NVML enumera as GPUs sem respeitar `CUDA_VISIBLE_DEVICES`. A GPU deve ser localizada pelo **UUID** do dispositivo que o PyTorch está usando, e não pelo índice.

**Rede:** a rede não tem contador por processo. Usa-se a diferença dos contadores do nó como limite superior. Como os dados estão na memória da GPU durante o treino, o tráfego do próprio job deve ser praticamente zero, e o termo de rede (0,001 kWh/GB) fica desprezível de qualquer forma.

## 4. Metadados do nó (`node.json`)

Coletados no início de cada execução, para caracterizar o ambiente virtual:

- hostname, partição, `SLURM_JOB_ID`, `SLURM_ARRAY_JOB_ID`, `SLURM_ARRAY_TASK_ID`, `SLURM_CPUS_PER_TASK`, `SLURM_MEM_PER_NODE`, `SLURM_JOB_GPUS`;
- saída de `scontrol show job`;
- `lscpu` (modelo, sockets, núcleos, threads), que dá o **total de vCPUs do nó**, usado no carbono embutido;
- memória total do nó;
- `nvidia-smi -q` da GPU alocada: modelo, UUID, driver, versão CUDA, limite de potência configurado;
- versões de kernel, Python, PyTorch, CUDA e cuDNN; commit do git do repositório.

## 5. Energia (CCF)

Para cada tipo de fase `f`, somam-se as durações de todas as suas ocorrências (`t_f`, em horas) e tiram-se as médias da telemetria sobre as amostras rotuladas com `f`. Fases muito curtas, como `val`, podem ter poucas amostras por ocorrência, mas a média agregada sobre todas as épocas é estável.

### 5.1 CPU

```
U_vCPU = média(proc_cpu_pct) / 100                    (em vCPUs: 250% → 2,5 vCPUs)
E_cpu  = (vCPU_min × N_vCPU + (vCPU_max − vCPU_min) × U_vCPU) × t_f
```

O artigo define `U_vCPU` como o % médio de carga sobre o total vezes o número de vCPUs. O % do processo do psutil já vem nessa escala (100% = 1 vCPU), então os dois são equivalentes.

| Constante | Valor | Fonte |
|---|---|---|
| `vCPU_min` | 0,7391 W | CCF, EPYC 4th Gen (`coefficients-aws-use.csv`) |
| `vCPU_max` | 2,2822 W | CCF, EPYC 4th Gen (`coefficients-aws-use.csv`) |
| `N_vCPU` | `--cpus-per-task` (8) | Alocação do SLURM |

**Escolha dos coeficientes.** O nó H100 tem 2× EPYC 9654, da 4ª geração (Genoa). A planilha oficial do CCF ([ccf-coefficients](https://github.com/cloud-carbon-footprint/ccf-coefficients), pasta `output/`) tem os seguintes valores por arquitetura:

| Arquitetura | Fonte | Min (W/vCPU) | Max (W/vCPU) |
|---|---|---|---|
| **EPYC 4th Gen** | AWS | **0,7391** | **2,2822** |
| EPYC 4th Gen | GCP | 0,7391 | 2,1959 |
| EPYC 3rd Gen | Azure | 0,4454 | 2,0193 |
| EPYC 3rd Gen | artigo | 0,434 | 1,948 |

- A planilha do Azure, que é a nuvem do artigo, não tem a 4ª geração. Usamos a da **AWS**.
- O mínimo é igual nas duas nuvens; a AWS tem o máximo maior, o que torna a estimativa conservadora.
- Os valores do artigo diferem um pouco dos atuais do CCF para Azure, provavelmente por usarem uma versão anterior da planilha.
- Como sensibilidade, o relatório pode recalcular `E_cpu` com os coeficientes do artigo, para isolar o efeito da troca de geração.

### 5.2 Memória

```
E_mem = 0,000392 kWh/(GB·h) × média(proc_rss_gb) × t_f
```

A memória da GPU (HBM) não entra aqui, porque já está incluída na potência da GPU.

### 5.3 GPU (extensão do método)

O artigo exclui GPU. Aplicamos a mesma forma linear do CCF, por GPU alocada:

```
U_gpu = média(gpu_util_pct) / 100
E_gpu = (P_min + (P_max − P_min) × U_gpu) × N_gpu × t_f
```

| Constante | Valor | Fonte |
|---|---|---|
| `P_min` | Potência média do NVML na fase `idle_baseline` da própria execução | Medido |
| `P_max` | 700 W | TDP da H100 SXM5 (datasheet NVIDIA) |
| `N_gpu` | 1 | Alocação do SLURM |

O `gpu_util_pct` do NVML indica a fração do tempo em que algum kernel estava rodando, não a fração da capacidade usada. Uma GPU pode marcar 100% de uso consumindo bem menos que o TDP. Por isso a validação contra o contador de energia (seção 7) é essencial.

### 5.4 Armazenamento

```
E_storage = 0,0012 kWh/(TB·h) × reservado (TB) × t_f
```

Reservado = tamanho de `data/processed` (cerca de 1,4 GB). O artigo usa o disco reservado do runner (14 GB). Valor desprezível.

### 5.5 Rede

```
E_net = 0,001 kWh/GB × (Δ node_net_sent_b + Δ node_net_recv_b) / 10⁹
```

Limite superior (seção 3).

### 5.6 Total

```
E_TI    = E_cpu + E_mem + E_gpu + E_storage + E_net
E_total = E_TI × PUE                                   PUE = 1,17 (artigo, Américas)
```

## 6. Carbono

### 6.1 Operacional

```
CO2_op = E_total (kWh) × CI(h_início) (gCO2e/kWh)
```

Como no artigo, usa-se a intensidade da **hora em que a execução começou**. Como registramos início e fim de cada fase, também é possível calcular uma variante ponderada hora a hora, mas o resultado principal segue o artigo.

### 6.2 Intensidade de carbono pelo ONS (Nordeste)

**Fonte:** ONS Dados Abertos, dataset *Geração por Usina em Base Horária*. São arquivos mensais em CSV, XLSX ou Parquet, e o mês corrente já é publicado:

```
https://ons-aws-prod-opendata.s3.amazonaws.com/dataset/geracao_usina_2_ho/GERACAO_USINA-2_<AAAA>_<MM>.parquet
```

Colunas relevantes: `din_instante` (hora, horário de Brasília), `id_subsistema` (`NE`), `nom_tipocombustivel`, `val_geracao` (MWmed na hora ≈ MWh).

**Cálculo:**

```
CI(h) = Σ_usinas∈NE val_geracao(h) × fator(combustível) / Σ_usinas∈NE val_geracao(h)
```

| `nom_tipocombustivel` | gCO₂e/kWh | Observação |
|---|---|---|
| Hidráulica | 24 | IPCC AR5 |
| Eólica | 11 | IPCC AR5 |
| Fotovoltaica | 48 | IPCC AR5 |
| Nuclear | 12 | IPCC AR5 (não há no NE) |
| Gás | 490 | IPCC AR5 |
| Carvão | 820 | IPCC AR5 |
| Biomassa | 230 | IPCC AR5 |
| Óleo Combustível / Óleo Diesel | 650 | Electricity Maps |
| Outras Multi-Combustível | 650 | Conservador, tratado como óleo |
| Resíduos Industriais / não mapeado | 700 | Conservador |

**Cuidados:**
- **Fuso horário:** `din_instante` está em UTC−3, e os carimbos da telemetria estão em UTC (epoch). Converter antes de cruzar os dois.
- **Produção × consumo:** o cálculo usa só a geração do Nordeste e não desconta o intercâmbio com outros subsistemas. O Nordeste costuma exportar energia (eólica e solar), então o resultado representa bem a matriz local. Essa limitação deve ser declarada.
- **MMGD:** o dataset inclui linhas de micro e minigeração distribuída (`Pequenas Usinas (MMGD)`). Elas entram no cálculo e devem ser mencionadas no texto.
- **Referência cruzada:** comparar a média do período com o fator médio de emissão do SIN publicado pelo MCTI.

### 6.3 Embutido

```
CO2_emb = 1,61079 tCO2e × (t / (4 anos)) × (N_vCPU / vCPUs_do_nó)
```

Esta é a fórmula do artigo, rateada por vCPU. Num servidor com 8 H100, a maior parte do carbono embutido está nas GPUs, então esse rateio provavelmente subestima o valor. Por fidelidade ao artigo, ele é o resultado principal. Um rateio alternativo por GPU (`N_gpu / 8`) pode ser reportado como sensibilidade.

## 7. Validação da estimativa

| Comparação | O que revela |
|---|---|
| `E_gpu` (CCF) × energia do NVML na fase | Erro do modelo linear por utilização quando aplicado à H100 |
| Σ das potências amostradas × Δ do contador NVML | Consistência interna da telemetria (taxa de amostragem suficiente?) |
| `TotalCPU` do `sacct` × ∫ `proc_cpu_pct` | Consistência da telemetria de CPU |
| `ConsumedEnergy` do `sacct` (se habilitado) × `E_TI` | Segunda referência medida, no nível do nó |

A energia do NVML, de cada fase, é a diferença entre as leituras do contador nas duas fronteiras da fase.

## 8. Saída por execução (`energy.json`)

Por tipo de fase (`train`, `val`, `idle_baseline`):

```
duration_h, n_samples,
U_vcpu, mem_gb, U_gpu,
E_cpu_kwh, E_mem_kwh, E_gpu_kwh, E_storage_kwh, E_net_kwh, E_it_kwh, E_total_kwh,
E_gpu_nvml_kwh, gpu_ccf_error_pct,
ci_g_per_kwh, co2_op_g, co2_emb_g
```

E também: as constantes usadas (cópia do `energy.yaml`) e a hora de início usada para buscar a intensidade de carbono.

## 9. Normalizações para a comparação

- kWh por execução (só `train`);
- J por imagem de treino processada = E_total / (épocas × 7.200);
- J por GFLOP de treino (FLOPs de *forward* × 3, a aproximação usual para *forward* + *backward*);
- kWh até a primeira época que atinge X% de acurácia de validação;
- média ± desvio padrão nas 5 seeds e diferença CNN − ViT com intervalo de confiança.
