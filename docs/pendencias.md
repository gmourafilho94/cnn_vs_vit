# Pendências, valores a confirmar e riscos

## A confirmar com o NPAD

| # | Pendência | Impacto | Como resolver |
|---|---|---|---|
| 1 | Os nós de computação têm acesso à internet? | Por padrão, tudo que precisa de rede roda no nó de login | Resposta do NPAD |
| 2 | Como o Python é provido (module, conda, container)? | Instalação do ambiente | Resposta do NPAD |
| 3 | Versão do driver NVIDIA nos nós H100 | Escolha da versão CUDA do PyTorch | Smoke test (`nvidia-smi`) |
| 4 | Tempo máximo da partição `gpu-8-h100` | `--time` do job array | `sinfo -p gpu-8-h100` |
| 5 | A contabilização de energia do SLURM está ativa? | Segunda referência de validação | `sacct -o ConsumedEnergy` no smoke test |
| 6 | PUE real do datacenter do NPAD | Substituir os valores das Américas do artigo | Perguntar ao NPAD |
| 7 | Leitura de RAPL nos nós (AMD) é permitida a usuários? | Possível medição real da CPU | Smoke test (`/sys/class/powercap`) |

## Valores a preencher em `configs/energy.yaml`

| Constante | Situação |
|---|---|
| `cpu.vcpu_min_w` / `vcpu_max_w` | **Resolvido:** CCF "EPYC 4th Gen" (AWS), 0,7391 / 2,2822 W. Confirmar no smoke test que o `lscpu` do nó mostra mesmo o EPYC 9654 |
| `datacenter.pue` | Ver item 6 acima |

## Riscos conhecidos

| Risco | Mitigação |
|---|---|
| **Nó compartilhado:** outros jobs no mesmo nó H100 afetam métricas do nó e podem afetar o desempenho (memória, PCIe, temperatura) | CCF usa métricas do processo e da GPU alocada; métricas do nó ficam registradas para detectar interferência; modelos intercalados no array |
| **Execuções curtas:** com 7.200 imagens e modelos pequenos, uma época na H100 leva poucos segundos | Telemetria a 1 s + contador de energia do NVML lido exatamente nas fronteiras; agregação por tipo de fase sobre todas as épocas |
| **Utilização ≠ potência na GPU:** o modelo linear do CCF pode errar bastante | É justamente o que a validação com o NVML mede; reportar o erro como resultado |
| **Acurácia baixa do ViT** com 80 imagens/classe do zero | Esperado; discutir. Não usar receitas diferentes por modelo |
| **Carbono pela produção, não pelo consumo** (intercâmbio entre subsistemas ignorado) | Declarar; comparar com o fator médio do SIN (MCTI) |
| **Dataset sem licença nem documentação** | Registrar no trabalho; versionar a lista exata de imagens usadas |
| **Carbono embutido rateado por vCPU** subestima um servidor com GPUs | Manter a fórmula do artigo como resultado principal e reportar o rateio por GPU como sensibilidade |
