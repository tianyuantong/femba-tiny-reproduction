# 正式线：三个任务 × 三档位宽

[INT8 结果](results/int8/README.md) · [INT4 结果](results/int4/README.md) · [评测指标与成本](results/metrics/README.md) · [QAP 逐层分析](results/layers/README.md) · [PTQ 掉点定位](results/ptq-diagnosis/README.md) · [激活位宽线](results/activation-sweep/README.md)

在 TUAR、TUAB、TUSL 三个任务上比较 FP32、PTQ、C、QAP 四组（位宽三档，种子 42 / 43 / 44，受试者独立划分）。模型源码基于上游 BioFoundation `a2a4d44` 依次应用三个补丁，见 [QUANTIZATION.md](../teammate_ptq/QUANTIZATION.md#独立准备源码)。

## 协议

| 项目 | 规定 |
|---|---|
| 划分 | 全部按受试者；TUAB 测试集 = 官方 eval；预训练语料 = TUAB 官方训练集，剔除 TUAR、TUSL 测试受试者 |
| 四种方法 | FP32；PTQ（在 FP32 微调检查点上校准）；C（FP32 预训练 + 量化微调）；QAP（量化预训练 + 量化微调，两阶段位宽相同） |
| 三档位宽 | int8 = W8A8；int4 = W4A8；三值 = 每层 scale = mean\|w\|，\|w\| > 0.7 × scale 记 ±1、否则记 0，激活 8 位 |
| 训练 | 预训练 2 轮（每轮存档）；整体微调 5 轮（TUSL 20 轮）；均按最低验证损失选检查点 |
| PTQ | 激活百分位七档（99.999 / 99.99 / 99.9 / 99 / 98 / 95 / 90，目录中 99.999 记为 `original`）。二分类用 512 个训练片段校准、按验证 AUROC 选档；TUSL 用 384 个、按验证宏平均 AUROC 选档；同折共用校准名单 |
| 评估 | TUAR / TUAB：AUROC、AP、默认阈值与验证集阈值下的 BAcc。TUSL：按事件（两片段概率平均）的宏平均 AUROC、宏平均 AP、BAcc |
| TUSL | 296 个事件、三分类，四组轮换。v1 验证集受单名受试者主导而早停在第 0–2 轮；v2 验证集改为从另外三组按人抽取（[resplit_validation.py](../tusl_events/resplit_validation.py)），v1 结果并列保留 |

## 环境与数据

- **环境**：Python 3.11，`pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements-formal.txt`（[版本清单](requirements-formal.txt)：torch 2.7.1 + CUDA 12.8，Brevitas 0.12.1；纯 PyTorch 实现，无需 `mamba-ssm`）。
- **数据**：用 [make_subject_splits.py](../subject_split/make_subject_splits.py) 按受试者生成划分文件，TUAR、TUAB 用 [run_subject_data.sh](../subject_split/run_subject_data.sh) 生成；TUAB 微调集 `tuab-ft` 用 [subset_h5.py](../subject_split/subset_h5.py) 从训练/验证集各抽 1/5 受试者（`--fraction 0.2 --seed 42`），测试集用完整官方 eval；TUSL 用 [make_tusl_events.py](../tusl_events/make_tusl_events.py)（v1）再经 `resplit_validation.py`（v2）。每个数据目录含 `train/val/test.h5`、`data-manifest.json` 和 `<split>-manifest.json`。
- **运行目录 `STUDY`**：把脚本拷到同一目录后运行：`cp ../teammate_ptq/*.py ../tusl_events/evaluate_events.py ./*.sh ./*.py "$STUDY"/`。
- **路径变量**：除 `pretrain_queue.sh` 外，脚本默认路径均可由环境变量覆盖（`PY`、`STUDY`、`SRC`、`DATA_ROOT`、`TUSL_ROOT`、`ART`）；三值预训练另需 `DATA`（TUAB 预训练语料）与 `SMOKE`（冒烟测试目录），冒烟测试需 `FORMAL_ART`（存放共用 FP32 检查点的 `$ART`）。

## 运行

```bash
# 预训练：FP32 / QAP8 × 三个种子
nohup ./pretrain_queue.sh > "$ART/logs/pretrain-queue.log" 2>&1 &
# 三值预训练：先跑 ternary_smoke.sh 冒烟测试，通过后自动开始
SMOKE_QUEUE=./finetune_queue.sh nohup ./pretrain_queue_ternary.sh > "$ART/logs/pretrain-queue-ternary.log" 2>&1 &
# 单种子、单位宽的微调、PTQ 与评估
SEED=42 BITS=2 SCHEME=ternary TASKS="tuar tuab tusl0-v2 tusl1-v2 tusl2-v2 tusl3-v2" LOG_SUFFIX=-all ./finetune_queue.sh
# 多批次串行启动（预训练之外 GPU 上只跑一个微调）
BATCHES="42:4:uniform 43:8:uniform" QUEUE=./finetune_queue.sh nohup ./launch_batches.sh >> "$ART/logs/launch-batches.log" 2>&1 &
# QAP 预训练未就绪时：RUN_QAP=no 先跑 C 和 PTQ，就绪后用 RUN_QAP=yes 补跑
FP32_PARENTS=given RUN_QAP=no SEED=43 BITS=4 TASKS=tuab ./finetune_queue.sh
# 激活位宽：ABITS 设激活位宽，RUN_QAP=no 只跑 C 和 PTQ，见 results/activation-sweep/README.md
FP32_PARENTS=given RUN_QAP=no SEED=42 BITS=8 ABITS=6 TASKS=tuab ./finetune_queue.sh
# 随机初始化基线（无预训练，三任务，BITS=8 或 4，在 STUDY 目录运行）
BITS=8 SEEDS="42 43 44" ./run_random_init.sh
# FP16 / BF16 重放（直接重放 FP32 微调检查点，在 STUDY 目录运行）
DTYPES="bfloat16 float16" ./run_half_precision.sh
# 汇总指标
python summarize_formal.py --artifacts "$ART" --output <输出目录>
```

产物位于 `$ART/seed<种子>/`：`fp32/`、`c_<档>/`、`qap_<档>/` 下为 `finetune-<任务>/`（评估在 `eval/`），`ptq_<档>/` 下为 `sweep-<任务>/`；档名为 `w8a8`、`w4a8`、`ternary_a8`。多队列共享阶段通过 `<目录>.lock` 互斥；运行中的 bash 脚本勿原地修改，改动另存新文件名。

## 多台机器分跑

FP32 预训练与微调只在一台机器跑，三档位宽共用该份 FP32：

1. **放置数据与共用 FP32 目录**：将数据目录与 `$ART/seed<种子>/fp32/` 放在同一根目录 `$ROOT` 下，并替换 `complete.json` 中的绝对路径前缀：
   ```bash
   grep -rl <原根目录> "$ROOT/formal-20261025" --include=complete.json | xargs sed -i "s#<原根目录>#$ROOT#g"
   ```
2. **设置环境变量**：
   ```bash
   export ROOT=<根目录> PY=<python> STUDY=<运行目录> SRC=<打好补丁的源码>
   export ART="$ROOT/formal-20261025"
   export DATA_ROOT="$ROOT/subject-data-20261025"
   export TUSL_ROOT="$ROOT/tusl-events-20261026-v2"
   export DATA="$DATA_ROOT/tuab"
   export FORMAL_ART="$ART"
   mkdir -p "$ART/logs"
   ```
3. **跨机对齐检查**：重跑一个短 FP32 阶段（如种子 42 的 `finetune-tusl0-v2`，约 5 分钟），核对 `epochs.jsonl` 逐轮验证损失。
4. **运行目标位宽**：
   ```bash
   cd "$STUDY"
   # INT4：先跑 QAP4 预训练，再跑微调、PTQ 与评估
   $PY train_stage.py --source "$SRC" --data-root "$DATA" --data-manifest "$DATA/data-manifest.json" \
     --output "$ART/seed43/qap_w4a8/pretrain" --group qap_b --stage pretrain --seed 43 --max-epochs 2 --save-every-epoch \
     --weight-bit-width 4 --activation-bit-width 8 > "$ART/logs/seed43-qap_w4a8-pretrain.log" 2>&1
   FP32_PARENTS=given SEED=43 BITS=4 SCHEME=uniform TASKS="tuar tuab tusl0-v2 tusl1-v2 tusl2-v2 tusl3-v2" LOG_SUFFIX=-all ./finetune_queue.sh
   # 三值：冒烟通过后跑三个种子的三值预训练，再跑各微调与评估
   SMOKE=$ROOT/ternary-smoke SMOKE_QUEUE=./finetune_queue.sh nohup ./pretrain_queue_ternary.sh > "$ART/logs/pretrain-queue-ternary.log" 2>&1 &
   FP32_PARENTS=given SEED=42 BITS=2 SCHEME=ternary TASKS="tuar tuab tusl0-v2 tusl1-v2 tusl2-v2 tusl3-v2" LOG_SUFFIX=-all ./finetune_queue.sh
   ```
   `FP32_PARENTS=given` 会在缺少 FP32 父目录时直接报错退出，防止误触发重训；重跑时已完成阶段自动跳过。用启动器时加 `START_AFTER=none`。
5. **回传汇总**：回传各阶段目录（不含检查点）与日志，合并至 `$ART` 后运行 `summarize_formal.py`。
