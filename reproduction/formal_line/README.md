# 正式线：三个任务 × 三档位宽

在 TUAR、TUAB、TUSL 三个任务上比较 FP32、PTQ、C、QAP 四组，位宽三档，种子 42 / 43 / 44，全部按受试者划分。源码是上游 BioFoundation `a2a4d44` 依次打三个补丁，命令见 [QUANTIZATION.md](../teammate_ptq/QUANTIZATION.md#独立准备源码)。

FP32 与 INT8 三个种子的测试结果、复核脚本和学习曲线见 [INT8 结果](results/int8/README.md)；TUAB 上 PTQ 的掉点定位见 [PTQ 掉点定位](results/ptq-diagnosis/README.md)。

## 协议

| 项目 | 规定 |
|---|---|
| 划分 | 全部按受试者；TUAB 测试集 = 官方 eval；预训练语料 = TUAB 官方训练集，剔除 TUAR、TUSL 的测试受试者 |
| 四组 | FP32；PTQ（在 FP32 微调检查点上校准）；C（FP32 预训练 + 量化微调）；QAP（量化预训练 + 量化微调，两阶段位宽相同） |
| 三档 | int8 = W8A8；int4 = W4A8；三值 = 每层 scale = mean\|w\|，\|w\| > 0.7 × scale 记 ±1、否则记 0，激活 8 位 |
| 训练 | 预训练 2 轮，每轮存档；整体微调 5 轮，TUSL 20 轮；都按最低验证损失选点 |
| PTQ | 激活百分位七档（99.999 … 90）。二分类用 512 个训练窗校准、按验证 AUROC 选档；TUSL 用 384 个、按验证宏平均 AUROC 选档；同一折各档共用一份校准名单 |
| 评估 | TUAR / TUAB：AUROC、AP、默认阈值和验证集阈值下的 BAcc。TUSL：按事件（两窗概率平均）的宏平均 AUROC、宏平均 AP、BAcc |
| TUSL | 296 个事件、三分类，四折轮换测试组。协议 v2 的验证集从另外三组按人抽（[resplit_validation.py](../tusl_events/resplit_validation.py)），是看过 v1 测试结果之后的探索性修订；v1 结果并列保留 |

## 环境与数据

- 环境：Python 3.11，`pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements-formal.txt`（[版本清单](requirements-formal.txt)：torch 2.7.1 + CUDA 12.8，Brevitas 0.12.1；量化版 FEMBA 是纯 PyTorch，不需要 mamba-ssm）。
- 数据：TUAR、TUAB 用 [run_subject_data.sh](../subject_split/run_subject_data.sh) 生成；TUSL 用 [make_tusl_events.py](../tusl_events/make_tusl_events.py)（v1）再经 `resplit_validation.py`（v2）。每个数据目录含 `train/val/test.h5`、`data-manifest.json` 和 `<split>-manifest.json`，运行脚本按清单核对文件大小、窗数和类别数。
- 运行目录 `STUDY`：队列在这个目录里调用各个脚本，所以先把它们拷到一起：`cp ../teammate_ptq/*.py ../tusl_events/evaluate_events.py ./*.sh ./*.py "$STUDY"/`。
- 路径：除 `pretrain_queue.sh` 外，脚本开头的默认路径都能用环境变量覆盖（`PY`、`STUDY`、`SRC`、`DATA_ROOT`、`TUSL_ROOT`、`ART`）；三值预训练还读 `DATA`（TUAB 预训练语料目录）和 `SMOKE`（冒烟测试目录），冒烟测试还读 `FORMAL_ART`（放 FP32 父目录的 `$ART`）。

## 运行

```bash
# 预训练：FP32 / QAP8 × 三个种子（本机专用，路径写死；FP32-42 在它之前单独跑过，QAP4-42 也在本机跑过）
nohup ./pretrain_queue.sh > "$ART/logs/pretrain-queue.log" 2>&1 &
# 三值预训练：先跑 ternary_smoke.sh（小数据上走一遍三值全流程），通过才开始
SMOKE_QUEUE=./finetune_queue.sh nohup ./pretrain_queue_ternary.sh > "$ART/logs/pretrain-queue-ternary.log" 2>&1 &
# 一个种子、一档位宽的微调、PTQ 和评估
SEED=42 BITS=2 SCHEME=ternary TASKS="tuar tuab tusl0-v2 tusl1-v2 tusl2-v2 tusl3-v2" LOG_SUFFIX=-all ./finetune_queue.sh
# 多个批次依次跑，预训练之外 GPU 上只有一个微调
BATCHES="42:4:uniform 43:8:uniform" QUEUE=./finetune_queue.sh nohup ./launch_batches.sh >> "$ART/logs/launch-batches.log" 2>&1 &
# 汇总
python summarize_formal.py --artifacts "$ART" --output <输出目录>
```

结果在 `$ART/seed<种子>/` 下：`fp32/`、`c_<档>/`、`qap_<档>/` 里是各任务的 `finetune-<任务>/`（评估在 `eval/`），`ptq_<档>/` 里是 `sweep-<任务>/`；档名为 `w8a8`、`w4a8`、`ternary_a8`。多条队列共用的阶段靠 `<目录>.lock` 互斥。正在运行的 bash 脚本不能原地修改，改动写成新文件名。

## 多台机器分跑

FP32 的预训练和微调只在一台机器上跑，三档位宽共用这一份 FP32。其他机器按下面做。

1. **放数据和 FP32 父目录**：数据目录和 `$ART/seed<种子>/fp32/` 下的阶段目录（`complete.json`、其中指向的检查点、`eval/`）放在同一个根目录 `$ROOT` 下，结构不变。`complete.json` 里是原机器的绝对路径，根目录不同就替换前缀：
   ```bash
   grep -rl <原根目录> "$ROOT/formal-20261025" --include=complete.json | xargs sed -i "s#<原根目录>#$ROOT#g"
   ```
2. **设路径**（每个终端；同一条 `export` 里的变量会先展开再赋值，所以互相引用的变量一行一个）：
   ```bash
   export ROOT=<根目录> PY=<python> STUDY=<运行目录> SRC=<打好补丁的源码>
   export ART="$ROOT/formal-20261025"
   export DATA_ROOT="$ROOT/subject-data-20261025"
   export TUSL_ROOT="$ROOT/tusl-events-20261026-v2"
   export DATA="$DATA_ROOT/tuab"
   export FORMAL_ART="$ART"
   mkdir -p "$ART/logs"
   ```
3. **先对照一次**：重跑一个已有的 FP32 阶段（比如种子 42 的 `finetune-tusl0-v2`，约 5 分钟），逐轮比较验证损失和原机器的 `epochs.jsonl`，记下两台机器的差别。
4. **跑自己那档**：
   ```bash
   cd "$STUDY"
   # int4：每个种子先做 QAP4 预训练，再跑微调、PTQ 和评估
   $PY train_stage.py --source "$SRC" --data-root "$DATA" --data-manifest "$DATA/data-manifest.json" \
     --output "$ART/seed43/qap_w4a8/pretrain" --group qap_b --stage pretrain --seed 43 --max-epochs 2 --save-every-epoch \
     --weight-bit-width 4 --activation-bit-width 8 > "$ART/logs/seed43-qap_w4a8-pretrain.log" 2>&1
   FP32_PARENTS=given SEED=43 BITS=4 SCHEME=uniform TASKS="tuar tuab tusl0-v2 tusl1-v2 tusl2-v2 tusl3-v2" LOG_SUFFIX=-all ./finetune_queue.sh
   # 三值：冒烟测试通过后依次跑三个种子的三值预训练；微调、PTQ 和评估每个种子一条
   SMOKE=$ROOT/ternary-smoke SMOKE_QUEUE=./finetune_queue.sh nohup ./pretrain_queue_ternary.sh > "$ART/logs/pretrain-queue-ternary.log" 2>&1 &
   FP32_PARENTS=given SEED=42 BITS=2 SCHEME=ternary TASKS="tuar tuab tusl0-v2 tusl1-v2 tusl2-v2 tusl3-v2" LOG_SUFFIX=-all ./finetune_queue.sh
   ```
   `FP32_PARENTS=given` 时，缺 FP32 预训练、微调或评估都会直接报错退出，不在本机重训；补齐后重跑，已完成的阶段会跳过。QAP 阶段会等同一档的 QAP 预训练完成再开始。用启动器时加 `START_AFTER=none`。
5. **回传**各阶段目录（检查点除外）和日志，合并到同一个 `$ART` 后用 `summarize_formal.py` 汇总。
