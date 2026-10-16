# 训练、校准与评估

环境与版本见 [seed-results.json](seed-results.json) 的 `runtime`。使用已应用 `source.patch` 的队友代码，运行目录与输出目录分开。

```bash
study=/path/to/this-repo/reproduction/teammate_ptq
teammate_source=/path/to/teammate-source
data=/path/to/TUAR_data
artifacts=/path/to/new-study-artifacts
mkdir -p "$artifacts"
```

## 数据校验

首次训练前核对三个 H5 的完整 SHA256，生成各训练阶段共用的清单。清单对应本次队友数据，不重新划分 `val.h5`、`test.h5`。

```bash
PYTHONPATH="$study" python - "$data" "$artifacts/data-manifest.json" <<'PY'
import hashlib, json, sys
from pathlib import Path
from train_stage import SPLITS
data, output = Path(sys.argv[1]), Path(sys.argv[2])
records = {}
for split, (windows, size, expected) in SPLITS.items():
    path = data / f"{split}.h5"
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    assert path.stat().st_size == size and digest == expected, split
    records[split] = {"windows": windows, "bytes": size, "sha256": digest}
with output.open("x") as stream:
    json.dump({"data": records}, stream, indent=2)
PY
```

## FP32 与 QAP B

以 seed43 的 FP32 为例。预训练 10 轮，微调 5 轮，均 batch32，最低验证 loss 选检查点。预训练在 `train.h5` 内按该种子划分 9:1；微调用完整 `train.h5` 和单独的 `val.h5`。

```bash
seed=43
group=fp32
run="$artifacts/seeds/$seed/$group"
python "$study/train_stage.py" --source "$teammate_source" --data-root "$data" \
  --data-manifest "$artifacts/data-manifest.json" --output "$run/pretrain" \
  --group "$group" --stage pretrain --seed "$seed"

pretrained=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["best_checkpoint"])' "$run/pretrain/complete.json")
pretrained_sha=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["best_checkpoint_sha256"])' "$run/pretrain/complete.json")
python "$study/train_stage.py" --source "$teammate_source" --data-root "$data" \
  --data-manifest "$artifacts/data-manifest.json" --output "$run/finetune" \
  --group "$group" --stage finetune --seed "$seed" \
  --pretrained "$pretrained" --pretrained-sha256 "$pretrained_sha"
```

将 `group=qap_b` 跑 QAP 预训练与量化微调；分别设置 `seed=43`、`seed=44`。同种子的 FP32/QAP 保留相同随机初始化与数据划分。`--plan-only` 可先打印解析后的 Hydra 配置，不启动训练。

## C42

使用 seed42 的 **FP32 预训练**检查点，开启量化微调 5 轮。`pretrained` 和 `pretrained_sha` 从对应预训练 `complete.json` 读取；入口检查父模型的任务、种子和量化状态。
先按上节设置 `seed=42`、`group=fp32` 完成预训练，或将 `parent` 指向已有的对应 `complete.json`。

```bash
parent="$artifacts/seeds/42/fp32/pretrain/complete.json"
pretrained=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["best_checkpoint"])' "$parent")
pretrained_sha=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["best_checkpoint_sha256"])' "$parent")
python "$study/train_stage.py" --source "$teammate_source" --data-root "$data" \
  --data-manifest "$artifacts/data-manifest.json" --output "$artifacts/seeds/42/c/finetune" \
  --group c --stage finetune --seed 42 \
  --pretrained "$pretrained" --pretrained-sha256 "$pretrained_sha"
```

`pretrained_handoff.json` 和 `first_training_batch.json` 记录加载后 observer 状态及首批重新统计结果；不消耗额外训练批次。

## 完整评估与七档 PTQ

从相应微调 `complete.json` 读取最佳检查点及哈希，按模型组运行。评估先在验证集锁定阈值，再测完整测试集，两次均严格重载。

```bash
seed=43
group=fp32
run="$artifacts/seeds/$seed/$group"
checkpoint=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["best_checkpoint"])' "$run/finetune/complete.json")
checkpoint_sha=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["best_checkpoint_sha256"])' "$run/finetune/complete.json")
python "$study/evaluate_one.py" --source "$teammate_source" --data-root "$data" \
  --checkpoint "$checkpoint" --checkpoint-sha256 "$checkpoint_sha" \
  --model-kind "$group" --seed "$seed" --output "$run/evaluation"
```

评估 QAP B 时设置 `group=qap_b`，C42 设置 `seed=42`、`group=c`，再重设 `run`。每个种子的 PTQ 从该种子 FP32 微调检查点校准，始终固定同一批 512 个训练窗口：

```bash
fp32_complete="$artifacts/seeds/$seed/fp32/finetune/complete.json"
fp32_checkpoint=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["best_checkpoint"])' "$fp32_complete")
python "$study/sweep.py" --source "$teammate_source" --data "$data" \
  --fp32 "$fp32_checkpoint" --output "$artifacts/seeds/$seed/ptq-sweep"
```

`sweep.py` 比较七档验证 AUROC，只对选定档位测试一次。`calibrate.py` 可单独重建某一档；`threshold_control.py` 可从保留的验证与测试 logits 复核旧四组阈值。

## 曲线与结果汇总

每次训练保存 `epochs.jsonl`、解析配置、最佳检查点信息及来源哈希。`export_curves.py` 接收 TensorBoard 清单，保留完整标量 CSV 并输出图：

```json
{"runs":[{"run":"fp32-seed43-finetune","stage":"finetune","seed":43,"epochs":5,"tensorboard_dir":"/path/to/exact/tensorboard/run"}]}
```

```bash
python "$study/export_curves.py" --manifest runs.json --output "$artifacts/curves"
python "$study/summarize_seeds.py" --raw /path/to/retained-stage-records \
  --history "$study/results.json" --output "$artifacts/seed-results.json" \
  --figures "$artifacts/seed-figures"
```

汇总输入包含本次队列的完整阶段记录、配对和独立指标核验；输出只提取可共享字段。CSV、数据集、权重及逐样本预测保存在实验目录。公开训练入口补充了父检查点参数保护；实际运行驱动哈希仍按原件记录。

## 验证

先设置本文开头的变量。测试使用 CPU，原生模型与数据 fixture 读取指定的队友代码和 H5：

```bash
FEMBA_SOURCE="$teammate_source" FEMBA_DATA_ROOT="$data" \
  python -m unittest discover -s "$study" -p 'test_*.py' -v
```
