# 单阶段训练、校准与评估命令

> **正式线批量队列命令见 [reproduction/formal_line/README.md](../formal_line/README.md)。** 本页说明单个训练、PTQ 扫描、评估与曲线导出脚本的直接调用方式（以按录音划分的 TUAR 试点目录为例）。

环境版本见 [seed-results.json](seed-results.json) 的 `runtime`。源码准备见 [独立准备源码](QUANTIZATION.md#独立准备源码)。省略位宽参数时默认运行 W8A8：

```bash
study=/path/to/this-repo/reproduction/teammate_ptq
teammate_source=/path/to/teammate-source
data=/path/to/TUAR_data
artifacts=/path/to/new-study-artifacts
mkdir -p "$artifacts"
```

## 数据校验

首次训练前核对 `train.h5`、`val.h5`、`test.h5` 的文件大小、片段数与 SHA256，生成各阶段共用的 `data-manifest.json`：

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

## 1. FP32 与 QAP（预训练 + 微调）

以种子 43 的 FP32 为例（在 TUAR 试点中预训练 10 轮、微调 5 轮，batch 32，按最低验证损失选检查点）：

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

把 `group=fp32` 换成 `group=qap_b` 即可运行量化预训练与量化微调。加上 `--plan-only` 可先查看解析后的 Hydra 配置而不启动训练。

## 2. C 组（复用同种子 FP32 预训练检查点）

加载同种子的 FP32 预训练检查点，直接开启量化微调 5 轮（若不传 `--pretrained`，则从随机初始化开始微调）：

```bash
seed=43
parent="$artifacts/seeds/$seed/fp32/pretrain/complete.json"
pretrained=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["best_checkpoint"])' "$parent")
pretrained_sha=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["best_checkpoint_sha256"])' "$parent")
python "$study/train_stage.py" --source "$teammate_source" --data-root "$data" \
  --data-manifest "$artifacts/data-manifest.json" --output "$artifacts/seeds/$seed/c/finetune" \
  --group c --stage finetune --seed "$seed" \
  --pretrained "$pretrained" --pretrained-sha256 "$pretrained_sha"
```

## 3. 评估与七档 PTQ 扫描

从微调目录的 `complete.json` 读取最佳检查点，先在验证集上确定分类阈值，再评估测试集：

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

PTQ 从同种子的 FP32 微调检查点出发，在固定的 512 个训练片段上扫描七档激活百分位，按验证集 AUROC 选出一档后评一次测试集：

```bash
fp32_complete="$artifacts/seeds/$seed/fp32/finetune/complete.json"
fp32_checkpoint=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["best_checkpoint"])' "$fp32_complete")
python "$study/sweep.py" --source "$teammate_source" --data "$data" \
  --fp32 "$fp32_checkpoint" --output "$artifacts/seeds/$seed/ptq-sweep"
```

## 4. 导出训练曲线与汇总

每次训练自动保存逐轮指标 `epochs.jsonl`、配置与检查点哈希。`export_curves.py` 从 TensorBoard 日志导出 CSV 与 SVG 曲线：

```json
{"runs":[{"run":"fp32-seed43-finetune","stage":"finetune","seed":43,"epochs":5,"tensorboard_dir":"/path/to/exact/tensorboard/run"}]}
```

```bash
python "$study/export_curves.py" --manifest runs.json --output "$artifacts/curves"
python "$study/summarize_seeds.py" --raw /path/to/retained-stage-records \
  --history "$study/results.json" --output "$artifacts/seed-results.json" \
  --figures "$artifacts/seed-figures"
python "$study/summarize_c_followup.py" --raw /path/to/c-followup-records \
  --previous "$study/seed-results.json" --output "$artifacts/c-seed-results.json" \
  --figures "$artifacts/c-figures"
```

## 验证

在 CPU 上运行本目录的全部单元测试：

```bash
FEMBA_SOURCE="$teammate_source" FEMBA_DATA_ROOT="$data" \
  python -m unittest discover -s "$study" -p 'test_*.py' -v
```
