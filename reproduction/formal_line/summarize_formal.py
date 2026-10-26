"""Collect formal-line results into one table.

Reads ``<artifacts>/seed<seed>/<arm>/...`` written by finetune_queue.sh:
binary tasks (tuar, tuab) from evaluate_one.py / sweep.py results, TUSL folds
from evaluate_events.py results. Arms: fp32 and, per weight setting
(w8a8, w4a8, ternary_a8), ptq / c / qap. Prints a Markdown summary and writes
``summary.json``. Seeds are the unit of variation: TUSL folds are averaged
within a seed first; differences are computed within the same seed (and fold).
"""
import argparse
import json
import statistics
from pathlib import Path

SETTINGS = ("w8a8", "w4a8", "ternary_a8")
BINARY_TASKS = ("tuar", "tuab")
BINARY_METRICS = ("auroc", "ap", "bacc_default", "bacc_val_thr")
TUSL_METRICS = ("macro_auroc", "macro_ap", "bacc")


def load(path: Path):
    return json.loads(path.read_text()) if path.is_file() else None


def binary_row(result: dict, is_ptq: bool) -> dict:
    # sweep.py keeps its test row flat (the threshold-0 decision sits at the top);
    # evaluate_one.py nests that decision under "default".
    test = result["test"] if is_ptq else result
    default = test if is_ptq else test["default"]
    return {"auroc": test["auroc"], "ap": test["ap"], "bacc_default": default["balanced_accuracy"],
            "bacc_val_thr": test["validation_selected"]["balanced_accuracy"]}


def tusl_row(result: dict) -> dict:
    events = result["splits"]["test"]["events"]
    return {"macro_auroc": events["macro_auroc"], "macro_ap": events["macro_ap"], "bacc": events["balanced_accuracy"]}


def arm_dirs(seed_dir: Path, setting: str) -> dict:
    return {"fp32": seed_dir / "fp32", "ptq": seed_dir / f"ptq_{setting}",
            "c": seed_dir / f"c_{setting}", "qap": seed_dir / f"qap_{setting}"}


def collect(artifacts: Path) -> dict:
    out = {}
    for seed_dir in sorted(artifacts.glob("seed*")):
        seed = int(seed_dir.name[4:])
        for setting in SETTINGS:
            dirs = arm_dirs(seed_dir, setting)
            for arm, root in dirs.items():
                for task in BINARY_TASKS:
                    result = load(root / f"sweep-{task}" / "results.json") if arm == "ptq" else load(root / f"finetune-{task}" / "eval" / "results.json")
                    if result:
                        out.setdefault(setting, {}).setdefault(task, {}).setdefault(arm, {})[seed] = binary_row(result, arm == "ptq")
                folds = {}
                for fold in range(4):
                    task = f"tusl{fold}"
                    result = load(root / f"sweep-{task}" / "test" / "results.json") if arm == "ptq" else load(root / f"finetune-{task}" / "eval" / "results.json")
                    if result:
                        folds[fold] = tusl_row(result)
                if folds:
                    out.setdefault(setting, {}).setdefault("tusl", {}).setdefault(arm, {})[seed] = {
                        "folds": folds, **{m: statistics.mean(f[m] for f in folds.values()) for m in TUSL_METRICS}}
    # fp32 is read under every setting; a setting with no quantized result yet is left out
    return {setting: tasks for setting, tasks in out.items() if any(arm != "fp32" for arms in tasks.values() for arm in arms)}


def mean_sd(values: list[float]) -> str:
    if not values:
        return "—"
    if len(values) == 1:
        return f"{values[0]:.4f}"
    return f"{statistics.mean(values):.4f} ± {statistics.stdev(values):.4f}"


def render(summary: dict) -> str:
    lines = []
    for setting, tasks in summary.items():
        lines.append(f"\n## {setting}\n")
        for task, arms in tasks.items():
            metrics = TUSL_METRICS if task == "tusl" else BINARY_METRICS
            lines.append(f"### {task}\n")
            lines.append("| arm | seeds | " + " | ".join(metrics) + " |")
            lines.append("| --- | --- | " + " | ".join("---:" for _ in metrics) + " |")
            for arm in ("fp32", "ptq", "c", "qap"):
                rows = arms.get(arm, {})
                lines.append(f"| {arm} | {','.join(str(s) for s in sorted(rows))} | " +
                             " | ".join(mean_sd([rows[s][m] for s in sorted(rows)]) for m in metrics) + " |")
            key = "macro_auroc" if task == "tusl" else "auroc"
            for a, b in (("qap", "ptq"), ("c", "ptq"), ("qap", "c")):
                shared = sorted(set(arms.get(a, {})) & set(arms.get(b, {})))
                if shared:
                    diffs = [arms[a][s][key] - arms[b][s][key] for s in shared]
                    lines.append(f"- {a} − {b} ({key}, seeds {shared}): " + ", ".join(f"{d:+.4f}" for d in diffs) +
                                 (f"; mean {statistics.mean(diffs):+.4f}" if len(diffs) > 1 else ""))
            if task == "tusl":
                for arm in ("fp32", "ptq", "c", "qap"):
                    for seed, row in sorted(arms.get(arm, {}).items()):
                        lines.append(f"- {arm} seed {seed} per fold macro AUROC: " +
                                     ", ".join(f"f{f} {v['macro_auroc']:.3f}" for f, v in sorted(row["folds"].items())))
            lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None, help="where to write summary.json and summary.md")
    args = parser.parse_args()
    summary = collect(args.artifacts)
    text = render(summary)
    print(text)
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        (args.output / "summary.md").write_text(text.strip() + "\n")


if __name__ == "__main__":
    main()
