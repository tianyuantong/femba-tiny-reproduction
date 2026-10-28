"""Independent recomputation of the formal-line TUAR / TUAB numbers from the logit captures.

For each arm: AUROC / AP with sklearn on float64(z1) - float64(z0), default balanced
accuracy (score > 0), the validation threshold re-derived by brute force and applied to
test, and for PTQ the validation-only candidate choice. Compares with results.json and
summary.json. Usage: verify_binary.py <artifacts> <summary dir> <seed> <setting> <task>...
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ART, SUMMARY, SEED, SETTING = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
TASKS = sys.argv[5:]
if not TASKS or any(task not in ("tuar", "tuab") for task in TASKS):
    raise ValueError("Specify one or both binary tasks: tuar tuab")
CANDIDATES = ("original", "p99_99", "p99_9", "p99", "p98", "p95", "p90")
KIND = {"fp32": "fp32", "c": "c", "qap": "qap_b"}
TOL = 1e-9
problems = []


def load(npz: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(npz, allow_pickle=False) as z:
        logits, labels = z["logits"].astype(np.float64), z["labels"].astype(np.int64)
    return labels, logits[:, 1] - logits[:, 0]


def bacc(labels: np.ndarray, score: np.ndarray, threshold: float) -> float:
    pred = score > threshold
    return 0.5 * (np.mean(pred[labels == 1]) + np.mean(~pred[labels == 0]))


def best_threshold(labels: np.ndarray, score: np.ndarray) -> float:
    candidates = np.unique(np.r_[score, 0.0, np.nextafter(score.min(), -np.inf)])
    pos, neg = np.sort(score[labels == 1]), np.sort(score[labels == 0])
    true_positive = len(pos) - np.searchsorted(pos, candidates, side="right")
    true_negative = np.searchsorted(neg, candidates, side="right")
    # balanced accuracy times 2*|pos|*|neg|: exact integers, so ties are exact
    numerator = true_positive * len(neg) + true_negative * len(pos)
    tied = candidates[numerator == numerator.max()]
    return float(sorted(tied, key=lambda t: (abs(t), t))[0])


def check(name: str, ours: float, theirs: float) -> None:
    if abs(ours - theirs) > TOL:
        problems.append(f"{name}: recomputed {ours:.10f} vs stored {theirs:.10f}")


summary = json.loads((SUMMARY / "summary.json").read_text())
for task in TASKS:
    manifest_dir = {"tuab": "tuab-ft"}.get(task, task)
    rows, test_labels = {}, {}
    for arm in ("fp32", "c", "qap", "ptq"):
        if arm == "ptq":
            d = ART / f"seed{SEED}" / f"ptq_{SETTING}" / f"sweep-{task}"
            result = json.loads((d / "results.json").read_text())
            val_rows = []
            for i, cand in enumerate(CANDIDATES):
                y, s = load(d / f"{cand}_val_logits.npz")
                loss = np.logaddexp(0.0, -(2 * y - 1) * s).mean()
                val_rows.append((-roc_auc_score(y, s), -average_precision_score(y, s), loss, i, cand, y, s))
            chosen = min(val_rows)
            if chosen[4] != result["selected"]["candidate"]:
                problems.append(f"{task} ptq: validation picks {chosen[4]}, sweep picked {result['selected']['candidate']}")
            threshold = best_threshold(chosen[5], chosen[6])
            check(f"{task} ptq threshold", threshold, result["selected"]["threshold_selection"]["threshold"])
            y, s = load(d / f"{result['selected']['candidate']}_test_logits.npz")
            stored = {"auroc": result["test"]["auroc"], "ap": result["test"]["ap"],
                      "bacc_default": result["test"]["balanced_accuracy"],
                      "bacc_val_thr": result["test"]["validation_selected"]["balanced_accuracy"]}
            label = f"{arm}({result['selected']['candidate']})"
        else:
            d = ART / f"seed{SEED}" / ("fp32" if arm == "fp32" else f"{arm}_{SETTING}") / f"finetune-{task}" / "eval"
            result = json.loads((d / "results.json").read_text())
            vy, vs = load(d / f"{KIND[arm]}_val_logits.npz")
            threshold = best_threshold(vy, vs)
            check(f"{task} {arm} threshold", threshold, json.loads((d / "selection.json").read_text())["threshold"])
            y, s = load(d / f"{KIND[arm]}_test_logits.npz")
            stored = {"auroc": result["auroc"], "ap": result["ap"], "bacc_default": result["default"]["balanced_accuracy"],
                      "bacc_val_thr": result["validation_selected"]["balanced_accuracy"]}
            label = arm
        ours = {"auroc": roc_auc_score(y, s), "ap": average_precision_score(y, s), "bacc_default": bacc(y, s, 0.0),
                "bacc_val_thr": bacc(y, s, threshold)}
        for m in ours:
            check(f"{task} {arm} {m} vs results.json", ours[m], stored[m])
            check(f"{task} {arm} {m} vs summary", ours[m], summary[SETTING][task][arm][str(SEED)][m])
        test_labels[arm] = y
        rows[label] = (ours, float(np.mean(s > 0)), float(np.mean(s > threshold)), len(y))
    if len({tuple(v) for v in test_labels.values()}) != 1:
        problems.append(f"{task}: arms scored on different test labels")
    expected = json.loads((ART.parent / "subject-data-20261025" / manifest_dir / "data-manifest.json").read_text())
    n_test = len(test_labels["fp32"])
    print(f"\n{task}: {n_test} test windows (manifest {expected['splits']['test']['windows']}), "
          f"{int(test_labels['fp32'].sum())} positive")
    if n_test != expected["splits"]["test"]["windows"]:
        problems.append(f"{task}: {n_test} test windows, manifest says {expected['splits']['test']['windows']}")
    print("arm            AUROC   AP      BAcc@0  BAcc@val  pos-rate@0  pos-rate@val")
    for label, (m, pos0, posv, _) in rows.items():
        print(f"{label:14s} {m['auroc']:.4f}  {m['ap']:.4f}  {m['bacc_default']:.4f}  {m['bacc_val_thr']:.4f}    "
              f"{pos0:.4f}      {posv:.4f}")

print()
print("PROBLEMS:" if problems else "ALL CHECKS PASSED", *problems, sep="\n  ")

if problems:
    raise SystemExit(1)
