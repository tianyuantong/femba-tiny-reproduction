"""Independent check of the protocol-v2 TUSL results (selected seed, W8A8).

Recomputes event metrics with sklearn from the score files, compares with results.json and
summary.json, checks every stage trained / validated / tested on the v2 fold files, PTQ
calibrated on 384 windows and picked its percentile on validation only. Checks two
windows per event plus 153 distinct test events across all folds.
"""
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score

ART, SUMMARY, DATA = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
SEED, SETTING, TOL = int(sys.argv[4]), "w8a8", 1e-9
STAGES = {"fp32": "fp32/finetune-tusl{k}-v2", "c": "c_w8a8/finetune-tusl{k}-v2", "qap": "qap_w8a8/finetune-tusl{k}-v2"}
problems = []


def events_from(npz: Path):
    with np.load(npz, allow_pickle=False) as z:
        probs = z["window_probs"].astype(np.float64)
        labels = z["window_labels"]
        names = [str(n) for n in z["window_names"]]
    grouped = defaultdict(list)
    for i, n in enumerate(names):
        grouped[n.rpartition("_w")[0]].append(i)
    for event, indices in grouped.items():
        if len(indices) != 2 or len(set(int(labels[i]) for i in indices)) != 1:
            raise ValueError(f"{npz}: invalid paired windows for {event}")
    ev = sorted(grouped)
    p = np.stack([probs[grouped[e]].mean(axis=0) for e in ev])
    y = np.array([labels[grouped[e][0]] for e in ev])
    return p, y, ev, probs, labels, names


def metrics(p, y):
    return {"macro_auroc": roc_auc_score(np.eye(3)[y], p, average="macro"),
            "macro_ap": average_precision_score(np.eye(3)[y], p, average="macro"),
            "bacc": balanced_accuracy_score(y, p.argmax(1)), "pred": np.bincount(p.argmax(1), minlength=3).tolist()}


summary = json.loads((SUMMARY / "summary.json").read_text())["w8a8"]["tusl-v2"]
manifests = {k: json.loads((DATA / f"fold{k}" / "data-manifest.json").read_text())["data"] for k in range(4)}
table, picks = {}, {}
for arm in ("fp32", "c", "qap", "ptq"):
    per_fold = {}
    seen_events = set()
    for k in range(4):
        if arm == "ptq":
            d = ART / f"seed{SEED}" / f"ptq_{SETTING}" / f"sweep-tusl{k}-v2"
            sel = json.loads((d / "selection.json").read_text())
            vals = {p: json.loads((d / f"p{p}" / "val" / "results.json").read_text())["splits"]["val"]["events"]["macro_auroc"]
                    for p in sel["candidates"].split()}
            first_best = next(p for p in sel["candidates"].split() if vals[p] == max(vals.values()))
            if float(first_best) != float(sel["selected_percentile"]):
                problems.append(f"ptq f{k}: selected {sel['selected_percentile']}, validation argmax {first_best}")
            index_lists = set()
            for p in sel["candidates"].split():
                cal = json.loads((d / f"p{p}" / "results.json").read_text())
                if cal["recipe"]["windows"] != 384 or cal["train_sha256"] != manifests[k]["train"]["sha256"]:
                    problems.append(f"ptq f{k} p{p}: {cal['recipe']['windows']} calibration windows or not the v2 train file")
                index_lists.add(cal["recipe"]["indices_sha256"])
            if len(index_lists) != 1:
                problems.append(f"ptq f{k}: percentiles calibrated on {len(index_lists)} different window lists")
            test_dir = d / "test"
            result = json.loads((test_dir / "results.json").read_text())
            if f"/p{first_best}/" not in result["checkpoint"]:
                problems.append(f"ptq f{k}: test checkpoint is not p{first_best}")
        else:
            d = ART / f"seed{SEED}" / STAGES[arm].format(k=k)
            run = json.loads((d / "run_manifest.json").read_text())["data"]
            for split in ("train", "val", "test"):
                if run[split]["sha256"] != manifests[k][split]["sha256"]:
                    problems.append(f"{arm} f{k}: trained with a {split} file that is not the v2 one")
            picks.setdefault(arm, {})[k] = json.loads((d / "complete.json").read_text())["best_checkpoint"].rsplit("epoch=", 1)[1].split("-")[0]
            test_dir = d / "eval"
            result = json.loads((test_dir / "results.json").read_text())
        for split in result["data"]:
            if result["data"][split]["sha256"] != manifests[k][split]["sha256"]:
                problems.append(f"{arm} f{k}: evaluated on a {split} file that is not the v2 one")
        p, y, ev, *_ = events_from(test_dir / "test_scores.npz")
        if seen_events.intersection(ev):
            raise ValueError(f"{arm}: an event appears in multiple test folds")
        seen_events.update(ev)
        mine, stored = metrics(p, y), result["splits"]["test"]["events"]
        for ours, theirs in (("macro_auroc", "macro_auroc"), ("macro_ap", "macro_ap"), ("bacc", "balanced_accuracy")):
            if abs(mine[ours] - stored[theirs]) > TOL:
                problems.append(f"{arm} f{k} {ours}: {mine[ours]:.6f} vs results.json {stored[theirs]:.6f}")
        per_fold[k] = mine
    if len(seen_events) != 153:
        raise ValueError(f"{arm}: expected 153 distinct test events, got {len(seen_events)}")
    means = {m: statistics.mean(per_fold[k][m] for k in range(4)) for m in ("macro_auroc", "macro_ap", "bacc")}
    for m, v in means.items():
        if abs(v - summary[arm][str(SEED)][m]) > TOL:
            problems.append(f"{arm} mean {m}: {v:.6f} vs summary {summary[arm][str(SEED)][m]:.6f}")
    table[arm] = (means, per_fold)

print("arm   AUROC   AP      BAcc   | per fold AUROC / BAcc / predicted-class counts")
for arm, (m, pf) in table.items():
    print(f"{arm:5s} {m['macro_auroc']:.4f}  {m['macro_ap']:.4f}  {m['bacc']:.4f} | "
          + "  ".join(f"f{k} {pf[k]['macro_auroc']:.3f}/{pf[k]['bacc']:.3f}/{pf[k]['pred']}" for k in range(4)))
print("single-class models:", [f"{a}-f{k}" for a, (_, pf) in table.items() for k in range(4) if sum(1 for c in pf[k]["pred"] if c) == 1])
print("selected epochs (val loss):", {a: picks[a] for a in picks})
print("\nPROBLEMS:" if problems else "\nALL CHECKS PASSED", *problems[:30], sep="\n  ")

if problems:
    raise SystemExit(1)
