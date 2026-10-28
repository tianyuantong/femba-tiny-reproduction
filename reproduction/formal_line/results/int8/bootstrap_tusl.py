"""How much the TUSL v2 differences between arms depend on which subjects are tested.

Keeps the trained models fixed and resamples the test set: within each fold, test
subjects are drawn with replacement and every drawn subject brings all of its events.
Each replicate recomputes the event-level macro AUROC of every fold for all arms and
seeds on the same resample, averages the four folds within a seed and then the three
seeds, exactly as the reported numbers. Macro AUROC is undefined when a fold lacks one
of the three classes (every real test fold has all three), so such replicates are drawn
again and the redraw rate is reported. An event-level bootstrap runs for comparison.
The observed differences are also given per fold and with each fold left out.
Usage: bootstrap_tusl.py <artifacts> <summary.json> <output.json>
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ART, SUMMARY, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
REPLICATES, RNG_SEED, FOLDS = 2000, 20261028, 4
SEEDS = ("42", "43", "44")
ARMS = ("fp32", "ptq", "c", "qap")
PAIRS = (("qap", "c"), ("qap", "ptq"), ("c", "ptq"), ("qap", "fp32"), ("ptq", "fp32"))
STAGE = {"fp32": "fp32/finetune-tusl{k}-v2/eval", "c": "c_w8a8/finetune-tusl{k}-v2/eval",
         "qap": "qap_w8a8/finetune-tusl{k}-v2/eval", "ptq": "ptq_w8a8/sweep-tusl{k}-v2/test"}


def event_probs(npz: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    with np.load(npz, allow_pickle=False) as z:
        window_probs, window_labels = z["window_probs"].astype(np.float64), z["window_labels"]
        names = [str(n) for n in z["window_names"]]
    grouped = defaultdict(list)
    for i, name in enumerate(names):
        grouped[name.rpartition("_w")[0]].append(i)
    events = sorted(grouped)
    probs = np.stack([window_probs[grouped[e]].mean(axis=0) for e in events])
    labels = np.array([int(window_labels[grouped[e][0]]) for e in events])
    return events, probs, labels


def load_fold(k: int) -> dict:
    probs, reference = {}, None
    for seed in SEEDS:
        for arm in ARMS:
            events, p, y = event_probs(ART / f"seed{seed}" / STAGE[arm].format(k=k) / "test_scores.npz")
            if reference is None:
                reference = (events, y)
            assert events == reference[0] and (y == reference[1]).all(), f"fold {k}: {arm} seed {seed} scored other events"
            probs[seed, arm] = p
    subjects = defaultdict(list)
    for i, event in enumerate(reference[0]):
        subjects[event.split("_", 1)[0]].append(i)
    return {"labels": reference[1], "probs": probs, "subjects": [np.array(v) for v in subjects.values()]}


def fold_auroc(folds: list[dict], index_sets: list[np.ndarray]) -> np.ndarray:
    """Event-level macro AUROC of every fold as a seeds x arms x folds array."""
    scores = np.empty((len(SEEDS), len(ARMS), FOLDS))
    for k, (fold, idx) in enumerate(zip(folds, index_sets)):
        onehot = np.eye(3)[fold["labels"][idx]]
        for i, seed in enumerate(SEEDS):
            for j, arm in enumerate(ARMS):
                scores[i, j, k] = roc_auc_score(onehot, fold["probs"][seed, arm][idx], average="macro")
    return scores


def draw(folds: list[dict], rng: np.random.Generator, unit: str) -> tuple[list[np.ndarray], int]:
    redraws = 0
    while True:
        index_sets = []
        for fold in folds:
            if unit == "subject":
                picks = rng.integers(len(fold["subjects"]), size=len(fold["subjects"]))
                index_sets.append(np.concatenate([fold["subjects"][p] for p in picks]))
            else:
                index_sets.append(rng.integers(len(fold["labels"]), size=len(fold["labels"])))
        if all(len(np.unique(fold["labels"][idx])) == 3 for fold, idx in zip(folds, index_sets)):
            return index_sets, redraws
        redraws += 1


def interval(values: np.ndarray) -> list[float]:
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def main() -> None:
    folds = [load_fold(k) for k in range(FOLDS)]
    observed_folds = fold_auroc(folds, [np.arange(len(f["labels"])) for f in folds])
    observed = observed_folds.mean(axis=2)
    summary = json.loads(SUMMARY.read_text())["w8a8"]["tusl-v2"]
    for i, seed in enumerate(SEEDS):
        for j, arm in enumerate(ARMS):
            assert abs(observed[i, j] - summary[arm][seed]["macro_auroc"]) < 1e-12, (arm, seed)
    report = {"replicates": REPLICATES, "rng_seed": RNG_SEED,
              "test": {"events": [len(f["labels"]) for f in folds], "subjects": [len(f["subjects"]) for f in folds]},
              "observed_mean": {arm: float(observed[:, j].mean()) for j, arm in enumerate(ARMS)},
              "by_fold": {}}
    print("observed difference per fold (seeds 42/43/44) and three-seed mean with each fold left out")
    for a, c in PAIRS:
        diff = observed_folds[:, ARMS.index(a)] - observed_folds[:, ARMS.index(c)]  # seeds x folds
        left_out = {f"without_fold{k}": {"mean": float(np.delete(diff, k, axis=1).mean()),
                                         "per_seed": dict(zip(SEEDS, np.delete(diff, k, axis=1).mean(axis=1).tolist()))}
                    for k in range(FOLDS)}
        report["by_fold"][f"{a}-{c}"] = {"per_fold": dict(zip(SEEDS, diff.tolist())), "leave_one_fold_out": left_out}
        print(f"{a}-{c}: " + "  ".join(f"f{k} " + "/".join(f"{x:+.3f}" for x in diff[:, k]) for k in range(FOLDS))
              + " | " + "  ".join(f"no f{k} {v['mean']:+.4f}" for k, v in enumerate(left_out.values())))
    for unit in ("subject", "event"):
        rng = np.random.default_rng(RNG_SEED)
        stats, redraws = np.empty((REPLICATES, len(SEEDS), len(ARMS))), 0
        for b in range(REPLICATES):
            index_sets, extra = draw(folds, rng, unit)
            redraws += extra
            stats[b] = fold_auroc(folds, index_sets).mean(axis=2)
        pairs = {}
        for a, c in PAIRS:
            per_seed = stats[:, :, ARMS.index(a)] - stats[:, :, ARMS.index(c)]
            observed_diff = observed[:, ARMS.index(a)] - observed[:, ARMS.index(c)]
            pairs[f"{a}-{c}"] = {"observed": float(observed_diff.mean()), "ci95": interval(per_seed.mean(axis=1)),
                                 "share_above_zero": float((per_seed.mean(axis=1) > 0).mean()),
                                 "per_seed": {seed: {"observed": float(observed_diff[i]), "ci95": interval(per_seed[:, i])}
                                              for i, seed in enumerate(SEEDS)}}
        report[unit] = {"redraw_rate": redraws / (redraws + REPLICATES), "pairs": pairs}
        print(f"\n{unit}-level bootstrap ({REPLICATES} replicates, {redraws} redrawn for a missing class)")
        print("pair       observed   95% interval        share > 0 | per seed: observed [95% interval]")
        for name, p in pairs.items():
            print(f"{name:9s} {p['observed']:+.4f}   [{p['ci95'][0]:+.4f}, {p['ci95'][1]:+.4f}]   {p['share_above_zero']:.3f}   | "
                  + "  ".join(f"{s}: {v['observed']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]" for s, v in p["per_seed"].items()))
    OUT.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
