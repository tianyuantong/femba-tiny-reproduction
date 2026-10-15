#*----------------------------------------------------------------------------*
#* Copyright (C) 2026.                                                        *
#* SPDX-License-Identifier: Apache-2.0                                        *
#*                                                                            *
#* Licensed under the Apache License, Version 2.0 (the "License");              *
#* you may not use this file except in compliance with the License.             *
#* You may obtain a copy of the License at                                      *
#*                                                                            *
#* http://www.apache.org/licenses/LICENSE-2.0                                   *
#*                                                                            *
#* Unless required by applicable law or agreed to in writing, software          *
#* distributed under the License is distributed on an "AS IS" BASIS,            *
#* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.    *
#* See the License for the specific language governing permissions and         *
#* limitations under the License.                                              *
#*----------------------------------------------------------------------------*
"""Export native TensorBoard training scalars without changing the training run.

Manifest: {"runs": [{"run": "fp32-s42", "stage": "finetune", "seed": 42,
                     "epochs": 5, "tensorboard_dir": "/path/to/exact/run"}]}.
The CSV retains duplicate events; plots omit post-fit validation using epoch tags.
"""

import argparse
import bisect
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


CSV_FIELDS = ("run", "stage", "seed", "tag", "kind", "event", "step", "wall_time", "value")


def load_manifest(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    runs = payload.get("runs") if isinstance(payload, dict) else None
    if not isinstance(runs, list) or not runs:
        raise ValueError("Manifest must contain a nonempty runs list")
    names = set()
    for run in runs:
        if not isinstance(run, dict) or not isinstance(run.get("run"), str) or not run["run"]:
            raise ValueError("Each run needs a nonempty run name")
        if run["run"] in names:
            raise ValueError(f"Duplicate run name: {run['run']}")
        names.add(run["run"])
        if run.get("stage") not in ("pretrain", "finetune"):
            raise ValueError("stage must be pretrain or finetune")
        if type(run.get("seed")) is not int or type(run.get("epochs")) is not int:
            raise ValueError("seed and epochs must be integers")
        if run["epochs"] <= 0 or not isinstance(run.get("tensorboard_dir"), str):
            raise ValueError("Each run needs positive epochs and a tensorboard_dir")
        directory = Path(run["tensorboard_dir"]).expanduser()
        directory = directory if directory.is_absolute() else path.parent / directory
        if not directory.is_dir() or not list(directory.glob("events.out.tfevents.*")):
            raise ValueError(f"Expected an exact TensorBoard run directory: {directory}")
        run["tensorboard_dir"] = str(directory.resolve())
    return runs


def read_events(run: dict) -> tuple[list[dict], list[str]]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from tensorboard.util import tensor_util

    accumulator = EventAccumulator(run["tensorboard_dir"], size_guidance={"scalars": 0, "tensors": 0},
                                   purge_orphaned_data=False)
    accumulator.Reload()
    rows, skipped = [], []
    for kind in ("scalars", "tensors"):
        for tag in sorted(accumulator.Tags().get(kind, [])):
            events = accumulator.Scalars(tag) if kind == "scalars" else accumulator.Tensors(tag)
            for ordinal, event in enumerate(events):
                value = event.value if kind == "scalars" else tensor_util.make_ndarray(event.tensor_proto)
                if kind == "tensors" and (value.size != 1 or value.dtype.kind not in "biuf"):
                    skipped.append(tag)
                    continue
                rows.append({"run": run["run"], "stage": run["stage"], "seed": run["seed"],
                             "tag": tag, "kind": kind, "event": ordinal, "step": event.step,
                             "wall_time": event.wall_time, "value": float(value)})
    if not rows:
        raise ValueError(f"No numeric scalar events found: {run['run']}")
    return rows, sorted(set(skipped))


def fit_epoch_series(rows: list[dict], tag: str, epochs: int) -> list[tuple[int, float]]:
    """Keep the first value per fit epoch; retain raw duplicates only in the CSV."""
    epoch_at_step = {}
    for row in rows:
        if row["tag"] == "epoch":
            value = row["value"]
            if not math.isfinite(value) or value != int(value) or value < 0:
                raise ValueError("TensorBoard epoch values must be nonnegative integers")
            epoch_at_step.setdefault(row["step"], int(value))
    if not epoch_at_step:
        raise ValueError("Epoch plots require the native epoch scalar tag")
    epoch_steps = sorted(epoch_at_step)
    values = {}
    for row in sorted(rows, key=lambda item: (item["wall_time"], item["event"])):
        if row["tag"] != tag:
            continue
        position = bisect.bisect_right(epoch_steps, row["step"]) - 1
        if position < 0:
            raise ValueError(f"No epoch tag at or before {tag} step {row['step']}")
        epoch = epoch_at_step[epoch_steps[position]]
        if epoch < epochs:
            values.setdefault(epoch + 1, row["value"])
    return sorted(values.items())


def _first_present(rows: list[dict], choices: tuple[str, ...]) -> str | None:
    tags = {row["tag"] for row in rows}
    return next((tag for tag in choices if tag in tags), None)


def _plot_loss_lr(runs: list[dict], by_run: dict, stage: str, output: Path, pyplot: object) -> str:
    figure, axes = pyplot.subplots(2, 1, figsize=(7.5, 6), layout="constrained")
    for run in runs:
        rows, label = by_run[run["run"]], f"{run['run']} (seed {run['seed']})"
        for choices, style, suffix in ((('train_loss_epoch', 'train_loss'), '-', 'train'),
                                       (('val_loss_epoch', 'val_loss'), '--', 'validation')):
            tag = _first_present(rows, choices)
            if tag:
                series = fit_epoch_series(rows, tag, run["epochs"])
                if series:
                    x, y = zip(*series)
                    axes[0].plot(x, y, style, marker=".", label=f"{label}: {suffix}")
        learning_rates = defaultdict(list)
        for row in rows:
            if row["tag"].startswith("lr-"):
                learning_rates[row["step"]].append(row["value"])
        steps = sorted(learning_rates)
        if steps:
            low = [min(learning_rates[step]) for step in steps]
            high = [max(learning_rates[step]) for step in steps]
            line, = axes[1].plot(steps, high, label=label)
            axes[1].fill_between(steps, low, high, color=line.get_color(), alpha=0.12)
    axes[0].set(xlabel="Fit epoch", ylabel="Native loss", title=f"{stage.capitalize()} loss")
    axes[1].set(xlabel="Optimizer step", ylabel="LR (parameter-group min/max)")
    for axis in axes:
        axis.grid(alpha=0.2)
        if axis.lines:
            axis.legend(fontsize=7)
    filename = f"{stage}-loss-lr.svg"
    figure.savefig(output / filename)
    pyplot.close(figure)
    return filename


def plot_curves(runs: list[dict], rows: list[dict], output: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot

    by_run = {run["run"]: [row for row in rows if row["run"] == run["run"]] for run in runs}
    paths = [_plot_loss_lr([run for run in runs if run["stage"] == stage], by_run,
                          stage, output, pyplot) for stage in ("pretrain", "finetune")
             if any(run["stage"] == stage for run in runs)]
    figure, axis = pyplot.subplots(figsize=(7.5, 3.6), layout="constrained")
    for run in runs:
        source = by_run[run["run"]]
        tag = _first_present(source, ("val_BinaryAUROC", "val_AUROC"))
        if tag:
            series = fit_epoch_series(source, tag, run["epochs"])
            if series:
                x, y = zip(*series)
                axis.plot(x, y, marker=".", label=f"{run['run']} (seed {run['seed']})")
    if axis.lines:
        axis.set(xlabel="Fit epoch", ylabel="Native positive-logit AUROC",
                 title="Native validation positive-logit AUROC")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7)
        filename = "native-positive-logit-auroc.svg"
        figure.savefig(output / filename)
        paths.append(filename)
    pyplot.close(figure)
    return paths


def export_curves(manifest: Path, output: Path) -> dict:
    runs = load_manifest(manifest)
    if output.exists():
        raise FileExistsError(f"Use a new output directory: {output}")
    rows, sources = [], []
    for run in runs:
        events, skipped = read_events(run)
        rows.extend(events)
        event_files = []
        for ordinal, path in enumerate(sorted(Path(run["tensorboard_dir"]).glob("events.out.tfevents.*"))):
            with path.open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            event_files.append({"event_file": ordinal, "bytes": path.stat().st_size, "sha256": digest})
        sources.append({key: run[key] for key in ("run", "stage", "seed", "epochs")}
                       | {"events": len(events), "source_files": event_files,
                          "skipped_nonscalar_tensor_tags": skipped})
    output.mkdir(parents=True)
    with (output / "scalars.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    figures = plot_curves(runs, rows, output)
    summary = {"runs": sources, "scalar_events": len(rows), "figures": figures,
               "nonfinite_events": sum(not math.isfinite(row["value"]) for row in rows),
               "csv": "scalars.csv", "csv_duplicates_preserved": True,
               "plot_rule": "First event per fit epoch; post-fit validation excluded by epoch",
               "auroc_score": "Native positive logit z1; not the corrected margin z1-z0",
               "learning_rate_plot": "Maximum across groups; shaded minimum to maximum"}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = export_curves(args.manifest.resolve(), args.output.resolve())
    print(json.dumps({"scalar_events": summary["scalar_events"], "figures": summary["figures"]}))


if __name__ == "__main__":
    main()
