"""Where quantization hurts a PTQ and a QAP model, by block and by site (validation only).

With the weights quantized, one group of activation quantizers stays on and the rest run in float;
the drop from the weights-only model is compared between the two models, block by block and, inside
the second block, site by site. Every pass runs on one device, so the references and drops stay
comparable; sensitivity.py's numbers are kept as a cross-check. On the first 512 validation windows with everything on, each activation site also records its
signal-to-quantization-noise ratio and its shares of zeroed and clipped values.
"""

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

import evaluate as capture
from compare_calibration import quantize_only, replay_split
from ptq_ablation import float_activations
from sweep import validation_metrics
from threshold_control import timestamp

SIGNAL_WINDOWS = 512


def group(name: str) -> str:
    return "block0" if name.startswith("mamba_blocks.0") else "block1" if name.startswith("mamba_blocks.1") else "other"


def run_windows(task, data: Path, limit: int) -> None:
    import torch
    from torch.utils.data import DataLoader, Subset
    from datasets.hdf5_dataset import HDF5Loader

    dataset = HDF5Loader(str(data / "val.h5"), use_cache=False)
    device = next(task.parameters()).device
    try:
        with torch.no_grad():
            for signals, _ in DataLoader(Subset(dataset, list(range(limit))), batch_size=32, shuffle=False):
                signals = task.normalize_fct(signals.to(device))
                task.model(signals, torch.zeros_like(signals, dtype=torch.bool))
    finally:
        dataset.data.close()


def signal_statistics(task, names: list[str], data: Path, limit: int) -> dict:
    """Noise, zeroed and clipped shares of each activation site with every quantizer on."""
    def value(tensor):
        return tensor.value if hasattr(tensor, "value") else tensor

    totals = {name: {"signal": 0.0, "noise": 0.0, "zeroed": 0, "clipped": 0, "elements": 0} for name in names}

    def record(name, inputs, output):
        x, y, row = value(inputs[0]).detach(), value(output).detach(), totals[name]
        row["signal"] += float(x.double().pow(2).sum())
        row["noise"] += float((x - y).double().pow(2).sum())
        row["zeroed"] += int(((y == 0) & (x != 0)).sum())
        row["clipped"] += int((x.abs() > y.abs().max()).sum())
        row["elements"] += x.numel()

    modules = dict(task.model.named_modules())
    hooks = [modules[name].register_forward_hook(lambda _, inputs, output, name=name: record(name, inputs, output))
             for name in names]
    try:
        run_windows(task, data, limit)
    finally:
        for hook in hooks:
            hook.remove()
    return {name: {"sqnr_db": 10 * math.log10(row["signal"] / row["noise"]) if row["noise"] > 0 else None,
                   "zeroed_share": row["zeroed"] / row["elements"], "clipped_share": row["clipped"] / row["elements"]}
            for name, row in totals.items()}


def run(args) -> dict:
    args.source, args.data, args.output = args.source.resolve(), args.data.resolve(), args.output.resolve()
    capture.use_data_splits(args.data)
    sys.path.insert(0, str(args.source))
    config = capture.argument_quantization_config(args)
    reference = json.loads(args.ptq_sensitivity.read_text())
    if capture.sha256(args.ptq) != reference["checkpoint_sha256"]:
        raise ValueError("The PTQ checkpoint is not the one the sensitivity results came from")
    names = list(reference["alone_validation_auroc"])
    groups = {key: [name for name in names if group(name) == key] for key in ("other", "block0", "block1")}
    assert [len(groups[key]) for key in ("other", "block0", "block1")] == [7, 14, 14]
    capture.configure_runtime()
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "results.json"
    with np.load(args.qap_val_logits, allow_pickle=False) as stored:
        qap_stored = validation_metrics(stored["labels"], stored["logits"])["auroc"]
    result = json.loads(path.read_text()) if path.is_file() else {
        "quantization_config": config, "split": "val", "groups": groups,
        "device": capture.configure_runtime().get("gpu"),
        "ptq_checkpoint_sha256": reference["checkpoint_sha256"], "qap_checkpoint_sha256": capture.sha256(args.qap),
        "stored": {"qap_all": qap_stored, "ptq": reference["reference_validation_auroc"],
                   "ptq_alone": reference["alone_validation_auroc"]},
        "ptq": {"alone": {}, "groups": {}}, "qap": {"alone": {}, "groups": {}}}
    capture.configure_runtime()

    def load(kind):
        checkpoint = args.qap if kind == "qap_b" else args.ptq
        return capture.load_task(checkpoint, kind, expected_config=config)[0]

    def score(kind, prepare):
        """AUROC, balanced accuracy at threshold 0 and at the threshold chosen on this split."""
        task = load(kind)
        prepare(task.model)
        arrays = replay_split(task, args.data, "val")
        row = validation_metrics(arrays["labels"], arrays["logits"])
        chosen = row["threshold_selection"]
        return {"auroc": row["auroc"], "bacc_default": capture.decision_metrics(arrays["labels"], arrays["logits"])["balanced_accuracy"],
                "bacc_chosen_threshold": chosen["objective_numerator"] / chosen["objective_denominator"]}

    def save():
        result["time_utc"] = timestamp()
        path.write_text(json.dumps(result, indent=2) + "\n")

    if "signal" not in result:
        result["signal"] = {"ptq": signal_statistics(load("ptq_original"), names, args.data, args.signal_windows),
                            "qap": signal_statistics(load("qap_b"), names, args.data, args.signal_windows)}
        save()
    if args.signal_only:
        return result
    models = (("qap_b", "qap"), ("ptq_original", "ptq"))
    for kind, key in models:
        if "all" not in result[key]:
            print(f"{key} all", flush=True)
            result[key]["all"] = score(kind, lambda model: None)
            save()
        if "weights_only" not in result[key]:
            print(f"{key} weights only", flush=True)
            result[key]["weights_only"] = score(kind, lambda model: quantize_only(model, "weights"))
            save()
        for name, members in groups.items():
            if name not in result[key]["groups"]:
                print(f"{key} group {name}", flush=True)
                off = [other for other in names if other not in members]
                result[key]["groups"][name] = score(kind, lambda model: float_activations(model, off))
                save()
    print("STAGE1_DONE", flush=True)
    for kind, key in models:
        for name in groups["block1"]:
            if name not in result[key]["alone"]:
                print(f"{key} alone {name}", flush=True)
                off = [other for other in names if other != name]
                result[key]["alone"][name] = score(kind, lambda model: float_activations(model, off))
                save()
    print("STAGE2_DONE", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "data", "qap", "ptq", "ptq-sensitivity", "qap-val-logits", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    capture.add_quantization_arguments(parser)
    parser.add_argument("--signal-windows", type=int, default=SIGNAL_WINDOWS)
    parser.add_argument("--signal-only", action="store_true", help="Stop after the signal statistics")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
