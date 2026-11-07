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
"""Rotate the second Mamba block's scan outputs with H128 Hadamard matrices before PTQ.

For one FP32 parent of a binary task: R^T is folded into each target's out_proj and the gated
scan outputs of mamba_blocks.1 are rotated before their activation quantizers
(hadamard_rotation.py). The float model is first checked against the unrotated parent with
TF32 off: fixed-basis reconstruction, negative controls, per-vector RMS on the out_proj outputs
and elementwise logits on frozen train and validation windows; a failure blocks the run. Each
arm then follows the formal sweep: seven activation percentiles calibrated on the protocol's
512 training windows, the candidate with the highest validation AUROC evaluated on test with all
quantizers and with weight quantizers only. Arms: rotation alone, and rotation stacked on the
SmoothQuant channel migration of channel_migration.py at a given alpha.
"""

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys

import calibrate
import evaluate as capture
from channel_migration import FLOAT_TOLERANCE, channel_maxima, float_logits, migrate
from compare_calibration import evaluate_task, quantize_only, replay_split
from hadamard_rotation import (TARGETS, compare_parity, negative_controls, prepare_rotation,
                               require_fixed_basis, require_parity, verify_fixed_basis)
from sweep import CANDIDATES, validation_metrics
from threshold_control import timestamp

DEVICE = "cuda"
PROBE_WINDOWS = 32
VALIDATION_PROBE_SEED = 43
MODEL_CONTROLS = ("weights_transposed", "activation_shifted", "activation_only", "weights_only")


def load_parent(fp32: Path, config: dict):
    task, _ = capture.load_task(fp32, "fp32", target_config=config, num_classes=2, classification_type="bc",
                                device=DEVICE)
    return task


@contextmanager
def strict_fp32_runtime():
    """TF32 off and deterministic cuDNN for the parity gate; the formal flags return afterwards."""
    import torch

    saved = (torch.get_float32_matmul_precision(), torch.backends.cudnn.allow_tf32,
             torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        yield {"matmul_precision": "highest", "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
               "tf32_cudnn": False, "cudnn_benchmark": False, "cudnn_deterministic": True}
    finally:
        torch.set_float32_matmul_precision(saved[0])
        torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = saved[1:]


def capture_outputs(task, signals) -> dict:
    """Logits and each target's out_proj output as [B, L, 385], quantization off."""
    import torch

    modules, captured = dict(task.model.named_modules()), {}

    def keep(name: str, output) -> None:
        if not isinstance(output, torch.Tensor):
            raise TypeError(f"{name}.out_proj returned {type(output).__name__}; run the gate with quantization off")
        captured[name] = output.detach().reshape(signals.shape[0], -1, output.shape[-1]).cpu()

    handles = [modules[name + ".out_proj"].register_forward_hook(lambda module, inputs, output, name=name: keep(name, output))
               for name in TARGETS]
    try:
        with torch.no_grad():
            captured["logits"] = task.model(signals, torch.zeros_like(signals, dtype=torch.bool))[0].detach().cpu()
    finally:
        for handle in handles:
            handle.remove()
    return captured


def apply_variant(task, variant: str):
    """Fold and hook a fresh float parent as the gate variant asks; returns the plan."""
    import torch

    model = task.model
    if variant == "reference":
        return None
    plan = prepare_rotation(model, "identity" if variant == "identity" else "h128")
    if variant == "weights_transposed":  # W @ R^T with activations through R
        for name, projection in plan.projections(model).items():
            with torch.no_grad():
                projection.weight.copy_(projection.weight @ plan.matrix(name).T)
        plan.install(model)
    elif variant == "activation_shifted":  # correct weights, rotated activation moved by one feature
        plan.fold_into_model_(model)
        modules = dict(model.named_modules())
        for name in TARGETS:
            modules[name + ".ssm.output_quant"].register_forward_pre_hook(
                lambda module, inputs, name=name: (torch.roll(plan.rotate_activation(name, inputs[0]), 1, dims=-1),))
    elif variant == "activation_only":
        plan.install(model)
    elif variant == "weights_only":
        plan.fold_into_model_(model)
    else:  # identity, h128
        plan.fold_into_model_(model)
        plan.install(model)
    return plan


def run_gate(args, config: dict, probes: dict) -> dict:
    """Float parity of the rotated parent on frozen windows; raises when any gate fails."""
    with strict_fp32_runtime() as runtime:
        probes = {split: signals.to(DEVICE) for split, signals in probes.items()}
        reference = load_parent(args.fp32, config)
        captured = {split: capture_outputs(reference, signals) for split, signals in probes.items()}
        del reference
        structure_plan = prepare_rotation(load_parent(args.fp32, config).model, "h128")
        basis = verify_fixed_basis(structure_plan)
        require_fixed_basis(basis)
        controls = negative_controls(structure_plan)
        report = {"runtime": runtime, "probe_windows": {split: int(signals.shape[0]) for split, signals in probes.items()},
                  "fixed_basis": basis, "negative_controls": controls, "variants": {}, "model_negative_controls": {}}
        for variant in ("identity", "h128", *MODEL_CONTROLS):
            task = load_parent(args.fp32, config)
            apply_variant(task, variant)
            rows = {split: compare_parity(captured[split], capture_outputs(task, signals)) for split, signals in probes.items()}
            del task
            if variant in MODEL_CONTROLS:
                report["model_negative_controls"][variant] = {
                    "rejected": not any(row["allclose"] for row in rows.values()), "splits": rows}
            else:
                report["variants"][variant] = rows
        for variant in ("identity", "h128"):
            for row in report["variants"][variant].values():
                require_parity(row)
        if not controls["passed"] or not all(row["rejected"] for row in report["model_negative_controls"].values()):
            raise RuntimeError("A negative control was not rejected")
        report["passed"] = True
    return report


def prepare_candidate(args, config: dict, dataset, indices: list[int], percentile: float, migration=None) -> tuple:
    """A fresh parent, migrated when asked, rotated and calibrated at one percentile."""
    from models.femba_ptq import calibrate_quantized

    task, record = load_parent(args.fp32, config), {"percentile": percentile}
    if migration is not None:
        maxima, alpha = migration
        factors = migrate(task.model, maxima, alpha)
        record["migration_alpha"] = alpha
        record["migration_factors"] = {prefix: {"min": factor.min().item(), "median": factor.median().item(),
                                                "max": factor.max().item()} for prefix, factor in factors.items()}
    plan = prepare_rotation(task.model, "h128")
    plan.fold_into_model_(task.model)
    plan.install(task.model)
    calibrate.set_percentile(task.model, percentile)
    report = calibrate_quantized(task.model, calibrate.batches(task, dataset, indices))
    record["rotated_scales"] = {name: report["scales"][name + ".ssm.output_quant.act_quant"] for name in TARGETS}
    record["out_proj_weight_scales"] = {name: value for name, value in report["scales"].items()
                                        if any(name.startswith(target + ".out_proj") for target in TARGETS)}
    return task, record


def rotated_candidate(args, config: dict, dataset, indices: list[int], percentile: float, migration=None) -> tuple:
    """prepare_candidate scored on validation."""
    task, record = prepare_candidate(args, config, dataset, indices, percentile, migration)
    val = replay_split(task, args.data, "val")
    metrics = validation_metrics(val["labels"], val["logits"])
    record["validation"] = {key: metrics[key] for key in ("auroc", "ap", "loss")}
    record["validation_default_balanced_accuracy"] = metrics["threshold_selection"]["validation_default"]["balanced_accuracy"]
    return task, record


def choose(records: list[dict]) -> int:
    """The formal sweep's rule: highest validation AUROC, then AP, then lowest loss, then preset order."""
    return min(range(len(records)), key=lambda index: (
        -records[index]["validation"]["auroc"], -records[index]["validation"]["ap"],
        records[index]["validation"]["loss"], index))


def run_arm(args, config: dict, dataset, indices: list[int], migration=None) -> dict:
    tasks, records = {}, []
    for name, percentile in CANDIDATES:
        print(f"CANDIDATE {'rotation+migration' if migration else 'rotation'} {name}", flush=True)
        tasks[name], record = rotated_candidate(args, config, dataset, indices, percentile, migration)
        records.append({"candidate": name, **record})
    chosen = records[choose(records)]
    print(f"TEST {chosen['candidate']}", flush=True)
    best = tasks[chosen["candidate"]]
    evaluation = {"all": evaluate_task(best, args.data)}
    quantize_only(best.model, "weights")
    evaluation["weights_only"] = evaluate_task(best, args.data)
    tasks.clear()
    return {"candidates": records, "chosen_candidate": chosen["candidate"], "chosen_percentile": chosen["percentile"],
            "chosen": evaluation}


def baseline_check(args, config: dict, dataset, indices: list[int], percentile: float = 95.0, score: bool = True) -> dict:
    """The unrotated parent at the formal line's selected percentile, to tie this run to the sweep."""
    from models.femba_ptq import calibrate_quantized

    task = load_parent(args.fp32, config)
    calibrate.set_percentile(task.model, percentile)
    report = calibrate_quantized(task.model, calibrate.batches(task, dataset, indices))
    record = {"percentile": percentile,
              "scan_output_scales": {name: report["scales"][name + ".ssm.output_quant.act_quant"] for name in TARGETS}}
    if score:
        val = replay_split(task, args.data, "val")
        metrics = validation_metrics(val["labels"], val["logits"])
        record["validation"] = {key: metrics[key] for key in ("auroc", "ap", "loss")}
    return record


def run(args) -> dict:
    import torch

    args.source, args.data, args.output = args.source.resolve(), args.data.resolve(), args.output.resolve()
    capture.use_data_splits(args.data)
    sys.path.insert(0, str(args.source))
    from datasets.hdf5_dataset import HDF5Loader

    config = capture.argument_quantization_config(args)
    args.output.mkdir(parents=True, exist_ok=False)
    runtime = capture.configure_runtime()
    dataset, val_dataset = (HDF5Loader(str(args.data / f"{split}.h5"), use_cache=False) for split in ("train", "val"))
    try:
        indices = calibrate.calibration_indices(len(dataset))
        parent = load_parent(args.fp32, config)
        val_indices = torch.randperm(len(val_dataset), generator=torch.Generator().manual_seed(VALIDATION_PROBE_SEED))
        probes = {"train": next(calibrate.batches(parent, dataset, indices[:PROBE_WINDOWS])).cpu(),
                  "val": next(calibrate.batches(parent, val_dataset, val_indices[:PROBE_WINDOWS].tolist())).cpu()}
        print("GATE", flush=True)
        gate = run_gate(args, config, probes)
        reference = float_logits(parent, probes["train"].to(DEVICE))
        rotated = load_parent(args.fp32, config)
        apply_variant(rotated, "h128")
        change = (float_logits(rotated, probes["train"].to(DEVICE)) - reference).abs().max().item()
        if change > FLOAT_TOLERANCE:
            raise RuntimeError(f"Rotation changed the float logits by {change} under the formal runtime")
        rotation = prepare_rotation(rotated.model, "h128").to_dict()
        del rotated
        result = {"time_utc": timestamp(), "seed": args.seed, "fp32_sha256": capture.sha256(args.fp32),
                  "quantization_config": config, "rotation": rotation, "targets": list(TARGETS), "parity": gate,
                  "float_logit_change": change, "float_logit_tolerance": FLOAT_TOLERANCE,
                  "calibration": {"split": "train", "windows": len(indices), "train_sha256": capture.sha256(args.data / "train.h5")},
                  "percentiles": [percentile for _, percentile in CANDIDATES],
                  "selection_rule": "highest validation AUROC among the seven formal percentiles, then AP, then loss"}
        if args.smoke:  # calibrate once at 95 with and without rotation; no validation or test replay
            print("SMOKE", flush=True)
            result["baseline"] = baseline_check(args, config, dataset, indices, score=False)
            task, record = prepare_candidate(args, config, dataset, indices, 95.0)
            with torch.no_grad():
                logits = task.model(probes["train"].to(DEVICE), torch.zeros_like(probes["train"], dtype=torch.bool).to(DEVICE))[0]
            record["probe_logits_finite"] = bool(torch.isfinite(logits).all())
            result["smoke"] = record
        else:
            result["baseline"] = baseline_check(args, config, dataset, indices)
            result["arms"] = {"rotation": run_arm(args, config, dataset, indices)}
            if args.migration_alpha is not None:
                maxima = channel_maxima(parent, calibrate.batches(parent, dataset, indices))
                result["arms"]["rotation_migration"] = {"migration_alpha": args.migration_alpha,
                                                        **run_arm(args, config, dataset, indices, (maxima, args.migration_alpha))}
    finally:
        dataset.data.close()
        val_dataset.data.close()
    result.update(data={split: capture.sha256(args.data / f"{split}.h5") for split in ("val", "test")},
                  runtime=runtime, source=capture.source_record(args.source),
                  runner_sha256={name: capture.sha256(Path(__file__).with_name(name))
                                 for name in ("rotation_ptq.py", "hadamard_rotation.py", "channel_migration.py")})
    capture.write_json(args.output / "results.json", result)
    return result


def summary(result: dict) -> dict:
    if "arms" not in result:
        return {"smoke": True, "parity_passed": result["parity"]["passed"], "float_logit_change": result["float_logit_change"],
                "scan_output_scales": result["baseline"]["scan_output_scales"], "rotated_scales": result["smoke"]["rotated_scales"],
                "probe_logits_finite": result["smoke"]["probe_logits_finite"]}
    return {arm: {"chosen_percentile": record["chosen_percentile"],
                  "validation": {row["candidate"]: round(row["validation"]["auroc"], 4) for row in record["candidates"]},
                  "test_auroc": round(record["chosen"]["all"]["test"]["auroc"], 4),
                  "test_default_bacc": round(record["chosen"]["all"]["test"]["default"]["balanced_accuracy"], 4),
                  "weights_only_test_auroc": round(record["chosen"]["weights_only"]["test"]["auroc"], 4)}
            for arm, record in result["arms"].items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "data", "fp32", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True, help="Formal-line seed of the FP32 parent, recorded only")
    parser.add_argument("--migration-alpha", type=float, default=None,
                        help="Also run rotation stacked on the channel migration at this alpha")
    parser.add_argument("--smoke", action="store_true", help="Only the parity gate and the float check")
    capture.add_quantization_arguments(parser)
    result = run(parser.parse_args())
    print(json.dumps(summary(result)))


if __name__ == "__main__":
    main()
