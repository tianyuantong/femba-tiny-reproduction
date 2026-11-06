"""Storage and operation counts of the classifier at each weight setting, from the model itself.

Effective bits per parameter: quantized weights at their stored width; the positional embedding
at the activation width, because it passes through pos_quant; everything else (biases, state-space
parameters, depthwise convolutions, normalization, scales) at 32 bits, plus one 32-bit scale per
quantized weight tensor. Operations for one 5 s window: MACs of the Linear and Conv layers from
forward hooks, plus the state-space scan counted from its loop. BOPs multiply each quantized MAC by
weight bits and activation bits, and every floating-point operation by 32 x 32.
    usage: model_cost.py --source <pinned source> --output <json>
"""
import argparse
import json
import math
from pathlib import Path
import sys

SETTINGS = {"fp32": (32, 32, "uniform"), "w8a8": (8, 8, "uniform"), "w4a8": (4, 8, "uniform"),
            "ternary_a8": (2, 8, "ternary")}
SCAN_MULTIPLIES = 5  # per state element and step: dt*A, dt*B, dA*h, dB*x, h*C; plus D*x per width element


def linear_macs(model, torch) -> dict:
    counts = {}

    def hook(name, module, inputs, output):
        tensor = output.value if hasattr(output, "value") else output
        if isinstance(module, torch.nn.Linear):
            macs = module.in_features * module.out_features * tensor.numel() // module.out_features
        else:
            kernel = math.prod(module.kernel_size)
            macs = kernel * module.in_channels // module.groups * tensor.numel()
        counts[name] = counts.get(name, 0) + int(macs)

    handles = [module.register_forward_hook(lambda m, i, o, n=name: hook(n, m, i, o))
               for name, module in model.named_modules()
               if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d))]
    probe = torch.zeros(1, 22, 1280)
    with torch.no_grad():
        model(probe, torch.zeros_like(probe, dtype=torch.bool))
    for handle in handles:
        handle.remove()
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source))
    import torch
    from models.FEMBA_int8 import FEMBATinyInt8Classifier

    model = FEMBATinyInt8Classifier(weight_bit_width=8, activation_bit_width=8, quantization_enabled=False).eval()
    total = sum(parameter.numel() for parameter in model.parameters())
    quantized = [(name, module.weight.numel()) for name, module in model.named_modules()
                 if getattr(module, "weight_quant", None) is not None and hasattr(module, "weight")]
    assert len(quantized) == 18, len(quantized)
    weights = sum(count for _, count in quantized)
    macs = linear_macs(model, torch)
    quantized_names = {name for name, _ in quantized}
    quantized_macs = sum(count for name, count in macs.items() if name in quantized_names)
    other_macs = sum(macs.values()) - quantized_macs
    ssm = [module for module in model.modules() if hasattr(module, "d_state") and hasattr(module, "A_log")]
    length, width, state = model.seq_len, model.mamba_blocks[0].mamba_fwd.ssm.A_log.shape[1], ssm[0].d_state
    scan = len(ssm) * length * width * (state * SCAN_MULTIPLIES + 1)
    positional = model.pos_embed.numel()
    exponentials = len(ssm) * length * width * state
    result = {"parameters": total, "quantized_weights": weights, "other_parameters": total - weights,
              "quantized_weight_tensors": len(quantized), "macs_quantized_layers": quantized_macs,
              "macs_float_layers": other_macs, "scan_multiplies": scan, "scan_exponentials": exponentials,
              "positional_embedding": positional,
              "scan": {"blocks": len(ssm), "steps": length, "width": width, "state": state}, "settings": {}}
    for setting, (weight, activation, scheme) in SETTINGS.items():
        fp32 = setting == "fp32"
        fixed = activation * positional + 32 * (total - weights - positional) + 32 * len(quantized)
        bits = 32 * total if fp32 else weight * weights + fixed
        result["settings"][setting] = {
            "weight_bits": weight, "activation_bits": activation, "storage_megabytes": bits / 8e6,
            "effective_bits_per_parameter": bits / total, "bops_quantized_layers": quantized_macs * weight * activation,
            "bops_total": quantized_macs * weight * activation + (other_macs + scan) * 32 * 32}
        if scheme == "ternary":  # three levels need log2(3) bits at best
            entropy = math.log2(3) * weights + fixed
            result["settings"][setting]["effective_bits_entropy_coded"] = entropy / total
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
