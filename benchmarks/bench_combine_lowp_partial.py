#!/usr/bin/env python3

import argparse
import json
import statistics

import torch

from mate.mha_interface import flash_attn_combine


def run_case(
    partial_dtype: torch.dtype,
    seq_len: int,
    heads: int,
    iterations: int,
) -> dict:
    torch.manual_seed(31)
    num_splits = 2
    batch = 1
    head_dim = 128
    partial = torch.randn(
        (num_splits, batch, seq_len, heads, head_dim),
        dtype=torch.float32,
        device="musa",
    ).to(partial_dtype)
    lse_storage = torch.randn(
        (num_splits, batch, heads, seq_len),
        dtype=torch.float32,
        device="musa",
    )
    lse_partial = lse_storage.transpose(2, 3)

    output, lse = flash_attn_combine(partial, lse_partial, out_dtype=torch.bfloat16)
    max_lse = lse_partial.amax(dim=0, keepdim=True)
    weights = torch.exp(lse_partial - max_lse)
    denominator = weights.sum(dim=0)
    reference_output = (partial.float() * weights.unsqueeze(-1)).sum(
        dim=0
    ) / denominator.unsqueeze(-1)
    reference_lse = max_lse.squeeze(0) + denominator.log()
    output_error = (output.float() - reference_output).abs().max().item()
    lse_error = (lse - reference_lse).abs().max().item()
    assert output_error <= 0.02
    assert lse_error <= 1e-4

    for _ in range(5):
        flash_attn_combine(partial, lse_partial, out_dtype=torch.bfloat16)
    torch.musa.synchronize()
    times = []
    for _ in range(iterations):
        start = torch.musa.Event(enable_timing=True)
        end = torch.musa.Event(enable_timing=True)
        start.record()
        flash_attn_combine(partial, lse_partial, out_dtype=torch.bfloat16)
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))

    return {
        "dtype": str(partial_dtype),
        "output_max_abs": output_error,
        "lse_max_abs": lse_error,
        "median_ms": statistics.median(times),
        "min_ms": min(times),
        "max_ms": max(times),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()

    print(
        json.dumps(
            {
                "torch": torch.__version__,
                "device": torch.musa.get_device_name(0),
                "seq_len": args.seq_len,
                "heads": args.heads,
                "f32": run_case(
                    torch.float32, args.seq_len, args.heads, args.iterations
                ),
                "bf16": run_case(
                    torch.bfloat16, args.seq_len, args.heads, args.iterations
                ),
                "fp16": run_case(
                    torch.float16, args.seq_len, args.heads, args.iterations
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
