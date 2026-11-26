#!/usr/bin/env python3

import argparse
import json
import statistics

import torch
from flashinfer.cascade import merge_state
from flashinfer.prefill import BatchPrefillWithRaggedKVCacheWrapper

from mate.mha_interface import flash_attn_combine, flash_attn_varlen_func


def _sync() -> None:
    torch.musa.synchronize()


def _timings(fn, flush: torch.Tensor, warmup: int, iterations: int) -> dict:
    for _ in range(warmup):
        fn()
    _sync()

    values = []
    for _ in range(iterations):
        flush.zero_()
        _sync()
        start = torch.musa.Event(enable_timing=True)
        end = torch.musa.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end))

    ordered = sorted(values)
    return {
        "iterations": iterations,
        "median_ms": statistics.median(values),
        "min_ms": ordered[0],
        "p10_ms": ordered[max(0, int(len(ordered) * 0.1) - 1)],
        "p90_ms": ordered[min(len(ordered) - 1, int(len(ordered) * 0.9))],
        "max_ms": ordered[-1],
    }


def _reference_varlen(q, k, v, q_lens, k_lens, causal, sm_scale):
    q = q.float().cpu()
    k = k.float().cpu()
    v = v.float().cpu()
    outputs = []
    lses = []
    q_offset = 0
    k_offset = 0
    for q_len, k_len in zip(q_lens, k_lens, strict=True):
        q_i = q[q_offset : q_offset + q_len].transpose(0, 1).float()
        k_i = k[k_offset : k_offset + k_len].transpose(0, 1).float()
        v_i = v[k_offset : k_offset + k_len].transpose(0, 1).float()
        scores = torch.matmul(q_i, k_i.transpose(1, 2)) * sm_scale
        if causal:
            q_pos = torch.arange(q_len)[:, None]
            k_pos = torch.arange(k_len)[None, :]
            valid = k_pos <= q_pos + k_len - q_len
            assert scores.shape[-2:] == valid.shape, (
                q_i.shape,
                k_i.shape,
                scores.shape,
                valid.shape,
            )
            scores = scores.masked_fill(~valid.unsqueeze(0), -torch.inf)
        lse = torch.logsumexp(scores, dim=-1)
        probs = torch.softmax(scores, dim=-1)
        outputs.append(torch.matmul(probs, v_i).transpose(0, 1))
        lses.append(lse.transpose(0, 1))
        q_offset += q_len
        k_offset += k_len
    return torch.cat(outputs, dim=0), torch.cat(lses, dim=0)


def _max_abs_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return (actual.float() - expected.float()).abs().max().item()


def correctness(backend: str) -> dict:
    torch.manual_seed(7)
    q_lens = [3, 5]
    k_lens = [3, 5]
    num_qo_heads = 2
    num_kv_heads = 2
    head_dim = 8
    sm_scale = head_dim**-0.5
    qo_indptr = torch.tensor(
        [0, q_lens[0], sum(q_lens)], dtype=torch.int32, device="musa"
    )
    kv_indptr = torch.tensor(
        [0, k_lens[0], sum(k_lens)], dtype=torch.int32, device="musa"
    )
    q = torch.randn(
        (sum(q_lens), num_qo_heads, head_dim),
        dtype=torch.bfloat16,
        device="musa",
    )
    k = torch.randn(
        (sum(k_lens), num_kv_heads, head_dim),
        dtype=torch.bfloat16,
        device="musa",
    )
    v = torch.randn_like(k)
    wrapper = BatchPrefillWithRaggedKVCacheWrapper(
        torch.empty(1024, device="musa"), backend=backend
    )
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        causal=True,
        sm_scale=sm_scale,
        q_data_type=torch.bfloat16,
    )
    output, lse = wrapper.forward_return_lse(q, k, v, causal=True, sm_scale=sm_scale)
    reference_output, reference_lse = _reference_varlen(
        q, k, v, q_lens, k_lens, True, sm_scale
    )

    torch.manual_seed(11)
    seq_len = 65
    num_heads = 8
    head_dim_v = 128
    v_a = torch.randn(
        (seq_len, num_heads, head_dim_v),
        dtype=torch.bfloat16,
        device="musa",
    )
    v_b = torch.randn_like(v_a)
    s_a = torch.randn((seq_len, num_heads), dtype=torch.float32, device="musa")
    s_b = torch.randn_like(s_a)
    merged_v, merged_lse = merge_state(v_a, s_a, v_b, s_b)
    max_lse = torch.maximum(s_a, s_b)
    weight_a = torch.exp(s_a - max_lse)
    weight_b = torch.exp(s_b - max_lse)
    denominator = weight_a + weight_b
    reference_merged_v = (
        weight_a.unsqueeze(-1) * v_a.float() + weight_b.unsqueeze(-1) * v_b.float()
    ) / denominator.unsqueeze(-1)
    reference_merged_lse = max_lse + denominator.log()

    return {
        "prefill_output_max_abs": _max_abs_error(output, reference_output),
        "prefill_lse_max_abs": _max_abs_error(lse, reference_lse),
        "cascade_output_max_abs": _max_abs_error(merged_v, reference_merged_v),
        "cascade_lse_max_abs": _max_abs_error(merged_lse, reference_merged_lse),
    }


def benchmark(iters: int, warmup: int, backend: str) -> dict:
    torch.manual_seed(17)
    q_lens = [128, 257]
    k_lens = [128, 257]
    num_qo_heads = 16
    num_kv_heads = 8
    head_dim = 128
    sm_scale = head_dim**-0.5
    qo_indptr = torch.tensor(
        [0, q_lens[0], sum(q_lens)], dtype=torch.int32, device="musa"
    )
    kv_indptr = torch.tensor(
        [0, k_lens[0], sum(k_lens)], dtype=torch.int32, device="musa"
    )
    q = torch.randn(
        (sum(q_lens), num_qo_heads, head_dim),
        dtype=torch.bfloat16,
        device="musa",
    )
    k = torch.randn(
        (sum(k_lens), num_kv_heads, head_dim),
        dtype=torch.bfloat16,
        device="musa",
    )
    v = torch.randn_like(k)
    wrapper = BatchPrefillWithRaggedKVCacheWrapper(
        torch.empty(1 << 24, dtype=torch.uint8, device="musa"),
        backend=backend,
    )
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        causal=True,
        sm_scale=sm_scale,
        q_data_type=torch.bfloat16,
    )
    scheduler_metadata = wrapper._scheduler_metadata
    direct_qo_indptr = wrapper._qo_indptr_device
    direct_kv_indptr = wrapper._kv_indptr_device

    def direct_prefill():
        return flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=direct_qo_indptr,
            cu_seqlens_k=direct_kv_indptr,
            max_seqlen_q=257,
            max_seqlen_k=257,
            softmax_scale=sm_scale,
            causal=True,
            scheduler_metadata=scheduler_metadata,
            return_softmax_lse=True,
            backend=backend,
        )

    def wrapped_prefill():
        return wrapper.forward_return_lse(q, k, v, causal=True, sm_scale=sm_scale)

    seq_len = 1024
    num_heads = 32
    head_dim_v = 128
    v_a = torch.randn(
        (seq_len, num_heads, head_dim_v),
        dtype=torch.bfloat16,
        device="musa",
    )
    v_b = torch.randn_like(v_a)
    s_a = torch.randn((seq_len, num_heads), dtype=torch.float32, device="musa")
    s_b = torch.randn_like(s_a)
    direct_partial = torch.stack((v_a, v_b)).unsqueeze(1).float()
    direct_lse = torch.stack((s_a, s_b)).unsqueeze(1)

    def direct_cascade():
        return flash_attn_combine(direct_partial, direct_lse, out_dtype=torch.bfloat16)

    def wrapped_cascade():
        return merge_state(v_a, s_a, v_b, s_b)

    flush = torch.empty(1 << 27, dtype=torch.uint8, device="musa")
    return {
        "prefill_shape": {
            "q_lens": q_lens,
            "k_lens": k_lens,
            "num_qo_heads": num_qo_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "dtype": "bfloat16",
        },
        "prefill_direct": _timings(direct_prefill, flush, warmup, iters),
        "prefill_wrapper": _timings(wrapped_prefill, flush, warmup, iters),
        "cascade_shape": {
            "seq_len": seq_len,
            "num_heads": num_heads,
            "head_dim": head_dim_v,
            "output_dtype": "bfloat16",
        },
        "cascade_direct": _timings(direct_cascade, flush, warmup, iters),
        "cascade_wrapper": _timings(wrapped_cascade, flush, warmup, iters),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--backend", choices=("auto", "mutlass", "mubin"), default="auto"
    )
    parser.add_argument("--correctness-only", action="store_true")
    args = parser.parse_args()

    result = {
        "torch": torch.__version__,
        "torch_musa": str(getattr(torch.version, "musa", None)),
        "device": torch.musa.get_device_name(0),
        "backend": args.backend,
        "correctness": correctness(args.backend),
    }
    if not args.correctness_only:
        result["benchmark"] = benchmark(args.iterations, args.warmup, args.backend)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
