#!/usr/bin/env python3

import argparse
import json
import statistics

import torch
from flashinfer.cascade import merge_state, merge_state_in_place, merge_states
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
    if actual.shape != expected.shape:
        raise ValueError(
            f"shape mismatch: actual={tuple(actual.shape)} "
            f"expected={tuple(expected.shape)}"
        )
    actual_f32 = actual.float()
    expected_f32 = expected.to(device=actual.device, dtype=torch.float32)
    return (actual_f32 - expected_f32).abs().max().item()


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
    direct_output, direct_prefill_lse = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=wrapper._qo_indptr_device,
        cu_seqlens_k=wrapper._kv_indptr_device,
        max_seqlen_q=wrapper._max_qo_len,
        max_seqlen_k=wrapper._max_kv_len,
        softmax_scale=sm_scale,
        causal=True,
        scheduler_metadata=wrapper._scheduler_metadata,
        return_softmax_lse=True,
        backend=backend,
    )
    direct_prefill_lse = direct_prefill_lse.transpose(0, 1).contiguous()
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
    direct_v, direct_cascade_lse = flash_attn_combine(
        torch.stack((v_a, v_b)).unsqueeze(1).float(),
        torch.stack((s_a, s_b))
        .unsqueeze(1)
        .transpose(2, 3)
        .contiguous()
        .transpose(2, 3),
        out_dtype=torch.bfloat16,
    )
    max_lse = torch.maximum(s_a, s_b)
    weight_a = torch.exp(s_a - max_lse)
    weight_b = torch.exp(s_b - max_lse)
    denominator = weight_a + weight_b
    reference_merged_v = (
        weight_a.unsqueeze(-1) * v_a.float() + weight_b.unsqueeze(-1) * v_b.float()
    ) / denominator.unsqueeze(-1)
    reference_merged_lse = max_lse + denominator.log()

    mask = torch.arange(seq_len, device="musa") % 2 == 0
    inplace_v = v_a.clone()
    inplace_lse = s_a.clone()
    merge_state_in_place(inplace_v, inplace_lse, v_b, s_b, mask=mask)
    reference_inplace_v = torch.where(
        mask.unsqueeze(-1).unsqueeze(-1),
        reference_merged_v,
        v_a.float(),
    )
    reference_inplace_lse = torch.where(mask.unsqueeze(-1), reference_merged_lse, s_a)

    states_v = torch.stack((v_a, v_b, v_a * 0.5)).permute(1, 0, 2, 3)
    states_lse = torch.stack((s_a, s_b, s_a - 0.25)).permute(1, 0, 2)
    merged_states_v, merged_states_lse = merge_states(states_v, states_lse)
    states_v_cpu = torch.stack(
        (v_a.float().cpu(), v_b.float().cpu(), v_a.float().cpu() * 0.5),
        dim=1,
    )
    states_lse_cpu = torch.stack((s_a.cpu(), s_b.cpu(), s_a.cpu() - 0.25), dim=1)
    states_max = states_lse_cpu.max(dim=1, keepdim=True).values
    states_weights = torch.exp(states_lse_cpu - states_max)
    states_denominator = states_weights.sum(dim=1, keepdim=True)
    reference_states_v = (states_weights.unsqueeze(-1) * states_v_cpu).sum(
        dim=1
    ) / states_denominator.squeeze(1).unsqueeze(-1)
    reference_states_lse = states_max.squeeze(1) + states_denominator.squeeze(1).log()

    result = {
        "prefill_output_max_abs": _max_abs_error(output, reference_output),
        "prefill_lse_max_abs": _max_abs_error(lse, reference_lse),
        "prefill_wrapper_vs_direct_output": _max_abs_error(output, direct_output),
        "prefill_wrapper_vs_direct_lse": _max_abs_error(lse, direct_prefill_lse),
        "cascade_output_max_abs": _max_abs_error(merged_v, reference_merged_v),
        "cascade_lse_max_abs": _max_abs_error(merged_lse, reference_merged_lse),
        "cascade_wrapper_vs_direct_output": _max_abs_error(
            merged_v, direct_v.squeeze(0)
        ),
        "cascade_wrapper_vs_direct_lse": _max_abs_error(
            merged_lse, direct_cascade_lse.squeeze(0)
        ),
        "cascade_inplace_output_max_abs": _max_abs_error(
            inplace_v, reference_inplace_v
        ),
        "cascade_inplace_lse_max_abs": _max_abs_error(
            inplace_lse, reference_inplace_lse
        ),
        "cascade_multi_output_max_abs": _max_abs_error(
            merged_states_v, reference_states_v
        ),
        "cascade_multi_lse_max_abs": _max_abs_error(
            merged_states_lse, reference_states_lse
        ),
    }
    assert result["prefill_output_max_abs"] <= 0.02
    assert result["prefill_lse_max_abs"] <= 1e-5
    assert result["prefill_wrapper_vs_direct_output"] <= 1e-4
    assert result["prefill_wrapper_vs_direct_lse"] <= 1e-5
    assert result["cascade_output_max_abs"] <= 0.02
    assert result["cascade_lse_max_abs"] <= 1e-5
    assert result["cascade_wrapper_vs_direct_output"] <= 1e-4
    assert result["cascade_wrapper_vs_direct_lse"] <= 1e-5
    assert result["cascade_inplace_output_max_abs"] <= 0.02
    assert result["cascade_inplace_lse_max_abs"] <= 1e-5
    assert result["cascade_multi_output_max_abs"] <= 0.02
    assert result["cascade_multi_lse_max_abs"] <= 1e-5
    result["passed"] = True
    return result


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
    direct_lse = direct_lse.transpose(2, 3).contiguous().transpose(2, 3)

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
