import pytest
import torch
from flashinfer import prefill


def _fake_scheduler_metadata(*args, **kwargs):
    assert args[0] == 2
    assert kwargs["qkv_dtype"] == torch.float16
    return torch.tensor([7])


def _fake_varlen(
    q,
    k,
    v,
    *,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    softmax_scale,
    causal,
    window_size,
    softcap,
    scheduler_metadata,
    return_softmax_lse,
    out,
    backend,
):
    assert cu_seqlens_q.tolist() == [0, 2, 5]
    assert cu_seqlens_k.tolist() == [0, 3, 6]
    assert max_seqlen_q == 3
    assert max_seqlen_k == 3
    assert softmax_scale == pytest.approx(0.5)
    assert causal is True
    assert window_size == (-1, -1)
    assert softcap == 0.0
    assert scheduler_metadata.tolist() == [7]
    assert backend == "auto"
    output = torch.zeros_like(q) if out is None else out
    softmax_lse = torch.arange(q.shape[1] * q.shape[0], dtype=torch.float32).reshape(
        q.shape[1], q.shape[0]
    )
    return (output, softmax_lse) if return_softmax_lse else output


def _make_wrapper(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(prefill, "get_scheduler_metadata", _fake_scheduler_metadata)
    monkeypatch.setattr(prefill, "flash_attn_varlen_func", _fake_varlen)
    wrapper = prefill.BatchPrefillWithRaggedKVCacheWrapper(
        float_workspace_buffer=torch.empty(1)
    )
    wrapper.plan(
        torch.tensor([0, 2, 5], dtype=torch.int32),
        torch.tensor([0, 3, 6], dtype=torch.int32),
        num_qo_heads=2,
        num_kv_heads=2,
        head_dim_qk=4,
        causal=True,
        sm_scale=0.5,
        q_data_type=torch.float16,
    )
    return wrapper


def test_ragged_prefill(monkeypatch: pytest.MonkeyPatch) -> None:
    wrapper = _make_wrapper(monkeypatch)
    q = torch.zeros((5, 2, 4), dtype=torch.float16)
    k = torch.zeros((6, 2, 4), dtype=torch.float16)
    v = torch.zeros((6, 2, 4), dtype=torch.float16)

    output = wrapper.forward(q, k, v, causal=True, sm_scale=0.5)

    assert output.shape == q.shape


def test_ragged_prefill_returns_lse(monkeypatch: pytest.MonkeyPatch) -> None:
    wrapper = _make_wrapper(monkeypatch)
    q = torch.zeros((5, 2, 4), dtype=torch.float16)
    k = torch.zeros((6, 2, 4), dtype=torch.float16)
    v = torch.zeros((6, 2, 4), dtype=torch.float16)

    output, lse = wrapper.forward_return_lse(q, k, v, causal=True, sm_scale=0.5)

    assert output.shape == q.shape
    assert lse.shape == (5, 2)
    assert torch.equal(
        lse, torch.arange(10, dtype=torch.float32).reshape(2, 5).transpose(0, 1)
    )


def test_custom_mask_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(prefill, "get_scheduler_metadata", _fake_scheduler_metadata)
    wrapper = prefill.BatchPrefillWithRaggedKVCacheWrapper(
        float_workspace_buffer=torch.empty(1)
    )

    with pytest.raises(NotImplementedError, match="custom attention masks"):
        wrapper.plan(
            torch.tensor([0, 1], dtype=torch.int32),
            torch.tensor([0, 1], dtype=torch.int32),
            num_qo_heads=1,
            num_kv_heads=1,
            head_dim_qk=4,
            custom_mask=torch.ones((1, 1), dtype=torch.bool),
        )
