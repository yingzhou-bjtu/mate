import pytest
import torch
from flashinfer import cascade


def _fake_combine(
    out_partial: torch.Tensor,
    lse_partial: torch.Tensor,
    out: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
):
    assert out is None
    assert lse_partial.stride(-2) == 1
    lse = lse_partial.to(torch.float32)
    max_lse = lse.amax(dim=0, keepdim=True)
    weights = torch.exp(lse - max_lse)
    denominator = weights.sum(dim=0)
    merged = (out_partial.to(torch.float32) * weights.unsqueeze(-1)).sum(
        dim=0
    ) / denominator.unsqueeze(-1)
    merged = merged.to(out_dtype or out_partial.dtype)
    merged_lse = max_lse.squeeze(0) + denominator.log()
    return merged, merged_lse


def test_merge_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cascade, "flash_attn_combine", _fake_combine)
    v_a = torch.tensor([[[1.0, 2.0]]])
    s_a = torch.tensor([[0.0]])
    v_b = torch.tensor([[[3.0, 6.0]]])
    s_b = torch.tensor([[0.0]])

    v, s = cascade.merge_state(v_a, s_a, v_b, s_b)

    assert v.shape == v_a.shape
    assert s.shape == s_a.shape
    assert s.dtype == torch.float32
    assert torch.allclose(v, torch.tensor([[[2.0, 4.0]]]))
    assert torch.allclose(s, torch.log(torch.tensor([[2.0]])))


def test_merge_state_in_place_with_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cascade, "flash_attn_combine", _fake_combine)
    v = torch.tensor([[[1.0]], [[3.0]]])
    s = torch.tensor([[0.0], [1.0]])
    v_other = torch.tensor([[[3.0]], [[7.0]]])
    s_other = torch.tensor([[0.0], [1.0]])
    expected_v = torch.tensor([[[2.0]], [[3.0]]])
    expected_s = torch.tensor([[torch.log(torch.tensor(2.0))], [1.0]])

    cascade.merge_state_in_place(
        v, s, v_other, s_other, mask=torch.tensor([True, False])
    )

    assert torch.allclose(v, expected_v)
    assert torch.allclose(s, expected_s)


def test_merge_states(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cascade, "flash_attn_combine", _fake_combine)
    v = torch.tensor([[[[1.0]], [[3.0]], [[5.0]]]])
    s = torch.zeros((1, 3, 1), dtype=torch.float32)

    merged_v, merged_s = cascade.merge_states(v, s)

    assert merged_v.shape == (1, 1, 1)
    assert merged_s.shape == (1, 1)
    assert torch.allclose(merged_v, torch.tensor([[[3.0]]]))
    assert torch.allclose(merged_s, torch.log(torch.tensor([[3.0]])))
