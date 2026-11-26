"""FlashInfer cascade compatibility backed by MATE FMHA combine."""

from __future__ import annotations

import torch

from mate.mha_interface import flash_attn_combine

__all__ = ["merge_state", "merge_state_in_place", "merge_states"]


def _validate_state(v: torch.Tensor, s: torch.Tensor, name: str) -> None:
    if v.ndim != 3:
        raise ValueError(f"{name}_v must have shape [seq_len, heads, head_dim]")
    if s.shape != v.shape[:-1]:
        raise ValueError(
            f"{name}_s must have shape {tuple(v.shape[:-1])}, got {tuple(s.shape)}"
        )


def _prepare_lse_partial(lse: torch.Tensor) -> torch.Tensor:
    storage = torch.empty(
        (lse.shape[1], 1, lse.shape[2], lse.shape[0]),
        dtype=torch.float32,
        device=lse.device,
    )
    storage.copy_(lse.permute(1, 2, 0).unsqueeze(1))
    return storage.transpose(2, 3)


def _combine_two_states(
    v_a: torch.Tensor,
    s_a: torch.Tensor,
    v_b: torch.Tensor,
    s_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    out_partial = torch.empty(
        (2, 1, *v_a.shape), dtype=torch.float32, device=v_a.device
    )
    out_partial[0, 0].copy_(v_a)
    out_partial[1, 0].copy_(v_b)

    lse_storage = torch.empty(
        (2, 1, s_a.shape[1], s_a.shape[0]),
        dtype=torch.float32,
        device=s_a.device,
    )
    lse_storage[0, 0].copy_(s_a.transpose(0, 1))
    lse_storage[1, 0].copy_(s_b.transpose(0, 1))
    lse_partial = lse_storage.transpose(2, 3)
    out, lse = flash_attn_combine(out_partial, lse_partial, out_dtype=v_a.dtype)
    return out.squeeze(0), lse.squeeze(0)


def merge_state(
    v_a: torch.Tensor,
    s_a: torch.Tensor,
    v_b: torch.Tensor,
    s_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge attention output and LSE from two KV segments."""
    _validate_state(v_a, s_a, "a")
    _validate_state(v_b, s_b, "b")
    if v_a.shape != v_b.shape:
        raise ValueError(
            f"state shapes differ: {tuple(v_a.shape)} and {tuple(v_b.shape)}"
        )
    if v_a.dtype != v_b.dtype or v_a.device != v_b.device:
        raise ValueError("both states must use the same dtype and device")
    return _combine_two_states(v_a, s_a, v_b, s_b)


def merge_state_in_place(
    v: torch.Tensor,
    s: torch.Tensor,
    v_other: torch.Tensor,
    s_other: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> None:
    """Merge another attention state into ``(v, s)``."""
    v_merged, s_merged = merge_state(v, s, v_other, s_other)
    if mask is None:
        v.copy_(v_merged)
        s.copy_(s_merged)
        return

    if mask.shape != (v.shape[0],):
        raise ValueError(f"mask must have shape [{v.shape[0]}]")
    mask_v = mask.to(device=v.device, dtype=torch.bool)[:, None, None]
    mask_s = mask_v.squeeze(-1)
    v.copy_(torch.where(mask_v, v_merged, v))
    s.copy_(torch.where(mask_s, s_merged, s))


def merge_states(v: torch.Tensor, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge attention output and LSE from multiple KV segments."""
    if v.ndim != 4:
        raise ValueError("v must have shape [seq_len, num_states, heads, head_dim]")
    if s.shape != v.shape[:2] + v.shape[2:3]:
        raise ValueError("s must have shape [seq_len, num_states, heads]")
    if v.shape[1] == 0:
        raise ValueError("num_states must be greater than zero")

    out_partial = v.permute(1, 0, 2, 3).unsqueeze(1).to(torch.float32)
    lse_partial = _prepare_lse_partial(s)
    out, lse = flash_attn_combine(out_partial, lse_partial, out_dtype=v.dtype)
    return out.squeeze(0), lse.squeeze(0)
