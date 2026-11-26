"""FlashInfer ragged prefill compatibility backed by MATE FA3 attention."""

from __future__ import annotations

from typing import Any

import torch

from mate.mha_interface import flash_attn_varlen_func, get_scheduler_metadata

__all__ = ["BatchPrefillWithRaggedKVCacheWrapper"]


def _torch_dtype(value: str | torch.dtype) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    dtype = getattr(torch, value, None)
    if isinstance(dtype, torch.dtype):
        return dtype
    raise ValueError(f"unsupported dtype {value!r}")


def _max_seqlen(indptr: torch.Tensor) -> int:
    if indptr.ndim != 1 or indptr.numel() == 0:
        raise ValueError("indptr must be a non-empty one-dimensional tensor")
    return int(indptr.diff().max().item())


class BatchPrefillWithRaggedKVCacheWrapper:
    """FlashInfer-compatible ragged prefill wrapper backed by MATE."""

    def __init__(
        self,
        float_workspace_buffer: torch.Tensor,
        kv_layout: str = "NHD",
        use_cuda_graph: bool = False,
        qo_indptr_buf: torch.Tensor | None = None,
        kv_indptr_buf: torch.Tensor | None = None,
        custom_mask_buf: torch.Tensor | None = None,
        mask_indptr_buf: torch.Tensor | None = None,
        backend: str = "auto",
        jit_args: list[Any] | None = None,
        jit_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if kv_layout != "NHD":
            raise NotImplementedError("only NHD KV layout is supported")
        if backend not in {"auto", "mutlass", "mubin"}:
            raise ValueError(f"unsupported MATE backend {backend!r}")
        if custom_mask_buf is not None or mask_indptr_buf is not None:
            raise NotImplementedError("custom mask buffers are not supported")

        self._float_workspace_buffer = float_workspace_buffer
        self._int_workspace_buffer = None
        self._kv_layout = kv_layout
        self._use_cuda_graph = use_cuda_graph
        self._qo_indptr_buf = qo_indptr_buf
        self._kv_indptr_buf = kv_indptr_buf
        self._backend = backend
        self._jit_args = jit_args
        self._jit_kwargs = jit_kwargs
        self._scheduler_metadata = None
        self._qo_indptr_device = None
        self._kv_indptr_device = None

    def is_cuda_graph_enabled(self) -> bool:
        return self._use_cuda_graph

    def reset_workspace_buffer(
        self, float_workspace_buffer: torch.Tensor, int_workspace_buffer
    ) -> None:
        self._float_workspace_buffer = float_workspace_buffer
        self._int_workspace_buffer = int_workspace_buffer
        self._scheduler_metadata = None

    def plan(
        self,
        qo_indptr: torch.Tensor,
        kv_indptr: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim_qk: int,
        head_dim_vo: int | None = None,
        custom_mask: torch.Tensor | None = None,
        packed_custom_mask: torch.Tensor | None = None,
        causal: bool = False,
        pos_encoding_mode: str = "NONE",
        use_fp16_qk_reduction: bool = False,
        window_left: int = -1,
        logits_soft_cap: float | None = None,
        sm_scale: float | None = None,
        rope_scale: float | None = None,
        rope_theta: float | None = None,
        q_data_type: str | torch.dtype = "float16",
        kv_data_type: str | torch.dtype | None = None,
        non_blocking: bool = True,
        prefix_len_ptr: torch.Tensor | None = None,
        token_pos_in_items_ptr: torch.Tensor | None = None,
        token_pos_in_items_len: int = 0,
        max_item_len_ptr: torch.Tensor | None = None,
    ) -> None:
        if custom_mask is not None or packed_custom_mask is not None:
            raise NotImplementedError("custom attention masks are not supported")
        if pos_encoding_mode != "NONE":
            raise NotImplementedError(
                f"position encoding mode {pos_encoding_mode!r} is not supported"
            )
        if rope_scale is not None or rope_theta is not None:
            raise NotImplementedError("RoPE in the prefill wrapper is not supported")
        if (
            prefix_len_ptr is not None
            or token_pos_in_items_ptr is not None
            or max_item_len_ptr is not None
        ):
            raise NotImplementedError(
                "prefix and packed-item metadata are not supported"
            )
        if non_blocking is not True:
            raise NotImplementedError("non_blocking=False is not supported")
        if qo_indptr.numel() != kv_indptr.numel():
            raise ValueError("qo_indptr and kv_indptr must have the same batch size")

        self._qo_indptr_buf = qo_indptr
        self._kv_indptr_buf = kv_indptr
        self._num_qo_heads = num_qo_heads
        self._num_kv_heads = num_kv_heads
        self._head_dim_qk = head_dim_qk
        self._head_dim_vo = head_dim_vo or head_dim_qk
        self._causal = causal
        self._window_left = window_left
        self._logits_soft_cap = logits_soft_cap
        self._sm_scale = sm_scale
        self._q_data_type = _torch_dtype(q_data_type)
        self._kv_data_type = _torch_dtype(kv_data_type or q_data_type)
        if self._q_data_type != self._kv_data_type:
            raise NotImplementedError("different Q and KV dtypes are not supported")
        self._max_qo_len = _max_seqlen(qo_indptr)
        self._max_kv_len = _max_seqlen(kv_indptr)
        self._batch_size = qo_indptr.numel() - 1
        device = self._float_workspace_buffer.device
        self._qo_indptr_device = qo_indptr.to(device=device, non_blocking=True)
        self._kv_indptr_device = kv_indptr.to(device=device, non_blocking=True)
        self._scheduler_metadata = get_scheduler_metadata(
            self._batch_size,
            self._max_qo_len,
            self._max_kv_len,
            self._num_qo_heads,
            self._num_kv_heads,
            self._head_dim_qk,
            qkv_dtype=self._q_data_type,
            headdim_v=self._head_dim_vo,
            cu_seqlens_q=self._qo_indptr_device,
            cu_seqlens_k=self._kv_indptr_device,
            causal=self._causal,
            window_size=self._window_size(self._window_left),
            attention_chunk=0,
            has_softcap=bool(self._logits_soft_cap),
            num_splits=0,
            pack_gqa=None,
        )

    begin_forward = plan

    @staticmethod
    def _window_size(window_left: int) -> tuple[int, int]:
        if window_left < 0:
            return (-1, -1)
        return (window_left, 0)

    def run(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *args,
        out: torch.Tensor | None = None,
        lse: torch.Tensor | None = None,
        return_lse: bool = False,
        enable_pdl: bool | None = None,
        causal: bool | None = None,
        window_left: int | None = None,
        logits_soft_cap: float | None = None,
        sm_scale: float | None = None,
    ):
        if args:
            raise TypeError("legacy positional run arguments are not supported")
        if self._qo_indptr_buf is None or self._kv_indptr_buf is None:
            raise RuntimeError("plan or begin_forward must be called before run")
        if enable_pdl is not None:
            raise NotImplementedError("enable_pdl is not supported")
        if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
            raise ValueError("q, k, and v must use varlen [tokens, heads, dim] layout")
        if q.shape[1] != self._num_qo_heads:
            raise ValueError("Q head count does not match the planned shape")
        if k.shape[1] != self._num_kv_heads or v.shape[1] != self._num_kv_heads:
            raise ValueError("K/V head count does not match the planned shape")
        if q.shape[-1] != self._head_dim_qk or k.shape[-1] != self._head_dim_qk:
            raise ValueError("Q/K head dimension does not match the planned shape")
        if v.shape[-1] != self._head_dim_vo:
            raise ValueError("V head dimension does not match the planned shape")
        if (
            q.dtype != self._q_data_type
            or k.dtype != self._kv_data_type
            or v.dtype != self._kv_data_type
        ):
            raise ValueError("Q/K/V dtype does not match the planned dtype")
        if q.device != self._float_workspace_buffer.device:
            raise ValueError("Q device does not match the planned workspace device")
        if self._qo_indptr_device is None or self._kv_indptr_device is None:
            raise RuntimeError("plan or begin_forward must be called before run")
        causal = self._causal if causal is None else causal
        window_left = self._window_left if window_left is None else window_left
        logits_soft_cap = (
            self._logits_soft_cap if logits_soft_cap is None else logits_soft_cap
        )
        sm_scale = self._sm_scale if sm_scale is None else sm_scale
        requested_lse = return_lse or lse is not None

        result = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=self._qo_indptr_device,
            cu_seqlens_k=self._kv_indptr_device,
            max_seqlen_q=self._max_qo_len,
            max_seqlen_k=self._max_kv_len,
            softmax_scale=sm_scale,
            causal=causal,
            window_size=self._window_size(window_left),
            softcap=logits_soft_cap or 0.0,
            scheduler_metadata=self._scheduler_metadata,
            return_softmax_lse=requested_lse,
            out=out,
            backend=self._backend,
        )
        if not requested_lse:
            return result

        output, softmax_lse = result
        softmax_lse = softmax_lse.transpose(0, 1)
        if lse is not None:
            lse.copy_(softmax_lse)
            softmax_lse = lse
        return (output, softmax_lse) if return_lse else output

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool = False,
        pos_encoding_mode: str = "NONE",
        use_fp16_qk_reduction: bool = False,
        window_left: int = -1,
        logits_soft_cap: float | None = None,
        sm_scale: float | None = None,
        rope_scale: float | None = None,
        rope_theta: float | None = None,
    ):
        self._validate_runtime_options(pos_encoding_mode, rope_scale, rope_theta)
        return self.run(
            q,
            k,
            v,
            causal=causal,
            window_left=window_left,
            logits_soft_cap=logits_soft_cap,
            sm_scale=sm_scale,
        )

    def forward_return_lse(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool = False,
        pos_encoding_mode: str = "NONE",
        use_fp16_qk_reduction: bool = False,
        window_left: int = -1,
        logits_soft_cap: float | None = None,
        sm_scale: float | None = None,
        rope_scale: float | None = None,
        rope_theta: float | None = None,
    ):
        self._validate_runtime_options(pos_encoding_mode, rope_scale, rope_theta)
        return self.run(
            q,
            k,
            v,
            causal=causal,
            window_left=window_left,
            logits_soft_cap=logits_soft_cap,
            sm_scale=sm_scale,
            return_lse=True,
        )

    @staticmethod
    def _validate_runtime_options(
        pos_encoding_mode: str,
        rope_scale: float | None,
        rope_theta: float | None,
    ) -> None:
        if pos_encoding_mode != "NONE":
            raise NotImplementedError(
                f"position encoding mode {pos_encoding_mode!r} is not supported"
            )
        if rope_scale is not None or rope_theta is not None:
            raise NotImplementedError("RoPE in the prefill wrapper is not supported")

    def end_forward(self) -> None:
        return None
