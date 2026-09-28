import base64
import os
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.online.fp8 import Fp8PerTensorOnlineLinearMethod
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    UnquantizedEmbeddingMethod,
    VocabParallelEmbedding,
)
from vllm.model_executor.parameter import GroupQuantScaleParameter, PackedvLLMParameter

from prism_ternary.kernels import (
    GROUP,
    int8_prepare,
    ternary_linear_q,
    ternary_matmul_q,
    HADAMARD_BLOCK,
    LANES,
    block_layout,
    signed_hadamard,
    ternary_linear,
    ternary_matmul,
)

HALF_BLOCK = HADAMARD_BLOCK // 2
# Opt-in int8 prefill GEMM (see kernels.int8_matmul). Decode (<=64 rows) is unchanged.
INT8_PREFILL = os.environ.get("PRISM_INT8_PREFILL", "0") == "1"

QUANT_METHOD = "prism_ternary"
TERNARY_SUFFIXES = frozenset(
    {"qkv_proj", "o_proj", "gate_up_proj", "down_proj", "in_proj_qkvz", "out_proj"}
)


def pack_signs(signs: np.ndarray) -> str:
    return base64.b64encode(np.packbits(signs < 0).tobytes()).decode()


def unpack_signs(encoded: str, width: int) -> torch.Tensor:
    bits = np.unpackbits(np.frombuffer(base64.b64decode(encoded), dtype=np.uint8))[:width]
    return torch.from_numpy(np.where(bits == 1, -1.0, 1.0).astype(np.float32))


@register_quantization_config(QUANT_METHOD)
class PrismTernaryConfig(QuantizationConfig):
    def __init__(self, signs: dict[int, torch.Tensor]) -> None:
        super().__init__()
        self.signs = signs

    @classmethod
    def get_name(cls) -> str:
        return QUANT_METHOD

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "PrismTernaryConfig":
        if config["group_size"] != GROUP or config["hadamard_block"] != HADAMARD_BLOCK:
            raise ValueError(f"unsupported prism_ternary layout: {config}")
        return cls(
            {int(width): unpack_signs(encoded, int(width)) for width, encoded in config["signs"].items()}
        )

    def get_quant_method(self, layer: torch.nn.Module, prefix: str) -> QuantizeMethodBase | None:
        if prefix.startswith("mtp."):
            return Fp8PerTensorOnlineLinearMethod() if isinstance(layer, LinearBase) else None
        if isinstance(layer, ParallelLMHead):
            return PrismTernaryLinearMethod(self)
        if isinstance(layer, LinearBase):
            return (
                PrismTernaryLinearMethod(self)
                if prefix.rsplit(".", 1)[-1] in TERNARY_SUFFIXES
                else UnquantizedLinearMethod()
            )
        return UnquantizedEmbeddingMethod() if isinstance(layer, VocabParallelEmbedding) else None


class PrismTernaryLinearMethod(LinearMethodBase):
    def __init__(self, quant_config: PrismTernaryConfig) -> None:
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs: Any,
    ) -> None:
        weight_loader = extra_weight_attrs["weight_loader"]
        rows = sum(output_partition_sizes)
        if input_size_per_partition % GROUP:
            raise ValueError(f"input width {input_size_per_partition} is not a multiple of {GROUP}")
        layer.register_parameter(
            "weight",
            PackedvLLMParameter(
                data=torch.empty(rows, input_size_per_partition // LANES, dtype=torch.int32),
                input_dim=1,
                output_dim=0,
                packed_dim=1,
                packed_factor=LANES,
                weight_loader=weight_loader,
            ),
        )
        layer.register_parameter(
            "scales",
            GroupQuantScaleParameter(
                data=torch.empty(rows, input_size_per_partition // GROUP, dtype=torch.float16),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
            ),
        )
        # The rotation is a block-diagonal Hadamard over the *full* input width, so a
        # row-parallel shard takes its slice of the full sign vector. When the shard
        # boundary cuts a 1024 block exactly in half (TP=2 on down_proj: 17408 -> 8704),
        # each rank transforms its half and the two halves are exchanged in apply().
        full = self.quant_config.signs[input_size]
        width = input_size_per_partition
        start = 0 if width == input_size else get_tensor_model_parallel_rank() * width
        local = full[start : start + width]
        layer.prism_straddle = -1
        layer.prism_i8 = False  # set in process_weights_after_loading when PRISM_INT8_PREFILL=1
        if start % HADAMARD_BLOCK == 0 and width % HADAMARD_BLOCK == 0:
            signs = local
        elif get_tensor_model_parallel_world_size() == 2 and width % HADAMARD_BLOCK == HALF_BLOCK:
            rank = start // width
            layer.prism_straddle = rank
            half = local[width - HALF_BLOCK :] if rank == 0 else local[:HALF_BLOCK]
            signs = local[: width - HALF_BLOCK] if rank == 0 else local[HALF_BLOCK:]
            half_signs = torch.ones(HADAMARD_BLOCK, dtype=torch.float32)
            half_signs[:HALF_BLOCK].copy_(half)
            layer.register_buffer("half_signs", half_signs)
        else:
            raise ValueError(f"shard [{start}, {start + width}) of width {input_size} breaks hadamard blocks")
        # allocate on the default (model-load) device, as upstream does, then copy from the CPU table
        buf = torch.empty(signs.numel(), dtype=torch.float32)
        buf.copy_(signs)
        layer.register_buffer("hadamard_signs", buf)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight, scales = block_layout(layer.weight.data, layer.scales.data)
        layer.weight = torch.nn.Parameter(weight, requires_grad=False)
        layer.scales = torch.nn.Parameter(scales, requires_grad=False)
        layer.prism_i8 = INT8_PREFILL
        if INT8_PREFILL:
            q, sm127 = int8_prepare(scales)
            layer.register_buffer("i8_q", q)
            layer.register_buffer("i8_sm127", sm127)

    def apply(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        if layer.prism_straddle < 0 and layer.prism_i8:
            y = ternary_linear_q(x, layer.weight, layer.scales, layer.i8_q, layer.i8_sm127, layer.hadamard_signs)
        elif layer.prism_straddle < 0:
            y = ternary_linear(x, layer.weight, layer.scales, layer.hadamard_signs)
        else:
            y = self._apply_straddle(layer, x)
        return y if bias is None else y + bias

    @staticmethod
    def _apply_straddle(layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        # H1024 [a; b] = [Ha + Hb; Ha - Hb] (Sylvester, same 1/32 scale as the kernel).
        # Rank 0 owns a (its last 512 cols), rank 1 owns b (its first 512 cols).
        rank = layer.prism_straddle
        x2 = x.reshape(-1, x.shape[-1])
        width = x2.shape[1]
        if rank == 0:
            aligned, half = x2[:, : width - HALF_BLOCK], x2[:, width - HALF_BLOCK :]
        else:
            half, aligned = x2[:, :HALF_BLOCK], x2[:, HALF_BLOCK:]
        rotated = signed_hadamard(aligned.contiguous(), layer.hadamard_signs)
        own = signed_hadamard(F.pad(half, (0, HALF_BLOCK)).contiguous(), layer.half_signs)[:, :HALF_BLOCK]
        both = tensor_model_parallel_all_gather(own.contiguous(), dim=-1).float()
        ha, hb = both[:, :HALF_BLOCK], both[:, HALF_BLOCK:]
        mixed = (ha + hb if rank == 0 else ha - hb).to(x2.dtype)
        rotated = torch.cat([rotated, mixed] if rank == 0 else [mixed, rotated], dim=1)
        if layer.prism_i8:
            y = ternary_matmul_q(rotated, layer.weight, layer.scales, layer.i8_q, layer.i8_sm127)
        else:
            y = ternary_matmul(rotated, layer.weight, layer.scales)
        return y.reshape(*x.shape[:-1], layer.weight.shape[0])


def register() -> None:
    return None
