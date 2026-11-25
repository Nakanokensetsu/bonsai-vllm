import base64
import os
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
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
MTP_BF16 = os.environ.get("PRISM_MTP_BF16", "0") == "1"
# Small bf16 linears (GDN in_proj_ba, 5120 -> 48 per rank): with a row-major weight cuBLAS picks a
# 4-CTA kernel at 2-8 rows (~20 us); a column-major copy of the same values runs in ~4 us.
BA_COLMAJOR = os.environ.get("PRISM_BA_COLMAJOR", "0") == "1"
COLMAJOR_SUFFIXES = frozenset({"in_proj_ba"})


class ColMajorUnquantizedLinearMethod(UnquantizedLinearMethod):
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)
        w = layer.weight.data
        if not getattr(layer, "use_skinny_gemm", False) and w.ndim == 2 and w.is_cuda and w.stride(1) == 1:
            layer.weight.data = w.t().contiguous().t()

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
            # The MTP head ships in bf16. Upstream quantizes it to fp8 online; PRISM_MTP_BF16=1 keeps
            # it in bf16 (about +0.2 GB per GPU) in case fp8 costs draft acceptance.
            if not isinstance(layer, LinearBase):
                return None
            return UnquantizedLinearMethod() if MTP_BF16 else Fp8PerTensorOnlineLinearMethod()
        if isinstance(layer, ParallelLMHead):
            return PrismTernaryLinearMethod(self)
        if isinstance(layer, LinearBase):
            suffix = prefix.rsplit(".", 1)[-1]
            if suffix in TERNARY_SUFFIXES:
                return PrismTernaryLinearMethod(self)
            if BA_COLMAJOR and suffix in COLMAJOR_SUFFIXES:
                return ColMajorUnquantizedLinearMethod()
            return UnquantizedLinearMethod()
        return UnquantizedEmbeddingMethod() if isinstance(layer, VocabParallelEmbedding) else None


def _with_neighbour_columns(layer: torch.nn.Module, inner):
    """Wrap a weight loader so a straddling layer also keeps the other rank's 512 boundary columns.

    The loader is handed the full on-disk tensor, so the neighbour's slice is just another narrow().
    """

    def load(param, loaded_weight, *args, **kwargs):
        inner(param, loaded_weight, *args, **kwargs)
        rank = layer.prism_straddle
        if rank < 0:
            return
        if param is layer.weight:
            size, per = param.data.shape[1], HALF_BLOCK // LANES
            layer.weight_x.copy_(loaded_weight.narrow(1, size if rank == 0 else size - per, per))
        elif param is layer.scales:
            size, per = param.data.shape[1], HALF_BLOCK // GROUP
            layer.scales_x.copy_(loaded_weight.narrow(1, size if rank == 0 else size - per, per))

    return load


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
        weight_loader = _with_neighbour_columns(layer, extra_weight_attrs["weight_loader"])
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
            layer.register_buffer("weight_x", torch.empty(rows, HALF_BLOCK // LANES, dtype=torch.int32))
            layer.register_buffer("scales_x", torch.empty(rows, HALF_BLOCK // GROUP, dtype=torch.float16))
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
        if layer.prism_straddle >= 0:
            layer.weight_x, layer.scales_x = block_layout(layer.weight_x.data, layer.scales_x.data)
            if INT8_PREFILL:
                qx, smx = int8_prepare(layer.scales_x)
                layer.register_buffer("i8_qx", qx)
                layer.register_buffer("i8_sm127x", smx)

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
        # The 1024 block that the shard boundary cuts is H1024 [a; b] = [Ha + Hb; Ha - Hb] (Sylvester),
        # a on rank 0 (its last 512 cols), b on rank 1 (its first 512). Instead of exchanging Ha / Hb,
        # each rank feeds its own half-transform into the local product with the sign its column block
        # sees (+Ha for rank 0, -Hb for rank 1) and adds the neighbour's boundary columns times its own
        # half-transform (W_B Ha on rank 0, W_A Hb on rank 1). The all-reduce that follows sums both ranks
        # to exactly W (Ha + Hb) + W (Ha - Hb) cut at the boundary, with no collective of its own.
        rank = layer.prism_straddle
        x2 = x.reshape(-1, x.shape[-1])
        width = x2.shape[1]
        if rank == 0:
            aligned, half = x2[:, : width - HALF_BLOCK], x2[:, width - HALF_BLOCK :]
        else:
            half, aligned = x2[:, :HALF_BLOCK], x2[:, HALF_BLOCK:]
        rotated = signed_hadamard(aligned.contiguous(), layer.hadamard_signs)
        own = signed_hadamard(F.pad(half, (0, HALF_BLOCK)).contiguous(), layer.half_signs)[:, :HALF_BLOCK]
        local = torch.cat([rotated, own] if rank == 0 else [-own, rotated], dim=1)
        if layer.prism_i8:
            y = ternary_matmul_q(local, layer.weight, layer.scales, layer.i8_q, layer.i8_sm127)
            y = y + ternary_matmul_q(own.contiguous(), layer.weight_x, layer.scales_x, layer.i8_qx, layer.i8_sm127x)
        else:
            y = ternary_matmul(local, layer.weight, layer.scales)
            y = y + ternary_matmul(own.contiguous(), layer.weight_x, layer.scales_x)
        return y.reshape(*x.shape[:-1], layer.weight.shape[0])


def register() -> None:
    # PRISM_AR_INT8=1: TP=2 の prefill AllReduce を int8 で交換(既定は無効)。記録: notes/20261029-tp-opt/README.md §8,10 / 2026-11-03 Triton 化
    from prism_ternary import ar_int8
    if ar_int8.ENABLED:
        ar_int8.patch_vllm()
