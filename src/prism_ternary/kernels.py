from functools import cache
from pathlib import Path
from types import ModuleType

import torch
import triton
import triton.language as tl
from torch.utils.cpp_extension import load

HADAMARD_BLOCK = 1024
GROUP = 128
LANES = 16
WORDS_PER_GROUP = GROUP // LANES
ROW_BLOCK = 32
GEMV_MAX_ROWS = 64
# Decode batches up to this many rows use the int8 dp4a GEMV when PRISM_DP4A_DECODE=1
# (RTX 5070: 2.1x at 1 row, 1.7x at 2, 1.3x at 4, slower at 8; per-layer error 1.7e-3 -> 6.7e-3).
DP4A_MAX_ROWS = 4
DP4A_DECODE = __import__("os").environ.get("PRISM_DP4A_DECODE", "0") == "1"
# int8 prefill for small chunks (65..N rows): unpack the 2-bit codes inside the GEMM instead of
# expanding the whole weight to int8 first. Bit-identical to int8_matmul. On RTX 5070 (sm_120):
# 1.4-1.9x at <=128 rows, 1.0-1.2x at 256-384 rows, slower from 512 rows. 0 = off.
FUSED_I8_MAX_ROWS = int(__import__("os").environ.get("PRISM_FUSED_I8_MAX", "0"))


@cache
def _cuda() -> ModuleType:
    return load(
        name="prism_ternary_cuda",
        sources=[str(Path(__file__).with_name("kernels.cu"))],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )


def block_layout(weight: torch.Tensor, scales: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    out, words = weight.shape
    groups = scales.shape[1]
    if out % ROW_BLOCK:
        raise ValueError(f"output width {out} is not a multiple of {ROW_BLOCK}")
    blocked_w = weight.view(out // ROW_BLOCK, ROW_BLOCK, groups, WORDS_PER_GROUP).permute(0, 2, 1, 3)
    blocked_s = scales.view(out // ROW_BLOCK, ROW_BLOCK, groups).permute(0, 2, 1)
    return blocked_w.contiguous().view(out, words), blocked_s.contiguous().view(out, groups)


@triton.jit
def _dequant_kernel(w_ptr, s_ptr, out_ptr, groups, stride_on, BLOCK_N: tl.constexpr):
    pid_n = tl.program_id(0)
    g = tl.program_id(1)
    rows = tl.arange(0, BLOCK_N)
    words = tl.arange(0, 8)
    lanes = tl.arange(0, 16)
    w_base = w_ptr + (pid_n * groups + g) * (BLOCK_N * 8)
    w = tl.load(w_base + rows[:, None] * 8 + words[None, :])
    sc = tl.load(s_ptr + (pid_n * groups + g) * BLOCK_N + rows).to(tl.float32)
    codes = ((w[:, :, None] >> (lanes * 2)[None, None, :]) & 3).to(tl.float32) - 1.0
    vals = codes * sc[:, None, None]
    offs_n = pid_n * BLOCK_N + rows
    out_ptrs = out_ptr + offs_n[:, None, None] * stride_on + g * 128 + (words * 16)[None, :, None] + lanes[None, None, :]
    tl.store(out_ptrs, vals.to(out_ptr.dtype.element_ty))


def dequantize(weight: torch.Tensor, scales: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    out, words = weight.shape
    width = words * LANES
    groups = width // GROUP
    full = torch.empty((out, width), dtype=dtype, device=weight.device)
    _dequant_kernel[(out // ROW_BLOCK, groups)](weight, scales, full, groups, full.stride(0), BLOCK_N=ROW_BLOCK, num_warps=4)
    return full


# ---- int8 prefill path (opt-in: PRISM_INT8_PREFILL=1) --------------------------------------
# Group scales are split at load time into a per-row max and an int8 ratio
#     q[n, g] = round(127 * s[n, g] / smax[n]),
# so W_int8[n, k] = code[n, k] * q[n, g] is exact int8 and a whole prefill GEMM becomes one cuBLAS
# int8 GEMM plus a per-row / per-token rescale. Activations are quantized per token to int8 after
# the Hadamard rotation (which spreads outliers). On RTX 5070 (sm_120, M=512) the linear layers go
# from 242 ms to 114 ms per step; relative error per layer ~9.5e-3 vs 2.4e-3 for the bf16 path.


def int8_prepare(scales: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Blocked fp16 scales [N, G] -> (q: int8 [N, G] in the same blocked order, smax/127: fp32 [N])."""
    n, groups = scales.shape
    sb = scales.float().view(n // ROW_BLOCK, groups, ROW_BLOCK)
    smax = sb.amax(dim=1).clamp_min(1e-12)
    q = torch.round(127.0 * sb / smax[:, None, :]).clamp_(0, 127).to(torch.int8)
    return q.view(n, groups).contiguous(), (smax.reshape(n) / 127.0).contiguous()


@triton.jit
def _expand_i8_kernel(w_ptr, q_ptr, out_ptr, groups, stride_on, BLOCK_N: tl.constexpr):
    pid_n = tl.program_id(0)
    g = tl.program_id(1)
    rows = tl.arange(0, BLOCK_N)
    words = tl.arange(0, 8)
    lanes = tl.arange(0, 16)
    w = tl.load(w_ptr + (pid_n * groups + g) * (BLOCK_N * 8) + rows[:, None] * 8 + words[None, :])
    q = tl.load(q_ptr + (pid_n * groups + g) * BLOCK_N + rows).to(tl.int32)
    codes = ((w[:, :, None] >> (lanes * 2)[None, None, :]) & 3) - 1
    offs_n = pid_n * BLOCK_N + rows
    tl.store(out_ptr + offs_n[:, None, None] * stride_on + g * 128 + (words * 16)[None, :, None] + lanes[None, None, :],
             (codes * q[:, None, None]).to(tl.int8))


@triton.jit
def _quant_rows_kernel(x_ptr, xq_ptr, xs_ptr, K, stride_x, stride_q, BLOCK_K: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_K)
    amax = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        v = tl.load(x_ptr + row * stride_x + k0 + offs, mask=k0 + offs < K, other=0.0).to(tl.float32)
        amax = tl.maximum(amax, tl.abs(v))
    scale = tl.maximum(tl.max(amax, axis=0), 1e-8) / 127.0
    inv = 1.0 / scale
    for k0 in range(0, K, BLOCK_K):
        v = tl.load(x_ptr + row * stride_x + k0 + offs, mask=k0 + offs < K, other=0.0).to(tl.float32) * inv
        v = tl.where(v >= 0, v + 0.5, v - 0.5).to(tl.int32)
        v = tl.minimum(tl.maximum(v, -127), 127)
        tl.store(xq_ptr + row * stride_q + k0 + offs, v.to(tl.int8), mask=k0 + offs < K)
    tl.store(xs_ptr + row, scale)


@triton.jit
def _rescale_kernel(acc_ptr, xs_ptr, sm_ptr, y_ptr, M, N, stride_a, stride_y, BM: tl.constexpr, BN: tl.constexpr):
    om = tl.program_id(0) * BM + tl.arange(0, BM)
    on = tl.program_id(1) * BN + tl.arange(0, BN)
    mask = (om[:, None] < M) & (on[None, :] < N)
    a = tl.load(acc_ptr + om[:, None] * stride_a + on[None, :], mask=mask, other=0).to(tl.float32)
    xs = tl.load(xs_ptr + om, mask=om < M, other=0.0)
    sm = tl.load(sm_ptr + on, mask=on < N, other=0.0)
    tl.store(y_ptr + om[:, None] * stride_y + on[None, :], (a * xs[:, None] * sm[None, :]).to(y_ptr.dtype.element_ty), mask=mask)


def int8_matmul(rotated: torch.Tensor, weight: torch.Tensor, q: torch.Tensor, sm127: torch.Tensor) -> torch.Tensor:
    m, k = rotated.shape
    n = weight.shape[0]
    groups = k // GROUP
    wi = torch.empty((n, k), dtype=torch.int8, device=rotated.device)
    _expand_i8_kernel[(n // ROW_BLOCK, groups)](weight, q, wi, groups, wi.stride(0), BLOCK_N=ROW_BLOCK, num_warps=4)
    xq = torch.empty((m, k), dtype=torch.int8, device=rotated.device)
    xs = torch.empty((m,), dtype=torch.float32, device=rotated.device)
    _quant_rows_kernel[(m,)](rotated, xq, xs, k, rotated.stride(0), xq.stride(0), BLOCK_K=1024, num_warps=4)
    acc = torch._int_mm(xq, wi.t())
    y = torch.empty((m, n), dtype=rotated.dtype, device=rotated.device)
    _rescale_kernel[(triton.cdiv(m, 64), triton.cdiv(n, 128))](acc, xs, sm127, y, m, n, acc.stride(0), y.stride(0),
                                                               BM=64, BN=128, num_warps=4)
    return y


@triton.jit
def _fused_i8_kernel(xq_ptr, w_ptr, q_ptr, acc_ptr, M, N, KK, groups, BM: tl.constexpr, BN: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    om = pid_m * BM + tl.arange(0, BM)
    on = pid_n * BN + tl.arange(0, BN)
    rb = on // 32
    row = on % 32
    words = tl.arange(0, 8)
    lanes = tl.arange(0, 16)
    kk = tl.arange(0, 128)
    acc = tl.zeros((BM, BN), tl.int32)
    for g in range(0, groups):
        x = tl.load(xq_ptr + om[:, None] * KK + g * 128 + kk[None, :], mask=om[:, None] < M, other=0)
        w = tl.load(w_ptr + ((rb * groups + g) * 32 + row)[:, None] * 8 + words[None, :])
        q = tl.load(q_ptr + (rb * groups + g) * 32 + row).to(tl.int32)
        c = ((w[:, :, None] >> (lanes * 2)[None, None, :]) & 3) - 1
        c = tl.reshape(c, (BN, 128)) * q[:, None]
        acc += tl.dot(x, tl.trans(c.to(tl.int8)))
    tl.store(acc_ptr + om[:, None] * N + on[None, :], acc, mask=om[:, None] < M)


def fused_int8_matmul(rotated: torch.Tensor, weight: torch.Tensor, q: torch.Tensor, sm127: torch.Tensor) -> torch.Tensor:
    m, k = rotated.shape
    n = weight.shape[0]
    xq = torch.empty((m, k), dtype=torch.int8, device=rotated.device)
    xs = torch.empty((m,), dtype=torch.float32, device=rotated.device)
    _quant_rows_kernel[(m,)](rotated, xq, xs, k, rotated.stride(0), xq.stride(0), BLOCK_K=1024, num_warps=4)
    acc = torch.empty((m, n), dtype=torch.int32, device=rotated.device)
    _fused_i8_kernel[(triton.cdiv(m, 128), n // 64)](xq, weight, q, acc, m, n, k, k // GROUP, BM=128, BN=64,
                                                     num_warps=4, num_stages=3)
    y = torch.empty((m, n), dtype=rotated.dtype, device=rotated.device)
    _rescale_kernel[(triton.cdiv(m, 64), triton.cdiv(n, 128))](acc, xs, sm127, y, m, n, acc.stride(0), y.stride(0),
                                                               BM=64, BN=128, num_warps=4)
    return y


@torch.library.custom_op("prism_ternary::matmul_q", mutates_args=())
def ternary_matmul_q(
    rotated: torch.Tensor, weight: torch.Tensor, scales: torch.Tensor, q: torch.Tensor, sm127: torch.Tensor
) -> torch.Tensor:
    if rotated.shape[0] <= GEMV_MAX_ROWS:
        return _small_batch_gemv(rotated, weight, scales)
    if rotated.shape[0] <= FUSED_I8_MAX_ROWS and weight.shape[0] % 64 == 0:
        return fused_int8_matmul(rotated.contiguous(), weight, q, sm127)
    return int8_matmul(rotated.contiguous(), weight, q, sm127)


@ternary_matmul_q.register_fake
def _ternary_matmul_q_fake(rotated, weight, scales, q, sm127):
    return rotated.new_empty(rotated.shape[0], weight.shape[0])


@torch.library.custom_op("prism_ternary::linear_q", mutates_args=())
def ternary_linear_q(
    x: torch.Tensor, weight: torch.Tensor, scales: torch.Tensor, q: torch.Tensor, sm127: torch.Tensor, signs: torch.Tensor
) -> torch.Tensor:
    rotated = _cuda().signed_hadamard(x.reshape(-1, x.shape[-1]), signs)
    y = ternary_matmul_q(rotated, weight, scales, q, sm127)
    return y.reshape(*x.shape[:-1], weight.shape[0])


@ternary_linear_q.register_fake
def _ternary_linear_q_fake(x, weight, scales, q, sm127, signs):
    return x.new_empty(*x.shape[:-1], weight.shape[0])


@torch.library.custom_op("prism_ternary::hadamard", mutates_args=())
def signed_hadamard(x: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    return _cuda().signed_hadamard(x, signs)


@signed_hadamard.register_fake
def _signed_hadamard_fake(x: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


def _small_batch_gemv(rotated: torch.Tensor, weight: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    if DP4A_DECODE and rotated.shape[0] <= DP4A_MAX_ROWS:
        xq, xs, gs = _cuda().dp4a_quant_acts(rotated.contiguous())
        return _cuda().dp4a_gemv(xq, xs, gs, weight, scales)
    return _cuda().ternary_gemv(rotated, weight, scales)


@torch.library.custom_op("prism_ternary::matmul", mutates_args=())
def ternary_matmul(rotated: torch.Tensor, weight: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    return (
        _small_batch_gemv(rotated, weight, scales)
        if rotated.shape[0] <= GEMV_MAX_ROWS
        else rotated @ dequantize(weight, scales, rotated.dtype).T
    )


@ternary_matmul.register_fake
def _ternary_matmul_fake(rotated: torch.Tensor, weight: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    return rotated.new_empty(rotated.shape[0], weight.shape[0])


@torch.library.custom_op("prism_ternary::linear", mutates_args=())
def ternary_linear(
    x: torch.Tensor, weight: torch.Tensor, scales: torch.Tensor, signs: torch.Tensor
) -> torch.Tensor:
    rotated = _cuda().signed_hadamard(x.reshape(-1, x.shape[-1]), signs)
    y = (
        _small_batch_gemv(rotated, weight, scales)
        if rotated.shape[0] <= GEMV_MAX_ROWS
        else rotated @ dequantize(weight, scales, rotated.dtype).T
    )
    return y.reshape(*x.shape[:-1], weight.shape[0])


@ternary_linear.register_fake
def _ternary_linear_fake(
    x: torch.Tensor, weight: torch.Tensor, scales: torch.Tensor, signs: torch.Tensor
) -> torch.Tensor:
    return x.new_empty(*x.shape[:-1], weight.shape[0])
