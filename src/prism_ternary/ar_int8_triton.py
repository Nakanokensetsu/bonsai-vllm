"""ar_int8 の pack / unpack+加算を Triton の 1 カーネルずつにしたもの(2026-11-03)。
PyTorch 版(ar_int8.py)は pack/unpack に 0.13〜0.5 ms かかり、通信の短縮(理論 2 倍)を 1.2〜1.35 倍に削っていた。
形式・丸めは PyTorch 版と同一(group ごとの absmax/127 を fp16 に丸めたスケールで rint)なので、結果はビット一致する。
"""
import torch
import triton
import triton.language as tl

GROUPS_PER_PROG = 16


@triton.jit
def _pack_kernel(X, OUT, s_xm, s_om, H: tl.constexpr, G: tl.constexpr, NG: tl.constexpr, GB: tl.constexpr):
    """X [M, H] bf16 -> OUT [M, H + 2*NG] int8 = [codes | fp16 scales]。grid (M, cdiv(NG, GB))。"""
    m = tl.program_id(0)
    g = tl.program_id(1) * GB + tl.arange(0, GB)
    gm = g < NG
    offs = g[:, None] * G + tl.arange(0, G)[None, :]
    x = tl.load(X + m * s_xm + offs, mask=gm[:, None], other=0.0).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(x), 1) / 127.0, 1e-8).to(tl.float16)
    s = scale.to(tl.float32)
    # 正確な除算(div_rn): 近似除算だと .5 の境目で PyTorch 版と丸めが分かれる
    q = tl.minimum(tl.maximum(tl.extra.cuda.libdevice.rint(tl.extra.cuda.libdevice.div_rn(x, s[:, None])), -127.0), 127.0)
    # 全要素 0 のグループはスケールが fp16 で 0 に潰れる(0/0)。逆量子化は 0 になるので符号も 0 にしておく
    q = tl.where(s[:, None] > 0, q, 0.0).to(tl.int8)
    row = OUT + m * s_om
    tl.store(row + offs, q, mask=gm[:, None])
    tl.store((row + H + 2 * g).to(tl.pointer_type(tl.float16), bitcast=True), scale, mask=gm)


@triton.jit
def _unpack_sum_kernel(P, Y, s_pr, s_pm, s_ym, H: tl.constexpr, G: tl.constexpr, NG: tl.constexpr, GB: tl.constexpr):
    """P [2, M, W] int8(2 ランク分)-> Y [M, H] bf16 = deq(rank0) + deq(rank1)。grid (M, cdiv(NG, GB))。"""
    m = tl.program_id(0)
    g = tl.program_id(1) * GB + tl.arange(0, GB)
    gm = g < NG
    offs = g[:, None] * G + tl.arange(0, G)[None, :]
    r0 = P + m * s_pm
    r1 = r0 + s_pr
    q0 = tl.load(r0 + offs, mask=gm[:, None], other=0).to(tl.float32)
    q1 = tl.load(r1 + offs, mask=gm[:, None], other=0).to(tl.float32)
    s0 = tl.load((r0 + H + 2 * g).to(tl.pointer_type(tl.float16), bitcast=True), mask=gm, other=0.0).to(tl.float32)
    s1 = tl.load((r1 + H + 2 * g).to(tl.pointer_type(tl.float16), bitcast=True), mask=gm, other=0.0).to(tl.float32)
    y = q0 * s0[:, None] + q1 * s1[:, None]
    tl.store(Y + m * s_ym + offs, y.to(Y.dtype.element_ty), mask=gm[:, None])


def pack(x: torch.Tensor, group: int, out: torch.Tensor | None = None) -> torch.Tensor:
    M, H = x.shape
    ng = H // group
    if out is None:
        out = torch.empty(M, H + 2 * ng, dtype=torch.int8, device=x.device)
    _pack_kernel[(M, triton.cdiv(ng, GROUPS_PER_PROG))](
        x, out, x.stride(0), out.stride(0), H=H, G=group, NG=ng, GB=GROUPS_PER_PROG, num_warps=4)
    return out


def unpack_sum(p: torch.Tensor, H: int, group: int, dtype) -> torch.Tensor:
    """p: [2, M, W] int8"""
    _, M, _ = p.shape
    ng = H // group
    y = torch.empty(M, H, dtype=dtype, device=p.device)
    _unpack_sum_kernel[(M, triton.cdiv(ng, GROUPS_PER_PROG))](
        p, y, p.stride(0), p.stride(1), y.stride(0), H=H, G=group, NG=ng, GB=GROUPS_PER_PROG, num_warps=4)
    return y


def allreduce_int8(x: torch.Tensor, all_gather_into, group: int) -> torch.Tensor:
    """all_gather_into(out, inp): out は [2*M, W] int8、inp は [M, W]。world=2 前提。"""
    M, H = x.shape
    p = pack(x.contiguous(), group)
    out = torch.empty(2 * M, p.shape[1], dtype=torch.int8, device=x.device)
    all_gather_into(out, p)
    return unpack_sum(out.view(2, M, -1), H, group, x.dtype)
