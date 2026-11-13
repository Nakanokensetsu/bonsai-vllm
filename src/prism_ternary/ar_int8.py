"""TP=2 prefill all-reduce を int8(128要素ごとのfp16スケール)で交換する。
WSL2はGPU P2Pが無くNCCLが約3GB/sなので、帯域を半分にして prefill/混在ステップを速くする狙い(2026-10-29)。

交換: 各ランクが自分の部分和を [int8 codes | fp16 scales] に詰めて all_gather し、
      両ランクが「両方を逆量子化して足す」。両ランクの結果が一致する(自分の側を厳密に使うとランク間でずれる)。
有効化: PRISM_AR_INT8=1。行数が PRISM_AR_INT8_MIN_ROWS(既定192)以上の bf16 2次元テンソルだけ。
        CUDAグラフのキャプチャ中は使わない(デコードは元の経路)。
"""
import os
import sys

import torch



GROUP = int(os.environ.get("PRISM_AR_INT8_GROUP", "64"))
ENABLED = os.environ.get("PRISM_AR_INT8", "0") == "1"
_used = False
MIN_ROWS = int(os.environ.get("PRISM_AR_INT8_MIN_ROWS", "192"))
IMPL = os.environ.get("PRISM_AR_INT8_IMPL", "triton")  # triton | torch


def pack(x: torch.Tensor) -> torch.Tensor:
    """x: [M, H] bf16 (H % GROUP == 0) -> [M, H + 2*H/GROUP] int8"""
    M, H = x.shape
    xg = x.view(M, H // GROUP, GROUP).float()
    scale = (xg.abs().amax(-1) / 127.0).clamp_min(1e-8).to(torch.float16)  # 逆量子化と同じ丸めのスケールで量子化する
    q = (xg / scale.float().unsqueeze(-1)).round().clamp_(-127, 127).to(torch.int8).view(M, H)
    return torch.cat([q, scale.view(torch.int8).view(M, -1)], dim=1)


def unpack(p: torch.Tensor, H: int) -> torch.Tensor:
    """p: [..., M, H + 2*H/GROUP] int8 -> [..., M, H] float32"""
    q = p[..., :H].reshape(*p.shape[:-1], H // GROUP, GROUP).float()
    s = p[..., H:].contiguous().view(torch.float16).float()  # [..., M, H/GROUP]
    return (q * s.unsqueeze(-1)).reshape(*p.shape[:-1], H)


def allreduce_int8(x: torch.Tensor, all_gather_into) -> torch.Tensor:
    """all_gather_into(out, inp): out は [world*M, W] の int8、inp は [M, W]。world=2 前提。"""
    if IMPL == "triton":
        from prism_ternary.ar_int8_triton import allreduce_int8 as tri  # 同じ形式・丸めの Triton 版(2026-11-03)
        return tri(x, all_gather_into, GROUP)
    M, H = x.shape
    p = pack(x.contiguous())
    out = torch.empty(2 * M, p.shape[1], dtype=torch.int8, device=x.device)
    all_gather_into(out, p)
    y = unpack(out.view(2, M, -1), H)
    return (y[0] + y[1]).to(x.dtype)


def patch_vllm():
    """CudaCommunicator.all_reduce を包む。デコード(CUDAグラフ)と小さい行数は元の経路。"""
    from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator

    orig = CudaCommunicator.all_reduce

    def all_reduce(self, input_):
        x = input_
        if (
            self.world_size == 2
            and self.pynccl_comm is not None
            and not self.pynccl_comm.disabled
            and x.dtype == torch.bfloat16
            and x.dim() == 2
            and x.shape[0] >= MIN_ROWS
            and x.shape[1] % GROUP == 0
            and not torch.cuda.is_current_stream_capturing()
        ):
            global _used
            if not _used:
                _used = True
                print(f"ar_int8: first int8 all-reduce, shape={tuple(x.shape)} pid={os.getpid()}", flush=True)
            return allreduce_int8(x, self.pynccl_comm.all_gather)
        return orig(self, input_)

    CudaCommunicator.all_reduce = all_reduce
    print(f"ar_int8: CudaCommunicator.all_reduce patched (group={GROUP}, min_rows={MIN_ROWS}) pid={os.getpid()}", flush=True)
