#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

namespace {

constexpr int kWarpsPerBlock = 2;
constexpr int kThreads = 32 * kWarpsPerBlock;
constexpr int kTileN = 8;
constexpr int kTilesPerWarp = 4;
constexpr int kRowBlock = kTileN * kTilesPerWarp;
constexpr int kRowsPerBlock = kWarpsPerBlock * kRowBlock;
constexpr int kWordsPerGroupBlock = kRowBlock * 8;
constexpr int kMaxRows = 64;
constexpr int kTargetWarps = 2048;
constexpr int kMaxSplitK = 16;

__device__ __forceinline__ uint32_t bf16_bits(int value) {
    return value < 0 ? 0xBF80u : (value > 0 ? 0x3F80u : 0u);
}

__device__ __forceinline__ void mma_bf16(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

template <int kMBlocks, bool kAtomic>
__global__ void __launch_bounds__(kThreads) ternary_mma_kernel(
    const __nv_bfloat16* __restrict__ x,
    const uint32_t* __restrict__ w,
    const __half* __restrict__ s,
    void* __restrict__ y,
    int rows,
    int N,
    int K,
    int groups_per_split) {
    __shared__ uint32_t lut[16];
    if (threadIdx.x < 16) lut[threadIdx.x] = bf16_bits((threadIdx.x & 3) - 1) | (bf16_bits((threadIdx.x >> 2) - 1) << 16);
    __syncthreads();
    const int groups = K / 128;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int gid = lane >> 2;
    const int tig = lane & 3;
    const int n0 = (blockIdx.x * kWarpsPerBlock + warp) * kRowBlock;
    if (n0 >= N) return;
    const int g_begin = blockIdx.y * groups_per_split;
    const int g_end = min(groups, g_begin + groups_per_split);
    const uint32_t* wblock = w + static_cast<size_t>(n0 / kRowBlock) * groups * kWordsPerGroupBlock + gid * 8;
    const __half* sblock = s + static_cast<size_t>(n0 / kRowBlock) * groups * kRowBlock + 2 * tig;
    float acc[kTilesPerWarp][kMBlocks][4];
#pragma unroll
    for (int t = 0; t < kTilesPerWarp; ++t)
#pragma unroll
        for (int b = 0; b < kMBlocks; ++b)
#pragma unroll
            for (int i = 0; i < 4; ++i) acc[t][b][i] = 0.f;
    const __nv_bfloat16* xlo[kMBlocks];
    bool a_lo[kMBlocks];
    bool a_hi[kMBlocks];
#pragma unroll
    for (int b = 0; b < kMBlocks; ++b) {
        const int m = b * 16 + gid;
        a_lo[b] = m < rows;
        a_hi[b] = m + 8 < rows;
        xlo[b] = x + static_cast<size_t>(min(m, rows - 1)) * K + 2 * tig;
    }
    uint4 pk[kTilesPerWarp][2];
    auto load_group = [&](int g, uint4 (*dst)[2]) {
        const uint32_t* src = wblock + static_cast<size_t>(g) * kWordsPerGroupBlock;
#pragma unroll
        for (int t = 0; t < kTilesPerWarp; ++t) {
#pragma unroll
            for (int h = 0; h < 2; ++h) dst[t][h] = *reinterpret_cast<const uint4*>(src + t * kTileN * 8 + h * 4);
        }
    };
    if (g_begin < g_end) load_group(g_begin, pk);
    for (int g = g_begin; g < g_end; ++g) {
        uint4 cur[kTilesPerWarp][2];
#pragma unroll
        for (int t = 0; t < kTilesPerWarp; ++t) {
            cur[t][0] = pk[t][0];
            cur[t][1] = pk[t][1];
        }
        if (g + 1 < g_end) load_group(g + 1, pk);
        float ctmp[kTilesPerWarp][kMBlocks][4];
#pragma unroll
        for (int t = 0; t < kTilesPerWarp; ++t)
#pragma unroll
            for (int b = 0; b < kMBlocks; ++b)
#pragma unroll
                for (int i = 0; i < 4; ++i) ctmp[t][b][i] = 0.f;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int k = g * 128 + h * 64 + j * 16;
                uint32_t a[kMBlocks][4];
#pragma unroll
                for (int b = 0; b < kMBlocks; ++b) {
                    const __nv_bfloat16* xp = xlo[b] + k;
                    a[b][0] = a_lo[b] ? *reinterpret_cast<const uint32_t*>(xp) : 0u;
                    a[b][2] = a_lo[b] ? *reinterpret_cast<const uint32_t*>(xp + 8) : 0u;
                    a[b][1] = a_hi[b] ? *reinterpret_cast<const uint32_t*>(xp + static_cast<size_t>(8) * K) : 0u;
                    a[b][3] = a_hi[b] ? *reinterpret_cast<const uint32_t*>(xp + static_cast<size_t>(8) * K + 8) : 0u;
                }
#pragma unroll
                for (int t = 0; t < kTilesPerWarp; ++t) {
                    const uint32_t word = j == 0 ? cur[t][h].x : (j == 1 ? cur[t][h].y : (j == 2 ? cur[t][h].z : cur[t][h].w));
                    const uint32_t b0 = lut[(word >> (4 * tig)) & 0xFu];
                    const uint32_t b1 = lut[(word >> (4 * tig + 16)) & 0xFu];
#pragma unroll
                    for (int b = 0; b < kMBlocks; ++b) mma_bf16(ctmp[t][b], a[b][0], a[b][1], a[b][2], a[b][3], b0, b1);
                }
            }
        }
        const __half* sg = sblock + static_cast<size_t>(g) * kRowBlock;
#pragma unroll
        for (int t = 0; t < kTilesPerWarp; ++t) {
            const __half2 sc = *reinterpret_cast<const __half2*>(sg + t * kTileN);
            const float sc0 = __low2float(sc);
            const float sc1 = __high2float(sc);
#pragma unroll
            for (int b = 0; b < kMBlocks; ++b) {
                acc[t][b][0] = fmaf(sc0, ctmp[t][b][0], acc[t][b][0]);
                acc[t][b][1] = fmaf(sc1, ctmp[t][b][1], acc[t][b][1]);
                acc[t][b][2] = fmaf(sc0, ctmp[t][b][2], acc[t][b][2]);
                acc[t][b][3] = fmaf(sc1, ctmp[t][b][3], acc[t][b][3]);
            }
        }
    }
#pragma unroll
    for (int t = 0; t < kTilesPerWarp; ++t) {
        const int nc0 = n0 + t * kTileN + 2 * tig;
        const int nc1 = nc0 + 1;
#pragma unroll
        for (int b = 0; b < kMBlocks; ++b) {
            const size_t m_lo = static_cast<size_t>(b * 16 + gid);
            const size_t m_hi = m_lo + 8;
            if constexpr (kAtomic) {
                float* out = static_cast<float*>(y) + static_cast<size_t>(blockIdx.y) * rows * N;
                if (a_lo[b]) out[m_lo * N + nc0] = acc[t][b][0];
                if (a_lo[b]) out[m_lo * N + nc1] = acc[t][b][1];
                if (a_hi[b]) out[m_hi * N + nc0] = acc[t][b][2];
                if (a_hi[b]) out[m_hi * N + nc1] = acc[t][b][3];
            } else {
                __nv_bfloat16* out = static_cast<__nv_bfloat16*>(y);
                if (a_lo[b]) *reinterpret_cast<__nv_bfloat162*>(out + m_lo * N + nc0) = __floats2bfloat162_rn(acc[t][b][0], acc[t][b][1]);
                if (a_hi[b]) *reinterpret_cast<__nv_bfloat162*>(out + m_hi * N + nc0) = __floats2bfloat162_rn(acc[t][b][2], acc[t][b][3]);
            }
        }
    }
}

__global__ void reduce_split_kernel(
    const float* __restrict__ partial,
    __nv_bfloat16* __restrict__ y,
    int elements,
    int split) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= elements) return;
    float sum = 0.f;
    for (int p = 0; p < split; ++p) sum += partial[static_cast<size_t>(p) * elements + i];
    y[i] = __float2bfloat16(sum);
}

template <int kMBlocks>
void launch_mma(const __nv_bfloat16* x, const uint32_t* w, const __half* s, void* y, int rows, int N, int K, int split, int groups_per_split, cudaStream_t stream) {
    const dim3 grid(N / kRowsPerBlock, split);
    if (split == 1) ternary_mma_kernel<kMBlocks, false><<<grid, kThreads, 0, stream>>>(x, w, s, y, rows, N, K, groups_per_split);
    else ternary_mma_kernel<kMBlocks, true><<<grid, kThreads, 0, stream>>>(x, w, s, y, rows, N, K, groups_per_split);
}

constexpr int kHadamardBlock = 1024;
constexpr int kHadamardThreads = 256;

__global__ void __launch_bounds__(kHadamardThreads) signed_hadamard_kernel(
    const __nv_bfloat16* __restrict__ x,
    const float* __restrict__ signs,
    __nv_bfloat16* __restrict__ y,
    int K) {
    __shared__ float buf[kHadamardBlock];
    const size_t base = static_cast<size_t>(blockIdx.y) * K + static_cast<size_t>(blockIdx.x) * kHadamardBlock;
    const int col0 = blockIdx.x * kHadamardBlock;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int idx = threadIdx.x + i * kHadamardThreads;
        buf[idx] = __bfloat162float(x[base + idx]) * signs[col0 + idx];
    }
    __syncthreads();
    float values[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) values[i] = buf[threadIdx.x + i * kHadamardThreads];
#pragma unroll
    for (int stride = 1; stride < 32; stride <<= 1) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float other = __shfl_xor_sync(0xffffffffu, values[i], stride);
            values[i] = (threadIdx.x & stride) ? other - values[i] : values[i] + other;
        }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) buf[threadIdx.x + i * kHadamardThreads] = values[i];
    __syncthreads();
    for (int stride = 32; stride < kHadamardBlock; stride <<= 1) {
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int p = threadIdx.x + i * kHadamardThreads;
            const int a = ((p / stride) * stride * 2) + (p % stride);
            const int b = a + stride;
            const float u = buf[a];
            const float v = buf[b];
            buf[a] = u + v;
            buf[b] = u - v;
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int idx = threadIdx.x + i * kHadamardThreads;
        y[base + idx] = __float2bfloat16(buf[idx] * 0.03125f);
    }
}


// ---- batched int8 dp4a GEMV (opt-in PRISM_DP4A_DECODE=1; idea from bonsai-turbo, Apache-2.0) ----
constexpr int kDp4aGroup = 128;
constexpr int kDp4aMaxB = 8;
constexpr int kDp4aWarps = 8;

__device__ __forceinline__ int dp4a_u8s8(uint32_t codes, uint32_t acts, int acc) {
    asm("dp4a.u32.s32 %0, %1, %2, %3;" : "=r"(acc) : "r"(codes), "r"(acts), "r"(acc));
    return acc;
}

// One warp per (row b, 128-group g). Output bytes of a group are in dp4a order:
// for word j (codes 16j..16j+15) and step i: bytes a[16j+i], a[16j+4+i], a[16j+8+i], a[16j+12+i].
__global__ void quant_acts_kernel(const __nv_bfloat16* __restrict__ x, int K, int groups,
                                  int8_t* __restrict__ xq, float* __restrict__ xs,
                                  int* __restrict__ gsum) {
    const int b = blockIdx.y;
    const int g = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (g >= groups) return;
    const int lane = threadIdx.x & 31;
    const __nv_bfloat16* xr = x + (size_t)b * K + g * kDp4aGroup;
    float v[4];
    float amax = 0.f;
#pragma unroll
    for (int t = 0; t < 4; ++t) {
        v[t] = __bfloat162float(xr[lane * 4 + t]);
        amax = fmaxf(amax, fabsf(v[t]));
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
    const float s = fmaxf(amax, 1e-8f) / 127.f;
    const float inv = 1.f / s;
    int lsum = 0;
    int8_t* out = xq + (size_t)b * K + g * kDp4aGroup;
#pragma unroll
    for (int t = 0; t < 4; ++t) {
        const int k = lane * 4 + t;               // position inside the group
        int q = __float2int_rn(v[t] * inv);
        q = max(-127, min(127, q));
        lsum += q;
        const int j = k >> 4, l = k & 15;         // word j, code l in the word
        const int i = l & 3, byte = l >> 2;       // dp4a step i, byte lane
        out[j * 16 + i * 4 + byte] = (int8_t)q;
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) lsum += __shfl_xor_sync(0xffffffffu, lsum, off);
    if (lane == 0) {
        xs[b * groups + g] = s;
        gsum[b * groups + g] = lsum;
    }
}

template <int B>
__global__ void __launch_bounds__(kDp4aWarps * 32) gemv_dp4a_kernel(
    const uint32_t* __restrict__ w, const __half* __restrict__ ws,
    const int8_t* __restrict__ xq, const float* __restrict__ xs, const int* __restrict__ gsum,
    __nv_bfloat16* __restrict__ y, int N, int K, int groups) {
    const int rb = blockIdx.x;                    // row block of 32 rows
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;            // row inside the block
    float acc[B];
#pragma unroll
    for (int b = 0; b < B; ++b) acc[b] = 0.f;
    for (int g = warp; g < groups; g += kDp4aWarps) {
        const size_t slot = ((size_t)rb * groups + g) * 32 + lane;
        const uint4* wp = reinterpret_cast<const uint4*>(w + slot * 8);
        const uint4 w0 = wp[0], w1 = wp[1];
        const uint32_t words[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
        const float wscale = __half2float(ws[slot]);
#pragma unroll
        for (int b = 0; b < B; ++b) {
            const uint32_t* ap = reinterpret_cast<const uint32_t*>(xq + (size_t)b * K + g * kDp4aGroup);
            int dot = 0;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const uint4 a = reinterpret_cast<const uint4*>(ap)[j];   // broadcast across lanes
                dot = dp4a_u8s8(words[j] & 0x03030303u, a.x, dot);
                dot = dp4a_u8s8((words[j] >> 2) & 0x03030303u, a.y, dot);
                dot = dp4a_u8s8((words[j] >> 4) & 0x03030303u, a.z, dot);
                dot = dp4a_u8s8((words[j] >> 6) & 0x03030303u, a.w, dot);
            }
            acc[b] += wscale * xs[b * groups + g] * (float)(dot - gsum[b * groups + g]);
        }
    }
    __shared__ float red[kDp4aWarps][32][B];
#pragma unroll
    for (int b = 0; b < B; ++b) red[warp][lane][b] = acc[b];
    __syncthreads();
    if (warp == 0) {
#pragma unroll
        for (int b = 0; b < B; ++b) {
            float s = 0.f;
#pragma unroll
            for (int ww = 0; ww < kDp4aWarps; ++ww) s += red[ww][lane][b];
            y[(size_t)b * N + rb * 32 + lane] = __float2bfloat16(s);
        }
    }
}

}  // namespace

torch::Tensor signed_hadamard(torch::Tensor x, torch::Tensor signs) {
    TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kBFloat16 && x.is_contiguous(), "x must be contiguous bf16 on cuda");
    TORCH_CHECK(signs.dtype() == torch::kFloat32 && signs.is_contiguous(), "signs must be contiguous fp32");
    const int rows = x.size(0);
    const int K = x.size(1);
    TORCH_CHECK(K % kHadamardBlock == 0 && signs.numel() == K, "K must be a multiple of 1024 and match signs");
    auto y = torch::empty_like(x);
    const dim3 grid(K / kHadamardBlock, rows);
    signed_hadamard_kernel<<<grid, kHadamardThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        signs.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(y.data_ptr()),
        K);
    return y;
}

torch::Tensor ternary_gemv(torch::Tensor x, torch::Tensor w, torch::Tensor s) {
    TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kBFloat16 && x.is_contiguous(), "x must be contiguous bf16 on cuda");
    TORCH_CHECK(w.dtype() == torch::kInt32 && w.is_contiguous(), "weight must be contiguous int32 in blocked layout");
    TORCH_CHECK(s.dtype() == torch::kFloat16 && s.is_contiguous(), "scales must be contiguous fp16 in blocked layout");
    const int rows = x.size(0);
    const int K = x.size(1);
    const int N = w.size(0);
    TORCH_CHECK(K % 128 == 0, "K must be a multiple of 128");
    TORCH_CHECK(N % kRowsPerBlock == 0, "N must be a multiple of 64");
    TORCH_CHECK(rows >= 1 && rows <= kMaxRows, "rows must be in [1, 64]");
    const int groups = K / 128;
    const int warps_n = N / kRowBlock;
    int split = min(kMaxSplitK, max(1, (kTargetWarps + warps_n - 1) / warps_n));
    const int groups_per_split = (groups + split - 1) / split;
    split = (groups + groups_per_split - 1) / groups_per_split;
    auto stream = at::cuda::getCurrentCUDAStream();
    const auto* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
    const auto* wp = reinterpret_cast<const uint32_t*>(w.data_ptr<int32_t>());
    const auto* sp = reinterpret_cast<const __half*>(s.data_ptr());
    auto y = torch::empty({rows, N}, x.options());
    auto partial = split == 1 ? torch::Tensor() : torch::empty({split, rows, N}, x.options().dtype(torch::kFloat32));
    void* output = split == 1 ? y.data_ptr() : partial.data_ptr();
    if (rows <= 16) launch_mma<1>(xp, wp, sp, output, rows, N, K, split, groups_per_split, stream);
    else if (rows <= 32) launch_mma<2>(xp, wp, sp, output, rows, N, K, split, groups_per_split, stream);
    else launch_mma<4>(xp, wp, sp, output, rows, N, K, split, groups_per_split, stream);
    if (split > 1) {
        const int elements = rows * N;
        reduce_split_kernel<<<(elements + 255) / 256, 256, 0, stream>>>(
            partial.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(y.data_ptr()),
            elements,
            split);
    }
    return y;
}

std::vector<torch::Tensor> quant_acts(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kBFloat16 && x.is_contiguous());
    const int B = x.size(0), K = x.size(1), groups = K / kDp4aGroup;
    TORCH_CHECK(K % kDp4aGroup == 0);
    auto xq = torch::empty({B, K}, x.options().dtype(torch::kInt8));
    auto xs = torch::empty({B, groups}, x.options().dtype(torch::kFloat32));
    auto gs = torch::empty({B, groups}, x.options().dtype(torch::kInt32));
    dim3 grid((groups + 7) / 8, B);
    quant_acts_kernel<<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), K, groups, xq.data_ptr<int8_t>(),
        xs.data_ptr<float>(), gs.data_ptr<int>());
    return {xq, xs, gs};
}

torch::Tensor gemv_dp4a(torch::Tensor xq, torch::Tensor xs, torch::Tensor gs, torch::Tensor w, torch::Tensor ws) {
    const int B = xq.size(0), K = xq.size(1), N = w.size(0), groups = K / kDp4aGroup;
    TORCH_CHECK(B >= 1 && B <= kDp4aMaxB && N % 32 == 0);
    auto y = torch::empty({B, N}, xq.options().dtype(torch::kBFloat16));
    auto stream = at::cuda::getCurrentCUDAStream();
    const auto* wp = reinterpret_cast<const uint32_t*>(w.data_ptr<int32_t>());
    const auto* sp = reinterpret_cast<const __half*>(ws.data_ptr());
    auto* yp = reinterpret_cast<__nv_bfloat16*>(y.data_ptr());
#define LAUNCH(NB) gemv_dp4a_kernel<NB><<<N / 32, kDp4aWarps * 32, 0, stream>>>(wp, sp, xq.data_ptr<int8_t>(), xs.data_ptr<float>(), gs.data_ptr<int>(), yp, N, K, groups)
    switch (B) {  // exact batch per instantiation: rows beyond B are never read or written
        case 1: LAUNCH(1); break;
        case 2: LAUNCH(2); break;
        case 3: LAUNCH(3); break;
        case 4: LAUNCH(4); break;
        case 5: LAUNCH(5); break;
        case 6: LAUNCH(6); break;
        case 7: LAUNCH(7); break;
        default: LAUNCH(8); break;
    }
#undef LAUNCH
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ternary_gemv", &ternary_gemv, "ternary 2-bit gemm for <=64 rows (bf16 activations, tensor cores, blocked weight layout)");
    m.def("dp4a_quant_acts", &quant_acts, "per-128-group int8 activations in dp4a byte order");
    m.def("dp4a_gemv", &gemv_dp4a, "batched (<=8 rows) int8 dp4a GEMV over the blocked 2-bit layout");
    m.def("signed_hadamard", &signed_hadamard, "blockwise signed Walsh-Hadamard transform (block 1024)");
}
