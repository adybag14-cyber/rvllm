// CUTLASS 3.x SM90 autotuned FP8 GEMM variants for rvLLM.
//
// 15 tile/cluster/schedule configurations compiled as separate extern "C"
// entry points. The Rust autotune engine benchmarks all variants per
// (M,N,K) shape and caches the winner.
//
// D[m,n] = cast_to_f16(a_scales[m] * b_scale[0] * sum_k(A_fp8[m,k] * B_fp8[k,n]))
//
// FP8 GEMM runs alpha=1, beta=0, then a post-kernel applies per-row A scales
// and per-tensor B scale (same pattern as cutlass_fp8_gemm.cu).
//
// A=[M,K] RowMajor FP8 E4M3, B=[N,K] RowMajor (ColumnMajor to CUTLASS) FP8 E4M3
// D=[M,N] RowMajor F16

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cute/tensor.hpp>
#include <cutlass/util/packed_stride.hpp>
#include <cuda_fp16.h>

using namespace cute;

using ElementA = cutlass::float_e4m3_t;
using ElementB = cutlass::float_e4m3_t;
using ElementC = cutlass::half_t;
using ElementD = cutlass::half_t;
using ElementAccum = float;
using ElementCompute = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

// ---------------------------------------------------------------------------
// Template: build a full CUTLASS 3.x FP8 GEMM type from tile/cluster/schedule
// ---------------------------------------------------------------------------

template<typename TileShape_, typename ClusterShape_, typename KernelSchedule_>
struct Fp8GemmType {
    using EpilogueOp = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
        TileShape_, ClusterShape_,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccum, ElementCompute,
        ElementC, LayoutC, 8,
        ElementD, LayoutD, 8,
        cutlass::epilogue::collective::EpilogueScheduleAuto
    >::CollectiveOp;

    using MainloopOp = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90,
        cutlass::arch::OpClassTensorOp,
        ElementA, LayoutA, 16,   // alignment 16 for FP8
        ElementB, LayoutB, 16,   // alignment 16 for FP8
        ElementAccum,
        TileShape_,
        ClusterShape_,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename EpilogueOp::SharedStorage))>,
        KernelSchedule_
    >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int, int, int, int>,
        MainloopOp,
        EpilogueOp
    >;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

// ---------------------------------------------------------------------------
// Post-GEMM kernel: apply per-row A scales and per-tensor B scale
// ---------------------------------------------------------------------------

__global__ void apply_fp8_scales_kernel_autotune(
    __half* output, const float* row_scales, const float* col_scale,
    int M, int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M * N) {
        int row = idx / N;
        float val = __half2float(output[idx]) * row_scales[row] * col_scale[0];
        val = fmaxf(-65504.0f, fminf(65504.0f, val));
        output[idx] = __float2half_rn(val);
    }
}

// ---------------------------------------------------------------------------
// Dispatch template: run an FP8 GEMM variant + apply scales
// ---------------------------------------------------------------------------

template<typename TileShape_, typename ClusterShape_, typename KernelSchedule_>
int fp8_gemm_dispatch(
    void* output, const void* a, const void* b,
    const void* a_scales, const void* b_scale,
    int M, int N, int K,
    void* workspace, size_t workspace_size,
    cudaStream_t stream)
{
    using G = Fp8GemmType<TileShape_, ClusterShape_, KernelSchedule_>;
    using Gemm = typename G::Gemm;

    auto prob_shape = cute::make_shape(M, N, K, 1);

    auto stride_A = cutlass::make_cute_packed_stride(
        typename Gemm::GemmKernel::StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(
        typename Gemm::GemmKernel::StrideB{}, {N, K, 1});
    auto stride_C = cutlass::make_cute_packed_stride(
        typename Gemm::GemmKernel::StrideC{}, {M, N, 1});
    auto stride_D = cutlass::make_cute_packed_stride(
        typename Gemm::GemmKernel::StrideD{}, {M, N, 1});

    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        prob_shape,
        {
            reinterpret_cast<const ElementA*>(a), stride_A,
            reinterpret_cast<const ElementB*>(b), stride_B,
        },
        {
            {ElementAccum(1.0f), ElementAccum(0.0f)},
            reinterpret_cast<const ElementC*>(output), stride_C,
            reinterpret_cast<ElementD*>(output), stride_D,
        }
    };

    Gemm gemm_op;

    cutlass::Status status = gemm_op.can_implement(args);
    if (status != cutlass::Status::kSuccess) return -1;

    status = gemm_op.initialize(args, workspace, stream);
    if (status != cutlass::Status::kSuccess) return -2;

    status = gemm_op(stream);
    if (status != cutlass::Status::kSuccess) return -3;

    // Apply per-row A scales and per-tensor B scale
    int total = M * N;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    apply_fp8_scales_kernel_autotune<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<__half*>(output),
        reinterpret_cast<const float*>(a_scales),
        reinterpret_cast<const float*>(b_scale),
        M, N
    );

    return 0;
}

// ---------------------------------------------------------------------------
// Workspace-size template
// ---------------------------------------------------------------------------

template<typename TileShape_, typename ClusterShape_, typename KernelSchedule_>
size_t fp8_gemm_ws_dispatch(int M, int N, int K)
{
    using G = Fp8GemmType<TileShape_, ClusterShape_, KernelSchedule_>;
    using Gemm = typename G::Gemm;

    auto prob_shape = cute::make_shape(M, N, K, 1);

    auto stride_A = cutlass::make_cute_packed_stride(
        typename Gemm::GemmKernel::StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(
        typename Gemm::GemmKernel::StrideB{}, {N, K, 1});
    auto stride_C = cutlass::make_cute_packed_stride(
        typename Gemm::GemmKernel::StrideC{}, {M, N, 1});
    auto stride_D = cutlass::make_cute_packed_stride(
        typename Gemm::GemmKernel::StrideD{}, {M, N, 1});

    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        prob_shape,
        {nullptr, stride_A, nullptr, stride_B},
        {{ElementAccum(1.0f), ElementAccum(0.0f)}, nullptr, stride_C, nullptr, stride_D}
    };

    Gemm gemm_op;
    return gemm_op.get_workspace_size(args);
}

// ---------------------------------------------------------------------------
// Schedule aliases
// ---------------------------------------------------------------------------

using WS   = cutlass::gemm::KernelTmaWarpSpecialized;
using Coop = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
using PP   = cutlass::gemm::KernelTmaWarpSpecializedPingpong;

// ---------------------------------------------------------------------------
// Macro to stamp out extern "C" entry points
// ---------------------------------------------------------------------------

#define FP8_GEMM_VARIANT(ID, TILE_M, TILE_N, TILE_K, CL_M, CL_N, CL_K, SCHED) \
extern "C" int cutlass_fp8_gemm_v##ID(                                          \
    void* o, const void* a, const void* b,                                      \
    const void* a_scales, const void* b_scale,                                  \
    int M, int N, int K, void* ws, size_t ws_sz, cudaStream_t s) {              \
    return fp8_gemm_dispatch<                                                   \
        Shape<_##TILE_M, _##TILE_N, _##TILE_K>,                                \
        Shape<_##CL_M, _##CL_N, _##CL_K>, SCHED>(                             \
            o, a, b, a_scales, b_scale, M, N, K, ws, ws_sz, s);               \
}                                                                               \
extern "C" size_t cutlass_fp8_gemm_v##ID##_workspace_size(int M, int N, int K) {\
    return fp8_gemm_ws_dispatch<                                                \
        Shape<_##TILE_M, _##TILE_N, _##TILE_K>,                                \
        Shape<_##CL_M, _##CL_N, _##CL_K>, SCHED>(M, N, K);                   \
}

// ---------------------------------------------------------------------------
// 15 variants: WS/Coop/PP x tile/cluster combos for FP8
// ---------------------------------------------------------------------------
//
// ID  Tile MxNxK       Cluster  Schedule   Notes
//  0  64x128x128       1x1x1   WS         small M baseline
//  1  64x128x128       1x1x1   Coop       small M cooperative
//  2  64x256x128       1x1x1   WS         small M, wide N
//  3  64x256x128       1x2x1   WS         small M, wide N, N-clustered
//  4  128x128x128      1x1x1   WS         balanced baseline
//  5  128x128x128      1x1x1   Coop       balanced cooperative
//  6  128x256x128      1x1x1   WS         wide N baseline
//  7  128x256x128      1x1x1   Coop       wide N cooperative
//  8  128x256x128      1x2x1   WS         wide N, N-clustered
//  9  128x256x128      1x2x1   Coop       wide N, N-clustered, cooperative
// 10  128x256x128      1x2x1   PP         wide N, N-clustered, pingpong
// 11  256x128x128      2x1x1   WS         tall M, M-clustered
// 12  128x256x128      2x2x1   WS         4-SM, balanced cluster
// 13  128x128x256      1x1x1   WS         deep K
// 14  128x256x128      1x4x1   WS         4-SM along N

FP8_GEMM_VARIANT( 0,  64, 128, 128, 1,1,1, WS)
FP8_GEMM_VARIANT( 1, 128, 128, 256, 1,1,1, Coop)  // K=256 Coop (pair of v13 WS)
FP8_GEMM_VARIANT( 2,  64, 256, 128, 1,1,1, WS)
FP8_GEMM_VARIANT( 3,  64, 256, 128, 1,2,1, WS)
FP8_GEMM_VARIANT( 4, 128, 128, 128, 1,1,1, WS)
FP8_GEMM_VARIANT( 5, 128, 128, 128, 1,1,1, Coop)
FP8_GEMM_VARIANT( 6, 128, 256, 128, 1,1,1, WS)
FP8_GEMM_VARIANT( 7, 128, 256, 128, 1,1,1, Coop)
FP8_GEMM_VARIANT( 8, 128, 256, 128, 1,2,1, WS)
FP8_GEMM_VARIANT( 9, 128, 256, 128, 1,2,1, Coop)
FP8_GEMM_VARIANT(10, 128, 256, 128, 1,2,1, PP)
FP8_GEMM_VARIANT(11, 256, 128, 128, 2,1,1, WS)
FP8_GEMM_VARIANT(12, 128, 256, 128, 2,2,1, WS)
FP8_GEMM_VARIANT(13, 128, 128, 256, 1,1,1, WS)
FP8_GEMM_VARIANT(14, 128, 256, 128, 1,4,1, WS)
