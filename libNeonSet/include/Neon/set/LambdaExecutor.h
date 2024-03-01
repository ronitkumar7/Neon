#pragma once
#include <functional>
#include "Neon/set/ExecutionThreadSpan.h"
#include <cub/cub.cuh>   // or equivalently <cub/device/device_radix_sort.cuh>

namespace Neon::set::details {

namespace denseSpan {

#ifdef NEON_COMPILER_CUDA
template <typename DataSetContainer,
          typename UserLambda>
NEON_CUDA_KERNEL auto launchLambdaOnSpanCUDA(typename DataSetContainer::Span span,
                                             UserLambda                      userLambdaTa)
    -> void
{
    typename DataSetContainer::Idx e;
    if constexpr (DataSetContainer::executionThreadSpan == ExecutionThreadSpan::d1) {
        if (span.setAndValidate(e,
                                threadIdx.x + blockIdx.x * blockDim.x)) {
            userLambdaTa(e);
        }
    }
    if constexpr (DataSetContainer::executionThreadSpan == ExecutionThreadSpan::d2) {
        if (span.setAndValidate(e,
                                threadIdx.x + blockIdx.x * blockDim.x,
                                threadIdx.y + blockIdx.y * blockDim.y)) {
            userLambdaTa(e);
        }
    }
    if constexpr (DataSetContainer::executionThreadSpan == ExecutionThreadSpan::d3) {
        if (span.setAndValidate(e,
                                threadIdx.x + blockIdx.x * blockDim.x,
                                threadIdx.y + blockIdx.y * blockDim.y,
                                threadIdx.z + blockIdx.z * blockDim.z)) {
            userLambdaTa(e);
        }
    }
}
#endif



#ifdef NEON_COMPILER_CUDA
template <typename ReduceOp, typename T, typename DataSetContainer, typename UserLambda, uint_32t blockDimX, uint_32t blockDimY, uint_32t blockDimZ>
NEON_CUDA_KERNEL auto launchLambdaOnSpanCUDAReduction(typename DataSetContainer::Span span,
                                             UserLambda                      userLambdaTa,
      ReduceOp reduce_op,
                                                      T*                              mem)
    -> void
{
    typename DataSetContainer::Idx e;
    T                              partial;
    if constexpr (DataSetContainer::executionThreadSpan == ExecutionThreadSpan::d1) {
        if (span.setAndValidate(e,
                                threadIdx.x + blockIdx.x * blockDim.x)) {
            partial = userLambdaTa(e);
        }
    }
    if constexpr (DataSetContainer::executionThreadSpan == ExecutionThreadSpan::d2) {
        if (span.setAndValidate(e,
                                threadIdx.x + blockIdx.x * blockDim.x,
                                threadIdx.y + blockIdx.y * blockDim.y)) {
            partial = userLambdaTa(e);
        }
    }
    if constexpr (DataSetContainer::executionThreadSpan == ExecutionThreadSpan::d3) {
        if (span.setAndValidate(e,
                                threadIdx.x + blockIdx.x * blockDim.x,
                                threadIdx.y + blockIdx.y * blockDim.y,
                                threadIdx.z + blockIdx.z * blockDim.z)) {
            partial = userLambdaTa(e);
 
 
            typedef cub::BlockReduce<T, blockDimX,
                                     cub::BLOCK_REDUCE_WARP_REDUCTIONS,
                                     blockDimY,
                                     blockDimZ>
                                                         BlockReduce;
            __shared__ typename BlockReduce::TempStorage temp_storage;
            T                                            block_reduce = BlockReduce(temp_storage).Reduce(partial, reduce_op);

            if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
                mem[blockIdx.x +
                       blockIdx.y * gridDim.x +
                       blockIdx.z * gridDim.x * gridDim.y] = block_reduce;
            }
            


        }
    }
}
#endif


template <typename IndexType,
          typename DataSetContainer,
          typename UserLambda_ta>
void launchLambdaOnSpanOMP(Neon::Integer_3d<IndexType> const&     gridDim,
                           typename DataSetContainer::Span const& span,
                           UserLambda_ta                          userLambdaTa)
{
    if constexpr (DataSetContainer::executionThreadSpan == ExecutionThreadSpan::d1) {
#ifdef NEON_OS_WINDOWS
//#pragma omp parallel for default(shared)
#else
 #pragma omp parallel for simd default(shared)
#endif
        for (IndexType x = 0; x < gridDim.x; x++) {
            typename DataSetContainer::Idx e;
            if (span.setAndValidate(e, x)) {
                userLambdaTa(e);
            }
        }
    }


    if constexpr (DataSetContainer::executionThreadSpan == ExecutionThreadSpan::d2) {
#ifdef NEON_OS_WINDOWS
#pragma omp parallel for default(shared)
#else
// #pragma omp parallel for simd collapse(2) default(shared)
#endif
        for (IndexType y = 0; y < gridDim.y; y++) {
            for (IndexType x = 0; x < gridDim.x; x++) {
                typename DataSetContainer::Idx e;
                if (span.setAndValidate(e, x, y)) {
                    userLambdaTa(e);
                }
            }
        }
    }

    if constexpr (DataSetContainer::executionThreadSpan == ExecutionThreadSpan::d3) {
#ifdef NEON_OS_WINDOWS
#pragma omp parallel for default(shared)
#else
// #pragma omp parallel for simd collapse(1) default(shared) schedule(guided)
#endif
        for (IndexType z = 0; z < gridDim.z; z++) {
            for (IndexType y = 0; y < gridDim.y; y++) {
                for (IndexType x = 0; x < gridDim.x; x++) {
                    typename DataSetContainer::Idx e;
                    if (span.setAndValidate(e, x, y, z)) {
                        userLambdaTa(e);
                    }
                }
            }
        }
    }
}
}  // namespace denseSpan

namespace blockSpan {

#ifdef NEON_COMPILER_CUDA
template <typename DataSetContainer,
          typename UserLambda>
NEON_CUDA_KERNEL auto launchLambdaOnSpanCUDA(typename DataSetContainer::Span span,
                                             UserLambda                      userLambdaTa)
    -> void
{
    typename DataSetContainer::Idx e;
    if constexpr (DataSetContainer::executionThreadSpan == ExecutionThreadSpan::d1b3) {
        if (span.setAndValidateGPUDevice(e)) {
            userLambdaTa(e);
        }
    }
}
#endif


template <typename IndexType, typename DataSetContainer, typename UserLambda_ta>
void launchLambdaOnSpanOMP(const Neon::Integer_3d<IndexType>& blockSize,
                           const Neon::Integer_3d<IndexType>& blockGridSize,
                           typename DataSetContainer::Span    span,
                           UserLambda_ta                      userLambdaTa)
{

    if constexpr (DataSetContainer::executionThreadSpan == ExecutionThreadSpan::d1b3) {
        // #pragma omp parallel for schedule(guided)
        for (IndexType bIdx = 0; bIdx < blockGridSize.x; bIdx++) {
            for (IndexType z = 0; z < blockSize.z; z++) {
                for (IndexType y = 0; y < blockSize.y; y++) {
#ifndef NEON_OS_WINDOWS
// #pragma omp simd
#endif
                    for (IndexType x = 0; x < blockSize.x; x++) {
                        typename DataSetContainer::Idx e;
                        if (span.setAndValidateCPUDevice(e, bIdx, x, y, z)) {
                            userLambdaTa(e);
                        }
                    }
                }
            }
        }
    }
}
}  // namespace blockSpan

}  // namespace Neon::set::details