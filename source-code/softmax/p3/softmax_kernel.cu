#include <cuda_runtime.h>
#include <math.h>
#include <float.h>
#include <cstdint>
/*
 * __global__ = This is a CUDA keyword that tells the compiler:
 *    "This function runs on the GPU and can be called from CPU!"
 *
 * Think of this as a tiny worker that gets copied thousands of times across
 * the GPU. Each copy (thread) handles ONE element of the arrays.
 *
 * Params:
 * - x, y: Input arrays (read-only, hense the const keyword)
 * - out: Output array (Obvious!)
 * - n: Size of our arrays, number of elements, used for masking
 */

__global__ void softmax_kernel_naive(const float *x, float *out, int input_row_stride, int output_row_stride,
                                     int n_rows, int n_cols)
{
    int row = blockIdx.x;
    if (row >= n_rows)
        return;

    const float *row_in = x + row * input_row_stride;
    float *row_out = out + row * output_row_stride;

    // Step 1: find max value for numerical stability
    float max_val = -FLT_MAX;
    for (int j = 0; j < n_cols; ++j)
        if (row_in[j] > max_val)
            max_val = row_in[j];

    // Step 2: compute sum of exp(x - max)
    float sum_exp = 0.0f;
    for (int j = 0; j < n_cols; ++j)
        sum_exp += expf(row_in[j] - max_val);

    // Step 3: normalize
    for (int j = 0; j < n_cols; ++j)
        row_out[j] = expf(row_in[j] - max_val) / sum_exp;
}

__global__ void softmax_kernel_step1(const float *x, float *out, int input_row_stride,
                                     int output_row_stride, int n_rows, int n_cols)
{
#define BLOCK_SIZE 256
    int row = blockIdx.x;
    if (row >= n_rows)
        return;

    const float *row_in = x + row * input_row_stride;
    float *row_out = out + row * output_row_stride;

    // step 1: Each thread finds it's local max
    float local_max = -FLT_MAX;
    for (int j = threadIdx.x; j < n_cols; j += blockDim.x)
        local_max = fmaxf(local_max, row_in[j]);

    // only one thread writes final max
    __shared__ float max_val;
    __shared__ float shared_max[BLOCK_SIZE];
    shared_max[threadIdx.x] = local_max;
    __syncthreads(); // make sure all writes are done

    if (threadIdx.x == 0)
    {
        float global_max = -FLT_MAX;
        for (int t = 0; t < blockDim.x; ++t)
            global_max = fmaxf(global_max, shared_max[t]);
        max_val = global_max;
    }
    __syncthreads();

    float local_sum = 0.f;
    for (int j = threadIdx.x; j < n_cols; j += blockDim.x)
        local_sum += expf(row_in[j] - max_val);

    __shared__ float sum_exp;
    __shared__ float shared_sum_exp[BLOCK_SIZE];
    shared_sum_exp[threadIdx.x] = local_sum;
    __syncthreads();
    if (threadIdx.x == 0)
    {
        float total = 0.f;
        for (int t = 0; t < blockDim.x; ++t)
            total += shared_sum_exp[t];
        sum_exp = total;
    }

    __syncthreads();

    for (int j = threadIdx.x; j < n_cols; j += blockDim.x)
        row_out[j] = expf(row_in[j] - max_val) / sum_exp;
}

__global__ void softmax_kernel_step2(const float *x, float *out, int input_row_stride,
                                     int output_row_stride, int n_rows, int n_cols)
{
    int row = blockIdx.x;
    if (row >= n_rows)
        return;

    extern __shared__ float shared[];

    const float *row_in = x + row * input_row_stride;
    float *row_out = out + row * output_row_stride;

    // step 1: Each thread finds it's local max
    float local_max = -FLT_MAX;
    for (int j = threadIdx.x; j < n_cols; j += blockDim.x)
        local_max = fmaxf(local_max, row_in[j]);

    // only one thread writes final max
    shared[threadIdx.x] = local_max;
    __syncthreads(); // make sure all writes are done

    // --- Parallel reduction for max
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1)
    {
        if (threadIdx.x < offset)
            shared[threadIdx.x] = fmaxf(shared[threadIdx.x], shared[threadIdx.x + offset]);
        __syncthreads();
    }
    float max_val = shared[0];
    __syncthreads();

    float local_sum = 0.f;
    for (int j = threadIdx.x; j < n_cols; j += blockDim.x)
        local_sum += expf(row_in[j] - max_val);

    shared[threadIdx.x] = local_sum; // reuse same buffer
    __syncthreads();

    // --- Parallel reduction for sum
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1)
    {
        if (threadIdx.x < offset)
            shared[threadIdx.x] += shared[threadIdx.x + offset];
        __syncthreads();
    }
    float sum_exp = shared[0];
    __syncthreads();

    for (int j = threadIdx.x; j < n_cols; j += blockDim.x)
        row_out[j] = expf(row_in[j] - max_val) / sum_exp;
}

__global__ void softmax_kernel_step3_vec4(
    const float *__restrict__ x,
    float *__restrict__ out,
    int in_stride, int out_stride,
    int n_rows, int n_cols)
{
    int row = blockIdx.x;
    if (row >= n_rows)
        return;

    extern __shared__ float shared[]; // reuse the same buffer
    const float *__restrict__ row_in = x + row * in_stride;
    float *__restrict__ row_out = out + row * out_stride;

    // pass 1, local max, vectorized
    float local_max = -FLT_MAX;
    int n_vec = n_cols >> 2; // n_cols / 4
    for (int j4 = threadIdx.x; j4 < n_vec; j4 += blockDim.x)
    {
        float4 v = reinterpret_cast<const float4 *>(row_in)[j4];
        local_max = fmaxf(local_max, v.x);
        local_max = fmaxf(local_max, v.y);
        local_max = fmaxf(local_max, v.z);
        local_max = fmaxf(local_max, v.w);
    }
    // tail, if any, handle with first few threads
    int tail_start = n_vec << 2;
    for (int j = tail_start + threadIdx.x; j < n_cols; j += blockDim.x)
    {
        local_max = fmaxf(local_max, row_in[j]);
    }

    shared[threadIdx.x] = local_max;
    __syncthreads();

    // parallel reduction for max
    for (int off = blockDim.x >> 1; off > 0; off >>= 1)
    {
        if (threadIdx.x < off)
            shared[threadIdx.x] = fmaxf(shared[threadIdx.x], shared[threadIdx.x + off]);
        __syncthreads();
    }
    float max_val = shared[0];
    __syncthreads();

    // pass 2, write exp(x - max) to out, accumulate sum, vectorized
    float local_sum = 0.f;
    for (int j4 = threadIdx.x; j4 < n_vec; j4 += blockDim.x)
    {
        float4 v = reinterpret_cast<const float4 *>(row_in)[j4];
        v.x = expf(v.x - max_val);
        v.y = expf(v.y - max_val);
        v.z = expf(v.z - max_val);
        v.w = expf(v.w - max_val);
        local_sum += v.x + v.y + v.z + v.w;
        reinterpret_cast<float4 *>(row_out)[j4] = v; // store unnormalized exp
    }
    for (int j = tail_start + threadIdx.x; j < n_cols; j += blockDim.x)
    {
        float e = expf(row_in[j] - max_val);
        local_sum += e;
        row_out[j] = e;
    }

    shared[threadIdx.x] = local_sum;
    __syncthreads();

    // parallel reduction for sum
    for (int off = blockDim.x >> 1; off > 0; off >>= 1)
    {
        if (threadIdx.x < off)
            shared[threadIdx.x] += shared[threadIdx.x + off];
        __syncthreads();
    }
    float sum_exp = shared[0];
    __syncthreads();

    // pass 3, normalize, vectorized
    for (int j4 = threadIdx.x; j4 < n_vec; j4 += blockDim.x)
    {
        float4 v = reinterpret_cast<const float4 *>(row_out)[j4];
        v.x /= sum_exp;
        v.y /= sum_exp;
        v.z /= sum_exp;
        v.w /= sum_exp;
        reinterpret_cast<float4 *>(row_out)[j4] = v;
    }
    for (int j = tail_start + threadIdx.x; j < n_cols; j += blockDim.x)
    {
        row_out[j] /= sum_exp;
    }
}

// warp reduction helpers
__inline__ __device__ float warp_reduce_max(float val)
{
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

__inline__ __device__ float warp_reduce_sum(float val)
{
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void softmax_kernel_step4(
    const float *__restrict__ x,
    float *__restrict__ out,
    int in_stride, int out_stride,
    int n_rows, int n_cols)
{
    int row = blockIdx.x;
    if (row >= n_rows)
        return;

    const float *row_in = x + row * in_stride;
    float *row_out = out + row * out_stride;

    float local_max = -FLT_MAX;
    for (int j = threadIdx.x; j < n_cols; j += blockDim.x)
        local_max = fmaxf(local_max, row_in[j]);

    // warp-level reduce first
    float warp_max = warp_reduce_max(local_max);

    // share warp results through shared memory
    __shared__ float warp_buf[32]; // one per warp (max 1024/32 = 32 warps)
    int warp_id = threadIdx.x / 32;
    if (threadIdx.x % 32 == 0)
        warp_buf[warp_id] = warp_max;
    __syncthreads();

    // first warp reduces warp results
    float block_max = -FLT_MAX;
    if (warp_id == 0)
        block_max = (threadIdx.x < (blockDim.x / 32)) ? warp_buf[threadIdx.x] : -FLT_MAX;
    block_max = warp_reduce_max(block_max);
    block_max = __shfl_sync(0xffffffff, block_max, 0); // broadcast

    // now compute exp and sum
    float local_sum = 0.f;
    for (int j = threadIdx.x; j < n_cols; j += blockDim.x)
    {
        float e = expf(row_in[j] - block_max);
        local_sum += e;
        row_out[j] = e;
    }

    float warp_sum = warp_reduce_sum(local_sum);
    if (threadIdx.x % 32 == 0)
        warp_buf[warp_id] = warp_sum;
    __syncthreads();

    float block_sum = 0.f;
    if (warp_id == 0)
        block_sum = (threadIdx.x < (blockDim.x / 32)) ? warp_buf[threadIdx.x] : 0.f;
    block_sum = warp_reduce_sum(block_sum);
    block_sum = __shfl_sync(0xffffffff, block_sum, 0);

    // normalize
    for (int j = threadIdx.x; j < n_cols; j += blockDim.x)
        row_out[j] /= block_sum;
}
/*
 * Kernel launcher: The bridge between CPU and GPU!
 *
 * extern "C" = prevents C++ name mangling, making this function callable from C code
 *              or other languages like Python
 *
 * This function tells GPU to execute the code and returns (asynchronous execution)
 */
extern "C" void launch_softmax_kernel(const float *x, float *out,
                                      int input_row_stride, int output_row_stride,
                                      int n_rows, int n_cols, int blocks, int threads,
                                      cudaStream_t stream)
{
    /*
     * The kernel launch: Cuda's special syntax
     * <<<blocks, threads, 0, stream>>>
     * ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ This is called execution configuration
     *
     * - blocks: how many blocks to launch
     * - threads: how many threads per block
     * - 0: shared memory size
     * - stream: Cuda stream for async execution (like a "lane" on the GPU)
     *
     * So total threads launched would be blocks*threads
     * Choose threads to be multiple of 32 (warp size)
     * Common choices are: 128, 256, 512, 1024
     */

    // softmax_kernel<<<blocks, threads, 0, stream>>>(x, out, input_row_stride, output_row_stride, n_rows, n_cols);
    // softmax_kernel_step1<<<blocks, threads, 0, stream>>>(x, out, input_row_stride, output_row_stride, n_rows, n_cols);
    // we reuse the shared memory for max array and sum_exp.
    // size_t shared_mem = threads * sizeof(float);
    // softmax_kernel_step2<<<blocks, threads, shared_mem, stream>>>(x, out, input_row_stride, output_row_stride, n_rows, n_cols);

    // step 3
    // size_t shmem = threads * sizeof(float);

    // uintptr_t x_addr = reinterpret_cast<uintptr_t>(x);
    // uintptr_t out_addr = reinterpret_cast<uintptr_t>(out);

    // bool base_aligned = ((x_addr % 16) == 0) && ((out_addr % 16) == 0);
    // bool stride_aligned = ((input_row_stride % 4) == 0) && ((output_row_stride % 4) == 0);
    // bool width_aligned = ((n_cols % 4) == 0);
    // bool use_vec4 = base_aligned && stride_aligned && width_aligned;

    // if (use_vec4)
    // {
    //     softmax_kernel_step3_vec4<<<blocks, threads, shmem, stream>>>(
    //         x, out, input_row_stride, output_row_stride, n_rows, n_cols);
    // }
    // else
    // {
    //     softmax_kernel_step2<<<blocks, threads, shmem, stream>>>(
    //         x, out, input_row_stride, output_row_stride, n_rows, n_cols);
    // }

    // step 4
    size_t shmem = 32 * sizeof(float); // small buffer (one per warp)
    softmax_kernel_step4<<<blocks, threads, shmem, stream>>>(
        x, out, input_row_stride, output_row_stride, n_rows, n_cols);
}
