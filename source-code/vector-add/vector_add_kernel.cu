#include <cuda_runtime.h>

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
__global__ void add_kernel(const float *x, const float *y, float *out, int n) {
  /* Cuda's magic formula, converting thread coordinates to array index
   *
   * Imagine your GPU as a 2D grid:
   * - blockIdx.x = Which "block" am I in? (like which neighborhood)
   * - blockDim.x = How many threads per block? (like house per neighborhood)
   * - threadIdx.x = Which thread I am within my block?
   *
   * Example: If blockDim.x = 256 and I'm thread 5 in block 2:
   * idx = 2 * 256 + 5 = 517
   * Means I'm responsible for processing element 517 of the array
   */
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  /* Bounds checking: the GPU's safety net
   * Here we make sure we never try to access a memory that doesn't belong to the arrays
   */
  if (idx < n){
    // Obvious?!
    out[idx] = x[idx] + y[idx];
  }
}

__global__ void add_kernel_vec4(const float *__restrict__ x,
                                const float *__restrict__ y,
                                float *__restrict__ out, int n) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx + 3 < n) {
        float4 a = reinterpret_cast<const float4*>(x)[idx / 4];
        float4 b = reinterpret_cast<const float4*>(y)[idx / 4];
        float4 c;
        c.x = a.x + b.x;
        c.y = a.y + b.y;
        c.z = a.z + b.z;
        c.w = a.w + b.w;
        reinterpret_cast<float4*>(out)[idx / 4] = c;
    }
}

__global__ void add_kernel_vec4_looped(const float *__restrict__ x,
                                       const float *__restrict__ y,
                                       float *__restrict__ out,
                                       int n) {
    // Thread’s global linear index
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // Total number of threads across the grid
    int stride = blockDim.x * gridDim.x;

    // Grid-stride loop: each iteration handles one float4 (4 floats)
    for (int i = tid * 4; i < n; i += stride * 4) {
        if (i + 3 < n) {
            // Vectorized 16-byte load
            float4 a = reinterpret_cast<const float4*>(x)[i / 4];
            float4 b = reinterpret_cast<const float4*>(y)[i / 4];

            // Compute
            float4 c;
            c.x = a.x + b.x;
            c.y = a.y + b.y;
            c.z = a.z + b.z;
            c.w = a.w + b.w;

            // Vectorized 16-byte store
            reinterpret_cast<float4*>(out)[i / 4] = c;
        }
    }
}

/*
 * Kernel launcher: The bridge between CPU and GPU!
 *
 * extern "C" = prevents C++ name mangling, making this function callable from C code
 *              or other languages like Python
 *
 * This function tells GPU to execute the code and returns (asynchronous execution)
 */
extern "C" void launch_add_kernel(const float *x, const float *y, float *out,
                                  int n, int blocks, int threads,
                                  cudaStream_t stream) {
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
  // add_kernel<<<blocks, threads, 0, stream>>>(x, y, out, n);
  // add_kernel_vec4<<<blocks, threads, 0, stream>>>(x, y, out, n);
 add_kernel_vec4_looped<<<blocks, threads, 0, stream>>>(x, y, out, n);

  // This function returns immediately, if you'd like to wait, use:
  // cudaStreamSynchronize(stream) or cudaDeviceSynchronize()
}
