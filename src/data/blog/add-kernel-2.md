---
title: "Vector Addition::P2::Cuda Kernel"
pubDatetime: 2025-10-05T12:00:00Z
modDatetime: 2025-10-05T12:00:00Z
author: "Saman"
published: true
tags:
  - Cuda kernel
  - tech
description: "Investing vector addition kernel in Cuda"
---

Now that we have implemented the Triton kernel (better to say we copied it) and investigated it, it's time for a Cuda version. Juicy stuff.

Do you remember we had add_kernel with @triton.jit decorator? Here that turns into a file by itself: vector_add_kernel.cu.

```c++
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
  add_kernel<<<blocks, threads, 0, stream>>>(x, y, out, n);

  // This function returns immediately, if you'd like to wait, use:
  // cudaStreamSynchronize(stream) or cudaDeviceSynchronize()
}
```

I really tried to be as good as possible in explaining concepts through comments; But still, 1 concept remaining: Warp.

Warp is a group of 32 consecutive threads that execute instructions together in lockstep. Think marching band of musicians
that must play the same note (same code) at the same time.

But why Warps? Because GPUs use SIMD (Single instruction, Multiple Data) execution. Each warp is the minimum unit of execution.

Now let's go to the next file, the caller:

```cpp
#include <ATen/ATen.h>          // ATen: PyTorch Tensor library (like numpy)
#include <c10/cuda/CUDAGuard.h> // Ensures we're on the right GPU device
#include <c10/cuda/CUDAStream.h>// manages Cuda execution streams
#include <cuda_runtime.h>       // Core Cuda functionality
#include <torch/extension.h>    // Magic glue between C++ and Python

/*
 * This tells C++ compiler: "Hey, there's a function called launch_add_kernel
 * defined somewhere else (in our .cu file). Trust me, it exists!"
 */

extern "C" void launch_add_kernel(const float *x, const float *y, float *out,
                                  int n, int blocks, int threads,
                                  cudaStream_t stream);

at::Tensor add_cuda(at::Tensor x, at::Tensor y) {
  TORCH_CHECK(x.sizes() == y.sizes(), "x and y must have the same shape")
  
  // Same as triton, pre-allocate the memory
  // torch::zeros_like(x) is the alternative, but slower
  auto out = torch::empty_like(x);

  // did you notice same interface as python? Check the previous post.
  int64_t n_elements = x.numel();

  const int threads = 1024; // Threads per block (multiple of 32: warp friendly!)

  const int blocks = (int)((n_elements + threads - 1) / threads);

  // data_ptr<T>() extracts the actual memory address
  // Cuda kernels need raw memory pointer
  const float *x_ptr = x.data_ptr<float>();
  const float *y_ptr = y.data_ptr<float>();
  float *out_ptr = out.data_ptr<float>();

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  launch_add_kernel(x_ptr, y_ptr, out_ptr, (int)n_elements,
                    blocks, threads, stream);
  return out;
}

/* PYBIND11_MODULE: This makes the function callable in Python
 * the 11 in the name means C++11 at minimum
 * Funny enough, there is no other numbers.
 *
 * TORCH_EXTENSION_NAME: Automatically set by Pytorch build system
 */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
  m.def("add_cuda", &add_cuda, "Vector add (CUDA)");
}
```

You have made it this far, just a bit remaining, let's execute it.

```python
import os
import torch
from torch.utils.cpp_extension import load

this_dir = os.path.dirname(__file__)

ext = load(
  name="vector_add_ext",
  sources=[
    os.path.join(this_dir, "vector_add.cpp"),
    os.path.join(this_dir, "vector_add_kernel.cu"),
  ],
  verbose=True,
)

def add(x: torch.Tensor, y: torch.Tensor):
  return ext.add_cuda(x, y)

device = torch.device("cuda:0")
torch.manual_seed(0)
size = 98_432
x = torch.rand(size, device=device, dtype=torch.float32)
y = torch.rand(size, device=device, dtype=torch.float32)

out_cuda = add(x, y)
torch.cuda.synchronize()
```

Aaah, I'm tired, going to call it a night!

Hope you enjoyed it. Next one, we will do hell lot of benchmarking! Let's see how these kernels compare.
