---
title: "Vector Addition::P3::Optimizing"
pubDatetime: 2025-10-26T12:00:00Z
modDatetime: 2025-10-26T12:00:00Z
author: "Saman"
draft: false
published: true
tags:
  - Benchmarking
description: "Optimizing Cuda vector addition kernels to match Triton & Torch"
---
I'm sure from last post you remember our vector addition kernel in Cuda was behind Triton & Torch version.
Well that shows the beauty of how Triton compiler automatically optimizes the kernels.
Here is our kernel performance on RTX5060:
![GTX5060](../../assets/images/GTX5060.png)

First things first, let's play with threads number. After all, that's the only number we have.

```cpp
#include <ATen/ATen.h>          // ATen: PyTorch Tensor library (like numpy)
#include <c10/cuda/CUDAGuard.h> // Ensures we're on the right GPU device
#include <c10/cuda/CUDAStream.h>// manages Cuda execution streams
#include <cuda_runtime.h>       // Core Cuda functionality
#include <torch/extension.h>    // Magic glue between C++ and Python

extern "C" void launch_add_kernel(const float *x, const float *y, float *out,
                                  int n, int blocks, int threads,
                                  cudaStream_t stream);

at::Tensor add_cuda(at::Tensor x, at::Tensor y) {
  TORCH_CHECK(x.sizes() == y.sizes(), "x and y must have the same shape")

  auto out = torch::empty_like(x);
  int64_t n_elements = x.numel();

  const int threads = 64; // Threads per block (multiple of 32: warp friendly!)
  const int blocks = (int)((n_elements + threads - 1) / threads);

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

With this, we do observe some perf improvement. We're closer to Triton and Torch version.
![threads-optim](../../assets/images/art3-threads.png)