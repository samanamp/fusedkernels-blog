#include <ATen/ATen.h>          // ATen: PyTorch Tensor library (like numpy)
#include <c10/cuda/CUDAGuard.h> // Ensures we're on the right GPU device
#include <c10/cuda/CUDAStream.h>// manages Cuda execution streams
#include <cuda_runtime.h>       // Core Cuda functionality
#include <torch/extension.h>    // Magic glue between C++ and Python
#include <algorithm> 
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

  // const int threads = 64; // Threads per block (multiple of 32: warp friendly!)

  // const int blocks = (int)((n_elements + threads - 1) / threads);

    // int threads = 64;
    // int blocks = (n_elements/4 + threads - 1) / threads;

    int threads = 128;
    // int blocks = std::min((n_elements / 4 + threads - 1) / threads, 65535);
    int64_t block_count = std::min<int64_t>(
    (n_elements / 4 + threads - 1) / threads, 65535
);
int blocks = static_cast<int>(block_count);


  // data_ptr<T>() extracts the actual memory address
  // Cuda kernels need raw memory pointer
  const float *x_ptr = x.data_ptr<float>();
  const float *y_ptr = y.data_ptr<float>();
  float *out_ptr = out.data_ptr<float>();

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  launch_add_kernel(x_ptr, y_ptr, out_ptr, (int)n_elements, blocks, threads, stream);
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
