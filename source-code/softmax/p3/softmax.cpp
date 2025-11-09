#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

/*
 * This tells C++ compiler: "Hey, there's a function called launch_add_kernel
 * defined somewhere else (in our .cu file). Trust me, it exists!"
 */

extern "C" void launch_softmax_kernel(const float *x, float *out,
                                      int in_stride, int out_stride,
                                      int n_rows, int n_cols,
                                      int blocks, int threads,
                                      cudaStream_t stream);

at::Tensor softmax_cuda(at::Tensor x)
{

    // Same as triton, pre-allocate the memory
    // torch::zeros_like(x) is the alternative, but slower
    auto out = torch::empty_like(x);
    int64_t M = x.size(0), N = x.size(1);

    // For softmax_naive
    // int threads = 1;
    // int64_t block_count = M;
    // int blocks = static_cast<int>(block_count);

    // For softmax_kernel_step1
    int threads = 512; // or 256
    if (N > 14080)
        threads = 1024;
    int blocks = M;

    // data_ptr<T>() extracts the actual memory address
    // Cuda kernels need raw memory pointer
    const float *x_ptr = x.data_ptr<float>();
    float *out_ptr = out.data_ptr<float>();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    launch_softmax_kernel(x_ptr, out_ptr, x.stride(0), out.stride(0), M, N, blocks, threads, stream);
    C10_CUDA_CHECK(cudaGetLastError());
    return out;
}

/* PYBIND11_MODULE: This makes the function callable in Python
 * the 11 in the name means C++11 at minimum
 * Funny enough, there is no other numbers.
 *
 * TORCH_EXTENSION_NAME: Automatically set by Pytorch build system
 */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("softmax_cuda", &softmax_cuda, "Softmax (CUDA)");
}
