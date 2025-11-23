#include <ATen/ATen.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/torch.h>

extern "C" void launch_matmul_kernel(float *out, const float *a, const float *b,
                                     int M, int N, int K, dim3 dimGrid,
                                     dim3 dimBlock, cudaStream_t stream);

at::Tensor matmul_cuda(at::Tensor a, at::Tensor b)
{
    int64_t M = a.size(0), K = a.size(1), N = b.size(1);
    // Create out tensor placeholder in memory of device
    auto out = torch::empty({M, N}, a.options());

    // Create pointers to the Address of the tensors on HBM
    const float *a_ptr = a.data_ptr<float>();
    const float *b_ptr = b.data_ptr<float>();
    float *out_ptr = out.data_ptr<float>();

    // prep stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    dim3 dimBlock(32, 32);
    dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x,
                 (M + dimBlock.y - 1) / dimBlock.y);
    launch_matmul_kernel(out_ptr, a_ptr, b_ptr, M, N, K, dimGrid, dimBlock, stream);

    return out;
}

extern "C" void launch_matmul_kernel_coalescing(float *out, const float4 *a,
                                                const float4 *b_T, int M, int N,
                                                int K, dim3 dimGrid, dim3 dimBlock,
                                                cudaStream_t stream);

at::Tensor matmul_cuda_coalescing(at::Tensor a, at::Tensor b)
{
    int64_t M = a.size(0), K = a.size(1), N = b.size(1);
    TORCH_CHECK(K % 4 == 0, "K must be multiple of 4 when using float4 packing");

    auto out = torch::empty({M, N}, a.options());
    TORCH_CHECK(reinterpret_cast<uintptr_t>(a.data_ptr<float>()) % 16 == 0,
                "Tensor A is not 16-byte aligned");

    const float4 *a_ptr = reinterpret_cast<const float4 *>(a.data_ptr<float>());
    auto bT = b.transpose(0, 1).contiguous();
    TORCH_CHECK(reinterpret_cast<uintptr_t>(bT.data_ptr<float>()) % 16 == 0,
                "Tensor B is not 16-byte aligned");
    const float4 *b_T_ptr = reinterpret_cast<const float4 *>(bT.data_ptr<float>());

    float *out_ptr = out.data_ptr<float>();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    dim3 dimBlock(32, 32);
    dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x,
                 (M + dimBlock.y - 1) / dimBlock.y);
    launch_matmul_kernel_coalescing(out_ptr, a_ptr, b_T_ptr, M, N, K, dimGrid, dimBlock, stream);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("matmul_cuda", &matmul_cuda, "MATMUL (CUDA)");
    m.def("matmul_cuda_coal", &matmul_cuda_coalescing, "MATMUL Coal (CUDA)");
}