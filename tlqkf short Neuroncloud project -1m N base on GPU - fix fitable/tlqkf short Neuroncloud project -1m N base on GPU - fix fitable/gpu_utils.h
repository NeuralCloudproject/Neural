#ifndef GPU_UTILS_H
#define GPU_UTILS_H

#include <torch/torch.h>
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

void test_gpu_computation() {
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    torch::Tensor tensor_a = torch::ones({ 10000, 10000 }).to(torch::kCUDA);
    torch::Tensor tensor_b = torch::ones({ 10000, 10000 }).to(torch::kCUDA);

    auto start = std::chrono::high_resolution_clock::now();

    {
        at::cuda::CUDAStreamGuard guard1(stream1);
        tensor_a = tensor_a * 2;
    }
    {
        at::cuda::CUDAStreamGuard guard2(stream2);
        tensor_b = tensor_b + 3;
    }

    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "GPU Computation completed in " << elapsed.count() << " seconds.\n";
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
}

#endif // GPU_UTILS_H
