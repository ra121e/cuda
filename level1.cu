#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iostream>

__global__ void vectorAdd(
    const float *A,
    const float *B,
    float       *C,
    int         N
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        C[i] = A[i] + B[i];

        if (blockIdx.x < 2 && threadIdx.x < 4) {
            printf(
                "GPU: block=%d thread=%d global_i=%d "
                "A=%f B=%f C=%f\n",
                blockIdx.x,
                threadIdx.x,
                i,
                A[i],
                B[i],
                C[i]
            );
        }
    }
}

int main()
{
    const int       N = 10000;
    const size_t    bytes = N * sizeof(float);

    float   *h_A = new float[N];
    float   *h_B = new float[N];
    float   *h_C = new float[N];

    for (int i = 0; i < N; i++) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i * 2);
    }

    float   *d_A = nullptr;
    float   *d_B = nullptr;
    float   *d_C = nullptr;

    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    const int   threadsPerBlock = 256; 
    const int   blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    bool    success = true;

    for (int i = 0; i < N; ++i) {
        float   expected = h_A[i] + h_B[i];

        if (std::fabs(h_C[i] - expected) > 1e-5f) {
            std::cout << "Error at " << i << std::endl;
            success = false;
            break;
        }
    }

    if (success) {
        std::cout << "Vector addtion successfull!" << std::endl;
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}