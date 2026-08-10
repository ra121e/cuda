#include <stdio.h>

// GPU側で実行される関数（カーネル関数）
__global__ void cuda_hello() {
    printf("Hello World from GPU thread %d!\n", threadIdx.x);
}

int main() {
    printf("Hello World from CPU!\n");

    cuda_hello<<<1, 4>>>();

    // 終了を待つ
    cudaDeviceSynchronize();

    // WSL2対策：GPUのバッファを強制的に画面へ吐き出す
    cudaDeviceReset();

    return 0;
}