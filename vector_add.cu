#include <stdio.h>

// エラーが起きたら場所と理由を表示するマクロ
#define CHECK_CUDA(call) \
{ \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error at line %d: %s\n", __LINE__, cudaGetErrorString(err)); \
        return -1; \
    } \
}

__global__ void add(float *a, float *b, float *c) {
    *c = *a + *b;
}

int main() {
    float h_a = 1.5f, h_b = 2.5f, h_c = 0.0f;
    float *d_a, *d_b, *d_c;

    // 各ステップでエラーがないかチェック
    CHECK_CUDA(cudaMalloc((void**)&d_a, sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_b, sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_c, sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_a, &h_a, sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, &h_b, sizeof(float), cudaMemcpyHostToDevice));

    // 計算実行
    add<<<1, 1>>>(d_a, d_b, d_c);
    
    // 計算自体のエラーと、終了を待つチェック
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(&h_c, d_c, sizeof(float), cudaMemcpyDeviceToHost));

    printf("GPUでの計算結果: %.1f + %.1f = %.1f\n", h_a, h_b, h_c);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return 0;
}