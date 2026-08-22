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

}


int main(void)
{
    return 0;
}