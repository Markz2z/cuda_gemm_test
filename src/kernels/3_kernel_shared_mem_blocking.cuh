#pragma once

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

template <int Stride>
__global__ void sgemm_shared_mem_block(
      int M, int N, int K, float alpha, const float* A,
      const float* B, float beta, float* C) {
  int x = Stride * blockIdx.x + threadIdx.x / Stride;
  int y = Stride * blockIdx.y + threadIdx.x % Stride;
  __shared__ float A_buf[Stride*Stride];
  __shared__ float B_buf[Stride*Stride];
  if (x < M && y < N) {
    int loop = K / Stride;
    int chunk_A_x = threadIdx.x % Stride;
    int chunk_A_y = threadIdx.x / Stride;
    float val = 0.f;
    for (int l = 0; l < loop; ++l) {
      // 1. gmem -> smem
      //   A [l*Stride+chunk_A_x, x]
      //   B [y, l*Stride+chunk_A_y]
      A_buf[chunk_A_x + Stride*chunk_A_y] = A[l * Stride + chunk_A_x + x * K];
      B_buf[chunk_A_x + Stride*chunk_A_y] = B[y + (l * Stride + chunk_A_y) * N];
      __syncthreads();

      // 2. compute
      for (int i = 0; i < Stride; ++i) {
        val += A_buf[chunk_A_y*Stride+i] * B_buf[chunk_A_x+i*Stride];
      }
      __syncthreads();
    }

    // 3. write back
    C[x * N + y] = alpha * val + beta * C[x * N + y];
  }
}

