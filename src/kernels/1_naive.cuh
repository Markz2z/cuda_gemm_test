#pragma once

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>


__global__ void sgemm_naive(int M, int N, int K, float alpha, const float* A,
                            const float* B, float beta, float* C) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x < M && y < N) {
    float val = 0.0;
    for (int i = 0; i < K; ++i) {
      val += A[x * K + i] * B[i * N + y];
    }
    C[x * N + y] = alpha * val + beta * C[x * N + y];
  }
}

