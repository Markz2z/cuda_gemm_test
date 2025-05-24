#pragma once

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

template <int Stride>
__global__ void sgemm_global_mem_coalesce(
      int M, int N, int K, float alpha, const float* A,
      const float* B, float beta, float* C) {
  int x = Stride * blockIdx.x + threadIdx.x / Stride;
  int y = Stride * blockIdx.y + threadIdx.x % Stride;
  float val = 0.f;
  if (x < M && y < N) {
    for (int i = 0; i < K; ++i) {
      val += A[x*K + i] * B[i*N + y];
    }
    C[x * N + y] = alpha * val + beta * C[x * N + y];
  }
}

