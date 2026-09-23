#include <math.h>
#define GPU_TS_MATRIX 4
#define GPU_BUILD_OPTS "-DTS=4 -cl-opt-disable"

#include "array.h"
#include "gpu.h"
#include "nn.h"
#include <OpenCL/cl.h>
#include <string.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

#define BYTEBUFFER_IMPLEMENTATION
#define ALLOCATOR_IMPLEMENATION
#include "bytebuffer.h"
#include "allocator.h"

void gemm_nn_debug_cpu(NMatrix mc, NMatrix ma, NMatrix mb) {
  for (size_t i = 0; i < mc.rows; i++) {
    for (size_t j = 0; j < mc.cols; j++) {
      float sum = 0.0f;
      for (size_t k = 0; k < ma.cols; k++)
        sum += MAT_AT(ma, i, k) * MAT_AT(mb, k, j);
      MAT_AT(mc, i, j) = sum;
    }
  }
}

float ulp(float x) {
  return powf(2.0f, floorf(log2f(fabsf(x)) - 23));
}

int64_t ulp_diff(float a, float b) {
    int32_t ia, ib;
    memcpy(&ia, &a, sizeof(ia));
    memcpy(&ib, &b, sizeof(ib));
    int64_t la = (ia < 0) ? (int64_t)INT32_MIN - ia : ia;   /* monotonic key */
    int64_t lb = (ib < 0) ? (int64_t)INT32_MIN - ib : ib;
    return la > lb ? la - lb : lb - la;                     /* 0 == bit-identical */
}

#define ROUND_UP(size, tile) (((size) + (tile) - 1) / (tile) * (tile))

Malloc_Allocator mallocator = MALLOC_CREATE();

int main(void) {
  if (!init_opencl()) {
    printf("Failed to init OpenCL\n");
    return 1;
  }

  cl_int err = CL_SUCCESS;
  cl_kernel gemm_nn_debug = clCreateKernel(program, "gemm_nn_debug", &err);
  assert(err == CL_SUCCESS);
  (void)gemm_nn_debug;

  size_t arena_size = 2L*1024*1024*1024;
  Arena_Allocator arena = ARENA_CREATE(&mallocator.alloc, arena_size);

  NMatrix out0 = mat_alloc2(&arena.alloc, 4, 4);
  NMatrix out1 = mat_alloc2(&arena.alloc, 4, 4);
  NMatrix a   = mat_alloc2(&arena.alloc, 4, 4);
  NMatrix b   = mat_alloc2(&arena.alloc, 4, 4);

  rng_state = 1234L << 32;
  for (size_t i = 0; i < 4; i++) {
    for (size_t j = 0; j < 4; j++) {
      MAT_AT(a, i, j) = rand_uniform();
      MAT_AT(b, i, j) = rand_uniform();
    }
  }

  cl_mem gpu = clCreateBuffer(context, CL_MEM_READ_WRITE, 3*sizeof(float)*4*4, NULL, NULL);

  size_t offset = 0;
  from_matrix2(gpu, &out0, &offset);
  from_matrix2(gpu, &a, &offset);
  from_matrix2(gpu, &b, &offset);
  assert(offset*sizeof(float) == 3*sizeof(float)*4*4);

  copy_cpu_to_gpu(a, NULL);
  copy_cpu_to_gpu(b, NULL);

  assert(call_matrix_mul_stride_kernel(out0, a, b) == CL_SUCCESS);

  // // outputs
  // err  = clSetKernelArg(gemm_nn_debug, 0, sizeof(cl_mem), &out0.buffer);
  // err |= clSetKernelArg(gemm_nn_debug, 1, sizeof(MatrixData), &out0.data);
  // assert(err == CL_SUCCESS);
  // // inputs
  // err |= clSetKernelArg(gemm_nn_debug, 2, sizeof(cl_mem), &a.buffer);
  // err |= clSetKernelArg(gemm_nn_debug, 3, sizeof(MatrixData), &a.data);
  // err |= clSetKernelArg(gemm_nn_debug, 4, sizeof(cl_mem), &b.buffer);
  // err |= clSetKernelArg(gemm_nn_debug, 5, sizeof(MatrixData), &b.data);
  // assert(err == CL_SUCCESS);
  //
  // size_t global_ws[2] = {out0.cols, out0.rows};
  // err = clEnqueueNDRangeKernel(
  //     commands,
  //     gemm_nn_debug,
  //     2,
  //     NULL,
  //     global_ws,
  //     NULL,
  //     0, NULL, NULL
  // );
  // assert(err == CL_SUCCESS);

  copy_gpu_to_cpu_sync(out0);

  for (size_t i = 0; i < 4; i++) {
    for (size_t j = 0; j < 4; j++) {
      printf("%28.25f ", MAT_AT(out0, i, j));
    }
    printf("\n");
  }
  printf("\n");

  // gemm_nn_debug_cpu(out1, a, b);
  gemm_nn(out1, a, b, NULL, false);

  for (size_t i = 0; i < 4; i++) {
    for (size_t j = 0; j < 4; j++) {
      printf("%28.25f ", MAT_AT(out1, i, j));
    }
    printf("\n");
  }
  printf("\n");

  for (size_t i = 0; i < 4; i++) {
    for (size_t j = 0; j < 4; j++) {
      printf("%lld ", ulp_diff(MAT_AT(out0, i, j), MAT_AT(out1, i, j)));
    }
    printf("\n");
  }
  printf("\n");

  // for (size_t i = 0; i < 4; i++) {
  //   for (size_t j = 0; j < 4; j++) {
  //     printf("%28.25f ", MAT_AT(out0, i, j) - MAT_AT(out1, i, j));
  //   }
  //   printf("\n");
  // }
  // printf("\n");

  destoy_opencl();

  return 0;
}

