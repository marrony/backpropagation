#include "gpu.h"
#include "nn.h"
#include "transformer.h"
#include <OpenCL/cl.h>
#include <string.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#define BYTEBUFFER_IMPLEMENTATION
#define ALLOCATOR_IMPLEMENATION

#include "bytebuffer.h"
#include "allocator.h"
#include "transformer.h"

#define ROUND_UP(size, tile) (((size) + (tile) - 1) / (tile) * (tile))

Malloc_Allocator mallocator = MALLOC_CREATE();

int main(void) {
  if (!init_opencl()) {
    printf("Failed to init OpenCL\n");
    return 1;
  }

  size_t arena_size = 2L*1024*1024*1024;
  Arena_Allocator arena = ARENA_CREATE(&mallocator.alloc, arena_size);

  NMatrix out = mat_alloc2(&arena.alloc, 4, 4);
  NMatrix a   = mat_alloc2(&arena.alloc, 4, 4);
  NMatrix b   = mat_alloc2(&arena.alloc, 4, 4);

  for (size_t i = 0; i < 4; i++) {
    for (size_t j = 0; j < 4; j++) {
      MAT_AT(a, i, j) = 1;
      MAT_AT(b, i, j) = 2;
    }
  }

  cl_mem gpu = clCreateBuffer(context, CL_MEM_READ_WRITE, 3*sizeof(float)*4*4, NULL, NULL);

  size_t offset = 0;
  from_matrix2(gpu, &out, &offset);
  from_matrix2(gpu, &a, &offset);
  from_matrix2(gpu, &b, &offset);
  assert(offset*sizeof(float) == 3*sizeof(float)*4*4);

  copy_cpu_to_gpu(a, NULL);
  copy_cpu_to_gpu(b, NULL);
  call_matrix_add_kernel(out, a, b);
  copy_gpu_to_cpu_sync(out);

  for (size_t i = 0; i < 4; i++) {
    for (size_t j = 0; j < 4; j++) {
      printf("%8.2f ", MAT_AT(out, i, j));
    }
    printf("\n");
  }

  destoy_opencl();

  return 0;
}

