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
  (void)arena;

  destoy_opencl();

  return 0;
}

