#include <stdint.h>
#include <stdlib.h>
#define BYTEBUFFER_IMPLEMENTATION
#define ALLOCATOR_IMPLEMENATION
#define HASHMAP_IMPLMENTATION

#include "tokenizer.h"
#include "array.h"
#include "bytebuffer.h"
#include "allocator.h"

Malloc_Allocator mallocator = MALLOC_CREATE();

DEFINE_ARRAY_ALIAS(Byte, char);

Byte_Array read_input(Allocator* alloc, FILE* input) {
  Byte_Array buff =  ARRAY_CREATE(alloc);

  array_ensure(&buff, 10*1024);

  while (true) {
    int ch = fgetc(input);
    if (ch == EOF) return buff;
    array_append(&buff, (char)ch);
  }

  return buff;
}

int main(int argc, const char* argv[]) {
  size_t max_vocab = 0;
  size_t dev_chars = 0;
  const char* train_ids = NULL;
  const char* dev_ids = NULL;

  for (int i = 1; i < argc; i++) {
    if (strncmp(argv[i], "--max-vocab=", 12) == 0) {
      max_vocab = strtoul(argv[i]+12, NULL, 10);
    }

    if (strncmp(argv[i], "--dev-chars=", 12) == 0) {
      dev_chars = strtoul(argv[i]+12, NULL, 10);
    }

    if (strncmp(argv[i], "--train-ids=", 12) == 0) {
      train_ids = argv[i] + 12;
    }

    if (strncmp(argv[i], "--dev-ids=", 10) == 0) {
      dev_ids = argv[i] + 10;
    }
  }

  if (max_vocab == 0) {
    fprintf(stderr, "--max-vocab is mandatory\n");
    return -1;
  }

  if (dev_chars == 0) {
    fprintf(stderr, "--dev-chars is mandatory\n");
    return -1;
  }

  size_t* pairs = ALLOC(&mallocator.alloc, max_vocab*max_vocab*sizeof(size_t)).ptr;
  size_t* ctx = ALLOC(&mallocator.alloc, max_vocab*sizeof(size_t)).ptr;

  memset(pairs, 0, max_vocab*max_vocab*sizeof(size_t));
  memset(ctx, 0, max_vocab*sizeof(size_t));

  Byte_Array dev = read_input(&mallocator.alloc, fopen(dev_ids, "rb"));
  Byte_Array train = read_input(&mallocator.alloc, fopen(train_ids, "rb"));

  TokenID_Array train_ids_array = {
    .elems = (void*)train.elems,
    .count = train.count / sizeof(int32_t),
    .capacity = train.count / sizeof(int32_t),
    .allocator = NULL,
  };
  TokenID_Array dev_ids_array = {
    .elems = (void*)dev.elems,
    .count = dev.count / sizeof(int32_t),
    .capacity = dev.count / sizeof(int32_t),
    .allocator = NULL,
  };

  BPC b = bpc(
      train_ids_array,
      dev_ids_array,
      pairs,
      ctx,
      max_vocab,
      dev_chars
  );

  printf("V=%zu chars/tok=%.2lf bits/char=%.2lf\n",
      max_vocab, b.cpt, b.bpc);
}

