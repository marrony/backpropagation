#include <stdio.h>
#include <stdlib.h>

#define BYTEBUFFER_IMPLEMENTATION
#define ALLOCATOR_IMPLEMENATION

#include "array.h"
#include "bytebuffer.h"
#include "allocator.h"
#include "tokenizer.h"

#include "generated/vocab.h"

Malloc_Allocator mallocator = MALLOC_CREATE();

DEFINE_ARRAY_ALIAS(Byte, char);

Byte_Array read_input(Allocator* alloc, FILE* input) {
  Byte_Array buff =  ARRAY_CREATE(alloc);

  array_ensure(&buff, 10*1024);

  while (true) {
    int ch = fgetc(input);
    if (ch == EOF || ch == '\n') return buff;
    array_append(&buff, (char)ch);
  }

  return buff;
}

int main(void) {
  TokenID_Array tokens = ARRAY_CREATE(&mallocator.alloc);
  Byte_Array prompt =  read_input(&mallocator.alloc, stdin);

  tokenize(
      &tokens,
      prompt.elems,
      prompt.count,
      vocabulary,
      vocabulary_by_size,
      MAX_VOCAB
  );

  for (size_t i = 0; i < tokens.count; i++) {
    printf("%d", tokens.elems[i]);

    if (i < tokens.count-1)
      printf(" ");
  }

  printf("\n");

  return 0;
}

