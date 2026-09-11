#include <stdio.h>
#include <stdlib.h>

#define BYTEBUFFER_IMPLEMENTATION
#define ALLOCATOR_IMPLEMENATION

#include "array.h"
#include "bytebuffer.h"
#include "allocator.h"
#include "tokenizer.h"

#undef STR
#undef STR0
#define STR0(x) #x
#define STR(x) STR0(x)

#ifdef VOCAB_HEADER
#include STR(VOCAB_HEADER)
#else
#error VOCAB_HEADER not defined
#endif

Malloc_Allocator mallocator = MALLOC_CREATE();

DEFINE_ARRAY_ALIAS(Byte, char);

Byte_Array read_input(Allocator* alloc, FILE* input) {
  Byte_Array buff =  ARRAY_CREATE(alloc);

  array_ensure(&buff, 1024*1024);

  while (true) {
    int ch = fgetc(input);
    if (ch == EOF) return buff;
    array_append(&buff, (char)ch);
  }

  return buff;
}

int main(void) {
  TokenID_Array tokens = ARRAY_CREATE(&mallocator.alloc);
  TokenID_Array buff = ARRAY_CREATE(&mallocator.alloc);
  Byte_Buffer_Array words = ARRAY_CREATE(&mallocator.alloc);
  int32_t* merge_to = ALLOC(&mallocator.alloc, sizeof(int32_t)*MAX_VOCAB*MAX_VOCAB).ptr;

  for (size_t i = 0; i < MAX_VOCAB*MAX_VOCAB; i++)
    merge_to[i] = -1;

  for (size_t c = 257; c < MAX_VOCAB; c++) {
    assert(vocabulary[c].a != -1 && vocabulary[c].b != -1);
    merge_to[vocabulary[c].a*MAX_VOCAB + vocabulary[c].b] = c;
  }

  Byte_Array prompt =  read_input(&mallocator.alloc, stdin);

  if (prompt.count > 0 && prompt.elems[prompt.count-1] != '\0')
    array_append(&prompt, '\0');

  for (size_t i = 0, start = 0; i < prompt.count; i++) {
    if (prompt.elems[i] != '\0')
      continue;

    tokens.count = 0;
    words.count = 0;
    buff.count = 0;

    tokenize(
        &tokens,
        &words,
        &buff,
        byte_buffer_from_parts(prompt.elems + start, i - start),
        merge_to,
        MAX_VOCAB
    );

    start = i + 1;

    for (size_t i = 0; i < tokens.count; i++) {
      printf("%d", tokens.elems[i]);

      if (i < tokens.count-1)
        printf(" ");
    }

    printf("\n");
  }

  return 0;
}

