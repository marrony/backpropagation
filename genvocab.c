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

int comp_token(const void* a, const void* b) {
  const Token* token_a = a;
  const Token* token_b = b;

  size_t len_a = strlen(token_a->token);
  size_t len_b = strlen(token_b->token);

  if (len_a > len_b) return -1;
  if (len_a < len_b) return +1;
  return 0;
}

int main(int argc, const char* argv[]) {
  size_t max_vocab = 0;

  for (int i = 1; i < argc; i++) {
    if (strncmp(argv[i], "--max-vocab=", 12) == 0) {
      max_vocab = strtoul(argv[i]+12, NULL, 10);
    }
  }

  if (max_vocab == 0) {
    fprintf(stderr, "--max-vocab is mandatory\n");
    return -1;
  }

  Byte_Array buf = read_input(&mallocator.alloc, stdin);
  Byte_Buffer_Array training = ARRAY_CREATE(&mallocator.alloc);

  for (size_t i = 0, start = 0; i < buf.count; i++) {
      if (buf.elems[i] != '\0')
        continue;
      Byte_Buffer b = byte_buffer_from_parts(buf.elems + start, i - start);
      array_append(&training, b);
      start = i + 1;
  }

  Token* vocabulary = ALLOC(&mallocator.alloc, sizeof(Token)*max_vocab).ptr;
  memset(vocabulary, 0, sizeof(Token)*max_vocab);

  int32_t* merge_to = ALLOC(&mallocator.alloc, sizeof(int32_t)*max_vocab*max_vocab).ptr;

  int32_t token_count = gen_vocabulary(
      &mallocator.alloc,
      training,
      vocabulary,
      max_vocab
  );

  build_merge_to(merge_to, vocabulary, max_vocab);

  Token* vocabulary_sorted = ALLOC(&mallocator.alloc, sizeof(Token)*max_vocab).ptr;
  memcpy(vocabulary_sorted, vocabulary, sizeof(Token)*max_vocab);

  qsort(vocabulary_sorted, token_count, sizeof(Token), comp_token);

  Token_Sorted* vocabulary_by_size =  ALLOC(&mallocator.alloc, sizeof(Token_Sorted)*max_vocab).ptr;
  for (int i = 0; i < token_count; i++) {
    vocabulary_by_size[i].id = vocabulary_sorted[i].id;
    vocabulary_by_size[i].size = strlen(vocabulary_sorted[i].token);
  }

  printf("#define MAX_VOCAB %d\n", token_count);
  printf("Token vocabulary[MAX_VOCAB] = {\n");
  for (int i = 0; i < token_count; i++) {
    printf("  { .id = %d, .a = %d, .b = %d, .token = {", vocabulary[i].id, vocabulary[i].a, vocabulary[i].b);
    for (int j = 0; j < MAX_TOKEN; j++) {
      if (vocabulary[i].token[j] == '\'')
        printf("'\\'', ");
      else if (vocabulary[i].token[j] == '\\')
        printf("'\\\\', ");
      else if (isprint(vocabulary[i].token[j]))
        printf("'%c', ", vocabulary[i].token[j]);
      else
        printf("0x%02x, ", 0xff & vocabulary[i].token[j]);
    }
    printf("} },\n");
  }
  printf("};\n");

  printf("Token_Sorted vocabulary_by_size[MAX_VOCAB] = {\n");
  for (int i = 0; i < token_count; i++) {
    printf("  { .id = %d, .size = %zu },\n",
        vocabulary_by_size[i].id, vocabulary_by_size[i].size);
  }
  printf("};\n");

  // printf("int32_t merge_to[MAX_VOCAB*MAX_VOCAB] = {\n");
  // printf("};\n");

  TokenID_Array tokens = ARRAY_CREATE(&mallocator.alloc);
  TokenID_Array buff = ARRAY_CREATE(&mallocator.alloc);
  Byte_Buffer_Array words = ARRAY_CREATE(&mallocator.alloc);

  size_t total_chars = 0;
  size_t total_tokens = 0;
  for (size_t i = 0; i < training.count; i++) {
    Byte_Buffer text = training.elems[i];

    tokens.count = 0;
    words.count = 0;
    buff.count = 0;

    tokenize(
        &tokens,
        &words,
        &buff,
        text,
        merge_to,
        max_vocab
    );

    total_chars += text.len;
    total_tokens += tokens.count;
  }

  printf("\n");
  printf("// vocab_size = %zu\n", max_vocab);
  printf("// training_size = %zu\n", training.count);
  printf("// total_chars = %zu\n", total_chars);
  printf("// total_tokens = %zu\n", total_tokens);
  printf("// chars/token = %.2f\n", (float)total_chars/(float)total_tokens);

  return 0;
}

