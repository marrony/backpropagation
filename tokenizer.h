#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BYTEBUFFER_IMPLEMENTATION
#include "bytebuffer.h"

#define ALLOCATOR_IMPLEMENATION
#include "allocator.h"

#define HASHMAP_IMPLMENTATION
#include "hashmap.h"

#if (SIZE_MAX == 0xffffffffL)
  #define __builtin_rotateleft __builtin_rotateleft32
#elif (SIZE_MAX == 0xffffffffffffffffL)
  #define __builtin_rotateleft __builtin_rotateleft64
#else
  #error Unknown size of size_t
#endif

size_t hash_blob(Byte_Buffer key) {
  size_t hash = 5381;
  for (size_t i = 0; i < key.len; i++) {
    size_t h;
    __builtin_mul_overflow(31, hash, &h);

    hash = h + AT(key, i);
  }
  return hash;
}

size_t hash_blob2(Byte_Buffer key) {
  size_t hash = 5381;
  for (size_t i = 0; i < key.len; i++) {
    size_t h;
    __builtin_add_overflow(__builtin_rotateleft(hash, 5), hash, &h);

    hash = h + AT(key, i);
  }
  return hash;
}

bool equal_blob(Byte_Buffer a, Byte_Buffer b) {
  if (a.len != b.len) return false;
  return byte_buffer_cmp(a, b) == 0;
}

void free_key(Allocator* alloc, Byte_Buffer key) {
  FREE(alloc, key);
}

void free_value(Allocator* alloc, Byte_Buffer value) {
  FREE(alloc, value);
}

int32_t byte_buffer_get_int(Byte_Buffer buf) {
  assert(buf.len == sizeof(int32_t));
  int32_t* ptr = (int32_t*)buf.ptr;
  return *ptr;
}

Byte_Buffer byte_buffer_set_int(Byte_Buffer buf, int32_t value) {
  assert(buf.len == sizeof(int32_t));
  int32_t* ptr = (int32_t*)buf.ptr;
  *ptr = value;
  return buf;
}

#define MAX_VOCAB 512
#define TOK_SIZE 31
#define MAX_TOKEN (TOK_SIZE+1)

typedef struct {
  int32_t id;
  char token[MAX_TOKEN];
} Token;

typedef struct {
  int32_t id;
  size_t size;
} Token_Sorted;

typedef struct {
  int32_t token0;
  int32_t token1;
} Token_Pair;

Byte_Buffer alloc_pair(Allocator* alloc, Token_Pair pair) {
  Byte_Buffer buf = byte_buffer_filled(alloc, sizeof(Token_Pair), 0);
  memcpy(buf.ptr, &pair, sizeof(Token_Pair));
  return buf;
}

bool contains_space(Token* token) {
  for (int i = 0; i < MAX_TOKEN; i++) {
    if (isspace(token->token[i])) return true;
  }
  return false;
}

#define PAD_TOKEN 0
#define EOS_TOKEN 1
#define BOS_TOKEN 2

int32_t gen_vocabulary(
    const char* text[],
    size_t text_len,
    int32_t** tokens_ptr,
    Token* vocabulary
) {
  assert(MAX_VOCAB > 256);

  Malloc_Allocator mallocator = MALLOC_CREATE();
  Arena_Allocator arena = ARENA_CREATE(&mallocator.alloc, 1024*1024);
  Allocator* alloc = &arena.alloc;

  int32_t vocabulary_count = 0;

  Token token = {0};

  token.id = PAD_TOKEN;
  memset(token.token, 0, MAX_TOKEN );
  strcpy(token.token, "<PAD>");
  vocabulary[vocabulary_count] = token;
  vocabulary_count += 1;

  token.id = EOS_TOKEN;
  memset(token.token, 0, MAX_TOKEN );
  strcpy(token.token, "<EOS>");
  vocabulary[vocabulary_count] = token;
  vocabulary_count += 1;

  token.id = BOS_TOKEN;
  memset(token.token, 0, MAX_TOKEN );
  strcpy(token.token, "<BOS>");
  vocabulary[vocabulary_count] = token;
  vocabulary_count += 1;

  int32_t start = vocabulary_count - 1;

  for (int i = 1; i < 256; i++) {
    vocabulary[vocabulary_count].id = vocabulary_count;
    vocabulary[vocabulary_count].token[0] = i;
    vocabulary_count += 1;
  }

  size_t tokens_count = 0;
  for (size_t i = 0; i < text_len; i++) {
    tokens_count += strlen(text[i]);
    tokens_count += 1; //<EOS>
  }

  int32_t *tokens = malloc(tokens_count * sizeof(int32_t));
  *tokens_ptr = tokens;

  size_t count = 0;
  for (size_t i = 0; i < text_len; i++) {
    for (size_t j = 0; j < strlen(text[i]); j++) {
      char ch = text[i][j];
      tokens[count++] = vocabulary[ch + start].id;
    }
    tokens[count++] = vocabulary[EOS_TOKEN].id;
  }

  // merge spaces in the beginning of the words
  int32_t last_token = vocabulary_count;
  int32_t space_token = vocabulary[' ' + start].id;
  for (size_t i = 0; i < tokens_count - 1; i++) {
    if (tokens[i] == space_token) {
      char* token_str = vocabulary[tokens[i+1]].token;
      if (token_str[0] == ' ') continue;

      int32_t found = -1;
      for (int32_t j = last_token; j < vocabulary_count; j++) {
        if (vocabulary[j].token[0] == ' ' && vocabulary[j].token[1] == token_str[0]) {
          found = j;
          break;
        }
      }

      if (found == -1) {
        found = vocabulary_count;
        vocabulary[vocabulary_count].id = vocabulary_count;
        vocabulary[vocabulary_count].token[0] = ' ';
        vocabulary[vocabulary_count].token[1] = token_str[0];
        vocabulary_count += 1;
      }

      for (size_t j = i; j < tokens_count-1; j++)
        tokens[j] = tokens[j+1];
      tokens[i] = found;
      tokens_count -= 1;
    }
  }

  bool punctuation[256] = {0};
  punctuation['.'] = true;
  punctuation[','] = true;
  punctuation['?'] = true;
  punctuation['!'] = true;
  punctuation[':'] = true;
  punctuation[';'] = true;
  punctuation['-'] = true;
  punctuation['\''] = true;
  punctuation['\"'] = true;
  punctuation['('] = true;
  punctuation[')'] = true;
  punctuation['\r'] = true;
  punctuation['\n'] = true;

  int32_t old_vocabulary_count = 0;
  while (vocabulary_count < MAX_VOCAB) {
    if (old_vocabulary_count == vocabulary_count)
      break;

    old_vocabulary_count = vocabulary_count;

    size_t saved = SAVE(alloc);

    HashMap* hashmap = hashmap_create(alloc, 2*MAX_VOCAB, hash_blob, equal_blob, free_key, free_value);

    for (int i = 0; i < vocabulary_count; i++) {
      Token_Pair pair = {
        .token0 = i,
        .token1 = -1,
      };
      hashmap_put(
          hashmap,
          alloc_pair(alloc, pair),
          byte_buffer_filled(alloc, sizeof(int32_t), 0)
      );
    }

    for (size_t i = 0; i < tokens_count - 1; i++) {
      Token_Pair pair = {
        .token0 = tokens[i+0],
        .token1 = tokens[i+1],
      };

      if (pair.token0 > start && pair.token0-start < 256 && punctuation[pair.token0-start]) continue;
      if (pair.token1 > start && pair.token1-start < 256 && punctuation[pair.token1-start]) continue;

      Byte_Buffer key = byte_buffer_from_parts(&pair, sizeof(pair));

      Byte_Buffer value = NULL_BYTE_BUFFER;
      if (hashmap_get(hashmap, key, &value)) {
        byte_buffer_set_int(value, byte_buffer_get_int(value) + 1);
      } else {
        hashmap_put(
            hashmap,
            alloc_pair(alloc, pair),
            byte_buffer_filled(alloc, sizeof(int32_t), 0)
        );
      }
    }

    Token_Pair max_pair = {0};
    int max_count = 0;
    for (size_t i = 0; i < tokens_count - 1; i++) {
      Token_Pair pair = {
        .token0 = tokens[i+0],
        .token1 = tokens[i+1],
      };

      //avoid merge
      if (contains_space(vocabulary+pair.token1))
        continue;

      Byte_Buffer key = byte_buffer_from_parts(&pair, sizeof(pair));
      Byte_Buffer value = NULL_BYTE_BUFFER;
      int32_t count = 0;
      if (hashmap_get(hashmap, key, &value)) {
        count = byte_buffer_get_int(value);
      }

      if (count > 10 && count > max_count) {
        max_count = count;
        max_pair = pair;
      }
    }

    if (max_count > 0) {
      int32_t new_token = vocabulary_count;
      vocabulary_count += 1;

      vocabulary[new_token].id = new_token;
      strncpy(vocabulary[new_token].token, vocabulary[max_pair.token0].token, MAX_TOKEN);
      strncat(vocabulary[new_token].token, vocabulary[max_pair.token1].token, MAX_TOKEN);

      for (size_t i = tokens_count - 2; i > 0; i--) {
        int32_t token0 = tokens[i+0];
        int32_t token1 = tokens[i+1];
        if (token0 == max_pair.token0 && token1 == max_pair.token1) {
          for (size_t j = i; j < tokens_count-1; j++)
            tokens[j] = tokens[j+1];
          tokens[i] = new_token;
          tokens_count -= 1;
        }
      }
    }

    RESTORE(alloc, saved);
  }

  assert(vocabulary_count == MAX_VOCAB);

  return tokens_count;
}

