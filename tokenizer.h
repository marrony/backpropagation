#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <pcre2.h>
#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "array.h"
#include "bytebuffer.h"
#include "allocator.h"
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

DEFINE_ARRAY(Byte_Buffer);
DEFINE_ARRAY_ALIAS(TokenID, int32_t);
DEFINE_ARRAY_ALIAS(Dataset, TokenID_Array);

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
#define EOS_TOKEN 256

bool pre_split(const char* text, Byte_Buffer_Array* array) {
  pcre2_code *re;
  PCRE2_SPTR pattern = (PCRE2_SPTR)"(?i:\'s|\'t|\'re|\'ve|\'m|\'ll|\'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s";
  PCRE2_SPTR subject = (PCRE2_SPTR)text;

  int errornumber;
  PCRE2_SIZE erroroffset;

  // Compile using both UTF support and Unicode Character Properties (UCP)
  re = pcre2_compile(
      pattern,               /* the pattern */
      PCRE2_ZERO_TERMINATED, /* indicates pattern is zero-terminated */
      PCRE2_UTF | PCRE2_UCP, /* REQUIRED options for \p{L} and \p{N} */
      &errornumber,          /* for error number */
      &erroroffset,          /* for error offset */
      NULL);                 /* use default compile context */

  if (re == NULL) {
    PCRE2_UCHAR buffer[256];
    pcre2_get_error_message(errornumber, buffer, sizeof(buffer));
    printf("PCRE2 compilation failed at offset %d: %s\n", (int)erroroffset, buffer);
    return false;
  }

  pcre2_match_data *match_data = pcre2_match_data_create_from_pattern(re, NULL);
  PCRE2_SIZE *ovector;
  size_t subject_length = strlen((char *)subject);
  size_t start_offset = 0;

  // Loop to extract tokens matching the cl100k pattern (mimicking finditer)
  while (start_offset < subject_length) {
    int rc = pcre2_match(
        re,                   /* the compiled pattern */
        subject,              /* the subject string */
        subject_length,       /* the length of the subject */
        start_offset,         /* start at offset from previous match */
        0,                    /* default options */
        match_data,           /* block for storing the match result */
        NULL);                /* use default match context */

    if (rc < 0) {
      if (rc == PCRE2_ERROR_NOMATCH) break; // No more matches found
      printf("Matching error %d\n", rc);
      break;
    }

    ovector = pcre2_get_ovector_pointer(match_data);

    size_t nbytes = ovector[1] - ovector[0];
    Byte_Buffer buf = byte_buffer_from_parts((char*)subject + ovector[0], nbytes);

    array_append(array, buf);

    // Advance past the match
    start_offset = ovector[1];
  }

  // Clean up memory
  pcre2_match_data_free(match_data);
  pcre2_code_free(re);

  return true;
}

void calculate_histogram(Dataset_Array ids_list, int32_t* histogram, int32_t vocab_size) {
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

  for (size_t i = 0; i < ids_list.count; i++) {
    TokenID_Array ids = ids_list.elems[i];

    for (size_t j = 0; j < ids.count-1; j++) {
      int32_t token0 = ids.elems[j + 0];
      int32_t token1 = ids.elems[j + 1];

      if (token0 > 0 && token0 < 256 && punctuation[token0]) continue;
      if (token1 > 0 && token1 < 256 && punctuation[token1]) continue;

      int32_t index = token0*vocab_size + token1;
      histogram[index] += 1;
    }
  }
}

void array_remove_ith(TokenID_Array* array, size_t ith) {
  for (size_t i = ith; i < array->count-1; i++)
    array->elems[i] = array->elems[i+1];
  array->count -=1;
}

void array_println(TokenID_Array array) {
  printf("[");
  for (size_t i = 0; i < array.count; i++)
    printf("%d ", array.elems[i]);
  printf("]\n");
}

void merge_ids(Dataset_Array* ids_list, Token_Pair pair, int32_t new_token) {
  for (size_t i = 0; i < ids_list->count; i++) {
    TokenID_Array* ids = &ids_list->elems[i];

    if (ids->count <= 1) continue;

    for (size_t j = ids->count - 1; j > 0; j--) {
      int32_t token0 = ids->elems[j-1];
      int32_t token1 = ids->elems[j-0];

      if (token0 == pair.token0 && token1 == pair.token1) {
        array_remove_ith(ids, j);
        ids->elems[j-1] = new_token;
      }
    }
  }
}

int32_t gen_vocabulary(
    Allocator* global,
    const char* text[],
    size_t text_len,
    Token* vocabulary,
    int max_vocab
) {
  assert(max_vocab > 256);

  Arena_Allocator arena = ARENA_CREATE(global, 5*1024*1024);
  Allocator* alloc = &arena.alloc;

  int32_t vocabulary_count = 0;

  Token token = {0};

  token.id = PAD_TOKEN;
  memset(token.token, 0, MAX_TOKEN );
  strcpy(token.token, "<PAD>");
  vocabulary[vocabulary_count] = token;
  vocabulary_count += 1;

  for (int i = 1; i < 256; i++) {
    vocabulary[vocabulary_count].id = vocabulary_count;
    vocabulary[vocabulary_count].token[0] = i;
    vocabulary_count += 1;
  }

  token.id = EOS_TOKEN;
  memset(token.token, 0, MAX_TOKEN);
  strcpy(token.token, "<EOS>");
  vocabulary[vocabulary_count] = token;
  vocabulary_count += 1;

  Dataset_Array ids_list = ARRAY_CREATE(global);
  Byte_Buffer_Array words_array = ARRAY_CREATE(global);

  for (size_t text_idx = 0; text_idx < text_len; text_idx++) {
    words_array.count = 0;

    if (!pre_split(text[text_idx], &words_array)) {
      return 0;
    }

    for (size_t word_idx = 0; word_idx < words_array.count; word_idx++) {
      Byte_Buffer word = words_array.elems[word_idx];
      TokenID_Array tokens = ARRAY_CREATE(global);

      // convert bytes to token id
      for (size_t i = 0; i < word.len; i++) {
        array_append(&tokens, vocabulary[word.ptr[i]].id);
      }

      array_append(&ids_list, tokens);
    }
  }

  size_t x = 256 + 1;
  size_t num_merges = max_vocab - x;

  for (size_t i = 0; i < num_merges; i++) {
    size_t saved = SAVE(alloc);

    size_t histogram_size = sizeof(int32_t)*vocabulary_count*vocabulary_count;
    int32_t* histogram = ALLOC(alloc, histogram_size).void_ptr;
    memset(histogram, 0, histogram_size);

    calculate_histogram(ids_list, histogram, vocabulary_count);

    int32_t max_count = 0;
    Token_Pair max_pair = {0};

    for (int32_t token0 = 0; token0 < vocabulary_count; token0++) {
      for (int32_t token1 = 0; token1 < vocabulary_count; token1++) {
        int32_t index = token0*vocabulary_count + token1;

        if (histogram[index] > max_count) {
          max_count = histogram[index];
          max_pair.token0 = token0;
          max_pair.token1 = token1;
        }
      }
    }

    if (max_count == 0) break;

    int32_t new_token = i + x;
    vocabulary[new_token].id = new_token;
    strncpy(vocabulary[new_token].token, vocabulary[max_pair.token0].token, MAX_TOKEN);
    strncat(vocabulary[new_token].token, vocabulary[max_pair.token1].token, MAX_TOKEN);
    vocabulary_count += 1;

    merge_ids(&ids_list, max_pair, new_token);

    RESTORE(alloc, saved);
  }

  assert(vocabulary_count == max_vocab);

  ARENA_DESTROY(global, &arena);

  return 0;
}

#endif // TOKENIZER_H
