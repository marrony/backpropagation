#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <time.h>
#include <pcre2.h>
#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

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
  int32_t a;
  int32_t b;
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

bool pre_split(Byte_Buffer text, Byte_Buffer_Array* array) {
  pcre2_code *re;
  PCRE2_SPTR pattern = (PCRE2_SPTR)"(?i:\'s|\'t|\'re|\'ve|\'m|\'ll|\'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s";
  PCRE2_SPTR subject = (PCRE2_SPTR)text.cptr;

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
  size_t subject_length = text.len;
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

static inline bool countable(int32_t t0, int32_t t1) {
  static bool punctuation[256] = {
    ['.'] = true,
    [','] = true,
    ['?'] = true,
    ['!'] = true,
    [':'] = true,
    [';'] = true,
    ['-'] = true,
    ['\''] = true,
    ['\"'] = true,
    ['('] = true,
    [')'] = true,
    ['\r'] = true,
    ['\n'] = true,
  };

  if (t0 == 0 || t1 == 0) { return false; }
  if (t0 > 0 && t0 < 256 && punctuation[t0]) { return false; }
  if (t1 > 0 && t1 < 256 && punctuation[t1]) { return false; }
  return true;
}

void calculate_histogram(
  TokenID_Array ids,
  int32_t* histogram,
  int32_t vocab_size
) {
  for (size_t i = 0; i < ids.count - 1; i++) {
    int32_t token0 = ids.elems[i + 0];
    int32_t token1 = ids.elems[i + 1];

    if (!countable(token0, token1)) continue;

    int32_t index = token0*vocab_size + token1;
    histogram[index] += 1;
  }
}

int32_t find_max(int32_t* histogram, size_t vocab_size, Token_Pair* max_pair_out) {
  int32_t max_count = 0;
  size_t max_pair_index = 0;

  for (size_t i = 0; i < vocab_size*vocab_size; i++) {
    if (histogram[i] > max_count) {
      max_count = histogram[i];
      max_pair_index = i;
    }
  }

  // max_pair.token0*max_vocab + max_pair.token1
  // 2*3072 + 10 = 6154
  // 2  = 6154 / 3072
  // 10 = 6154 % 3072
  max_pair_out->token0 = max_pair_index / vocab_size;
  max_pair_out->token1 = max_pair_index % vocab_size;

  return max_count;
}

void array_println(TokenID_Array array) {
  printf("[");
  for (size_t i = 0; i < array.count; i++)
    printf("%d ", array.elems[i]);
  printf("]\n");
}

void merge_ids(
    TokenID_Array* ids_array,
    Token_Pair pair,
    int32_t c,
    int32_t* histogram,
    size_t max_vocab
) {
  int32_t a = pair.token0;
  int32_t b = pair.token1;
  int32_t* ids = ids_array->elems;
  size_t n = ids_array->count;
  size_t w = 0;

  for (size_t r = 0; r < n; ) {
    if (r + 1 < n && ids[r] == a && ids[r + 1] == b) {
      if (w > 0) {
        int32_t left = ids[w - 1];
        if (countable(left, a)) histogram[left*max_vocab + a] -= 1;
        if (countable(left, c)) histogram[left*max_vocab + c] += 1;
      }

      if (r + 2 < n) {
        int32_t right = ids[r + 2];
        if (countable(b, right)) histogram[b*max_vocab + right] -= 1;
        if (countable(c, right)) histogram[c*max_vocab + right] += 1;
      }

      histogram[a*max_vocab + b] -= 1;
      ids[w++] = c;
      r += 2;
    } else {
      ids[w++] = ids[r++];
    }
  }
  ids_array->count = w;
}

void print_timestamp(FILE* fp) {
  time_t raw_time;
  struct tm tm;
  char buffer[128];

  time(&raw_time);
  localtime_r(&raw_time, &tm);
  strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", &tm);

  fprintf(fp, "%s", buffer);
}

int32_t gen_vocabulary(
    Allocator* global,
    Byte_Buffer_Array text,
    Token* vocabulary,
    size_t max_vocab
) {
  assert(max_vocab > 256);

  size_t vocabulary_count = 0;

  Token token = {0};

  token.id = PAD_TOKEN;
  memset(token.token, 0, MAX_TOKEN );
  strcpy(token.token, "<PAD>");
  vocabulary[vocabulary_count] = token;
  vocabulary_count += 1;

  for (int i = 1; i < 256; i++) {
    vocabulary[vocabulary_count].id = vocabulary_count;
    vocabulary[vocabulary_count].a = -1;
    vocabulary[vocabulary_count].b = -1;
    vocabulary[vocabulary_count].token[0] = i;
    vocabulary_count += 1;
  }

  token.id = EOS_TOKEN;
  memset(token.token, 0, MAX_TOKEN);
  strcpy(token.token, "<EOS>");
  vocabulary[vocabulary_count] = token;
  vocabulary[vocabulary_count].a = -1;
  vocabulary[vocabulary_count].b = -1;
  vocabulary_count += 1;

  TokenID_Array ids = ARRAY_CREATE(global);
  Byte_Buffer_Array words_array = ARRAY_CREATE(global);

  for (size_t text_idx = 0; text_idx < text.count; text_idx++) {
    Byte_Buffer str = text.elems[text_idx];

    words_array.count = 0;

    if (!pre_split(str, &words_array)) {
      return 0;
    }

    for (size_t word_idx = 0; word_idx < words_array.count; word_idx++) {
      Byte_Buffer word = words_array.elems[word_idx];

      // convert bytes to token id
      for (size_t i = 0; i < word.len; i++) {
        array_append(&ids, word.uptr[i]);
      }

      array_append(&ids, 0);
    }
  }

  size_t histogram_size = sizeof(int32_t)*max_vocab*max_vocab;
  int32_t* histogram = ALLOC(global, histogram_size).ptr;
  memset(histogram, 0, histogram_size);

  calculate_histogram(ids, histogram, max_vocab);

  for (size_t c = 256+1; c < max_vocab; c++) {
    Token_Pair max_pair = {0};
    int32_t count = find_max(histogram, max_vocab, &max_pair);

    if (count <= 0) break;

    vocabulary[c].id = c;
    vocabulary[c].a = max_pair.token0;
    vocabulary[c].b = max_pair.token1;
    snprintf(vocabulary[c].token, MAX_TOKEN, "%s%s",
        vocabulary[max_pair.token0].token, vocabulary[max_pair.token1].token);
    vocabulary_count += 1;

    merge_ids(&ids, max_pair, c, histogram, max_vocab);

    assert(histogram[(size_t)max_pair.token0*max_vocab + max_pair.token1] == 0);
  }

  assert(vocabulary_count <= max_vocab);

  return vocabulary_count;
}

void build_merge_to(int32_t* merge_to, Token* vocabulary, size_t max_vocab) {
  for (size_t i = 0; i < max_vocab*max_vocab; i++)
    merge_to[i] = -1;

  for (size_t c = 257; c < max_vocab; c++) {
    assert(vocabulary[c].a != -1 && vocabulary[c].b != -1);
    merge_to[vocabulary[c].a*max_vocab + vocabulary[c].b] = c;
  }
}

void bpe_word(TokenID_Array* ids, const int32_t *merge_to, int32_t V) {
  for (;;) {
    int32_t best_id = INT32_MAX;
    size_t  best_pos = (size_t)-1;

    for (size_t j = 0; j + 1 < ids->count; j++) {
      int32_t m = merge_to[(size_t)ids->elems[j] * V + ids->elems[j + 1]];

      if (m >= 0 && m < best_id) {
        best_id = m;
        best_pos = j;
      }
    }

    if (best_pos == (size_t)-1)
      break;

    ids->elems[best_pos] = best_id;
    memmove(&ids->elems[best_pos + 1], &ids->elems[best_pos + 2],
            (ids->count - best_pos - 2) * sizeof(int32_t));
    ids->count -= 1;
  }
}

size_t tokenize(
  Arena_Allocator* arena,
  TokenID_Array* sequence,
  Byte_Buffer text,
  int32_t* merge_to,
  size_t max_vocab
) {
  size_t tokens_count = 0;

  Byte_Buffer_Array words = ARRAY_CREATE(&arena->alloc);
  TokenID_Array buf = ARRAY_CREATE(&arena->alloc);

  pre_split(text, &words);

  for (size_t w = 0; w < words.count; w++) {
    Byte_Buffer word = words.elems[w];

    buf.count = 0;

    for (size_t i = 0; i < word.len; i++) {
      array_append(&buf, word.uptr[i]);
    }

    bpe_word(&buf, merge_to, max_vocab);

    for (size_t i = 0; i < buf.count; i++) {
      array_append(sequence, buf.elems[i]);
    }

    tokens_count += buf.count;
  }

  return tokens_count;
}

typedef struct {
  double bpc;
  double cpt;
} BPC;

BPC bpc(
  TokenID_Array train_ids,
  TokenID_Array dev_ids,
  size_t* pairs,
  size_t* ctx,
  size_t max_vocab,
  size_t dev_chars
) {
  for (size_t i = 0; i + 1 < train_ids.count; i++) {
    int32_t a = train_ids.elems[i+0];
    int32_t b = train_ids.elems[i+1];

    if (a >= (int32_t)max_vocab || b >= (int32_t)max_vocab) {
      printf("a=%d b=%d\n", a, b);
    }
    assert(a < (int32_t)max_vocab && b < (int32_t)max_vocab);

    pairs[a*max_vocab + b] += 1;
    ctx[a] += 1;
  }

  double total_nats = 0.0;
  size_t n_tok = 0;

  for (size_t i = 0; i + 1 < dev_ids.count; i++) {
    int32_t a = dev_ids.elems[i+0];
    int32_t b = dev_ids.elems[i+1];

    assert(a < (int32_t)max_vocab && b < (int32_t)max_vocab);

    int32_t p = pairs[a*max_vocab + b];
    total_nats -= log((double)(p + 1) / (double)(ctx[a] + max_vocab));
    n_tok += 1;
  }

  return (BPC) {
    .bpc = total_nats * 1.4142135624 / (double)dev_chars,
    .cpt = (double)dev_chars / (double)n_tok,
  };
}

#endif // TOKENIZER_H
