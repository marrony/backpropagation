#include "array.h"
#include "nn.h"
#include <time.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#define BYTEBUFFER_IMPLEMENTATION
#define ALLOCATOR_IMPLEMENATION

#include "bytebuffer.h"
#include "allocator.h"
#include "tokenizer.h"
#include "transformer.h"
#include "node.h"

#undef STR
#undef STR0
#define STR0(x) #x
#define STR(x) STR0(x)

#define CORPUS(x, y) "distill/cdata-" STR(MAX_VOCAB) "/" #x "." #y ".bin"

Byte_Buffer model_name(Allocator* alloc, size_t V, size_t D, size_t F, size_t b, const char* ext) {
  Byte_Buffer buf = NULL_BYTE_BUFFER;

  int nbytes = -1;

  for (int i = 0; i < 2; i++) {
    nbytes = snprintf(buf.cptr, nbytes+1, "models/V=%zu-D=%zu-F=%zu-b=%zu.%s", V, D, F, b, ext);
    if (nbytes < 0 || buf.cptr != NULL) break;
    buf = ALLOC(alloc, nbytes+1);
  }

  return buf;
}

#ifdef VOCAB_HEADER
#include VOCAB_HEADER
#else
#error VOCAB_HEADER not defined
#endif

void hsl_to_rgb(float h, float s, float l, int *r, int *g, int *b) {
  // Calculate Chroma
  float c = (1.0f - fabsf(2.0f * l - 1.0f)) * s;

  // Calculate intermediate value X
  // fmodf(h / 60.0f, 2.0f) handles the modulo behavior for floating points
  float x = c * (1.0f - fabsf(fmodf(h / 60.0f, 2.0f) - 1.0f));

  // Match value m
  float m = l - c / 2.0f;

  float rp = 0, gp = 0, bp = 0;

  if (h >= 0.0f && h < 60.0f) {
    rp = c;
    gp = x;
  } else if (h >= 60.0f && h < 120.0f) {
    rp = x;
    gp = c;
  } else if (h >= 120.0f && h < 180.0f) {
    gp = c;
    bp = x;
  } else if (h >= 180.0f && h < 240.0f) {
    gp = x;
    bp = c;
  } else if (h >= 240.0f && h < 300.0f) {
    rp = x;
    bp = c;
  } else if (h >= 300.0f && h <= 360.0f) {
    rp = c;
    bp = x;
  }

  // Scale values to standard 8-bit integers [0, 255]
  *r = (int)roundf((rp + m) * 255.0f);
  *g = (int)roundf((gp + m) * 255.0f);
  *b = (int)roundf((bp + m) * 255.0f);
}

void set_color(int r, int g, int b) {
  printf("\033[38;2;%d;%d;%dm", r, g, b);
}

void rst_color(void) {
  printf("\033[38;0m");
}

#undef MIN
#undef MAX

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

Malloc_Allocator mallocator = MALLOC_CREATE();

size_t count_parameters(Optimizer* optmizer) {
  size_t count = 0;

  for (size_t i = 0; i < optmizer->tensors.count; i++) {
    NMatrix value = optmizer->tensors.elems[i].value;
    count += value.rows*value.cols;
  }

  return count;
}

int32_t sample(NMatrix probs) {
  float r = rand_uniform();
  float cumulative = 0.0f;

  for (int i = 0; i < MAX_VOCAB; i++) {
    cumulative += VEC_AT(probs, i);
    if (r <= cumulative)
      return i;
  }

  return MAX_VOCAB - 1;
}

typedef struct {
  float prob;
  size_t index;
} Entry;

DEFINE_ARRAY(Entry);

int compare_entry(const void* a, const void* b) {
  const Entry* entry_a = a;
  const Entry* entry_b = b;

  if (entry_a->prob < entry_b->prob) return +1;
  if (entry_a->prob > entry_b->prob) return -1;
  return 0;
}

int32_t sample_topp(Arena_Allocator* arena, NMatrix probs, float topp) {
  Entry_Array entries = ARRAY_CREATE(&arena->alloc);
  array_ensure(&entries, MAX_VOCAB);

  for (size_t i = 0; i < MAX_VOCAB; i++) {
    Entry entry = {
      .prob = VEC_AT(probs, i),
      .index = i,
    };

    array_append(&entries, entry);
  }

  qsort(entries.elems, entries.count, sizeof(Entry), compare_entry);

  Entry_Array nucleus = ARRAY_CREATE(&arena->alloc);
  array_ensure(&nucleus, MAX_VOCAB);

  float cummulative = 0;
  for (size_t i = 0; i < MAX_VOCAB; i++) {
    Entry entry = entries.elems[i];
    array_append(&nucleus, entry);
    cummulative += entry.prob;
    if (cummulative >= topp)
      break;
  }

  float r = rand_uniform() * cummulative;
  int32_t sampled = nucleus.elems[0].index;

  cummulative = 0;
  for (size_t i = 0; i < nucleus.count; i++) {
    Entry entry = nucleus.elems[i];
    cummulative += entry.prob;
    if (r <= cummulative) {
      return entry.index;
    }
  }

  return sampled;
}

void save_model(Transformer* trans_in, Byte_Buffer model_name) {
  FILE* fp = fopen(model_name.cptr, "wb");
  mat_write(trans_in->tok_emb.value, fp);
  mat_write(trans_in->ln.gamma.value, fp);
  mat_write(trans_in->ln.beta.value, fp);
  if (!trans_in->tie_embeddings)
    mat_write(trans_in->H.weight.value, fp);
  mat_write(trans_in->H.bias.value, fp);

  fwrite(&trans_in->num_blocks, sizeof(size_t), 1, fp);

  for (size_t i = 0; i < trans_in->num_blocks; i++) {
    mat_write(trans_in->blocks[i].ln1.gamma.value, fp);
    mat_write(trans_in->blocks[i].ln1.beta.value, fp);
    mat_write(trans_in->blocks[i].ln2.gamma.value, fp);
    mat_write(trans_in->blocks[i].ln2.beta.value, fp);
    mat_write(trans_in->blocks[i].ff1.weight.value, fp);
    mat_write(trans_in->blocks[i].ff1.bias.value, fp);
    mat_write(trans_in->blocks[i].ff2.weight.value, fp);
    mat_write(trans_in->blocks[i].ff2.bias.value, fp);
    mat_write(trans_in->blocks[i].attn.K.weight.value, fp);
    mat_write(trans_in->blocks[i].attn.K.bias.value, fp);
    mat_write(trans_in->blocks[i].attn.Q.weight.value, fp);
    mat_write(trans_in->blocks[i].attn.Q.bias.value, fp);
    mat_write(trans_in->blocks[i].attn.V.weight.value, fp);
    mat_write(trans_in->blocks[i].attn.V.bias.value, fp);
    mat_write(trans_in->blocks[i].attn.O.weight.value, fp);
    mat_write(trans_in->blocks[i].attn.O.bias.value, fp);
  }

  fclose(fp);
}

bool load_model(Transformer* trans_in, Byte_Buffer model_name) {
  FILE* fp = fopen(model_name.cptr, "rb");
  if (fp == NULL) {
    printf("model %.*s doesn't exists\n", (int)model_name.len, model_name.cptr);
    return false;
  }

  printf("model %.*s exists, reading\n", (int)model_name.len, model_name.cptr);
  mat_read(trans_in->tok_emb.value, fp);
  mat_read(trans_in->ln.gamma.value, fp);
  mat_read(trans_in->ln.beta.value, fp);
  if (!trans_in->tie_embeddings)
    mat_read(trans_in->H.weight.value, fp);
  mat_read(trans_in->H.bias.value, fp);

  fread(&trans_in->num_blocks, sizeof(size_t), 1, fp);

  for (size_t i = 0; i < trans_in->num_blocks; i++) {
    mat_read(trans_in->blocks[i].ln1.gamma.value, fp);
    mat_read(trans_in->blocks[i].ln1.beta.value, fp);
    mat_read(trans_in->blocks[i].ln2.gamma.value, fp);
    mat_read(trans_in->blocks[i].ln2.beta.value, fp);
    mat_read(trans_in->blocks[i].ff1.weight.value, fp);
    mat_read(trans_in->blocks[i].ff1.bias.value, fp);
    mat_read(trans_in->blocks[i].ff2.weight.value, fp);
    mat_read(trans_in->blocks[i].ff2.bias.value, fp);
    mat_read(trans_in->blocks[i].attn.K.weight.value, fp);
    mat_read(trans_in->blocks[i].attn.K.bias.value, fp);
    mat_read(trans_in->blocks[i].attn.Q.weight.value, fp);
    mat_read(trans_in->blocks[i].attn.Q.bias.value, fp);
    mat_read(trans_in->blocks[i].attn.V.weight.value, fp);
    mat_read(trans_in->blocks[i].attn.V.bias.value, fp);
    mat_read(trans_in->blocks[i].attn.O.weight.value, fp);
    mat_read(trans_in->blocks[i].attn.O.bias.value, fp);
  }

  fclose(fp);

  return true;
}

void set_dropout(Transformer* trans_in, float p) {
  trans_in->x0_p = p;
  for (size_t i = 0; i < trans_in->num_blocks; i++) {
    trans_in->blocks[i].attn.p = p;
  }
}

void generate_text(
  Arena_Allocator* arena,
  Transformer* trans_in,
  Byte_Buffer prompt,
  TokenID_Array* tokens,
  Byte_Buffer_Array* words,
  TokenID_Array* buf,
  int32_t* merge_to,
  size_t max_tokens
) {
  tokens->count = 0;
  words->count = 0;
  buf->count = 0;
  tokenize(
      tokens,
      words,
      buf,
      prompt,
      merge_to,
      MAX_VOCAB
  );

  size_t lines_count = 0;

  //const char* separator = "\u22c5";
  const char* separator = "";

  for (size_t i = 0; i < tokens->count; i++)
      printf("%s%s", vocabulary[tokens->elems[i]].token, separator);

  for (size_t i = 0; i < max_tokens; i++) {
    size_t saved = SAVE(&arena->alloc);

    size_t N = tokens->count;
    Transformer_Output trans_out = {0};
    init_transformer_output(
        .arena = arena,
        .trans_out = &trans_out,
        .num_blocks = trans_in->num_blocks,
        .vocab_size = trans_in->vocab_size,
        .heads_count = trans_in->heads_count,
        .emb_size = trans_in->emb_size,
        .ff_size = trans_in->ff_size,
        .sequence_size = N,
    );

    // 0.5 = deterministic
    // 1.0 = normal
    // 1.5 = creative
    // 3.0 = nonsensical
    transformer_forward(*tokens, &trans_out, trans_in, 1.0f);

    NMatrix last_token = mat_row(trans_out.probs, N - 1);

    int32_t next = mat_row_argmax(last_token);
    // int32_t next = sample(last_token);
    // int32_t next = sample_topp(arena, last_token, 0.9f);

    if (next == '\r') next = '\n';

    float red = 0.0f;
    float green = 120.0f;
    float alpha = VEC_AT(last_token, next);

    int r, g, b;
    hsl_to_rgb(green*alpha + (1.0f - alpha)*red, 1.0f, 0.5f, &r, &g, &b);
    set_color(r, g, b);
    printf("%s%s", vocabulary[next].token, separator);
    fflush(stdout);

    // if (tokens->count >= C) {
    //   for (size_t c = 0; c < C - 1; c++)
    //     tokens->elems[c] = tokens->elems[c+1];
    //   tokens->count -= 1;
    // }

    array_append(tokens, next);

    RESTORE(&arena->alloc, saved);

    if (next == EOS_TOKEN)
      break;

    if (next == '\n')
      lines_count += 1;

    if (lines_count >= 5)
      break;
  }

  rst_color();
  printf("\n");
}

// Helper function to shuffle an array of indices (Fisher-Yates)
void shuffle_indices(Index32_Array indices) {
  for (size_t i = indices.count - 1; i > 0; i--) {
    size_t j = rand_between(0, i);
    size_t temp = indices.elems[i];
    indices.elems[i] = indices.elems[j];
    indices.elems[j] = temp;
  }
}

int find_token(const char* token) {
  for (int i = 0; i < MAX_VOCAB; i++) {
    if (strcmp(token, vocabulary[i].token) == 0)
      return i;
  }
  return -1;
}

float cosine_learning_rate(
  size_t t,
  size_t t_warmup,
  size_t t_total,
  float lr_max,
  float lr_min
) {
  if (t < t_warmup) {
    return lr_max * (float)(t + 1) / (float)t_warmup;
  }

  float alpha = (float)(t - t_warmup) / (float)(t_total - t_warmup);
  if (alpha > 1.0f) alpha = 1.0f;
  return lr_min + 0.5f*(lr_max - lr_min) * (1.0f + cosf(alpha * M_PI));
}

void print_tokens(Transformer* trans_in) {
  (void)trans_in;
  // printf("\033[38;5;130m");
  // printf("desert  = %.5f\n",
  //   mat_row_similarity(
  //     mat_row(trans_in->tok_emb.value, find_token("desert")),
  //     mat_row(trans_in->tok_emb.value, find_token(" desert"))
  //   )
  // );
  // printf("\033[0m");
}

typedef struct {
  int32_t offset;
  int32_t prompt_len;
  int32_t total_len;
} Rec;
_Static_assert(sizeof(Rec) == 12, "Rec must be 3 packed int32");

typedef struct {
  int32_t *ids;
  size_t n_ids;
  Rec *recs;
  size_t n_recs;
} Corpus;

void *slurp(const char *path, size_t *nbytes) {
  FILE *f = fopen(path, "rb");
  if (!f) { perror(path); exit(1); }
  fseek(f, 0, SEEK_END);
  long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  void *p = malloc((size_t)n);
  if (fread(p, 1, (size_t)n, f) != (size_t)n) { perror(path); exit(1); }
  fclose(f);
  *nbytes = (size_t)n;
  return p;
}

Corpus corpus_load(const char *ids_path, const char *index_path) {
  Corpus c;
  size_t nb;
  c.ids   = slurp(ids_path, &nb);
  c.n_ids = nb / sizeof(int32_t);
  c.recs   = slurp(index_path, &nb);
  c.n_recs = nb / sizeof(Rec);
  return c;
}

const int32_t *example(const Corpus *c, size_t i, int *prompt_len, int *total_len) {
  Rec r = c->recs[i];
  *prompt_len = r.prompt_len;
  *total_len  = r.total_len;
  return c->ids + r.offset;
}

// mean = 0 and variance = 1 over D components
// Σ_d x_d² = D
// ‖x‖ = √D
//
// σ is the standard deviation
// σ² is the variance
//
// var(logits) = σ_E² · D
// std(logits) = σ_E  · √D
//
// h is LayerNorm output, std = 1  ‖h‖ = √D
// std_v(h · E[v]) = σ_E · ‖h‖ = σ_E · √D
// std(logits) = σ_E · √D = ‖E[v]‖
//
// σ = √(2/(V+D))
float logits_std(NMatrix logits) {
  float std_acc = 0.0;
  for (uint32_t i = 0; i < logits.rows; i++) {
    float mean = 0.0;
    for (uint32_t v = 0; v < logits.cols; v++) {
      mean += MAT_AT(logits, i, v);
    }
    mean /= logits.cols;

    float var = 0.0;
    for (uint32_t v = 0; v < logits.cols; v++) {
      float d = MAT_AT(logits, i, v) - mean; // std
      var += d * d;
    }
    std_acc += sqrtf(var / logits.cols);
  }
  return std_acc / logits.rows;
}

float evaluate(
  Arena_Allocator* arena,
  Transformer *trans_in,
  Corpus *dev,
  size_t nrecs
) {
  const float saved_p = trans_in->x0_p;
  set_dropout(trans_in, 0.0f);

  double loss_sum = 0.0;
  long   tokens_scored = 0;

  size_t saved = SAVE(&arena->alloc);

  TokenID_Array tokens = ARRAY_CREATE(&arena->alloc);
  TokenID_Array targets = ARRAY_CREATE(&arena->alloc);

  for (size_t e = 0; e < nrecs; e++) {
    int prompt_len, total_len;
    const int32_t *ids = example(dev, e, &prompt_len, &total_len);

    const int first = prompt_len - 1;
    const int npos  = total_len - prompt_len;
    if (npos <= 0) continue;

    tokens.count = 0;
    targets.count = 0;

    for (int c = 0; c < total_len - 1; c++) {
      array_append(&tokens, ids[c]);
      array_append(&targets, ids[c+1]);
    }

    size_t saved = SAVE(&arena->alloc);

    size_t N = tokens.count;

    Transformer_Output trans_out = {0};
    init_transformer_output(
        .arena = arena,
        .trans_out = &trans_out,
        .num_blocks = trans_in->num_blocks,
        .vocab_size = trans_in->vocab_size,
        .heads_count = trans_in->heads_count,
        .emb_size = trans_in->emb_size,
        .ff_size = trans_in->ff_size,
        .sequence_size = N,
    );

    transformer_forward(tokens, &trans_out, trans_in, 1.0f);

    for (size_t i = (size_t)first; i < N; i++) {
      const int32_t target = targets.elems[i];
      loss_sum -= log((double)MAT_AT(trans_out.probs, i, target) + 1e-10);
      tokens_scored++;
    }

    RESTORE(&arena->alloc, saved);
  }

  RESTORE(&arena->alloc, saved);

  set_dropout(trans_in, saved_p);
  return loss_sum / (double)tokens_scored;
}

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

static int cmp_size(const void *a, const void *b) {
  size_t x = *(const size_t *)a, y = *(const size_t *)b;
  return (x > y) - (x < y);          /* NOT x - y: size_t underflows */
}

size_t histogram[MAX_VOCAB] = {0};
size_t histogram_sorted[MAX_VOCAB] = {0};

int main(int argc, char* argv[]) {
  srand(getpid());

  Optimizer optimizer = (Optimizer) {
    .tensors = ARRAY_CREATE(&mallocator.alloc),
    .history = ARRAY_CREATE(&mallocator.alloc),
    .second = ARRAY_CREATE(&mallocator.alloc),
    .decay = ARRAY_CREATE(&mallocator.alloc),
    .learning_rate = 0.05f,
    .decay_factor = 0.00001f,
    .updates = 0,
  };
  size_t arena_size = MAX_VOCAB*MAX_VOCAB*sizeof(float);
  Arena_Allocator arena = ARENA_CREATE(&mallocator.alloc, arena_size + 2L*1024*1024*1024);
  // TokenID_Array sequence = ARRAY_CREATE(&mallocator.alloc);
  TokenID_Array tokens = ARRAY_CREATE(&mallocator.alloc);
  TokenID_Array buff = ARRAY_CREATE(&mallocator.alloc);
  Byte_Buffer_Array words = ARRAY_CREATE(&mallocator.alloc);
  TokenID_Array targets = ARRAY_CREATE(&mallocator.alloc);
  Index32_Array order_indices = ARRAY_CREATE(&mallocator.alloc);

  int32_t* merge_to = ALLOC(&mallocator.alloc, sizeof(int32_t)*MAX_VOCAB*MAX_VOCAB).ptr;

  build_merge_to(merge_to, vocabulary, MAX_VOCAB);

  Corpus corpus = corpus_load(CORPUS(train, ids), CORPUS(train, index));
  Corpus corpus_dev = corpus_load(CORPUS(dev, ids), CORPUS(dev, index));

  array_ensure(&order_indices, corpus.n_recs);

  size_t sum_tokens = 0;
  size_t scored_tokens = 0;
  for (size_t i = 0; i < corpus.n_recs; ++i) {
    sum_tokens += corpus.recs[i].total_len;
    scored_tokens += corpus.recs[i].total_len - corpus.recs[i].prompt_len;
    array_append(&order_indices, i);
  }

  shuffle_indices(order_indices);

  for (size_t i = 0; i < corpus.n_ids; i++)
    histogram[corpus.ids[i]] += 1;
  memcpy(histogram_sorted, histogram, sizeof(histogram));
  qsort(histogram_sorted, MAX_VOCAB, sizeof(size_t), cmp_size);

  size_t dead = 0;
  size_t rare = 0;
  for (size_t v = 0; v < MAX_VOCAB; v++) {
    if (histogram[v] == 0) {
      dead += 1;
    }

    if (histogram[v] < 100) {
      rare += 1;
    }
  }

  // for (size_t i = 0; i < corpus.n_recs; i++) {
  //   int prompt_len = 0;
  //   int total_len = 0;
  //   const int32_t *ids = example(&corpus, i, &prompt_len, &total_len);
  //
  //   printf("prompt_len = %d | total_len = %d\n", prompt_len, total_len);
  //
  //   // const char* sep = "\u22c5";
  //   const char* sep = "";
  //
  //   for (int j = 0; j < total_len; j++) {
  //     // if (j == prompt_len) printf(" -> ");
  //     printf("%s%s", vocabulary[ids[j]].token, sep);
  //   }
  //
  //   printf("\n");
  // }
  // return 0;

  size_t D = 192;
  size_t heads_count = 4;
  size_t F = D*heads_count;
  size_t V = MAX_VOCAB;
  size_t num_blocks = 4;

  Byte_Buffer model_name_par = model_name(&mallocator.alloc, V, D, F, num_blocks, "adam");
  Byte_Buffer model_name_bin = model_name(&mallocator.alloc, V, D, F, num_blocks, "bin");

  float peak_lr = 1e-3;
  size_t total_epochs = 2;
  size_t target_scored = 1638;
  size_t total_steps = (scored_tokens * total_epochs + target_scored - 1) / target_scored;
  // w0 = ~1/(1−β₂)
  // w1 = epochs*(2%-5%)
  // w = max(w0, w1)
  size_t warmup_steps = MAX(1000, (size_t)(total_epochs * 0.02f));
  float dropout_pct = 0.0;

  // optimize learning rate
  // int patience = 100;
  // float decay_factor = 0.9f;
  // float min_lr = 0.0001f;
  // float min_delta = 0.0005f;
  // float best_epoch_loss = INFINITY;
  // int plateau_epochs = 0;

  // array_ensure(&tokens, C);
  // array_ensure(&targets, C);

  Transformer trans_in = {0};

  init_transformer(
      .alloc = &arena.alloc,
      .trans = &trans_in,
      .tie_embeddings = true,
      .num_blocks = num_blocks,
      .heads_count = heads_count,
      .vocab_size = V,
      .emb_size = D,
      .ff_size = F,
  );

  set_color(255, 165, 0);
  printf("D = %zu\n", D);
  printf("V = %zu\n", V);
  printf("F = %zu\n", F);
  printf("tie_embeddings = %s\n", trans_in.tie_embeddings ? "yes" : "no");
  printf("heads_count = %zu\n", heads_count);
  printf("head_dim = %zu\n", D / heads_count);
  printf("block_count = %zu\n", num_blocks);
  {
    size_t params = count_parameters(&optimizer);
    register_tensor(&optimizer, trans_in.tok_emb, true);
    printf("token_emdeddings = %zu\n", count_parameters(&optimizer) - params);
  }
  {
    size_t params1 = count_parameters(&optimizer);
    if (!trans_in.tie_embeddings)
      register_tensor(&optimizer, trans_in.H.weight, true);
    printf("output_head = %zu\n", count_parameters(&optimizer) - params1);
    size_t params2 = count_parameters(&optimizer);
    register_tensor(&optimizer, trans_in.H.bias, false);
    printf("output_bias = %zu\n", count_parameters(&optimizer) - params2);
  }

  {
    size_t params = count_parameters(&optimizer);
    for (size_t i = 0; i < num_blocks; i++) {
      register_tensor(&optimizer, trans_in.blocks[i].ln1.gamma, false);
      register_tensor(&optimizer, trans_in.blocks[i].ln1.beta, false);
      register_tensor(&optimizer, trans_in.blocks[i].ln2.gamma, false);
      register_tensor(&optimizer, trans_in.blocks[i].ln2.beta, false);

      register_tensor(&optimizer, trans_in.blocks[i].ff1.weight, true);
      register_tensor(&optimizer, trans_in.blocks[i].ff1.bias, false);
      register_tensor(&optimizer, trans_in.blocks[i].ff2.weight, true);
      register_tensor(&optimizer, trans_in.blocks[i].ff2.bias, false);

      register_tensor(&optimizer, trans_in.blocks[i].attn.K.weight, true);
      register_tensor(&optimizer, trans_in.blocks[i].attn.K.bias, false);
      register_tensor(&optimizer, trans_in.blocks[i].attn.Q.weight, true);
      register_tensor(&optimizer, trans_in.blocks[i].attn.Q.bias, false);
      register_tensor(&optimizer, trans_in.blocks[i].attn.V.weight, true);
      register_tensor(&optimizer, trans_in.blocks[i].attn.V.bias, false);
      register_tensor(&optimizer, trans_in.blocks[i].attn.O.weight, true);
      register_tensor(&optimizer, trans_in.blocks[i].attn.O.bias, false);
    }
    printf("transformer_block = %zu\n", (count_parameters(&optimizer) - params) / num_blocks);
  }

  {
    size_t params0 = count_parameters(&optimizer);
    register_tensor(&optimizer, trans_in.ln.gamma, false);
    register_tensor(&optimizer, trans_in.ln.beta, false);
    printf("final_layer_norm = %zu\n", count_parameters(&optimizer) - params0);
  }

  size_t params = count_parameters(&optimizer);
  printf("total_parameters = %zu\n", params);
  printf("corpus_size (train) = %zu\n", corpus.n_recs);
  printf("corpus_size (dev)   = %zu\n", corpus_dev.n_recs);
  printf("tokens_count = %zu\n", sum_tokens);
  printf("scored_tokens = %zu (%.2f%%)\n", scored_tokens, (float)scored_tokens/(float)sum_tokens * 100.0f);
  printf("target_scored = %zu\n", target_scored);
  printf("total_steps = %zu\n", total_steps);
  printf("type frequency: p1=%zu p10=%zu p50=%zu p90=%zu max=%zu\n",
       histogram_sorted[V/100], histogram_sorted[V/10], histogram_sorted[V/2],
       histogram_sorted[9*V/10], histogram_sorted[V-1]);
  printf("  %zu never appear, %zu appear <100x (%.1f%%)\n", dead, rare, 100.0*rare/MAX_VOCAB);
  printf("chinchilla target = %zu tokens\n", 20*params);
  printf("ratio = %.2f (total) / %.2f (scored) tokens per parameter\n",
      sum_tokens / (float)params, scored_tokens / (float)params);
  printf("epochs to get there = %.2f (total) / %0.2f (scored)\n",
      (20*params) / (float)sum_tokens, (20*params) / (float)scored_tokens);
  rst_color();

  // Muennighoff et al. (2023), Scaling Data-Constrained Language Models, is the empirical answer:
  //
  //  - Up to ~4 epochs: repeated tokens are worth nearly as much as fresh ones
  //  - 4 to ~16 epochs: returns decay steadily
  //  - Past ~16: essentially zero value
  size_t epoch = 0;
  size_t optimizer_steps = 0;
  size_t cursor = 0;

  bool loaded = load_model(&trans_in, model_name_bin);
  load_optimizer(&optimizer, model_name_par, &epoch, &optimizer_steps, order_indices, &cursor);

  // printf("cursor = %zu\n", cursor);
  // for (size_t i = cursor; i < cursor+10; i++) {
  //   printf("%zu ", order_indices.elems[i]);
  // }
  // printf("\n");
  // return 0;

  Byte_Buffer prompt = from_cstring(&mallocator.alloc, "What is data science?");
  bool train = false;
  bool gen_text = false;

  Byte_Array buf = {0};
  for (int i = 1; i < argc; i++) {
    if (strncmp(argv[i], "--train", 7) == 0)
      train = true;

    if (strncmp(argv[i], "--epoch=", 8) == 0) {
      sscanf(argv[i], "--epoch=%zu", &epoch);
    }

    if (strncmp(argv[i], "--prompt", 8) == 0) {
      FREE(&mallocator.alloc, prompt);
      buf = read_input(&mallocator.alloc, stdin);
      prompt = byte_buffer_from_parts(buf.elems, buf.count);
      gen_text = true;
    }
  }

  if (prompt.len == 0) return -1;

  size_t prompt_len = prompt.len;
  while (prompt_len > 0 && prompt.cptr[prompt_len - 1] == '\n')
    prompt_len -= 1;

  Byte_Buffer prompt_fmt = NULL_BYTE_BUFFER;
  int nbytes = -1;
  for (int i = 0; i < 2; i++) {
    nbytes = snprintf(prompt_fmt.cptr, nbytes+1, "User: %.*s\nAssistant: ", (int)prompt_len, prompt.cptr);
    if (nbytes < 0 || prompt_fmt.cptr != NULL) break;
    prompt_fmt = ALLOC(&mallocator.alloc, nbytes);
  }
  array_destroy(&buf);

  if ((loaded && !train) || gen_text) {
    goto generate_text;
  }

  init_opencl();
  ensure_buffer_size(sizeof(float)*MAX_VOCAB*MAX_VOCAB);

  mat_zero(mat_row(trans_in.tok_emb.value, 0));
  print_tokens(&trans_in);

  set_color(0, 255, 0);
  print_timestamp(stdout);
  printf(" Start training\n");
  rst_color();

  optimizer.learning_rate = cosine_learning_rate(
      optimizer_steps,
      warmup_steps,
      total_steps,
      peak_lr,
      peak_lr / 10.0f
  );

  {
    size_t saved = SAVE(&arena.alloc);
    set_dropout(&trans_in, 0.0f);
    generate_text(&arena, &trans_in, prompt_fmt, &tokens, &words, &buff, merge_to, 100);
    RESTORE(&arena.alloc, saved);
  }

  float win_loss = 0;
  size_t win_tok = 0;

  while (optimizer_steps < total_steps) {
    float loss_sum = 0;
    size_t tokens_scored = 0;
    size_t tokens_count = 0;

    set_dropout(&trans_in, dropout_pct);

    while (tokens_scored < target_scored) {
      if (cursor == order_indices.count) {
        shuffle_indices(order_indices);
        cursor = 0;
        epoch += 1;

        save_model(&trans_in, model_name_bin);
        save_optimizer(&optimizer, model_name_par, epoch, optimizer_steps, order_indices, cursor);

        size_t saved = SAVE(&arena.alloc);
        set_dropout(&trans_in, 0.0f);
        generate_text(&arena, &trans_in, prompt_fmt, &tokens, &words, &buff, merge_to, 100);
        RESTORE(&arena.alloc, saved);
      }

      int prompt_len = 0;
      int total_len = 0;
      const int32_t *ids = example(&corpus, order_indices.elems[cursor], &prompt_len, &total_len);
      cursor += 1;

      const int first = prompt_len - 1;             /* first loss position */
      const int npos  = total_len - prompt_len;     /* how many */

      if (npos <= 0)
        continue;

      tokens.count = 0;
      targets.count = 0;

      for (int c = 0; c < total_len - 1; c++) {
        array_append(&tokens, ids[c]);
        array_append(&targets, ids[c+1]);
      }

      size_t N = tokens.count;

      size_t saved = SAVE(&arena.alloc);
      {
        Transformer_Output trans_out = {0};
        init_transformer_output(
            .arena = &arena,
            .trans_out = &trans_out,
            .num_blocks = trans_in.num_blocks,
            .vocab_size = trans_in.vocab_size,
            .heads_count = trans_in.heads_count,
            .emb_size = trans_in.emb_size,
            .ff_size = trans_in.ff_size,
            .sequence_size = N,
        );

        transformer_forward(tokens, &trans_out, &trans_in, 1.0f);

        if (optimizer_steps == 0) {
          set_color(0, 0, 255);
          // σ = √(2/(V+D))
          // std(logits) = σ · √D
          float std = sqrtf(2.0f / (D+MAX_VOCAB));

          printf("std(logits) = %.4f  (target ~%.4f)\n",
              logits_std(trans_out.logits.value), std*sqrtf(D));
          rst_color();

          NMatrix x = trans_out.ln.out.value;     /* whatever the head reads */
          double n2 = 0.0;
          for (uint32_t d = 0; d < x.cols; d++) {
              double t = MAT_AT(x, 0, d);
              n2 += t * t;
          }
          printf("||x[0]|| = %.4f  (expect %.4f)   gamma[0]=%.3f beta[0]=%.3f\n",
                 sqrt(n2), sqrt((double)x.cols),
                 VEC_AT(trans_in.ln.gamma.value, 0), VEC_AT(trans_in.ln.beta.value, 0));        
        }

        if (optimizer_steps == 0) {
            NMatrix z = trans_out.logits.value;
            int zero_rows = 0;
            double lo = 1e9, hi = -1e9;
            for (uint32_t i = 0; i < z.rows; i++) {
                double mean = 0.0;
                for (uint32_t d = 0; d < z.cols; d++) mean += MAT_AT(z, i, d);
                mean /= z.cols;
                double var = 0.0;
                for (uint32_t d = 0; d < z.cols; d++) {
                    double t = MAT_AT(z, i, d) - mean; var += t * t;
                }
                double s = sqrt(var / z.cols);
                if (s < 0.01) zero_rows++;
                if (s < lo) lo = s;
                if (s > hi) hi = s;
            }
            printf("N=%zu rows=%u  row_std range [%.4f, %.4f]  zero_rows=%d\n",
                   N, z.rows, lo, hi, zero_rows);
        }

        if (optimizer_steps == 0) {
            NMatrix x = trans_out.ln.out.value;
            uint32_t rows[3] = { 0, x.rows/2, x.rows-1 };
            for (int k = 0; k < 3; k++) {
                double n2 = 0.0;
                for (uint32_t d = 0; d < x.cols; d++) {
                    double t = MAT_AT(x, rows[k], d); n2 += t*t;
                }
                printf("||x[%u]|| = %.4f\n", rows[k], sqrt(n2));
            }
        }

        // dLoss/dlogits = softmax(logits) - onehot(target).
        NMatrix dlogits = trans_out.logits.grad;
        mat_copy(dlogits, trans_out.probs);

        tokens_count += total_len;
        tokens_scored += npos;

        for (size_t i = 0; i < N; i++) {
          int32_t target = targets.elems[i];

          if (i >= (size_t)first) {
            MAT_AT(dlogits, i, target) -= 1.0f;
            loss_sum -= logf(MAT_AT(trans_out.probs, i, target) + 1e-10f);
          } else {
            mat_zero(mat_row(dlogits, i));
          }
        }

        transformer_backward(tokens, &trans_out, &trans_in);
      }
      RESTORE(&arena.alloc, saved);
    }

    update_grads_adam(&optimizer, 1.0f / (float)tokens_scored);

    // cosine annealing
    optimizer.learning_rate = cosine_learning_rate(
        optimizer_steps,
        warmup_steps,
        total_steps,
        peak_lr,
        peak_lr / 10.0f
    );

    win_loss += loss_sum;
    win_tok += tokens_scored;

    float loss = loss_sum / tokens_scored;

    set_color(0, 255, 0);
    print_timestamp(stdout);
    printf(" epoch=%zu/%zu step=%zu/%zu cursor=%zu lr=%f loss=%f perp=%f tokens=%zu scored=%zu (%.2f%% scored)\n",
        epoch+1, total_epochs, optimizer_steps+1, total_steps, cursor, optimizer.learning_rate, loss, expf(loss),
        tokens_count, tokens_scored, (float)tokens_scored/(float)tokens_count * 100.0f);
    rst_color();

    optimizer_steps += 1;

    if (optimizer_steps % 50 == 0) {
      save_model(&trans_in, model_name_bin);
      save_optimizer(&optimizer, model_name_par, epoch, optimizer_steps, order_indices, cursor);

      size_t saved = SAVE(&arena.alloc);
      set_dropout(&trans_in, 0.0f);
      generate_text(&arena, &trans_in, prompt_fmt, &tokens, &words, &buff, merge_to, 100);
      RESTORE(&arena.alloc, saved);
    }

    if (optimizer_steps % 1000 == 0) {
      float dev = evaluate(&arena, &trans_in, &corpus_dev, corpus_dev.n_recs);
      float train = win_loss / (float)win_tok;
      win_loss = 0;
      win_tok = 0;

      set_color(66, 140, 211);
      printf("  eval opt=%zu  train=%.4f  dev=%.4f  gap=%+.4f\n",
              optimizer_steps, train, dev, dev - train);
      rst_color();
    }
  }

  save_model(&trans_in, model_name_bin);
  save_optimizer(&optimizer, model_name_par, epoch, optimizer_steps, order_indices, cursor);

generate_text:
  generate_text(&arena, &trans_in, prompt_fmt, &tokens, &words, &buff, merge_to, 2048);

  return 0;
}

