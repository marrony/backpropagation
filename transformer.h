#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include "array.h"
#include "nn.h"
#include "allocator.h"
#include "tokenizer.h"
#include <math.h>
#include <stdint.h>

typedef struct {
  NMatrix* k;
  NMatrix* v;
  size_t len;
} KVCache;

typedef struct {
  Tensor gamma;
  Tensor beta;
} Layer_Norm;

typedef struct {
  Tensor  out;      // [NxD]
  NMatrix mean;     // [1xN]
  NMatrix rstd;     // [1xN]
  NMatrix xhat;     // [NxD]
} Layer_Norm_Output;

typedef struct {
  Tensor weight;      // [DxD]
  Tensor bias;        // [1xD]
} Linear_Layer;

typedef struct {
  Linear_Layer Q;       // [DxD]
  Linear_Layer K;       // [DxD]
  Linear_Layer V;       // [DxD]
  Linear_Layer O;       // [DxD]
  //float p;
} Attention;

typedef struct {
  size_t heads_count;
  NMatrix Q;             // [NxD]
  NMatrix dQ;            // [NxD]
  NMatrix dK;            // [NxD]
  NMatrix dV;            // [NxD]
  NMatrix vals;          // [NxD]
  NMatrix dvals;         // [NxD]
  NMatrix out;           // [NxD]
} Attention_Output;

typedef struct {
  Layer_Norm ln1;       // [1xD]
  Attention attn;
  Layer_Norm ln2;       // [1xD]
  Linear_Layer ff1;     // [DxF]
  Linear_Layer ff2;     // [FxD]
} Block;

typedef struct {
  Layer_Norm_Output ln1; // [NxD]
  Attention_Output attn;
  NMatrix x1;            // [NxD]
  Layer_Norm_Output ln2; // [NxD]
  Tensor ff1_out;        // [NxF]
  Tensor  relu_out;      // [NxF]
  NMatrix ff2_out;       // [NxD]
  Tensor out;            // [NxD]
} Block_Output;

struct Layer_Norm_Forward_Opts {
  Tensor in;              // [NxD]
  Layer_Norm* ln_in;
  Layer_Norm_Output* ln_out;
};

struct Layer_Norm_Backward_Opts {
  Tensor in;              // [NxD]
  Layer_Norm* ln_in;
  Layer_Norm_Output* ln_out;
};

#define layer_norm_forward(...) \
  layer_norm_forward_opts((struct Layer_Norm_Forward_Opts){ __VA_ARGS__ })

#define layer_norm_backward(...) \
  layer_norm_backward_opts((struct Layer_Norm_Backward_Opts){ __VA_ARGS__ })

// ensure means=0 and variance=1 over D components
// Σ_d x_d² = D
// ‖x‖ = √D
void layer_norm_forward_opts(struct Layer_Norm_Forward_Opts opts) {
  NMatrix gamma = opts.ln_in->gamma.value;
  NMatrix beta = opts.ln_in->beta.value;
  NMatrix out = opts.ln_out->out.value;
  NMatrix mean = opts.ln_out->mean;
  NMatrix rstd = opts.ln_out->rstd;
  NMatrix xhat = opts.ln_out->xhat;
  NMatrix in = opts.in.value;

  for (uint32_t i = 0; i < out.rows; i++) { // 0..N
    float m = 0;
    for (uint32_t d = 0; d < out.cols; d++) // 0..D
      m += MAT_AT(in, i, d);
    m /= out.cols;
    VEC_AT(mean, i) = m;

    float v = 0;
    for (uint32_t d = 0; d < out.cols; d++) { // 0..D
      float diff = MAT_AT(in, i, d) - m;
      v += diff * diff;
    }
    v /= out.cols;

    float inv_std = 1.0f / sqrtf(v + 1e-5f);
    VEC_AT(rstd, i) = inv_std;

    for (uint32_t d = 0; d < out.cols; d++) { // 0..D
      float hat = (MAT_AT(in, i, d) - m) * inv_std;
      MAT_AT(xhat, i, d) = hat;
      MAT_AT(out, i, d) = VEC_AT(gamma, d) * hat + VEC_AT(beta, d);
    }
  }
}

void layer_norm_backward_opts(struct Layer_Norm_Backward_Opts opts) {
  NMatrix gamma = opts.ln_in->gamma.value;
  NMatrix dgamma = opts.ln_in->gamma.grad;
  NMatrix dbeta = opts.ln_in->beta.grad;
  NMatrix rstd = opts.ln_out->rstd;
  NMatrix xhat = opts.ln_out->xhat;
  NMatrix dout = opts.ln_out->out.grad;
  NMatrix din = opts.in.grad;

  for (uint32_t i = 0; i < dout.rows; i++) { // 0..N
    float inv_std = VEC_AT(rstd, i);

    float sum_dx_hat = 0;
    float sum_dx_hat_x_hat = 0;
    for (uint32_t d = 0; d < dout.cols; d++) { // 0..D
      float dx_hat = MAT_AT(dout, i, d) * VEC_AT(gamma, d);
      sum_dx_hat += dx_hat;
      sum_dx_hat_x_hat += dx_hat * MAT_AT(xhat, i, d);
    }

    for (uint32_t d = 0; d < dout.cols; d++) { // 0..D
      float dx_hat = MAT_AT(dout, i, d) * VEC_AT(gamma, d);
      float dx_ = dx_hat - sum_dx_hat/dout.cols - MAT_AT(xhat, i, d) * sum_dx_hat_x_hat/dout.cols;
      MAT_AT(din, i, d) += dx_ * inv_std;
    }
  }

  for (uint32_t i = 0; i < dout.rows; i++) {    // 0..N
    for (uint32_t d = 0; d < dout.cols; d++) {  // 0..D
      VEC_AT(dgamma, d) += MAT_AT(dout, i, d) * MAT_AT(xhat, i, d);
      VEC_AT(dbeta, d) += MAT_AT(dout, i, d);
    }
  }
}

// Glorot's target:
//
// σ² = 2/f
//
// Normal = N(0, σ²)
// σ² = 2/f
// σ = √(2/f)
//
// Uniform = U(-a, +a)
// w = 2a
//
// σ² = w²/12 = (2a)²/12 = a²/3
// 
// a²/3 = 2/f
// a² = 6/f
// a = √(6/f)
float xavier_glorot_limit(size_t fan_in, size_t fan_out) {
  return sqrtf(6.0f / (float)(fan_in + fan_out));
}

void init_xavier_glorot(NMatrix mat, size_t fan_in, size_t fan_out) {
  float std = xavier_glorot_limit(fan_in, fan_out);
  mat_rand_uniform(mat, -std, +std);
}

void init_transformer_output_projections(NMatrix mat, size_t fan_in, size_t fan_out, size_t num_layers) {
  float std = xavier_glorot_limit(fan_in, fan_out);
  float depth_scale = 1.0f / sqrtf(2.0f * (float)num_layers);
  float final_std = std * depth_scale;

  mat_rand_uniform(mat, -final_std, +final_std);
}

void alloc_tensor(Allocator* alloc, Tensor* tensor, size_t N, size_t D) {
  tensor->value = mat_alloc2(alloc, N, D);
  tensor->grad = mat_alloc2(alloc, N, D);

  mat_zero(tensor->value);
  mat_zero(tensor->grad);
}

void init_linear_layer(Allocator* alloc, Linear_Layer* linear_layer, size_t N, size_t D) {
  alloc_tensor(alloc, &linear_layer->weight, N, D);
  alloc_tensor(alloc, &linear_layer->bias, 1, D);
  init_xavier_glorot(linear_layer->weight.value, N, D);
  mat_zero(linear_layer->bias.value);
}

// attn_in->O.weight (Attention Output projection)
// block_in->ff2.weight (Second Feed-Forward projection)
void init_output_linear_layer(Allocator* alloc, Linear_Layer* linear_layer, size_t N, size_t D, size_t num_layers) {
  alloc_tensor(alloc, &linear_layer->weight, N, D);
  alloc_tensor(alloc, &linear_layer->bias, 1, D);
  init_transformer_output_projections(linear_layer->weight.value, N, D, num_layers);
  mat_zero(linear_layer->bias.value);
}

void init_gamma_beta(Allocator* alloc, Tensor* gamma, Tensor* beta, size_t D) {
  alloc_tensor(alloc, gamma, 1, D);
  alloc_tensor(alloc, beta, 1, D);
  mat_fill(gamma->value, 1);
  mat_zero(beta->value);
}

void init_block(Allocator* alloc, Block* block, size_t D, size_t F, size_t num_layers) {
  init_gamma_beta(alloc, &block->ln1.gamma, &block->ln1.beta, D);
  init_linear_layer(alloc, &block->attn.Q, D, D);
  init_linear_layer(alloc, &block->attn.K, D, D);
  init_linear_layer(alloc, &block->attn.V, D, D);
  init_output_linear_layer(alloc, &block->attn.O, D, D, num_layers);
  init_gamma_beta(alloc, &block->ln2.gamma, &block->ln2.beta, D);
  init_linear_layer(alloc, &block->ff1, D, F);
  init_output_linear_layer(alloc, &block->ff2, F, D, num_layers);
}

void init_block_output(Arena_Allocator* arena, Block_Output* block_out, size_t N, size_t D, size_t H, size_t F) {
  alloc_tensor(&arena->alloc, &block_out->ln1.out, N, D);
  block_out->ln1.mean     = mat_alloc2(&arena->alloc, 1, N);
  block_out->ln1.rstd     = mat_alloc2(&arena->alloc, 1, N);
  block_out->ln1.xhat     = mat_alloc2(&arena->alloc, N, D);
  mat_zero(block_out->ln1.mean);
  mat_zero(block_out->ln1.rstd);
  mat_zero(block_out->ln1.xhat);
  block_out->attn.heads_count = H;
  block_out->attn.Q  = mat_alloc2(&arena->alloc, N, D);
  block_out->attn.dQ = mat_alloc2(&arena->alloc, N, D);
  block_out->attn.dK = mat_alloc2(&arena->alloc, N, D);
  block_out->attn.dV = mat_alloc2(&arena->alloc, N, D);
  mat_zero(block_out->attn.Q);
  mat_zero(block_out->attn.dQ);
  mat_zero(block_out->attn.dK);
  mat_zero(block_out->attn.dV);
  block_out->attn.vals = mat_alloc2(&arena->alloc, N, D);
  block_out->attn.dvals = mat_alloc2(&arena->alloc, N, D);
  mat_zero(block_out->attn.vals);
  mat_zero(block_out->attn.dvals);
  block_out->attn.out     = mat_alloc2(&arena->alloc, N, D);
  block_out->x1           = mat_alloc2(&arena->alloc, N, D);
  mat_zero(block_out->attn.out);
  mat_zero(block_out->x1);
  alloc_tensor(&arena->alloc, &block_out->ln2.out, N, D);
  block_out->ln2.mean     = mat_alloc2(&arena->alloc, 1, N);
  block_out->ln2.rstd     = mat_alloc2(&arena->alloc, 1, N);
  block_out->ln2.xhat     = mat_alloc2(&arena->alloc, N, D);
  mat_zero(block_out->ln2.mean);
  mat_zero(block_out->ln2.rstd);
  mat_zero(block_out->ln2.xhat);
  alloc_tensor(&arena->alloc, &block_out->ff1_out, N, F);
  alloc_tensor(&arena->alloc, &block_out->relu_out, N, F);
  block_out->ff2_out      = mat_alloc2(&arena->alloc, N, D);
  mat_zero(block_out->ff2_out);
  alloc_tensor(&arena->alloc, &block_out->out, N, D);
}

struct Project_Forward_Opts {
  NMatrix out;
  Tensor x;
  Tensor W;
  Tensor b;
  bool transpose_w;
};

struct Project_Backward_Opts {
  Tensor x;
  Tensor W;
  Tensor b;
  NMatrix dout;
  bool transpose_w;
};

#define project(...) project_opts((struct Project_Forward_Opts){ __VA_ARGS__ })
#define dproject(...) dproject_opts((struct Project_Backward_Opts){ __VA_ARGS__ })

void project_opts(struct Project_Forward_Opts opts) {
  // out = x*W + b
  NMatrix out = opts.out;
  NMatrix x = opts.x.value;
  NMatrix W = opts.W.value;
  NMatrix b = opts.b.value;

  if (opts.transpose_w) {
    // dst = x * W^T + b
    ASSERT_MATRIX_MULT(out.rows, out.cols, x.rows, x.cols, W.cols, W.rows);
    gemm_nt(out, x, W, b.elems, false);
  } else {
    // dst = x * W + b
    ASSERT_MATRIX_MULT(out.rows, out.cols, x.rows, x.cols, W.rows, W.cols);
    gemm_nn(out, x, W, b.elems, false);
  }
}

void dproject_opts(struct Project_Backward_Opts opts) {
  NMatrix dout = opts.dout;

  // db += dout
  NMatrix db = opts.b.grad;
  for (uint32_t i = 0; i < dout.rows; i++)
    mat_add(db, db, mat_row(dout, i));

  if (opts.transpose_w) {
    // out = x*W^T + b
    //
    // dW^T += x^T * dout
    // dW += (x^T * dout)^T = dout^T * x
    NMatrix x = opts.x.value;
    NMatrix dW = opts.W.grad;
    mat_mult_A_transposed_and_B_acc(dW, dout, x);

    // dx += dout * (W^T)^T = dout * W
    NMatrix dx = opts.x.grad;
    NMatrix W = opts.W.value;
    mat_mult_acc(dx, dout, W);
  } else {
    // out = x*W + b

    // dW += x^T * dout
    NMatrix x = opts.x.value;
    NMatrix dW = opts.W.grad;
    mat_mult_A_transposed_and_B_acc(dW, x, dout);

    // dx += dout * W^T
    NMatrix dx = opts.x.grad;
    NMatrix W = opts.W.value;
    mat_mult_A_and_B_transposed_acc(dx, dout, W);
  }
}

// linear2(dropout(relu_out))
// layer_norm(x + dropout(ff2_out))
// layer_norm(x + dropout(attn_out))
// attn_weights = dropout(softmax(Q @ K.T / sqrt(d)))
// x0 = dropout(tok_emb)

void dropout_forward(NMatrix x, NMatrix mask, float p) {
  if (p <= 0.0f) return;

  float m = 1 / (1 - p);

  for (uint32_t i = 0; i < x.rows; i++) {
    for (uint32_t j = 0; j < x.cols; j++) {
      MAT_AT(mask, i, j) = rand_uniform() < p ? 0 : m;
      MAT_AT(x, i, j) *= MAT_AT(mask, i, j);
    }
  }
}

void dropout_backward(NMatrix dx, NMatrix mask, float p) {
  if (p <= 0.0f) return;

  for (uint32_t i = 0; i < dx.rows; i++) {
    for (uint32_t j = 0; j < dx.cols; j++) {
      float m = MAT_AT(mask, i, j);
      MAT_AT(dx, i, j) *= m;
    }
  }
}

// tensor = [NxD]
// d_head = D/H
//
// R(θ) = [ cos θ   −sin θ ]
//        [ sin θ    cos θ ]
//
// (R(mθ)·q)ᵀ (R(nθ)·k)  =  qᵀ · R(mθ)ᵀ · R(nθ) · k
//                       =  qᵀ · R(−mθ) · R(nθ) · k
//                       =  qᵀ · R((n−m)θ) · k
void rope(NMatrix tensor, size_t base, size_t d_head) {
  // // ln(10000.0)
  // const double log_base = 9.210340371976184;
  //
  // InvFreq inv_freq;
  // for (size_t i = 0; i < d_head/2; i++) {
  //   inv_freq.v[i] = (float)exp(-2.0 * i / (double)d_head * log_base);
  // }

  const float log_base = 9.21034037197f;
  float inv_d = -1.0f / (float)d_head;

  for (uint32_t m = 0; m < tensor.rows; m++) {
    for (uint32_t i = 0; i < tensor.cols; i += 2) {
      size_t local_i = i % d_head;
      // float theta = inv_freq.v[local_i >> 1];
      float theta = expf((float)local_i * inv_d * log_base);
      float phi = (float)(base+m) * theta;

      float p0 = MAT_AT(tensor, m, i+0);
      float p1 = MAT_AT(tensor, m, i+1);

      float cos_phi = cosf(phi);
      float sin_phi = sinf(phi);

      MAT_AT(tensor, m, i+0) = p0*cos_phi - p1*sin_phi;
      MAT_AT(tensor, m, i+1) = p0*sin_phi + p1*cos_phi;
    }
  }
}

// tensor = [NxD]
// d_head = D/H
// dout = [NxD]
void drope(NMatrix dtensor, size_t base, size_t d_head, NMatrix dout) {
  // // ln(10000.0)
  // const double log_base = 9.210340371976184;
  //
  // InvFreq inv_freq;
  // for (size_t i = 0; i < d_head/2; i++) {
  //   inv_freq.v[i] = (float)exp(-2.0 * i / (double)d_head * log_base);
  // }

  const float log_base = 9.21034037197f;
  float inv_d = -1.0f / (float)d_head;

  for (uint32_t m = 0; m < dout.rows; m++) {
    for (uint32_t i = 0; i < dout.cols; i += 2) {
      size_t local_i = i % d_head;
      // float theta = inv_freq.v[local_i >> 1];
      float theta = expf((float)local_i * inv_d * log_base);
      float phi = (float)(base+m) * theta;

      float dout0 = MAT_AT(dout, m, i+0);
      float dout1 = MAT_AT(dout, m, i+1);

      float cos_phi = cosf(phi);
      float sin_phi = sinf(phi);

      MAT_AT(dtensor, m, i+0) = dout0*cos_phi + dout1*sin_phi;
      MAT_AT(dtensor, m, i+1) = dout1*cos_phi - dout0*sin_phi;
    }
  }
}

struct Attention_Forward_Opts {
  Attention_Output* attn_out;
  Attention* attn_in;
  KVCache* kv_cache;
  size_t block;
  size_t base;
  size_t total;
  NMatrix scores;
  Tensor in_q;
  Tensor in_k;
  Tensor in_v;
};

struct Attention_Backward_Opts {
  Tensor in_q;
  Tensor in_k;
  Tensor in_v;
  Attention_Output* attn_out;
  Attention* attn_in;
  KVCache* kv_cache;
  size_t block;
  size_t total;
  NMatrix dout;
  NMatrix scores;
  NMatrix dscores;
};

#define attention_forward(...) attention_forward_opts((struct Attention_Forward_Opts){ __VA_ARGS__ })
#define attention_backward(...) attention_backward_opts((struct Attention_Backward_Opts){ __VA_ARGS__ })

void attention_forward_opts(struct Attention_Forward_Opts opts) {
  Attention_Output* attn_out = opts.attn_out;
  NMatrix scores = opts.scores;
  Attention* attn_in = opts.attn_in;
  Tensor in_q = opts.in_q;
  Tensor in_k = opts.in_k;
  Tensor in_v = opts.in_v;
  size_t base = opts.base;
  size_t block = opts.block;
  size_t total = opts.total;
  KVCache* cache = opts.kv_cache;

  size_t N = in_q.value.rows;
  int32_t D = in_q.value.cols;
  int32_t H = attn_out->heads_count;
  int32_t d_head = D / H;

  // Q = input*wQ + bQ
  project(
      .out = attn_out->Q,
      .x   = in_q,
      .W   = attn_in->Q.weight,
      .b   = attn_in->Q.bias,
  );

  NMatrix Kwrite = mat_rows(cache->k[block], cache->len, N);
  NMatrix Vwrite = mat_rows(cache->v[block], cache->len, N);

  NMatrix Qall = attn_out->Q;
  NMatrix Kall = mat_rows(cache->k[block], 0, total);
  NMatrix Vall = mat_rows(cache->v[block], 0, total);

  // K = input*wK + bK
  project(
      .out = Kwrite,
      .x   = in_k,
      .W   = attn_in->K.weight,
      .b   = attn_in->K.bias
  );

  // V = input*wV + bV
  project(
      .out = Vwrite,
      .x   = in_v,
      .W   = attn_in->V.weight,
      .b   = attn_in->V.bias
  );

  rope(Qall, base, d_head);
  rope(Kwrite, base, d_head);

  float scale = 1.0f / sqrtf(d_head);

  for (int32_t h = 0; h < H; h++) {      // 0..H
    int head_start = h*d_head;

    NMatrix Qh = mat_cols(Qall, head_start, d_head);
    NMatrix Kh = mat_cols(Kall, head_start, d_head);
    NMatrix Vh = mat_cols(Vall, head_start, d_head);
    NMatrix Ah = mat_cols(attn_out->vals, head_start, d_head);

    // scores = Q[h] * K[h]^T
    mat_mult_A_and_B_transposed(scores, Qh, Kh);

    // scores = scores / sqrt(d_head)
    mat_scale(scores, scores, scale);

    //    K    e    y
    // Q  s00  -inf -inf -inf -inf
    // u  s10  s11  -inf -inf -inf
    // e  s20  s21  s22  -inf -inf
    // r  s30  s31  s32  s33  -inf
    // y  s40  s41  s42  s43  s44
    for (size_t i = 0; i < N; i++) {    // 0..N
      for (size_t j = base+i+1; j < total; j++) {  // i..N
        MAT_AT(scores, i, j) = -INFINITY;
      }
    }

    // scores = softmax(scores)
    softmax_by_row(
        .out = scores,
        .x = scores,
        .temperature = 1.0f,
    );

    // A[h] = scores * V[h]
    mat_mult(Ah, scores, Vh);
  }

  // out = A*wO + bO
  project(
      .out = attn_out->out,
      .x   = tensor(attn_out->vals, attn_out->dvals),
      .W   = attn_in->O.weight,
      .b   = attn_in->O.bias,
  );
}

void attention_backward_opts(struct Attention_Backward_Opts opts) {
  Tensor in_q = opts.in_q;
  Tensor in_k = opts.in_k;
  Tensor in_v = opts.in_v;
  Attention_Output* attn_out = opts.attn_out;
  Attention* attn_in = opts.attn_in;
  NMatrix dout = opts.dout;
  NMatrix scores = opts.scores;
  NMatrix dscores = opts.dscores;
  KVCache* kv_cache = opts.kv_cache;
  size_t block = opts.block;
  size_t total = opts.total;
  size_t base = 0;

  size_t N = in_q.value.rows;
  size_t D = in_q.value.cols;
  size_t H = attn_out->heads_count;
  size_t d_head = D / H;

  assert((size_t)N == total);

  NMatrix Qall = attn_out->Q;
  NMatrix Kall = mat_rows(kv_cache->k[block], 0, total);
  NMatrix Vall = mat_rows(kv_cache->v[block], 0, total);

  // attn_out = attn_vals*wO + bO
  dproject(
      .x    = tensor(attn_out->vals, attn_out->dvals),
      .W    = attn_in->O.weight,
      .b    = attn_in->O.bias,
      .dout = dout,
  );

  float scale = 1.0f / sqrtf(d_head);

  for (size_t h = 0; h < H; h++) {      // 0..H
    int head_start = h*d_head;

    NMatrix Qh = mat_cols(Qall, head_start, d_head);
    NMatrix Kh = mat_cols(Kall, head_start, d_head);
    NMatrix Vh = mat_cols(Vall, head_start, d_head);

    NMatrix dQh = mat_cols(attn_out->dQ, head_start, d_head);
    NMatrix dKh = mat_cols(attn_out->dK, head_start, d_head);
    NMatrix dVh = mat_cols(attn_out->dV, head_start, d_head);

    NMatrix dAh = mat_cols(attn_out->dvals, head_start, d_head);

///////////////
    // scores = Q[h] * K[h]^T
    mat_mult_A_and_B_transposed(scores, Qh, Kh);

    // scores = scores / sqrt(d_head)
    mat_scale(scores, scores, scale);

    //    K    e    y
    // Q  s00  -inf -inf -inf -inf
    // u  s10  s11  -inf -inf -inf
    // e  s20  s21  s22  -inf -inf
    // r  s30  s31  s32  s33  -inf
    // y  s40  s41  s42  s43  s44
    for (size_t i = 0; i < N; i++) {    // 0..N
      for (size_t j = base+i+1; j < total; j++) {  // i..N
        MAT_AT(scores, i, j) = -INFINITY;
      }
    }

    // scores = softmax(scores)
    softmax_by_row(
        .out = scores,
        .x = scores,
        .temperature = 1.0f,
    );
///////////////

    // A[h] = scores * V[h]
    //
    // dAw[h] = dA[h] * V[h]^T
    // dV[h] = scores^T * dA[h]
    mat_mult_A_and_B_transposed(dscores, dAh, Vh);
    mat_mult_A_transposed_and_B_acc(dVh, scores, dAh);

    // attn_weights = softmax(scores)
    //
    // dscores = dsoftmax(attn_weights, dweights)
    dsoftmax_by_row(
        .dx = dscores,
        .y = scores,
        .dy = dscores,
        .temperature = 1.0f,
    );

    for (size_t i = 0; i < N; i++) {
      for (size_t j = i+1; j < N; j++) {
        MAT_AT(dscores, i, j) = 0.0f; 
      }
    }

    mat_scale(dscores, dscores, scale);

    // scores = Q[h] * K[h]^T
    //
    // dQ[h] = dScores * K[h]
    // dK[h] = dScore^T * Q[h]

    mat_mult_acc(dQh, dscores, Kh);
    mat_mult_A_transposed_and_B_acc(dKh, dscores, Qh);
  }

  drope(attn_out->dQ, 0, d_head, attn_out->dQ);
  drope(attn_out->dK, 0, d_head, attn_out->dK);

  // V = input*wV + bV
  dproject(
      .x    = in_v,
      .W    = attn_in->V.weight,
      .b    = attn_in->V.bias,
      .dout = attn_out->dV,
  );

  // K = input*wK + bK
  dproject(
      .x    = in_k,
      .W    = attn_in->K.weight,
      .b    = attn_in->K.bias,
      .dout = attn_out->dK,
  );

  // Q = input*wQ + bQ
  dproject(
      .x    = in_q,
      .W    = attn_in->Q.weight,
      .b    = attn_in->Q.bias,
      .dout = attn_out->dQ,
  );
}

struct Feed_Forward_Opts {
  Block* in;
  Block_Output* out;
};

struct Feed_Backward_Opts {
  Block* in;
  Block_Output* out;
  NMatrix dout;
};

#define feed_forward(...) feed_forward_opts((struct Feed_Forward_Opts){ __VA_ARGS__ })
#define feed_backward(...) feed_backward_opts((struct Feed_Backward_Opts){ __VA_ARGS__ })

void feed_forward_opts(struct Feed_Forward_Opts opts) {
  Block_Output* block_out = opts.out;
  Block* block_in = opts.in;

  // ff1_out = ln2_out*ff1W + ff1b
  project(
      .out = block_out->ff1_out.value,
      .x   = block_out->ln2.out,
      .W   = block_in->ff1.weight,
      .b   = block_in->ff1.bias,
  );

  // relu_out = relu(ff1_out)
  relu(block_out->relu_out.value, block_out->ff1_out.value, 0.0f);

  // mask = dropout(relu_out)

  // ff2_out = relu_out*ff2W + ff2b
  project(
      .out = block_out->ff2_out,
      .x   = block_out->relu_out,
      .W   = block_in->ff2.weight,
      .b   = block_in->ff2.bias,
  );
}

void feed_backward_opts(struct Feed_Backward_Opts opts) {
  Block_Output* block_out = opts.out;
  Block* block_in = opts.in;
  NMatrix dout = opts.dout;

  // ff2_out = project(relu_out, ff2W, ff2b)
  // ff2_out = relu_out*ff2W + ff2b
  //
  // dff2b     += dff2_out
  // dff2W     += relu_out^T * dff2_out
  // drelu_out += dff2_out * ff2W^T
  dproject(
      .x    = block_out->relu_out,
      .W    = block_in->ff2.weight,
      .b    = block_in->ff2.bias,
      .dout = dout,
  );

  // relu_out = relu(ff1_out)
  //
  // dff1_out = drelu(ff1_out)
  drelu(block_out->ff1_out.grad, block_out->ff1_out.value, block_out->relu_out.grad, 0.0f);

  // ff1_out = project(ln2_out, ff1W, ff1b)
  // ff1_out = ln2_out*ff1W + ff1b
  //
  // dff1b    += dff1_out
  // dff1W    += ln2_out^T * dff1_out
  // dln2_out += dff1_out * dff1W^T
  dproject(
      .x    = block_out->ln2.out,
      .W    = block_in->ff1.weight,
      .b    = block_in->ff1.bias,
      .dout = block_out->ff1_out.grad,
  );
}

struct Block_Forward_Opts {
  Block_Output* block_out;
  Block* block_in;
  KVCache* kv_cache;
  size_t block;
  size_t base;
  size_t total;
  Tensor in;
  NMatrix scores;
};

struct Block_Backward_Opts {
  Block* block_in;
  Block_Output* block_out;
  KVCache* kv_cache;
  size_t block;
  size_t total;
  Tensor in;
  NMatrix dout;
  NMatrix scores;
  NMatrix dscores;
};

#define block_forward(...) block_forward_opts((struct Block_Forward_Opts){ __VA_ARGS__ })
#define block_backward(...) block_backward_opts((struct Block_Backward_Opts){ __VA_ARGS__ })

// x = x + attn(ln1(x))
// x = x + ff(ln2(x))
//
// ln1_out  = norm(x0)
// attn_out = attention(ln1_out)
// x1       = x0 + attn_out
// ln2_out  = norm(x1)
// ff1_out  = project(ln2_out, ff1W, ff1b)
// relu_out = relu(ff1_out)
// ff2_out  = project(relu_out, ff2W, ff2b)
// out      = x1 + ff2_out
void block_forward_opts(struct Block_Forward_Opts opts) {
  Block_Output* block_out = opts.block_out;
  Block* block_in = opts.block_in;
  Tensor x0 = opts.in;
  NMatrix scores = opts.scores;

  ////////////////////////////////////////////////////
  // x1 = x0 + attention(norm(x0))
  ////////////////////////////////////////////////////

  // ln1_out = norm(x0)
  layer_norm_forward(
      .ln_out = &block_out->ln1,
      .ln_in  = &block_in->ln1,
      .in     = x0,
  );

  // attn_out = attention(ln1_out)
  attention_forward(
      .kv_cache = opts.kv_cache,
      .block    = opts.block,
      .base     = opts.base,
      .total    = opts.total,
      .attn_out = &block_out->attn,
      .attn_in  = &block_in->attn,
      .scores   = scores,
      .in_q     = block_out->ln1.out,
      .in_k     = block_out->ln1.out,
      .in_v     = block_out->ln1.out,
  );

  // x1 = x0 + attn_out
  mat_add(block_out->x1, x0.value, block_out->attn.out);

  ////////////////////////////////////////////////////
  // x2 = x1 + feed_forward(norm(x1))
  ////////////////////////////////////////////////////

  // ln2_out = norm(x1)
  layer_norm_forward(
      .ln_out = &block_out->ln2,
      .ln_in  = &block_in->ln2,
      .in     = tensor(block_out->x1, NULL_MATRIX),
  );

  // ff2_out = feed_forward(ln2_out)
  feed_forward(
      .in  = block_in,
      .out = block_out,
  );

  // mask = dropout(ff2_out)

  // out = x1 + ff2_out
  mat_add(block_out->out.value, block_out->x1, block_out->ff2_out);
}

// forward:
// ln1_out  = norm(x0)
// attn_out = attention(ln1_out)
// x1       = x0 + attn_out
// ln2_out  = norm(x1)
// ff1_out  = project(ln2_out, ff1W, ff1b)
// relu_out = relu(ff1_out)
// ff2_out  = project(relu_out, ff2W, ff2b)
// out      = x1 + ff2_out
//
// backward:
// dff2_out  = dout                            |
// drelu_out = dff2_out * ff2W^T               | drelu_out = dproject(ff2_out, dout)
// dff1_out  = drelu(ff1_out, drelu_out)       | dff1_out  = drelu(ff1_out, drelu_out)
// dln2_out  = dff1_out * dff1W^T              | dln2_out  = dproject(ln2_out, dff1_out)
// dx1       = dout                            | din       += dout
// dx1       += dnorm(x1, dln2_out)            | din       += dnorm(x1, dln2_out)
// dattn_out = dx1                             |
// dln1_out  = dattention(ln1_out, dattn_out)  | dln1_out  = dattention(ln1_out, din)
// din       = dx1                             |
// din       += dnorm(x0, dln1_out)            | din       += dnorm(x0, dln1_out)
void block_backward_opts(struct Block_Backward_Opts opts) {
  Tensor x0 = opts.in;
  NMatrix dout = opts.dout;
  NMatrix scores = opts.scores;
  NMatrix dscores = opts.dscores;
  Block* block_in = opts.block_in;
  Block_Output* block_out = opts.block_out;

  // ff2_out = feed_forward(ln2_out)
  //
  // dln2_out = feed_backward(ln2_out)
  feed_backward(
      .in   = block_in,
      .out  = block_out,
      .dout = dout,
  );

  // ln2_out = norm(x1)
  //
  // din += dout
  // din += dnorm(x1, dln2_out)
  mat_add(x0.grad, x0.grad, dout);

  layer_norm_backward(
      .in     = x0,
      .ln_in  = &block_in->ln2,
      .ln_out = &block_out->ln2,
  );

  // mask = dropout(attn_out)
  //
  // dattn_out = ddropout(attn_out)
  // dropout_backward(x0.grad, block_out->attn.mask, block_in->attn.p);

  // attn_out = attention(ln1_out)
  //
  // dln1_out = dattention(ln1_out, din)
  attention_backward(
      .kv_cache = opts.kv_cache,
      .block    = opts.block,
      .total    = opts.total,
      .in_q     = block_out->ln1.out,
      .in_k     = block_out->ln1.out,
      .in_v     = block_out->ln1.out,
      .attn_out = &block_out->attn,
      .attn_in  = &block_in->attn,
      .scores   = scores,
      .dscores  = dscores,
      .dout     = x0.grad,
  );

  // ln1_out = norm(x0)
  //
  // din += dnorm(x0, dln1_out)
  layer_norm_backward(
      .in     = x0,
      .ln_in  = &block_in->ln1,
      .ln_out = &block_out->ln1,
  );
}

void init_kv_cache(
  Allocator* alloc,
  KVCache* kv_cache,
  size_t num_blocks,
  size_t cache_size,
  size_t emb_size
) {
  kv_cache->k = ALLOC(alloc, sizeof(NMatrix)*num_blocks).ptr;
  kv_cache->v = ALLOC(alloc, sizeof(NMatrix)*num_blocks).ptr;
  kv_cache->len = 0;

  for (size_t i = 0; i < num_blocks; i++) {
    kv_cache->k[i] = mat_alloc2(alloc, cache_size, emb_size);
    kv_cache->v[i] = mat_alloc2(alloc, cache_size, emb_size);
  }
}

typedef struct {
  Tensor tok_emb;
  Block* blocks;
  KVCache* kv_cache;
  size_t num_blocks;
  Layer_Norm ln;
  Linear_Layer H;

  // configs
  bool tie_embeddings;
  size_t vocab_size;
  size_t heads_count;
  size_t emb_size;
  size_t ff_size;
} Transformer;

typedef struct {
  Tensor x0;
  Block_Output* blocks;
  size_t num_blocks;
  Layer_Norm_Output ln;
  Tensor logits;
  NMatrix probs;
  NMatrix scores;
  NMatrix dscores;

  // configs
  size_t sequence_size;
} Transformer_Output;

struct Init_Transformer_Opts {
  Allocator* alloc;
  Transformer* trans;
  KVCache* kv_cache;
  bool tie_embeddings;
  size_t num_blocks;
  size_t vocab_size;
  size_t heads_count;
  size_t emb_size;
  size_t ff_size;
};

struct Init_Transformer_Output_Opts {
  Arena_Allocator* arena;
  Transformer_Output* trans_out;
  KVCache* kv_cache;
  size_t num_blocks;
  size_t sequence_size;
  size_t vocab_size;
  size_t emb_size;
  size_t ff_size;
  size_t heads_count;
};

#define init_transformer(...) init_transformer_opts((struct Init_Transformer_Opts){ __VA_ARGS__ })
#define init_transformer_output(...) init_transformer_output_opts((struct Init_Transformer_Output_Opts){ __VA_ARGS__ })

void init_transformer_opts(struct Init_Transformer_Opts opts) {
  Allocator* alloc = opts.alloc;
  Transformer* trans = opts.trans;
  size_t V = opts.vocab_size;
  size_t D = opts.emb_size;
  size_t F = opts.ff_size;

  trans->kv_cache = opts.kv_cache;
  trans->tie_embeddings = opts.tie_embeddings;
  trans->vocab_size = opts.vocab_size;
  trans->heads_count = opts.heads_count;
  trans->emb_size = opts.emb_size;
  trans->ff_size = opts.ff_size;

  alloc_tensor(alloc, &trans->tok_emb, V, D);
  init_xavier_glorot(trans->tok_emb.value, V, D);

  init_gamma_beta(alloc, &trans->ln.gamma, &trans->ln.beta, D);

  if (!trans->tie_embeddings) {
    alloc_tensor(alloc, &trans->H.weight, D, V);
    init_xavier_glorot(trans->H.weight.value, D, V);
  }

  alloc_tensor(alloc, &trans->H.bias, 1, V);
  mat_zero(trans->H.bias.value);

  trans->num_blocks = opts.num_blocks;
  trans->blocks = ALLOC(alloc, sizeof(Block)*trans->num_blocks).ptr;

  for (size_t i = 0; i < trans->num_blocks; i++) {
    init_block(alloc, &trans->blocks[i], D, F, trans->num_blocks);
  }
}

void init_transformer_output_opts(struct Init_Transformer_Output_Opts opts) {
  Arena_Allocator* arena = opts.arena;
  Transformer_Output* trans_out = opts.trans_out;
  size_t V = opts.vocab_size;
  size_t N = opts.sequence_size;
  size_t D = opts.emb_size;
  size_t H = opts.heads_count;
  size_t F = opts.ff_size;
  KVCache* kv_cache = opts.kv_cache;

  trans_out->sequence_size = opts.sequence_size;

  alloc_tensor(&arena->alloc, &trans_out->x0, N, D);

  trans_out->num_blocks = opts.num_blocks;
  trans_out->blocks = ALLOC(&arena->alloc, sizeof(Block_Output)*trans_out->num_blocks).ptr;

  for (size_t i = 0; i < trans_out->num_blocks; i++) {
    init_block_output(arena, &trans_out->blocks[i], N, D, H, F);
  }

  alloc_tensor(&arena->alloc, &trans_out->ln.out, N, D);
  trans_out->ln.mean     = mat_alloc2(&arena->alloc, 1, N);
  trans_out->ln.rstd     = mat_alloc2(&arena->alloc, 1, N);
  trans_out->ln.xhat     = mat_alloc2(&arena->alloc, N, D);
  mat_zero(trans_out->ln.mean);
  mat_zero(trans_out->ln.rstd);
  mat_zero(trans_out->ln.xhat);

  alloc_tensor(&arena->alloc, &trans_out->logits, N, V);
  trans_out->probs = mat_alloc2(&arena->alloc, N, V);
  trans_out->scores = mat_alloc2(&arena->alloc, N, N + kv_cache->len);
  trans_out->dscores = mat_alloc2(&arena->alloc, N, N + kv_cache->len);
  mat_zero(trans_out->probs);
  mat_zero(trans_out->scores);
  mat_zero(trans_out->dscores);
}

void transformer_forward(
  TokenID_Array tokens,
  Transformer_Output* out,
  Transformer* in,
  float temperature
) {
  size_t N = tokens.count;

  assert(N == out->sequence_size);

  // x0 = tok_embs
  for (size_t i = 0; i < N; i++) {
    mat_copy(
        mat_row(out->x0.value, i),
        mat_row(in->tok_emb.value, tokens.elems[i])
    );
  }

  Tensor x = out->x0;

  size_t base = in->kv_cache->len;
  size_t total = base + N;

  // x[i+1] = block(x[i])
  for (size_t i = 0; i < in->num_blocks; i++) {
    block_forward(
        .kv_cache  = in->kv_cache,
        .block     = i,
        .base      = base,
        .total     = total,
        .block_out = &out->blocks[i],
        .block_in  = &in->blocks[i],
        .in        = x,
        .scores    = out->scores,
    );
    x = out->blocks[i].out;
  }

  in->kv_cache->len = total;

  // out_ln = norm(x[N])
  layer_norm_forward(
      .ln_out = &out->ln,
      .ln_in  = &in->ln,
      .in     = x,
  );

  // logits = out_ln*hW + bW
  //
  // logits = out_ln*tok_embs^T + bW (tied embeddings)
  //
  // What it means geometrically
  //
  // Under tying, the logit for token v is out_ln[i] · E[v].
  // So the model's last-layer job becomes:
  //
  // >> produce a hidden vector that points at the next token's embedding.
  project(
      .out = out->logits.value,
      .x   = out->ln.out,
      .W   = in->tie_embeddings ? in->tok_emb : in->H.weight,
      .b   = in->H.bias,
      .transpose_w = in->tie_embeddings,
  );

  // probs = softmax(logits)
  softmax_by_row(
      .out = out->probs,
      .x = out->logits.value,
      .temperature = temperature
  );
}

void transformer_backward(
  TokenID_Array tokens,
  Transformer_Output* out,
  Transformer* in
) {
  size_t N = tokens.count;
  NMatrix dlogits = out->logits.grad;

  // logits = out_ln*hW + bW
  //
  // dhb     += dlogits
  // dhW     += ln_out^T * dlogits
  // dout_ln += dlogits * hW^T
  dproject(
      .x    = out->ln.out,
      .W    = in->tie_embeddings ? in->tok_emb : in->H.weight,
      .b    = in->H.bias,
      .dout = dlogits,
      .transpose_w = in->tie_embeddings,
  );

  // out_ln = norm(x2)
  //
  // dx2 = dnorm(x2, dout_ln)
  Tensor x2 = out->blocks[out->num_blocks-1].out;
  layer_norm_backward(
      .in     = x2,
      .ln_in  = &in->ln,
      .ln_out = &out->ln,
  );

  for (size_t j = out->num_blocks; j > 0; j--) {
    size_t i = j - 1;

    Tensor x1 = i > 0 ? out->blocks[i-1].out : out->x0;

    // x[i+1] = block(x[i])
    //
    // dx[i] = dblock(x[i], dx[i+1])
    block_backward(
        .kv_cache  = in->kv_cache,
        .block     = i,
        .total     = N,
        .in        = x1,
        .dout      = x2.grad,
        .scores    = out->scores,
        .dscores   = out->dscores,
        .block_in  = &in->blocks[i],
        .block_out = &out->blocks[i],
    );

    x2 = x1;
  }

  // mask = dropout(x0)
  //
  // dx0 = ddropout(x0)
  // dropout_backward(out->x0.grad, out->x0_mask, in->x0_p);

  // x0 = tok_embs
  //
  // dtok_embs += dx0
  for (size_t i = 0; i < tokens.count; i++) {
    int32_t token = tokens.elems[i];
    mat_add(
        mat_row(in->tok_emb.grad, token),
        mat_row(in->tok_emb.grad, token),
        mat_row(out->x0.grad, i)
    );
  }
}

float cross_entropy(NMatrix probs, TokenID_Array targets) {
  size_t N = probs.rows;

  float loss = 0;

  for (size_t i = 0; i < N; i++)
    loss -= logf(MAT_AT(probs, i, targets.elems[i]) + 1e-10f);

  return loss;
}

#endif // TRANSFORMER_H
