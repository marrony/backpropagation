#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include "array.h"
#include "nn.h"
#include "allocator.h"
#include "tokenizer.h"
#include <math.h>
#include <stdint.h>

typedef struct {
  Tensor gamma;
  Tensor beta;
} Layer_Norm;

typedef struct {
  Tensor  out;      // [NxD]
  NMatrix mean;     // [1xN]
  NMatrix var;      // [1xN]
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
} Attention;

typedef struct {
  Tensor Q;              // [NxD]
  Tensor K;              // [NxD]
  Tensor V;              // [NxD]
  Tensor weights;        // [HxNxN]
  Tensor vals;           // [NxD]
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

void layer_norm_forward_opts(struct Layer_Norm_Forward_Opts opts) {
  NMatrix gamma = opts.ln_in->gamma.value;
  NMatrix beta = opts.ln_in->beta.value;
  NMatrix out = opts.ln_out->out.value;
  NMatrix mean = opts.ln_out->mean;
  NMatrix var = opts.ln_out->var;
  NMatrix xhat = opts.ln_out->xhat;
  NMatrix in = opts.in.value;

  for (int32_t i = 0; i < out.rows; i++) { // 0..N
    float m = 0;
    for (int32_t d = 0; d < out.cols; d++) // 0..D
      m += MAT_AT(in, i, d);
    m /= out.cols;
    VEC_AT(mean, i) = m;

    float v = 0;
    for (int32_t d = 0; d < out.cols; d++) { // 0..D
      float diff = MAT_AT(in, i, d) - m;
      v += diff * diff;
    }
    v /= out.cols;
    VEC_AT(var, i) = v;

    float inv_std = 1.0f / sqrtf(v + 1e-5f);
    for (int32_t d = 0; d < out.cols; d++) { // 0..D
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
  NMatrix var = opts.ln_out->var;
  NMatrix xhat = opts.ln_out->xhat;
  NMatrix dout = opts.ln_out->out.grad;
  NMatrix din = opts.in.grad;

  for (int32_t i = 0; i < dout.rows; i++) { // 0..N
    float inv_std = 1.0f / sqrtf(VEC_AT(var, i) + 1e-5f);

    float sum_dx_hat = 0;
    float sum_dx_hat_x_hat = 0;
    for (int32_t d = 0; d < dout.cols; d++) { // 0..D
      float dx_hat = MAT_AT(dout, i, d) * VEC_AT(gamma, d);
      sum_dx_hat += dx_hat;
      sum_dx_hat_x_hat += dx_hat * MAT_AT(xhat, i, d);
    }

    for (int32_t d = 0; d < dout.cols; d++) { // 0..D
      float dx_hat = MAT_AT(dout, i, d) * VEC_AT(gamma, d);
      float dx_ = dx_hat - sum_dx_hat/dout.cols - MAT_AT(xhat, i, d) * sum_dx_hat_x_hat/dout.cols;
      MAT_AT(din, i, d) += dx_ * inv_std;
    }
  }

  for (int32_t i = 0; i < dout.rows; i++) {    // 0..N
    for (int32_t d = 0; d < dout.cols; d++) {  // 0..D
      VEC_AT(dgamma, d) += MAT_AT(dout, i, d) * MAT_AT(xhat, i, d);
      VEC_AT(dbeta, d) += MAT_AT(dout, i, d);
    }
  }
}

void init_xavier_glorot(NMatrix mat, size_t fan_in, size_t fan_out) {
  float std = sqrtf(6.0f / (fan_in + fan_out));
  mat_rand_uniform(mat, -std, +std);
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

void init_gamma_beta(Allocator* alloc, Tensor* gamma, Tensor* beta, size_t D) {
  alloc_tensor(alloc, gamma, 1, D);
  alloc_tensor(alloc, beta, 1, D);
  mat_fill(gamma->value, 1);
  mat_zero(beta->value);
}

void init_block(Allocator* alloc, Block* block, size_t D, size_t F) {
  init_gamma_beta(alloc, &block->ln1.gamma, &block->ln1.beta, D);
  init_linear_layer(alloc, &block->attn.Q, D, D);
  init_linear_layer(alloc, &block->attn.K, D, D);
  init_linear_layer(alloc, &block->attn.V, D, D);
  init_linear_layer(alloc, &block->attn.O, D, D);
  init_gamma_beta(alloc, &block->ln2.gamma, &block->ln2.beta, D);
  init_linear_layer(alloc, &block->ff1, D, F);
  init_linear_layer(alloc, &block->ff2, F, D);
}

void init_block_output(Arena_Allocator* arena, Block_Output* block_out, size_t N, size_t D, size_t H, size_t F) {
  alloc_tensor(&arena->alloc, &block_out->ln1.out, N, D);
  block_out->ln1.mean     = mat_alloc2(&arena->alloc, 1, N);
  block_out->ln1.var      = mat_alloc2(&arena->alloc, 1, N);
  block_out->ln1.xhat     = mat_alloc2(&arena->alloc, N, D);
  mat_zero(block_out->ln1.mean);
  mat_zero(block_out->ln1.var);
  mat_zero(block_out->ln1.xhat);
  alloc_tensor(&arena->alloc, &block_out->attn.Q, N, D);
  alloc_tensor(&arena->alloc, &block_out->attn.K, N, D);
  alloc_tensor(&arena->alloc, &block_out->attn.V, N, D);
  alloc_tensor(&arena->alloc, &block_out->attn.weights, H, N*N);
  alloc_tensor(&arena->alloc, &block_out->attn.vals, N, D);
  block_out->attn.out     = mat_alloc2(&arena->alloc, N, D);
  block_out->x1           = mat_alloc2(&arena->alloc, N, D);
  mat_zero(block_out->attn.out);
  mat_zero(block_out->x1);
  alloc_tensor(&arena->alloc, &block_out->ln2.out, N, D);
  block_out->ln2.mean     = mat_alloc2(&arena->alloc, 1, N);
  block_out->ln2.var      = mat_alloc2(&arena->alloc, 1, N);
  block_out->ln2.xhat     = mat_alloc2(&arena->alloc, N, D);
  mat_zero(block_out->ln2.mean);
  mat_zero(block_out->ln2.var);
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
};

struct Project_Backward_Opts {
  Tensor x;
  Tensor W;
  Tensor b;
  NMatrix dout;
};

#define project(...) project_opts((struct Project_Forward_Opts){ __VA_ARGS__ })
#define dproject(...) dproject_opts((struct Project_Backward_Opts){ __VA_ARGS__ })

void project_opts(struct Project_Forward_Opts opts) {
  // out = x*W + b
  NMatrix out = opts.out;
  NMatrix x = opts.x.value;
  NMatrix W = opts.W.value;
  NMatrix b = opts.b.value;

  ASSERT_MATRIX_MULT(out.rows, out.cols, x.rows, x.cols, W.rows, W.cols);

  for (int i = 0; i < out.rows; i++) {
    for (int j = 0; j < out.cols; j++) {
      MAT_AT(out, i, j) = mat_dot_row_col(x, W, i, j) + MAT_AT(b, 0, j);
    }
  }
}

void dproject_opts(struct Project_Backward_Opts opts) {
  // out = x*W + b

  NMatrix dout = opts.dout;

  // db += dout
  NMatrix db = opts.b.grad;
  for (int32_t i = 0; i < dout.rows; i++)
    mat_add(db, db, mat_row(dout, i));

  // dW += x^T * dout
  NMatrix x = opts.x.value;
  NMatrix dW = opts.W.grad;
  mat_mult_A_transposed_and_B_acc(dW, x, dout);

  // dx += dout * W^T
  NMatrix dx = opts.x.grad;
  NMatrix W = opts.W.value;
  mat_mult_A_and_B_transposed_acc(dx, dout, W);
}

struct Attention_Forward_Opts {
  Attention_Output* attn_out;
  NMatrix scores;
  Attention* attn_in;
  Tensor in;
};

struct Attention_Backward_Opts {
  Tensor in;
  Attention_Output* attn_out;
  Attention* attn_in;
  NMatrix dout;
  NMatrix dscores;
};

#define attention_forward(...) attention_forward_opts((struct Attention_Forward_Opts){ __VA_ARGS__ })
#define attention_backward(...) attention_backward_opts((struct Attention_Backward_Opts){ __VA_ARGS__ })

void attention_forward_opts(struct Attention_Forward_Opts opts) {
  Attention_Output* attn_out = opts.attn_out;
  NMatrix scores = opts.scores;
  Attention* attn_in = opts.attn_in;
  Tensor in = opts.in;

  int32_t N = in.value.rows;
  int32_t D = in.value.cols;
  int32_t H = attn_out->weights.value.rows;
  int32_t head_dim = D / H;

  // Q = input*wQ + bQ
  project(
      .out = attn_out->Q.value,
      .x   = in,
      .W   = attn_in->Q.weight,
      .b   = attn_in->Q.bias,
  );

  // K = input*wK + bK
  project(
      .out = attn_out->K.value,
      .x   = in,
      .W   = attn_in->K.weight,
      .b   = attn_in->K.bias
  );

  // V = input*wV + bV
  project(
      .out = attn_out->V.value,
      .x   = in,
      .W   = attn_in->V.weight,
      .b   = attn_in->V.bias
  );

  float scale = 1.0f / sqrt(head_dim);

  for (int32_t h = 0; h < H; h++) {      // 0..H
    // scores = Q * K^T
    mat_mult_A_and_B_transposed(
        scores,
        mat_cols(attn_out->Q.value, h*head_dim, h*head_dim + head_dim),
        mat_cols(attn_out->K.value, h*head_dim, h*head_dim + head_dim)
    );

    // scores = scores / sqrt(h_dim)
    mat_scale(scores, scores, scale);

    for (int32_t i = 0; i < N; i++) {    // 0..N
      for (int32_t j = 0; j < N; j++) {  // 0..N
        if (j > i) MAT_AT(scores, i, j) = -INFINITY;
      }
    }

    // attn_weitghs = softmax(scores)
    NMatrix attn_weights = mat_row_as(attn_out->weights.value, h, N, N);
    softmax_by_row(attn_weights, scores, 1);

    // attn_vals = attn_weitghs * V
    mat_mult(
        mat_cols(attn_out->vals.value, h*head_dim, h*head_dim + head_dim),
        attn_weights,
        mat_cols(attn_out->V.value,    h*head_dim, h*head_dim + head_dim)
    );
  }

  // out = vals*wO + bO
  project(
      .out = attn_out->out,
      .x   = attn_out->vals,
      .W   = attn_in->O.weight,
      .b   = attn_in->O.bias,
  );
}

void attention_backward_opts(struct Attention_Backward_Opts opts) {
  Tensor in = opts.in;
  Attention_Output* attn_out = opts.attn_out;
  Attention* attn_in = opts.attn_in;
  NMatrix dout = opts.dout;
  NMatrix dscores = opts.dscores;

  int32_t N = in.value.rows;
  int32_t D = in.value.cols;
  int32_t H = attn_out->weights.value.rows;
  int32_t head_dim = D / H;

  // attn_out = attn_vals*wO + bO
  dproject(
      .x    = attn_out->vals,
      .W    = attn_in->O.weight,
      .b    = attn_in->O.bias,
      .dout = dout,
  );

  for (int32_t h = 0; h < H; h++) {      // 0..H
    int head_start = h*head_dim;
    int head_end = h*head_dim + head_dim;

    // [NxN]
    NMatrix attn_weights = mat_row_as(attn_out->weights.value, h, N, N);
    NMatrix dweights = mat_row_as(attn_out->weights.grad, h, N, N);

    // A = weight * V
    //
    // dweight = dA * V^T
    // dV = weight^T * dA

    mat_mult_A_and_B_transposed_acc(
        dweights,
        mat_cols(attn_out->vals.grad, head_start, head_end),
        mat_cols(attn_out->V.value,   head_start, head_end)
    );

    // [NxHdim] = [NxN] * [NxHdim]
    mat_mult_A_transposed_and_B_acc(
        mat_cols(attn_out->V.grad,    head_start, head_end),
        attn_weights,
        mat_cols(attn_out->vals.grad, head_start, head_end)
    );

    // attn_weights = softmax(scores)
    //
    // dscores = dsoftmax(attn_weights, dweights)

    mat_zero(dscores);
    dsoftmax_by_row(dscores, attn_weights, dweights, 1.0f);

    // scores = Q * K^T
    //
    // dQ = dScores * K
    // dK = dScore^T * Q

    mat_mult_acc(
        mat_cols(attn_out->Q.grad,  head_start, head_end),
        dscores,
        mat_cols(attn_out->K.value, head_start, head_end)
    );

    mat_mult_A_transposed_and_B_acc(
        mat_cols(attn_out->K.grad, head_start, head_end),
        dscores,
        mat_cols(attn_out->Q.value, head_start, head_end)
    );
  }

  float scale = 1.0f / sqrt(head_dim);
  mat_scale(attn_out->Q.grad, attn_out->Q.grad, scale);
  mat_scale(attn_out->K.grad, attn_out->K.grad, scale);

  // V = input*wV + bV
  dproject(
      .x    = in,
      .W    = attn_in->V.weight,
      .b    = attn_in->V.bias,
      .dout = attn_out->V.grad,
  );

  // K = input*wK + bK
  dproject(
      .x    = in,
      .W    = attn_in->K.weight,
      .b    = attn_in->K.bias,
      .dout = attn_out->K.grad,
  );

  // Q = input*wQ + bQ
  dproject(
      .x    = in,
      .W    = attn_in->Q.weight,
      .b    = attn_in->Q.bias,
      .dout = attn_out->Q.grad,
  );
}

struct Block_Forward_Opts {
  Block_Output* block_out;
  Block* block_in;
  Tensor in;
  NMatrix scores;
};

struct Block_Backward_Opts {
  Tensor in;
  NMatrix dout;
  NMatrix dscores;
  Block* block_in;
  Block_Output* block_out;
};

#define block_forward(...) block_forward_opts((struct Block_Forward_Opts){ __VA_ARGS__ })
#define block_backward(...) block_backward_opts((struct Block_Backward_Opts){ __VA_ARGS__ })

// ln1_out  = norm(input)
// attn_out = attention(ln1_out)
// x1       = input + attn_out
// ln2_out  = norm(x1)
// ff1_out  = project(ln2_out, ff1W, ff1b)
// relu_out = relu(ff1_out)
// ff2_out  = project(relu_out, ff2W, ff2b)
// out      = x1 + ff2_out
void block_forward_opts(struct Block_Forward_Opts opts) {
  Block_Output* block_out = opts.block_out;
  Block* block_in = opts.block_in;
  Tensor in = opts.in;
  NMatrix scores = opts.scores;

  // ln1_out = norm(input)
  layer_norm_forward(
      .ln_out = &block_out->ln1,
      .ln_in  = &block_in->ln1,
      .in     = in,
  );

  // attn_out = attention(ln1_out)
  attention_forward(
      .attn_out = &block_out->attn,
      .attn_in  = &block_in->attn,
      .scores   = scores,
      .in       = block_out->ln1.out,
  );

  // x1 = input + attn_out
  mat_add(block_out->x1, in.value, block_out->attn.out);

  // ln2_out = norm(x1)
  layer_norm_forward(
      .ln_out = &block_out->ln2,
      .ln_in  = &block_in->ln2,
      .in     = tensor(block_out->x1, NULL_MATRIX),
  );

  // ff1_out = ln2_out*ff1W + ff1b
  project(
      .out = block_out->ff1_out.value,
      .x   = block_out->ln2.out,
      .W   = block_in->ff1.weight,
      .b   = block_in->ff1.bias,
  );

  // relu_out = relu(ff1_out)
  relu(block_out->relu_out.value, block_out->ff1_out.value, 0.0f);

  // ff2_out = relu_out*ff2W + ff2b
  project(
      .out = block_out->ff2_out,
      .x   = block_out->relu_out,
      .W   = block_in->ff2.weight,
      .b   = block_in->ff2.bias,
  );

  // out = x1 + ff2_out
  mat_add(block_out->out.value, block_out->x1, block_out->ff2_out);
}

// forward:
// ln1_out  = norm(input)
// attn_out = attention(ln1_out)
// x1       = input + attn_out
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
// din       += dnorm(input, dln1_out)         | din       += dnorm(input, dln1_out)
void block_backward_opts(struct Block_Backward_Opts opts) {
  Tensor in = opts.in;
  NMatrix dout = opts.dout;
  NMatrix dscores = opts.dscores;
  Block* block_in = opts.block_in;
  Block_Output* block_out = opts.block_out;

  // ff2_out = project(relu_out, ff2W, ff2b)
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

  // ln2_out = norm(x1)
  //
  // din += dout
  // din += dnorm(x1, dln2_out)
  mat_add(in.grad, in.grad, dout);

  layer_norm_backward(
      .in     = in,
      .ln_in  = &block_in->ln2,
      .ln_out = &block_out->ln2,
  );

  // attn_out = attention(ln1_out)
  //
  // dln1_out = dattention(ln1_out, din)
  attention_backward(
      .in       = block_out->ln1.out,
      .attn_out = &block_out->attn,
      .attn_in  = &block_in->attn,
      .dscores  = dscores,
      .dout     = in.grad,
  );

  // ln1_out = norm(input)
  //
  // din += dnorm(input, dln1_out)
  layer_norm_backward(
      .in     = in,
      .ln_in  = &block_in->ln1,
      .ln_out = &block_out->ln1,
  );
}

#define NUM_BLOCKS 4

typedef struct {
  Tensor tok_emb;
  Tensor pos_emb;
  Block blocks[NUM_BLOCKS];
  Layer_Norm ln;
  Linear_Layer H;

  // configs
  size_t vocab_size;
  size_t context_size;
  size_t emb_size;
  size_t ff_size;
} Transformer;

typedef struct {
  Tensor x0;
  Block_Output blocks[NUM_BLOCKS];
  Layer_Norm_Output ln;
  Tensor logits;
  NMatrix probs;
  NMatrix scores;

  // configs
  size_t sequence_size;
  size_t heads_count;
} Transformer_Output;

struct Init_Transformer_Opts {
  Allocator* alloc;
  Transformer* trans;
  size_t vocab_size;
  size_t context_size;
  size_t emb_size;
  size_t ff_size;
};

struct Init_Transformer_Output_Opts {
  Arena_Allocator* arena;
  Transformer_Output* trans_out;
  size_t sequence_size;
  size_t vocab_size;
  size_t context_size;
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
  size_t C = opts.context_size;
  size_t D = opts.emb_size;
  size_t F = opts.ff_size;

  trans->vocab_size = opts.vocab_size;
  trans->context_size = opts.context_size;
  trans->emb_size = opts.emb_size;
  trans->ff_size = opts.ff_size;

  alloc_tensor(alloc, &trans->tok_emb, V, D);
  init_xavier_glorot(trans->tok_emb.value, V, D);

  alloc_tensor(alloc, &trans->pos_emb, C, D);
  init_xavier_glorot(trans->pos_emb.value, C, D);

  init_gamma_beta(alloc, &trans->ln.gamma, &trans->ln.beta, D);
  init_linear_layer(alloc, &trans->H, D, V);

  for (size_t i = 0; i < NUM_BLOCKS; i++) {
    init_block(alloc, &trans->blocks[i], D, F);
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

  trans_out->sequence_size = opts.sequence_size;
  trans_out->heads_count = opts.heads_count;

  alloc_tensor(&arena->alloc, &trans_out->x0, N, D);

  for (size_t i = 0; i < NUM_BLOCKS; i++) {
    init_block_output(arena, &trans_out->blocks[i], N, D, H, F);
  }

  alloc_tensor(&arena->alloc, &trans_out->ln.out, N, D);
  trans_out->ln.mean     = mat_alloc2(&arena->alloc, 1, N);
  trans_out->ln.var      = mat_alloc2(&arena->alloc, 1, N);
  trans_out->ln.xhat     = mat_alloc2(&arena->alloc, N, D);
  mat_zero(trans_out->ln.mean);
  mat_zero(trans_out->ln.var);
  mat_zero(trans_out->ln.xhat);

  alloc_tensor(&arena->alloc, &trans_out->logits, N, V);
  trans_out->probs = mat_alloc2(&arena->alloc, N, V);
  trans_out->scores = mat_alloc2(&arena->alloc, N, N);
  mat_zero(trans_out->probs);
  mat_zero(trans_out->scores);
}

void transformer_forward(
  TokenID_Array tokens,
  Transformer_Output* out,
  Transformer* in,
  float temperature
) {
  assert(tokens.count <= in->context_size);
  assert(tokens.count == out->sequence_size);

  // x0 = tok_embs + pos_embs
  for (size_t i = 0; i < tokens.count; i++) {
    mat_add(
        mat_row(out->x0.value, i),
        mat_row(in->tok_emb.value, tokens.elems[i]),
        mat_row(in->pos_emb.value, i)
    );
  }

  Tensor x = out->x0;

  // x[i+1] = block(x[i])
  for (size_t i = 0; i < NUM_BLOCKS; i++) {
    block_forward(
        .block_out = &out->blocks[i],
        .block_in  = &in->blocks[i],
        .in        = x,
        .scores    = out->scores,
    );
    x = out->blocks[i].out;
  }

  // out_ln = norm(x[N])
  layer_norm_forward(
      .ln_out = &out->ln,
      .ln_in  = &in->ln,
      .in     = x,
  );

  // logits = out_ln*hW + bW
  project(
      .out = out->logits.value,
      .x   = out->ln.out,
      .W   = in->H.weight,
      .b   = in->H.bias,
  );

  // probs = softmax(logits)
  softmax_by_row(out->probs, out->logits.value, temperature);

  // printf("x0     = "); mat_println(out->x0.value, 4);
  // // printf("x1     = "); mat_println(x1.value, 4);
  // // printf("x2     = "); mat_println(x2.value, 4);
  // printf("out_ln = "); mat_println(out->ln.out.value, 4);
  // printf("logits = "); mat_println(out->logits.value, 4);
  // printf("probs  = "); mat_println(out->probs, 4);
}

void transformer_backward(
  TokenID_Array tokens,
  TokenID_Array targets,
  Transformer_Output* out,
  Transformer* in
) {
  float scale = 1.0f / (float)tokens.count;

  // dlogits = probs - target
  NMatrix dlogits = out->logits.grad;
  mat_copy(dlogits, out->probs);
  mat_scale(dlogits, dlogits, scale);

  for (size_t i = 0; i < tokens.count; i++)
    MAT_AT(dlogits, i, targets.elems[i]) -= scale;

  // logits = out_ln*hW + bW
  //
  // dhb     += dlogits
  // dhW     += ln_out^T * dlogits
  // dout_ln += dlogits * hW^T
  dproject(
      .x    = out->ln.out,
      .W    = in->H.weight,
      .b    = in->H.bias,
      .dout = dlogits,
  );

  // out_ln = norm(x2)
  //
  // dx2 = dnorm(x2, dout_ln)
  Tensor x2 = out->blocks[NUM_BLOCKS-1].out;
  layer_norm_backward(
      .in     = x2,
      .ln_in  = &in->ln,
      .ln_out = &out->ln,
  );

  for (size_t j = NUM_BLOCKS; j > 0; j--) {
    size_t i = j - 1;

    Tensor x1 = i > 0 ? out->blocks[i-1].out : out->x0;

    // x[i+1] = block(x[i])
    //
    // dx[i] = dblock(x[i], dx[i+1])
    block_backward(
        .in        = x1,
        .dout      = x2.grad,
        .dscores   = out->scores,
        .block_in  = &in->blocks[i],
        .block_out = &out->blocks[i],
    );

    x2 = x1;
  }

  // x0 = tok_embs + pos_embs
  //
  // dtok_embs += dx0
  // dpos_embs += dx0
  for (size_t i = 0; i < tokens.count; i++) {
    int32_t token = tokens.elems[i];
    mat_add(
        mat_row(in->tok_emb.grad, token),
        mat_row(in->tok_emb.grad, token),
        mat_row(out->x0.grad, i)
    );
    mat_add(
        mat_row(in->pos_emb.grad, i),
        mat_row(in->pos_emb.grad, i),
        mat_row(out->x0.grad, i)
    );
  }
}

float cross_entropy(NMatrix probs, TokenID_Array targets) {
  size_t N = probs.rows;

  float loss = 0;

  for (size_t i = 0; i < N; i++)
    loss -= logf(MAT_AT(probs, i, targets.elems[i]) + 1e-10f);

  return loss / (float)N;
}

#endif // TRANSFORMER_H
