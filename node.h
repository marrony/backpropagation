#ifndef NODE_H
#define NODE_H

#include "nn.h"
#include "allocator.h"
#include "array.h"

typedef enum {
  NODE_CONSTANT,
  NODE_VARIABLE,
  NODE_EMBEDDING,
  NODE_LINEAR,
  NODE_SIGMOID,
  NODE_SOFTMAX,
  NODE_SOFTMAX_CROSS_ENTROPY,
  NODE_RELU,
  NODE_CONV2D,
  NODE_FLATTEN,
  NODE_SQUARE,
  NODE_CUBE,
  NODE_EXP,
  NODE_NEGATE,

  // 
  NODE_MULTIPLY,
  NODE_DIVIDE,
  NODE_ADD,
  NODE_SUB,

  __MAX_NODES = NODE_SUB,
} Node_Type;

#define STR(x) [x] = #x

static const char* Node_Type_Str[__MAX_NODES+1] = {
  STR(NODE_CONSTANT),
  STR(NODE_VARIABLE),
  STR(NODE_EMBEDDING),
  STR(NODE_LINEAR),
  STR(NODE_SIGMOID),
  STR(NODE_SOFTMAX),
  STR(NODE_SOFTMAX_CROSS_ENTROPY),
  STR(NODE_RELU),
  STR(NODE_CONV2D),
  STR(NODE_FLATTEN),
  STR(NODE_SQUARE),
  STR(NODE_CUBE),
  STR(NODE_EXP),
  STR(NODE_NEGATE),
  STR(NODE_MULTIPLY),
  STR(NODE_DIVIDE),
  STR(NODE_ADD),
  STR(NODE_SUB),
};

#define MAX_INPUTS 16

typedef struct {
  NMatrix value;
  NMatrix grad;
  NMatrix g_grad;
} Param;

typedef struct Node Node;
/*
 * Node Structure - Computation Graph Node for Automatic Differentiation
 *
 * Fields:
 *   type           - Node type (NODE_LINEAR, NODE_SIGMOID, NODE_ADD, etc.)
 *   input[]        - Input nodes (up to MAX_INPUTS = 16)
 *   input_size     - Number of input nodes
 *   output         - Output value/gradient
 *   weight         - Weights matrix/kernel (for NODE_LINEAR, NODE_CONV2D)
 *   bias           - Bias vector (for NODE_LINEAR, NODE_CONV2D)
 *   temperature    - Temperature for NODE_SOFTMAX
 *
 * Each Param has:
 *   .value  - Current parameter value (weights, biases)
 *   .grad   - Gradient (dL/dparam) computed by backward pass
 *   .g_grad - Gradient of gradient (for second-order methods)
 *
 * Forward pass:  node_forward(node)  -> output stored in node->output.value
 * Backward pass: node_backward(node, dL)  -> gradients stored in node->output.grad
 */
struct Node {
  Node_Type type;
  Node* input[MAX_INPUTS];
  int input_size;
  Param output;
  Param weight;
  Param bias;
  float temperature;

  //
  int32_t* context;
  size_t context_size;
  size_t* embedding_tokens_batch;
  NMatrix ada_grad;
};

#define NULL_MATRIX (NMatrix){NULL, 0, 0}
#define NULL_PARAM (Param) {0}

void init_param(Param* param, int rows, int cols) {
  param->value  = mat_alloc(rows, cols);
  param->grad   = mat_alloc(rows, cols);
  param->g_grad = mat_alloc(rows, cols);
  mat_fill(param->value, 0);
  mat_fill(param->grad, 0);
  mat_fill(param->g_grad, 0);
}

int node_get_value_rows(Node* node) {
  return node->output.value.rows;
}

int node_get_value_cols(Node* node) {
  return node->output.value.cols;
}

void destroy_param(Param* param) {
  mat_free(param->value);
  mat_free(param->grad);
  mat_free(param->g_grad);
}

void destroy_node(Node** node) {
  for (int i = 0; i < (*node)->input_size; i++)
    destroy_node(&(*node)->input[i]);
  destroy_param(&(*node)->output);
  destroy_param(&(*node)->weight);
  destroy_param(&(*node)->bias);
  free(*node);
  *node = NULL;
}

Node* create_node(Node_Type type) {
  Node* node = malloc(sizeof(Node));
  memset(node, 0, sizeof(Node));
  node->type = type;
  return node;
}

Node* create_constant(int rows, int cols) {
  Node* node = create_node(NODE_CONSTANT);
  node->input_size = 0;
  init_param(&node->output, rows, cols);
  
  return node;
}

Node* create_variable(int rows, int cols) {
  Node* node = create_node(NODE_VARIABLE);
  node->input_size = 0;
  init_param(&node->output, rows, cols);

  return node;
}

Node* create_embeddings(size_t max_vocab, size_t emb_dim, size_t context) {
  Node* node = create_node(NODE_EMBEDDING);
  node->input_size = 0;
  init_param(&node->output, 1, emb_dim*context);
  init_param(&node->weight, max_vocab, emb_dim);

  node->context = malloc(sizeof(int32_t)*context);
  node->context_size = context;
  memset(node->context, 0, sizeof(int32_t)*context);

  node->embedding_tokens_batch = malloc(sizeof(size_t)*max_vocab);
  memset(node->embedding_tokens_batch, 0, sizeof(size_t)*max_vocab);

  node->ada_grad = mat_alloc(max_vocab, emb_dim);
  mat_fill(node->ada_grad, 0);

  return node;
}

Node* create_linear(Node* input, int in_size, int out_size) {
  Node* node = create_node(NODE_LINEAR);
  node->input[0] = input;
  node->input_size = 1;

  int rows = node_get_value_rows(input);
  int cols = node_get_value_cols(input);
  assert(rows*cols == in_size);

  // node->output.value holds f(x) = x*W + b
  // node->output.grad holds dL/d(output)
  init_param(&node->output, 1, out_size);
  // Standard convention: W is (input_dim × output_dim) = (in_size × out_size)
  // Forward: output = input × W + b, where input is (1×in_size), W is (in_size×out_size), output is (1×out_size)
  init_param(&node->weight, in_size, out_size);
  init_param(&node->bias, 1, out_size);

  // mat_rand(node->weight.value);
  // mat_rand(node->bias.value);

  return node;
}

Node* create_unary(Node_Type type, Node* input, int rows, int cols) {
  Node* node = create_node(type);
  node->input[0] = input;
  node->input_size = 1;
  init_param(&node->output, rows, cols);
  return node;
}

Node* create_binary(Node_Type type, Node* input0, Node* input1, int rows, int cols) {
  Node* node = create_node(type);
  node->input[0] = input0;
  node->input[1] = input1;
  node->input_size = 2;
  init_param(&node->output, rows, cols);
  return node;
}

Node* create_sigmoid(Node* input) {
  int rows = node_get_value_rows(input);
  int cols = node_get_value_cols(input);
  return create_unary(NODE_SIGMOID, input, rows, cols);
}

Node* create_softmax(Node* input) {
  int rows = node_get_value_rows(input);
  int cols = node_get_value_cols(input);
  Node* node = create_unary(NODE_SOFTMAX, input, rows, cols);
  node->temperature = 1.0f;
  return node;
}

Node* create_softmax_cross_entropy(Node* input) {
  int rows = node_get_value_rows(input);
  int cols = node_get_value_cols(input);
  Node* node = create_unary(NODE_SOFTMAX_CROSS_ENTROPY, input, rows, cols);
  node->temperature = 1.0f;
  return node;
}

Node* create_relu(Node* input) {
  int rows = node_get_value_rows(input);
  int cols = node_get_value_cols(input);
  return create_unary(NODE_RELU, input, rows, cols);
}

Node* create_conv2d(Node* input, int img_size, int kern_size) {
  Node* node = create_node(NODE_CONV2D);
  node->input[0] = input;
  node->input_size = 1;

  int conv_size = (img_size - kern_size + 1);

  init_param(&node->output, conv_size, conv_size);
  init_param(&node->weight, kern_size, kern_size);
  init_param(&node->bias, 1, conv_size);

  // mat_rand(node->weight.value);
  // mat_rand(node->bias.value);

  return node;
}

// Create a flatten node that concatenates multiple input matrices.
// IMPORTANT: This uses mat_reshape which is a VIEW operation, not a copy.
// The flattened output shares the same data buffer with input matrices.
// This is safe as long as the input matrices are not modified after flattening.
Node* create_flatten(Node** inputs, int in_size) {
  assert(in_size < MAX_INPUTS);
  Node* node = create_node(NODE_FLATTEN);
  node->input_size = in_size;

  int size = 0;
  for (int i = 0; i < in_size; i++) {
    node->input[i] = inputs[i];
    int rows = node_get_value_rows(inputs[i]);
    int cols = node_get_value_cols(inputs[i]);
    size += rows * cols;
  }

  init_param(&node->output, 1, size);

  return node;
}

Node* create_square(Node* input, int size) {
  return create_unary(NODE_SQUARE, input, 1, size);
}

Node* create_cube(Node* input, int size) {
  return create_unary(NODE_CUBE, input, 1, size);
}

Node* create_exp(Node* input, int size) {
  return create_unary(NODE_EXP, input, 1, size);
}

Node* create_negate(Node* input, int size) {
  return create_unary(NODE_NEGATE, input, 1, size);
}

Node* create_multiply(Node* input0, Node* input1) {
  assert(node_get_value_cols(input0) == node_get_value_rows(input1));

  // [NxM] * [MxP] = [NxP]
  int N = node_get_value_rows(input0);
  int P = node_get_value_cols(input1);
  return create_binary(NODE_MULTIPLY, input0, input1, N, P);
}

Node* create_divide(Node* input0, Node* input1) {
  int N = node_get_value_rows(input0);
  int M = node_get_value_cols(input0);
  return create_binary(NODE_DIVIDE, input0, input1, N, M);
}

Node* create_add(Node* input0, Node* input1) {
  int N = node_get_value_rows(input0);
  int M = node_get_value_cols(input0);
  return create_binary(NODE_ADD, input0, input1, N, M);
}

Node* create_sub(Node* input0, Node* input1) {
  int N = node_get_value_rows(input0);
  int M = node_get_value_cols(input0);
  return create_binary(NODE_SUB, input0, input1, N, M);
}

typedef Node* Node_Ptr;
DEFINE_ARRAY(Node_Ptr);

void node_forward(Node* node, Node_Ptr_Array* order) {
  Param fx   = node->output;
  Param w    = node->weight;
  Param b    = node->bias;

  Param x = node->input[0] ? node->input[0]->output : NULL_PARAM;
  Param u = node->input[0] ? node->input[0]->output : NULL_PARAM;
  Param v = node->input[1] ? node->input[1]->output : NULL_PARAM;

  switch (node->type) {
    case NODE_CONSTANT:
    case NODE_VARIABLE:
      break;

    case NODE_EMBEDDING:
      for (size_t t = 0; t < node->context_size; t++) {
        int32_t token = node->context[t];
        node->embedding_tokens_batch[token] += 1;
        NMatrix value = mat_row(node->weight.value, token);
        NMatrix ctx = mat_row_slice(node->output.value, t*12, 12);
        mat_copy(ctx, value);
      }
      break;

    case NODE_LINEAR:
      // f(x) = x*W + b (standard convention: W is N×M, x is 1×N, output is 1×M)
      // x*W: [1×N] * [N×M] = [1×M]
      // x*W + b: [1×M] + [1×M] = [1×M]
      node_forward(node->input[0], order);
      mat_mult_add(fx.value, x.value, w.value, b.value);
      break;

    case NODE_SIGMOID:
      // f(x) = σ(x) (element-wise sigmoid)
      // σ(x) = 1 / (1 + e^(-x))
      node_forward(node->input[0], order);
      sigmoid(fx.value, x.value);
      break;

    case NODE_SOFTMAX:
    case NODE_SOFTMAX_CROSS_ENTROPY:
      // f(x) = softmax(x) (row-wise normalization)
      // softmax(x)_j = exp(x_j) / Σ_k exp(x_k)
      node_forward(node->input[0], order);
      softmax_temperature(fx.value, x.value, node->temperature);
      break;

    case NODE_RELU:
      // f(x) = ReLU(x) (element-wise rectified linear unit)
      // ReLU(x) = max(0, x)
      node_forward(node->input[0], order);
      relu(fx.value, x.value);
      break;

    case NODE_FLATTEN:
      // f(x₁, x₂, ..., xₙ) = concat(x₁, x₂, ..., xₙ)
      // Reshapes each input to (1, size_i) and concatenates horizontally
      // [size₁] ⊕ [size₂] ⊕ ... ⊕ [sizeₙ] = [Σ size_i]
      for (int i = 0; i < node->input_size; i++) {
        node_forward(node->input[i], order);

        Node* input = node->input[i];

        int rows = node_get_value_rows(input);
        int cols = node_get_value_cols(input);
        int size = rows * cols;

        NMatrix dst = mat_reshape(
            mat_row_slice(fx.value, i*size, size),
            rows,
            cols
        );

        mat_copy(dst, input->output.value);
      }
      break;

    case NODE_CONV2D:
    {
      // f(x) = conv2d(x, weight)
      node_forward(node->input[0], order);

      NMatrix x_ = mat_reshape(x.value,
          fx.value.rows + w.value.rows - 1,
          fx.value.cols + w.value.cols - 1
      );

      for (int i = 0; i < fx.value.rows; i++) {
        for (int j = 0; j < fx.value.cols; j++) {
          MAT_AT(fx.value, i, j) = 0; // bias

          for (int ki = 0; ki < w.value.rows; ki++) {
            for (int kj = 0; kj < w.value.cols; kj++) {
              float a = MAT_AT(x_, i+ki, j+kj);
              float b = MAT_AT(w.value, ki, kj);
              MAT_AT(fx.value, i, j) += a * b;
            }
          }
        }
      }
      break;
    }

    case NODE_SQUARE:
      // f(x) = x ⊙ x (element-wise squaring)
      // [N×M] ⊙ [N×M] = [N×M]
      node_forward(node->input[0], order);
      for (int i = 0; i < x.value.rows; i++) {
        for (int j = 0; j < x.value.cols; j++)
          MAT_AT(fx.value, i, j) = MAT_AT(x.value, i, j)*MAT_AT(x.value, i, j);
      }
      break;

    case NODE_CUBE:
      // f(x) = x ⊙ x ⊙ x (element-wise cubing)
      // [N×M] ⊙ [N×M] ⊙ [N×M] = [N×M]
      node_forward(node->input[0], order);
      for (int i = 0; i < x.value.rows; i++) {
        for (int j = 0; j < x.value.cols; j++)
          MAT_AT(fx.value, i, j) = MAT_AT(x.value, i, j)*MAT_AT(x.value, i, j)*MAT_AT(x.value, i, j);
      }
      break;

    case NODE_EXP:
      // f(x) = exp(x) (element-wise exponential)
      // exp(x)_ij = e^(x_ij)
      node_forward(node->input[0], order);
      for (int i = 0; i < x.value.rows; i++) {
        for (int j = 0; j < x.value.cols; j++)
          MAT_AT(fx.value, i, j) = expf(MAT_AT(x.value, i, j));
      }
      break;

    case NODE_NEGATE:
      // f(x) = -x (element-wise negation)
      // [N×M] -> [N×M]
      node_forward(node->input[0], order);
      mat_scale(fx.value, x.value, -1);
      break;

    case NODE_MULTIPLY:
      // f(u, v) = u * v (matrix multiplication)
      // [N×M] * [M×P] = [N×P]
      node_forward(node->input[0], order);
      node_forward(node->input[1], order);
      mat_mult(fx.value, u.value, v.value);
      break;

    case NODE_DIVIDE:
      // f(u, v) = u ⊘ v (element-wise division)
      // [N×M] ⊘ [N×M] = [N×M]
      node_forward(node->input[0], order);
      node_forward(node->input[1], order);
      mat_memberwise_div(fx.value, u.value, v.value, 1e-15f);
      break;

    case NODE_ADD:
      // f(u, v) = u + v
      node_forward(node->input[0], order);
      node_forward(node->input[1], order);
      mat_add(fx.value, u.value, v.value);
      break;

    case NODE_SUB:
      // f(u, v) = u - v
      node_forward(node->input[0], order);
      node_forward(node->input[1], order);
      mat_sub(fx.value, u.value, v.value);
      break;
  }
  array_append(order, node);
}

// y = f(x)
// upstream   = dL/dy
// local      = dy/dx
// downstream = dL/dx
//
// downstream = local * upstream
//
// Rule:
// A node should never write its downstream gradient into its own memory space (node->output.grad).
// It must write it directly into its input's gradient space (node->input[0]->output.grad).
// Exception:
// Variables
//
// void node_backward(Node* node, NMatrix dL) {
//   Node* input_x = node->input[0]
//
//   NMatrix dydx = f'(x)
//   input_x->output.grad += dydx * dL
//
//   node_backward(input_x, input_x->output.grad);
// }
//
void node_backward(Node_Ptr_Array* order) {
  if (order->count == 0) return;

  Node* node    = array_pop_last(order);
  Param fwd_out = node->output;
  Param w       = node->weight;
  Param b       = node->bias;
  NMatrix dL    = fwd_out.grad;

  Param x    = node->input[0] ? node->input[0]->output : NULL_PARAM;
  Param u    = node->input[0] ? node->input[0]->output : NULL_PARAM;
  Param v    = node->input[1] ? node->input[1]->output : NULL_PARAM;

  // printf("BACKWARD(%s) = [%dx%d]\n",
  //     Node_Type_Str[node->type],
  //     fwd_out.rows, fwd_out.cols
  // );

  switch (node->type) {
    case NODE_CONSTANT:
      // upstream:
      //   dL/d(c) = dL
      //
      // local:
      //   d(c)/dx = 0 (element-wise)
      //
      // downstream:
      //   dL/dx = 0 ⊙ dL = 0 (element-wise)
      break;

    case NODE_VARIABLE:
      // upstream:
      //   dL/d(x) = dL
      //
      // local:
      //   d(x)/dx = I (identity, element-wise)
      //
      // downstream:
      //   dL/dx = I ⊙ dL = dL (element-wise)
      break;

    case NODE_EMBEDDING:
      for (size_t t = 0; t < node->context_size; t++) {
        NMatrix grad = mat_row(node->weight.grad, node->context[t]);
        NMatrix ctx = mat_row_slice(dL, t*12, 12);
        mat_add(grad, grad, ctx);
      }
      break;

    case NODE_LINEAR:
      // upstream:
      //   dL/d(x*W + b) = dL
      //
      // local:
      //   d(x*W + b)/dW = x (outer product)
      //   d(x*W + b)/db = 1 (element-wise)
      //   d(x*W + b)/dx = W^T (matrix transpose)
      //
      // downstream (standard convention: W is N×M, x is 1×N, dL is 1×M):
      //   dL/dW = x^T * dL (matrix mult: [N×1] * [1×M] = [N×M])
      //   dL/db = dL (element-wise)
      //   dL/dx = dL * W^T (matrix mult: [1×M] * [M×N] = [1×N])
      //
      // dL/dW[i,j] = dL[j] * x[i]
      mat_mult_A_transposed_and_B_acc(w.grad, x.value, dL);
      // dL_db = I * dL
      mat_add(b.grad, b.grad, dL);
      // dL_dx = dL * W^T
      mat_mult_A_and_B_transposed_acc(x.grad, dL, w.value);
      break;

    case NODE_SIGMOID:
      // upstream:
      //   dL/d(σ(x)) = dL
      //
      // local:
      //   d(σ(x))/dx = σ(x) * (1 - σ(x)) (element-wise)
      //
      // downstream:
      //   dL/dx = (σ(x) * (1 - σ(x))) ⊙ dL (element-wise)
      dsigmoid(x.grad, fwd_out.value, dL);
      break;

    case NODE_SOFTMAX:
      // upstream:
      //   dL/d(softmax(x)) = dL
      //
      // local:
      //   d(softmax(x))/dx_ij = s_i * (δ_ij - s_i) (element-wise, row-wise)
      //
      // downstream:
      //   dL/dx_ij = s_i * (dL_ij - Σ_k dL_kj * s_i) (row-wise)
      // h = Softmax(x, t)                           | node->output.value = Softmax(x, t)
      // dL/dx = 1/t * h * [ dL/dh - dot(h, dL/dh) ] | x.grad = 1/t * node->output.value * [ dL - dot(node->output.value, dL) ]
      dsoftmax_temperature(x.grad, fwd_out.value, dL, node->temperature);
      break;

    case NODE_SOFTMAX_CROSS_ENTROPY:
      mat_sub(x.grad, fwd_out.value, dL);
      mat_scale(x.grad, x.grad, 1.0f / node->temperature);
      break;

    case NODE_RELU:
      // upstream:
      //   dL/d(ReLU(x)) = dL
      //
      // local:
      //   d(ReLU(x))/dx = 1 if x > 0 else 0 (element-wise)
      //
      // downstream:
      //   dL/dx = I(ReLU'(x)) ⊙ dL (element-wise), where I(...) is indicator
      drelu(x.grad, fwd_out.value, dL);
      break;

    case NODE_FLATTEN:
      // upstream:
      //   dL/d(concat(x₁, x₂, ..., xₙ)) = dL
      //
      // local:
      //   d(concat)/dx_i = I (identity, reshaping)
      //
      // downstream:
      //   dL/dx_i = dL_reshaped to shape of x_i (identity mapping)
      for (int i = 0; i < node->input_size; i++) {
        Node* input = node->input[i];

        int rows = node_get_value_rows(input);
        int cols = node_get_value_cols(input);
        int size = rows * cols;

        NMatrix ith_dL_dx = mat_reshape(
          mat_row_slice(dL, i*size, size),
          rows,
          cols
        );

        mat_copy(node->input[i]->output.grad, ith_dL_dx);
      }
      break;

    case NODE_CONV2D:
      // upstream:
      //   dL/d(conv2d(x, W)) = dL
      //
      // local:
      //   d(conv2d)/dW = x_padded ⊙ dL (element-wise)
      //   d(conv2d)/db = 1 (element-wise)
      //   d(conv2d)/dx = conv2d_transpose(dL, W)
      //
      // downstream:
      //   dL/dx = conv2d_transpose(dL, W)
      mat_fill(x.grad, 0);
      mat_fill(w.grad, 0);

      for (int i = 0; i < dL.rows; i++) {
        for (int j = 0; j < dL.cols; j++) {
          NMatrix dL_dx_reshaped = mat_reshape(
              x.grad,
              fwd_out.value.rows + w.grad.rows - 1,
              fwd_out.value.cols + w.grad.cols - 1
          );

          NMatrix x_reshaped = mat_reshape(
              x.value,
              fwd_out.value.rows + w.grad.rows - 1,
              fwd_out.value.cols + w.grad.cols - 1
          );

          float delta = MAT_AT(dL, i, j);
          // Bias gradient
          // grad_b[f] += delta;

          for(int u = 0; u < w.grad.rows; u++) {
            for(int v = 0; v < w.grad.cols; v++) {
              // Weight gradient
              MAT_AT(w.grad, u, v) += delta * MAT_AT(x_reshaped, i+u, j+v);

              // Input gradient
              MAT_AT(dL_dx_reshaped, i+u, j+v) += delta * MAT_AT(w.value, u, v);
            }
          }
        }
      }
      break;

    case NODE_SQUARE:
      // upstream:
      //   dL/d(x⊙x) = dL
      //
      // local:
      //   d(x⊙x)/dx = 2*x (element-wise)
      //
      // downstream:
      //   dL/dx = (2*x) ⊙ dL = 2*x*dL (element-wise)
      for (int i = 0; i < x.grad.rows; i++) {
        for (int j = 0; j < x.grad.cols; j++)
          MAT_AT(x.grad, i, j) += 2*MAT_AT(x.value, i, j)*MAT_AT(dL, i, j);
      }
      break;

    case NODE_CUBE:
      // upstream:
      //   dL/d(x⊙x⊙x) = dL
      //
      // local:
      //   d(x⊙x⊙x)/dx = 3*x⊙x (element-wise)
      //
      // downstream:
      //   dL/dx = (3*x⊙x) ⊙ dL = 3*x^2*dL (element-wise)
      for (int i = 0; i < x.grad.rows; i++) {
        for (int j = 0; j < x.grad.cols; j++)
          MAT_AT(x.grad, i, j) += 3*MAT_AT(x.value, i, j)*MAT_AT(x.value, i, j)*MAT_AT(dL, i, j);
      }
      break;

    case NODE_EXP:
      // upstream:
      //   dL/d(exp(x)) = dL
      //
      // local:
      //   d(exp(x))/dx = exp(x) (element-wise)
      //
      // downstream:
      //   dL/dx = exp(x) ⊙ dL (element-wise)
      for (int i = 0; i < dL.rows*dL.cols; i++) {
        x.grad.elems[i] += fwd_out.value.elems[i]*dL.elems[i];
      }
      break;

    case NODE_NEGATE:
      // upstream:
      //   dL/d(-x) = dL
      //
      // local:
      //   d(-x)/dx = -1 (element-wise)
      //
      // downstream:
      //   dL/dx = (-1) ⊙ dL = -dL (element-wise)
      for (int i = 0; i < dL.rows*dL.cols; i++) {
        x.grad.elems[i] += -dL.elems[i];
      }
      break;

    case NODE_MULTIPLY:
      // upstream:
      //   dL/d(u*v) = dL
      //
      // local:
      //   d(u*v)/du = v^T (matrix transpose)
      //   d(u*v)/dv = u^T (matrix transpose)
      //
      // downstream:
      //   dL/du = dL * v^T (matrix mult: [N×P] * [P×M] = [N×M])
      //   dL/dv = u^T * dL (matrix mult: [M×N] * [N×P] = [M×P])
      mat_mult_A_and_B_transposed(u.grad, dL, v.value);
      mat_mult_A_transposed_and_B(v.grad, u.value, dL);
      break;

    case NODE_DIVIDE:
      // upstream:
      //   dL/d(u⊘v) = dL
      //
      // local:
      //   d(u⊘v)/du = 1/v (element-wise)
      //   d(u⊘v)/dv = -u/v² (element-wise)
      //
      // downstream:
      //   dL/du = (1/v) ⊙ dL (element-wise)
      //   dL/dv = (-u/v²) ⊙ dL (element-wise)
      for (int i = 0; i < dL.rows*dL.cols; i++) {
        float inv_v = 1.0f / v.value.elems[i];

        u.grad.elems[i] += inv_v * dL.elems[i];
        v.grad.elems[i] += -u.value.elems[i] * inv_v * inv_v * dL.elems[i];
      }
      break;

    case NODE_ADD:
      // upstream:
      //   dL/d(u+v) = dL
      //
      // local:
      //   d(u+v)/du = I (identity, element-wise)
      //   d(u+v)/dv = I (identity, element-wise)
      //
      // downstream:
      //   dL/du = I ⊙ dL = dL (element-wise)
      //   dL/dv = I ⊙ dL = dL (element-wise)
      mat_add(u.grad, u.grad, dL);
      mat_add(v.grad, v.grad, dL);
      break;

    case NODE_SUB:
      // upstream:
      //   dL/d(u-v) = dL
      //
      // local:
      //   d(u-v)/du = I (identity, element-wise)
      //   d(u-v)/dv = -I (negative identity, element-wise)
      //
      // downstream:
      //   dL/du = I ⊙ dL = dL (element-wise)
      //   dL/dv = (-I) ⊙ dL = -dL (element-wise)
      for (int i = 0; i < dL.rows*dL.cols; i++) {
        u.grad.elems[i] += +dL.elems[i];
        v.grad.elems[i] += -dL.elems[i];
      }
      break;
  }

  node_backward(order);
}

// accumulate grads
void acc_grads(Node* node) {
  for (int i = 0; i < node->input_size; i++)
    acc_grads(node->input[i]);

  mat_add(node->output.g_grad, node->output.g_grad, node->output.grad);
  mat_add(node->weight.g_grad, node->weight.g_grad, node->weight.grad);
  mat_add(node->bias.g_grad, node->bias.g_grad, node->bias.grad);
}

void update_grads(Node* node, float lr, size_t batch_size) {
  for (int i = 0; i < node->input_size; i++)
    update_grads(node->input[i], lr, batch_size);

  if (node->type == NODE_EMBEDDING) {
    int max_vocab = 512;
    int embed_dim = 12;

    lr = 0.05f;

    for (int v = 0; v < max_vocab; v++) {
      if (v == 0) continue;
      if (node->embedding_tokens_batch[v] == 0) continue;

      node->embedding_tokens_batch[v] = 0;

      NMatrix emb = node->weight.value;
      NMatrix g_grad = node->weight.g_grad;
      NMatrix ada_grad = node->ada_grad;

      for (int d = 0; d < embed_dim; d++) {
        // average the gradient: ∇W = ∇W / B
        float g = MAT_AT(g_grad, v, d) / batch_size;

        // update AdaGrad: Gnew = Gold + ∇W^2
        MAT_AT(ada_grad, v, d) += g*g;

        // udpate the weight: Wnew = Wold - lr/sqrt(Gnew) * ∇W
        MAT_AT(emb, v, d) -= (lr / (sqrtf(MAT_AT(ada_grad, v, d)) + 1e-15f)) * g;

        MAT_AT(g_grad, v, d) = 0;
      }
    }
  } else {
    mat_weighted_add(node->output.value, node->output.value, node->output.g_grad, -lr);
    mat_weighted_add(node->weight.value, node->weight.value, node->weight.g_grad, -lr);
    mat_weighted_add(node->bias.value, node->bias.value, node->bias.g_grad, -lr);
  }
}

void zero_grads(Node* node) {
  for (int i = 0; i < node->input_size; i++)
    zero_grads(node->input[i]);

  mat_fill(node->output.grad, 0);
  mat_fill(node->weight.grad, 0);
  mat_fill(node->bias.grad, 0);
}

void zero_g_grads(Node* node) {
  for (int i = 0; i < node->input_size; i++)
    zero_g_grads(node->input[i]);

  mat_fill(node->output.g_grad, 0);
  mat_fill(node->weight.g_grad, 0);
  mat_fill(node->bias.g_grad, 0);
}

#endif // NODE_H
