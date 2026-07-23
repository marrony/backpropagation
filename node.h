#ifndef NODE_H
#define NODE_H

#include "nn.h"
#include "allocator.h"
#include "array.h"

typedef enum {
  NODE_CONSTANT,
  NODE_VARIABLE,
  NODE_EMBEDDING,
  NODE_EMBEDDING_CONV,
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
  STR(NODE_EMBEDDING_CONV),
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

#define MAX_INPUTS 4

typedef struct Node Node;
typedef struct Tensor Tensor;

typedef struct {
  Node* node;
  Tensor* output;
  Tensor* input[MAX_INPUTS];
} Tape_Node;

struct Tensor {
  NMatrix value;
  NMatrix grad;
};

/*
 * Node Structure - Computation Graph Node for Automatic Differentiation
 */
struct Node {
  Node_Type type;

  union {
    struct {
      Tensor value;
    } as_variable;

    struct {
      Node* input;
    } as_unary;

    struct {
      Node* input0;
      Node* input1;
    } as_binary;

    struct {
      Node* input;
      float a;
    } as_relu;

    struct {
      Node* input;
      float temperature;
      NMatrix target;
    } as_softmax;

    struct {
      Node* input;
      Tensor weight;
      Tensor bias;
    } as_conv2d;

    struct {
      Node* input;
      Tensor weight;
      Tensor bias;
    } as_linear;

    struct {
      Tensor embeddings;
      int32_t* context;
      size_t context_size;

      // conv1d
      size_t filter_count;
      Tensor* filters0;
      Tensor* filters1;
    } as_embedding;
  };
};

#define NULL_MATRIX (NMatrix){NULL, 0, 0}
#define NULL_PARAM (Param) {0}

void init_tensor(Tensor* param, int rows, int cols) {
  param->value  = mat_alloc(rows, cols);
  param->grad   = mat_alloc(rows, cols);
  mat_zero(param->value);
  mat_zero(param->grad);
}

void destroy_param(Tensor* param) {
  mat_free(&param->value);
  mat_free(&param->grad);
}

void destroy_node(Node** node) {
  // for (int i = 0; i < (*node)->input_size; i++)
  //   destroy_node(&(*node)->input[i]);
  // free(*node);
  *node = NULL;
}

DEFINE_ARRAY(Tape_Node);
DEFINE_ARRAY_ALIAS(Tensor_Ptr, Tensor*);
DEFINE_ARRAY(NMatrix);

typedef struct {
  Tensor_Ptr_Array tensors;
  NMatrix_Array history;
  NMatrix_Array second;
  float learning_rate;
  size_t updates;
} Optimizer;

void register_tensor(Optimizer* opt, Tensor* tensor) {
  NMatrix history = mat_alloc(tensor->grad.rows, tensor->grad.cols);
  NMatrix second = mat_alloc(tensor->grad.rows, tensor->grad.cols);
  mat_zero(history);
  mat_zero(second);

  array_append(&opt->tensors, tensor);
  array_append(&opt->history, history);
  array_append(&opt->second, second);
}

Node* create_node(Node_Type type) {
  Node* node = malloc(sizeof(Node));
  memset(node, 0, sizeof(Node));
  node->type = type;
  return node;
}

Tensor* create_tensor(int rows, int cols) {
  Tensor* tensor = malloc(sizeof(Tensor));
  tensor->value = mat_alloc(rows, cols);
  tensor->grad = mat_alloc(rows, cols);
  mat_zero(tensor->value);
  mat_zero(tensor->grad);
  return tensor;
}

Node* create_constant(int rows, int cols) {
  Node* node = create_node(NODE_CONSTANT);
  init_tensor(&node->as_variable.value, rows, cols);
  return node;
}

Node* create_variable(int rows, int cols) {
  Node* node = create_node(NODE_VARIABLE);
  init_tensor(&node->as_variable.value, rows, cols);
  return node;
}

Node* create_embeddings(Optimizer* optimizer, size_t max_vocab, size_t emb_dim, size_t context_size) {
  Node* node = create_node(NODE_EMBEDDING);
  init_tensor(&node->as_embedding.embeddings, max_vocab, emb_dim);

  node->as_embedding.context = malloc(sizeof(int32_t)*context_size);
  node->as_embedding.context_size = context_size;
  memset(node->as_embedding.context, 0, sizeof(int32_t)*context_size);

  register_tensor(optimizer, &node->as_embedding.embeddings);

  return node;
}

Node* create_embeddings_conv(Optimizer* optimizer, size_t max_vocab, size_t emb_dim, size_t context_size, size_t filter_count) {
  Node* node = create_node(NODE_EMBEDDING_CONV);
  init_tensor(&node->as_embedding.embeddings, max_vocab, emb_dim);

  node->as_embedding.context = malloc(sizeof(int32_t)*context_size);
  node->as_embedding.context_size = context_size;
  memset(node->as_embedding.context, 0, sizeof(int32_t)*context_size);

  register_tensor(optimizer, &node->as_embedding.embeddings);

  size_t kern_size = 3;
  node->as_embedding.filter_count = filter_count;
  node->as_embedding.filters0 = malloc(sizeof(Tensor)*filter_count);
  node->as_embedding.filters1 = malloc(sizeof(Tensor)*filter_count);
  memset(node->as_embedding.filters0, 0, sizeof(Tensor)*filter_count);
  memset(node->as_embedding.filters1, 0, sizeof(Tensor)*filter_count);
  for (size_t i = 0; i < filter_count; i++) {
    init_tensor(&node->as_embedding.filters0[i], kern_size, emb_dim);
    init_tensor(&node->as_embedding.filters1[i], kern_size, 48);

    register_tensor(optimizer, &node->as_embedding.filters0[i]);
    register_tensor(optimizer, &node->as_embedding.filters1[i]);
  }

  return node;
}

Node* create_linear(Optimizer* optimizer, Node* input, int in_size, int out_size) {
  Node* node = create_node(NODE_LINEAR);
  node->as_linear.input = input;

  // Standard convention: W is (input_dim × output_dim) = (in_size × out_size)
  // Forward: output = input × W + b, where input is (1×in_size), W is (in_size×out_size), output is (1×out_size)
  init_tensor(&node->as_linear.weight, in_size, out_size);
  init_tensor(&node->as_linear.bias, 1, out_size);

  register_tensor(optimizer, &node->as_linear.weight);
  register_tensor(optimizer, &node->as_linear.bias);

  return node;
}

Node* create_unary(Node_Type type, Node* input) {
  Node* node = create_node(type);
  node->as_unary.input = input;
  return node;
}

Node* create_binary(Node_Type type, Node* input0, Node* input1) {
  Node* node = create_node(type);
  node->as_binary.input0 = input0;
  node->as_binary.input1 = input1;
  return node;
}

Node* create_sigmoid(Node* input) {
  return create_unary(NODE_SIGMOID, input);
}

Node* create_softmax(Node* input) {
  Node* node = create_unary(NODE_SOFTMAX, input);
  node->as_softmax.temperature = 1.0f;
  return node;
}

Node* create_softmax_cross_entropy(Node* input, NMatrix target) {
  Node* node = create_unary(NODE_SOFTMAX_CROSS_ENTROPY, input);
  node->as_softmax.temperature = 1.0f;
  node->as_softmax.target = target;
  return node;
}

Node* create_relu(Node* input, float a) {
  Node* node = create_unary(NODE_RELU, input);
  node->as_relu.a = a;
  return node;
}

Node* create_conv2d(Node* input, int img_size, int kern_size) {
  Node* node = create_node(NODE_CONV2D);
  node->as_conv2d.input = input;

  int conv_size = (img_size - kern_size + 1);

  // init_param(&node->output, conv_size, conv_size);
  init_tensor(&node->as_conv2d.weight, kern_size, kern_size);
  init_tensor(&node->as_conv2d.bias, 1, conv_size);

  // mat_rand(node->weight.value);
  // mat_rand(node->bias.value);

  return node;
}

// Create a flatten node that concatenates multiple input matrices.
// IMPORTANT: This uses mat_reshape which is a VIEW operation, not a copy.
// The flattened output shares the same data buffer with input matrices.
// This is safe as long as the input matrices are not modified after flattening.
Node* create_flatten(Node** inputs, int in_size) {
  (void)inputs;
  assert(in_size < MAX_INPUTS);
  Node* node = create_node(NODE_FLATTEN);

  (void)in_size;

  // int size = 0;
  // for (int i = 0; i < in_size; i++) {
  //   node->input[i] = inputs[i];
  //   int rows = node_get_value_rows(inputs[i]);
  //   int cols = node_get_value_cols(inputs[i]);
  //   size += rows * cols;
  // }
  //
  return node;
}

Node* create_square(Node* input) {
  return create_unary(NODE_SQUARE, input);
}

Node* create_cube(Node* input) {
  return create_unary(NODE_CUBE, input);
}

Node* create_exp(Node* input) {
  return create_unary(NODE_EXP, input);
}

Node* create_negate(Node* input) {
  return create_unary(NODE_NEGATE, input);
}

Node* create_multiply(Node* input0, Node* input1) {
  // [NxM] * [MxP] = [NxP]
  return create_binary(NODE_MULTIPLY, input0, input1);
}

Node* create_divide(Node* input0, Node* input1) {
  return create_binary(NODE_DIVIDE, input0, input1);
}

Node* create_add(Node* input0, Node* input1) {
  return create_binary(NODE_ADD, input0, input1);
}

Node* create_sub(Node* input0, Node* input1) {
  return create_binary(NODE_SUB, input0, input1);
}

void node_conv1d(
  Tensor* out,
  Tensor* embeddings,
  Tensor* filters,
  int32_t* context,
  int32_t row_start,
  int32_t rows_out,
  int32_t filter_count) {
  for (int i = 0; i < rows_out; i++) {   // 0..ctx_size-kern_size+1
    int current_row = row_start + i;

    for (int j = 0; j < filter_count; j++) {      // 0..filter_count
      NMatrix w = filters[j].value;

      float sum = 0;
      for (int ki = 0; ki < w.rows; ki++) {     // 0..kern_size
        int32_t token = context != NULL ? context[current_row+ki] : current_row+ki;
        NMatrix x = mat_row(embeddings->value, token);

        for(int kj = 0; kj < w.cols; kj++)      // 0..emb_size
          sum += MAT_AT(x, 0, kj) * MAT_AT(w, ki, kj);
      }

      int row = rows_out == 1 ? 0 : current_row;
      MAT_AT(out->value, row, j) = sum;
    }
  }
}

void node_dconv1d(
  Tensor* embeddings,
  Tensor* filters,
  NMatrix dLdy,
  int32_t* context,
  int32_t row_start,
  int32_t rows_out,
  int32_t filter_count) {
  for (int i = 0; i < rows_out; i++) {    // 0..ctx_size-kern_size+1
    int current_row = row_start + i;

    for (int j = 0; j < filter_count; j++) {       // 0..filter_count
      int row = rows_out == 1 ? 0 : current_row;
      float delta = MAT_AT(dLdy, row, j);
      // Bias gradient
      // grad_b[f] += delta;

      Tensor w = filters[j];

      for(int ki = 0; ki < w.grad.rows; ki++) {     // 0..kern_size
        int32_t token = context != NULL ? context[current_row+ki] : current_row+ki;
        NMatrix x = mat_row(embeddings->value, token);
        NMatrix dLdx = mat_row(embeddings->grad, token);

        for(int kj = 0; kj < w.grad.cols; kj++) {   // 0..emb_size
          // Weight gradient
          MAT_AT(w.grad, ki, kj) += delta * MAT_AT(x, 0, kj);

          // Input gradient
          MAT_AT(dLdx, 0, kj) += delta * MAT_AT(w.value, ki, kj);
        }
      }
    }
  }
}

Tensor* node_forward(Arena_Allocator* arena, Node* node, Tape_Node_Array* tape) {
  size_t saved = SAVE(&arena->alloc);

  Tensor* out = ALLOC(&arena->alloc, sizeof(Tensor)).void_ptr;
  memset(out, 0, sizeof(Tensor));
  out->value = NULL_MATRIX;
  out->grad = NULL_MATRIX;

  Tape_Node tape_node = {
    .node = node,
    .output = out,
  };

  switch (node->type) {
    case NODE_CONSTANT:
    case NODE_VARIABLE:
      // delete Tensor allocated
      RESTORE(&arena->alloc, saved);
      out = &node->as_variable.value;
      break;

    case NODE_EMBEDDING: {
      int cols = node->as_embedding.embeddings.value.cols;
      out->value = mat_alloc2(&arena->alloc, 1, node->as_embedding.context_size*cols);
      out->grad = mat_alloc2(&arena->alloc, 1, node->as_embedding.context_size*cols);
      mat_zero(out->grad);

      int32_t* context = (int32_t*)ALLOC(&arena->alloc, sizeof(int32_t)*node->as_embedding.context_size).ptr;
      memcpy(context, node->as_embedding.context, sizeof(int32_t)*node->as_embedding.context_size);
      
      tape_node.input[0] = (Tensor*)context;

      for (size_t t = 0; t < node->as_embedding.context_size; t++) {
        mat_copy(
            mat_row_slice(mat_row(out->value, 0), t*cols, cols),
            mat_row(node->as_embedding.embeddings.value, context[t])
        );
      }
      break;
    }

    case NODE_EMBEDDING_CONV: {
      int kern_size = 3;
      int filter_count = node->as_embedding.filter_count;
      int rows_out_1 = node->as_embedding.context_size - kern_size + 1;
      int rows_out_2 = rows_out_1 - kern_size + 1;

      Tensor* layer1_out = ALLOC(&arena->alloc, sizeof(Tensor)).void_ptr;
      layer1_out->value = mat_alloc2(&arena->alloc, rows_out_1, filter_count);
      layer1_out->grad = mat_alloc2(&arena->alloc, rows_out_1, filter_count);
      mat_zero(layer1_out->value);
      mat_zero(layer1_out->grad);

      out->value = mat_alloc2(&arena->alloc, 1, filter_count);
      out->grad = mat_alloc2(&arena->alloc, 1, filter_count);
      mat_zero(out->value);
      mat_zero(out->grad);

      int32_t* context = (int32_t*)ALLOC(&arena->alloc, sizeof(int32_t)*node->as_embedding.context_size).ptr;
      memcpy(context, node->as_embedding.context, sizeof(int32_t)*node->as_embedding.context_size);

      tape_node.input[0] = (Tensor*)context;
      tape_node.input[1] = layer1_out;

      node_conv1d(layer1_out, &node->as_embedding.embeddings, node->as_embedding.filters0, context, 0, rows_out_1, filter_count);
      node_conv1d(out, layer1_out, node->as_embedding.filters1, NULL, rows_out_2-1, 1, filter_count);
      break;
    }

    case NODE_LINEAR: {
      // f(x) = x*W + b (standard convention: W is N×M, x is 1×N, output is 1×M)
      // x*W: [1×N] * [N×M] = [1×M]
      // x*W + b: [1×M] + [1×M] = [1×M]
      Tensor* x_tensor = node_forward(arena, node->as_linear.input, tape);

      tape_node.input[0] = x_tensor;
      out->value = mat_alloc2(&arena->alloc, x_tensor->value.rows, node->as_linear.weight.value.cols);
      out->grad = mat_alloc2(&arena->alloc, x_tensor->value.rows, node->as_linear.weight.value.cols);
      mat_zero(out->grad);

      // broadcasting the bias
      mat_mult(out->value, x_tensor->value, node->as_linear.weight.value);
      for (int i = 0; i < out->value.rows; i++) {
        NMatrix dst = mat_row(out->value, i);
        mat_add(dst, dst, node->as_linear.bias.value);
      }
      break;
    }

    case NODE_SIGMOID: {
      // f(x) = σ(x) (element-wise sigmoid)
      // σ(x) = 1 / (1 + e^(-x))
      Tensor* x_tensor = node_forward(arena, node->as_unary.input, tape);
      sigmoid(out->value, x_tensor->value);
      break;
    }

    case NODE_SOFTMAX: {
      // f(x) = softmax(x) (row-wise normalization)
      // softmax(x)_j = exp(x_j) / Σ_k exp(x_k)
      Tensor* logits_tensor = node_forward(arena, node->as_softmax.input, tape);
      tape_node.input[0] = logits_tensor;
      out->value = mat_alloc2(&arena->alloc, logits_tensor->value.rows, logits_tensor->value.cols);
      out->grad = mat_alloc2(&arena->alloc, logits_tensor->value.rows, logits_tensor->value.cols);

      for (int i = 0; i < out->value.rows; i++) {
        softmax_temperature(
            mat_row(out->value, i),
            mat_row(logits_tensor->value, i),
            node->as_softmax.temperature
        );
      }
      break;
    }

    case NODE_SOFTMAX_CROSS_ENTROPY: {
      Tensor* logits_tensor = node_forward(arena, node->as_softmax.input, tape);
      Tensor* target_tensor = ALLOC(&arena->alloc, sizeof(Tensor)).void_ptr;
      Tensor* probs_tensor = ALLOC(&arena->alloc, sizeof(Tensor)).void_ptr;

      target_tensor->value = mat_alloc2(&arena->alloc, logits_tensor->value.rows, logits_tensor->value.cols);
      target_tensor->grad = NULL_MATRIX;
      mat_copy(target_tensor->value, node->as_softmax.target);

      probs_tensor->value = mat_alloc2(&arena->alloc, logits_tensor->value.rows, logits_tensor->value.cols);
      probs_tensor->grad = NULL_MATRIX;

      tape_node.input[0] = logits_tensor;
      tape_node.input[1] = target_tensor;
      tape_node.input[2] = probs_tensor;

      int batch_size = probs_tensor->value.rows;

      out->value = mat_alloc2(&arena->alloc, batch_size, 1);
      out->grad = mat_alloc2(&arena->alloc, batch_size, 1);
      mat_zero(out->grad);

      for (int b = 0; b < batch_size; b++) {
        MAT_AT(out->value, b, 0) = softmax_cross_entropy_temperature(
            mat_row(probs_tensor->value, b),
            mat_row(logits_tensor->value, b),
            mat_row(target_tensor->value, b),
            node->as_softmax.temperature);
      }

      break;
    }

    case NODE_RELU: {
      // f(x) = ReLU(x) (element-wise rectified linear unit)
      // ReLU(x) = max(0, x)
      Tensor* x_tensor = node_forward(arena, node->as_relu.input, tape);
      tape_node.input[0] = x_tensor;
      out->value = mat_alloc2(&arena->alloc, x_tensor->value.rows, x_tensor->value.cols);
      out->grad = mat_alloc2(&arena->alloc, x_tensor->value.rows, x_tensor->value.cols);
      mat_zero(out->grad);
      relu(out->value, x_tensor->value, node->as_relu.a);
      break;
    }

    case NODE_FLATTEN:
      // // f(x₁, x₂, ..., xₙ) = concat(x₁, x₂, ..., xₙ)
      // // Reshapes each input to (1, size_i) and concatenates horizontally
      // // [size₁] ⊕ [size₂] ⊕ ... ⊕ [sizeₙ] = [Σ size_i]
      // for (int i = 0; i < node->input_size; i++) {
      //   Tensor* x = node_forward(node->input[i], tape);
      //
      //   Node* input = node->input[i];
      //
      //   int rows = node_get_value_rows(input);
      //   int cols = node_get_value_cols(input);
      //   int size = rows * cols;
      //
      //   NMatrix dst = mat_reshape(
      //       mat_row_slice(fx.value, i*size, size),
      //       rows,
      //       cols
      //   );
      //
      //   mat_copy(dst, input->output.value);
      // }
      break;

    case NODE_CONV2D:
      // // f(x) = conv2d(x, weight)
      // node_forward(node->input[0], order);
      //
      // NMatrix x_ = mat_reshape(x.value,
      //     fx.value.rows + w.value.rows - 1,
      //     fx.value.cols + w.value.cols - 1
      // );
      //
      // for (int i = 0; i < fx.value.rows; i++) {
      //   for (int j = 0; j < fx.value.cols; j++) {
      //     MAT_AT(fx.value, i, j) = 0; // bias
      //
      //     for (int ki = 0; ki < w.value.rows; ki++) {
      //       for (int kj = 0; kj < w.value.cols; kj++) {
      //         float a = MAT_AT(x_, i+ki, j+kj);
      //         float b = MAT_AT(w.value, ki, kj);
      //         MAT_AT(fx.value, i, j) += a * b;
      //       }
      //     }
      //   }
      // }
      break;

    case NODE_SQUARE:
      // // f(x) = x ⊙ x (element-wise squaring)
      // // [N×M] ⊙ [N×M] = [N×M]
      // node_forward(node->input[0], order);
      // for (int i = 0; i < x.value.rows; i++) {
      //   for (int j = 0; j < x.value.cols; j++)
      //     MAT_AT(fx.value, i, j) = MAT_AT(x.value, i, j)*MAT_AT(x.value, i, j);
      // }
      break;

    case NODE_CUBE:
      // // f(x) = x ⊙ x ⊙ x (element-wise cubing)
      // // [N×M] ⊙ [N×M] ⊙ [N×M] = [N×M]
      // node_forward(node->input[0], order);
      // for (int i = 0; i < x.value.rows; i++) {
      //   for (int j = 0; j < x.value.cols; j++)
      //     MAT_AT(fx.value, i, j) = MAT_AT(x.value, i, j)*MAT_AT(x.value, i, j)*MAT_AT(x.value, i, j);
      // }
      break;

    case NODE_EXP:
      // // f(x) = exp(x) (element-wise exponential)
      // // exp(x)_ij = e^(x_ij)
      // node_forward(node->input[0], order);
      // for (int i = 0; i < x.value.rows; i++) {
      //   for (int j = 0; j < x.value.cols; j++)
      //     MAT_AT(fx.value, i, j) = expf(MAT_AT(x.value, i, j));
      // }
      break;

    case NODE_NEGATE:
      // // f(x) = -x (element-wise negation)
      // // [N×M] -> [N×M]
      // node_forward(node->input[0], order);
      // mat_scale(fx.value, x.value, -1);
      break;

    case NODE_MULTIPLY: {
      // // f(u, v) = u * v (matrix multiplication)
      // // [N×M] * [M×P] = [N×P]
      Tensor* u_tensor = node_forward(arena, node->as_binary.input0, tape);
      Tensor* v_tensor = node_forward(arena, node->as_binary.input1, tape);
      tape_node.input[0] = u_tensor;
      tape_node.input[1] = v_tensor;
      out->value = mat_alloc2(&arena->alloc, u_tensor->value.rows, v_tensor->value.cols);
      out->grad  = mat_alloc2(&arena->alloc, u_tensor->value.rows, v_tensor->value.cols);
      mat_zero(out->grad);
      mat_mult(out->value, u_tensor->value, v_tensor->value);
    }

    case NODE_DIVIDE:
      // // f(u, v) = u ⊘ v (element-wise division)
      // // [N×M] ⊘ [N×M] = [N×M]
      // node_forward(node->input[0], order);
      // node_forward(node->input[1], order);
      // mat_memberwise_div(fx.value, u.value, v.value, 1e-15f);
      break;

    case NODE_ADD: {
      // // f(u, v) = u + v
      Tensor* u_tensor = node_forward(arena, node->as_binary.input0, tape);
      Tensor* v_tensor = node_forward(arena, node->as_binary.input1, tape);
      tape_node.input[0] = u_tensor;
      tape_node.input[1] = v_tensor;
      out->value = mat_alloc2(&arena->alloc, u_tensor->value.rows, v_tensor->value.cols);
      out->grad  = mat_alloc2(&arena->alloc, u_tensor->value.rows, v_tensor->value.cols);
      mat_zero(out->grad);
      mat_add(out->value, u_tensor->value, v_tensor->value);
      break;
   }

    case NODE_SUB:
      // // f(u, v) = u - v
      // node_forward(node->input[0], order);
      // node_forward(node->input[1], order);
      // mat_sub(fx.value, u.value, v.value);
      break;
  }

  // printf(
  //     "FORWARD(%s) = [%dx%d]\n",
  //     Node_Type_Str[node->type],
  //     out->value.rows, out->value.cols
  // );

  array_append(tape, tape_node);
  return out;
}

//          y = f(x)
//   upstream = dL/dy
//      local = dy/dx
// downstream = dL/dx
//
// downstream = local * upstream
//      dL/dx = dy/dx * dL/dy
void node_backward(Tape_Node_Array* tape) {
  if (tape->count == 0) return;

  Tape_Node tape_node = array_pop_last(tape);
  Node* node = tape_node.node;

  NMatrix dLdy = tape_node.output->grad;

  // printf("BACKWARD(%s) = [%dx%d]\n",
  //     Node_Type_Str[node->type],
  //     tape_node.output->value.rows,
  //     tape_node.output->value.cols
  // );

  switch (node->type) {
    case NODE_CONSTANT:
      // upstream:
      //   dL/d(c) = dL/dy
      //
      // local:
      //   d(c)/dx = 0 (element-wise)
      //
      // downstream:
      //   dL/dx = 0 ⊙ dL/dy = 0 (element-wise)
      mat_zero(node->as_variable.value.grad);
      break;

    case NODE_VARIABLE:
      // upstream:
      //   dL/d(x) = dL/dy
      //
      // local:
      //   d(x)/dx = I (identity, element-wise)
      //
      // downstream:
      //   dL/dx = I ⊙ dL/dy = dL/dy (element-wise)
      // mat_copy(node->value.grad, dLdy);
      break;

    case NODE_EMBEDDING: {
      int cols = node->as_embedding.embeddings.value.cols;
      int32_t* context = (int32_t*)tape_node.input[0];

      for (size_t t = 0; t < node->as_embedding.context_size; t++) {
        if (context[t] == 0) continue;
        NMatrix grad = mat_row(node->as_embedding.embeddings.grad, context[t]);
        NMatrix ctx = mat_row_slice(dLdy, t*cols, cols);
        mat_add(grad, grad, ctx);
      }
      break;
    }

    case NODE_EMBEDDING_CONV: {
      int kern_size = 3;
      int filter_count = node->as_embedding.filter_count;
      int rows_out_1 = node->as_embedding.context_size - kern_size + 1;
      int rows_out_2 = rows_out_1 - kern_size + 1;

      int32_t* context = (int32_t*)tape_node.input[0];
      Tensor* layer1_out = tape_node.input[1];

      node_dconv1d(layer1_out, node->as_embedding.filters1, dLdy, NULL, rows_out_2-1, 1, filter_count);
      node_dconv1d(&node->as_embedding.embeddings, node->as_embedding.filters0, layer1_out->grad, context, 0, rows_out_1, filter_count);
      break;
    }

    case NODE_LINEAR:
      // upstream:
      //   dL/d(x*W + b) = dL/dy
      //
      // local:
      //   d(x*W + b)/dW = x
      //   d(x*W + b)/db = I
      //   d(x*W + b)/dx = W^T
      //
      // downstream:
      //   dL/dW = x^T * dL/dy (matrix mult: [P×N] * [N×M] = [P×M])
      //   dL/db = dL/dy       (element-wise)
      //   dL/dx = dL/dy * W^T (matrix mult: [N×M] * [M×P] = [N×P])
      //
      // standard convention:
      //   x is [N×P], W is [P×M], b is [NxM], dL/dy is [N×M]

      // dL_dW = x^T * dL/dy
      mat_mult_A_transposed_and_B_acc(node->as_linear.weight.grad, tape_node.input[0]->value, dLdy);
      // dL_db = I * dL/dy
      mat_add(node->as_linear.bias.grad, node->as_linear.bias.grad, dLdy);
      // dL_dx = dL/dy * W^T
      mat_mult_A_and_B_transposed_acc(tape_node.input[0]->grad, dLdy, node->as_linear.weight.value);
      break;

    case NODE_SIGMOID:
      // upstream:
      //   dL/d(σ(x)) = dL/dy
      //
      // local:
      //   d(σ(x))/dx = σ(x) * (1 - σ(x)) (element-wise)
      //
      // downstream:
      //   dL/dx = (σ(x) * (1 - σ(x))) ⊙ dL/dy (element-wise)
      // dsigmoid(x.grad, fwd_out.value, dLdy);
      break;

    case NODE_SOFTMAX:
      // upstream:
      //   dL/d(softmax(x)) = dL/dy
      //
      // local:
      //   d(softmax(x))/dx_ij = s_i * (δ_ij - s_i) (element-wise, row-wise)
      //
      // downstream:
      //   dL/dx_ij = s_i * (dL_ij - Σ_k dL_kj * s_i) (row-wise)
      dsoftmax_temperature(
          tape_node.input[0]->grad,
          tape_node.output->value,
          dLdy,
          node->as_softmax.temperature);
      break;

    case NODE_SOFTMAX_CROSS_ENTROPY: {
      Tensor* logits_tensor = tape_node.input[0];
      Tensor* target_tensor = tape_node.input[1];
      Tensor* probs_tensor = tape_node.input[2];

      float inv_temperature = 1.0f / node->as_softmax.temperature;
      float dL_dLoss = MAT_AT(dLdy, 0, 0);

      // 1/t * (y - target) * dLdy
      for (int i = 0; i < logits_tensor->grad.cols; i++) {
        float prob = VEC_AT(probs_tensor->value, i);
        float target = VEC_AT(target_tensor->value, i);
        VEC_AT(logits_tensor->grad, i) += dL_dLoss * inv_temperature * (prob - target);
      }
      break;
    }

    case NODE_RELU:
      // upstream:
      //   dL/d(relu(x)) = dL/dy
      //
      // local:
      //   d(relu(x))/dx = 1 if x > 0 else 0 (element-wise)
      //
      // downstream:
      //   dL/dx = I(relu'(x)) ⊙ dL/dy (element-wise)
      drelu(tape_node.input[0]->grad, tape_node.input[0]->value, dLdy, node->as_relu.a);
      break;

    case NODE_FLATTEN:
      // upstream:
      //   dL/d(concat(x₁, x₂, ..., xₙ)) = dL/dy
      //
      // local:
      //   d(concat)/dx_i = I (identity, reshaping)
      //
      // downstream:
      //   dL/dx_i = dL_reshaped to shape of x_i (identity mapping)
      // for (int i = 0; i < node->input_size; i++) {
      //   Node* input = node->input[i];
      //
      //   int rows = node_get_value_rows(input);
      //   int cols = node_get_value_cols(input);
      //   int size = rows * cols;
      //
      //   NMatrix ith_dL_dx = mat_reshape(
      //     mat_row_slice(dLdy, i*size, size),
      //     rows,
      //     cols
      //   );
      //
      //   mat_copy(node->input[i]->output.grad, ith_dL_dx);
      // }
      break;

    case NODE_CONV2D:
      // upstream:
      //   dL/d(conv2d(x, W)) = dL/dy
      //
      // local:
      //   d(conv2d)/dW = x_padded ⊙ dL/dy (element-wise)
      //   d(conv2d)/db = 1 (element-wise)
      //   d(conv2d)/dx = conv2d_transpose(dL/dy, W)
      //
      // downstream:
      //   dL/dx = conv2d_transpose(dL/dy, W)
      // mat_zero(x.grad, 0);
      // mat_zero(w.grad, 0);
      //
      // for (int i = 0; i < dLdy.rows; i++) {
      //   for (int j = 0; j < dLdy.cols; j++) {
      //     NMatrix dL_dx_reshaped = mat_reshape(
      //         x.grad,
      //         fwd_out.value.rows + w.grad.rows - 1,
      //         fwd_out.value.cols + w.grad.cols - 1
      //     );
      //
      //     NMatrix x_reshaped = mat_reshape(
      //         x.value,
      //         fwd_out.value.rows + w.grad.rows - 1,
      //         fwd_out.value.cols + w.grad.cols - 1
      //     );
      //
      //     float delta = MAT_AT(dLdy, i, j);
      //     // Bias gradient
      //     // grad_b[f] += delta;
      //
      //     for(int u = 0; u < w.grad.rows; u++) {
      //       for(int v = 0; v < w.grad.cols; v++) {
      //         // Weight gradient
      //         MAT_AT(w.grad, u, v) += delta * MAT_AT(x_reshaped, i+u, j+v);
      //
      //         // Input gradient
      //         MAT_AT(dL_dx_reshaped, i+u, j+v) += delta * MAT_AT(w.value, u, v);
      //       }
      //     }
      //   }
      // }
      break;

    case NODE_SQUARE:
      // upstream:
      //   dL/d(x⊙x) = dL/dy
      //
      // local:
      //   d(x⊙x)/dx = 2*x (element-wise)
      //
      // downstream:
      //   dL/dx = (2*x) ⊙ dL/dy = 2*x*dL/dy (element-wise)
      // for (int i = 0; i < x.grad.rows; i++) {
      //   for (int j = 0; j < x.grad.cols; j++)
      //     MAT_AT(x.grad, i, j) += 2*MAT_AT(x.value, i, j)*MAT_AT(dLdy, i, j);
      // }
      break;

    case NODE_CUBE:
      // upstream:
      //   dL/d(x⊙x⊙x) = dL/dy
      //
      // local:
      //   d(x⊙x⊙x)/dx = 3*x⊙x (element-wise)
      //
      // downstream:
      //   dL/dx = (3*x⊙x) ⊙ dL/dy = 3*x^2*dL/dy (element-wise)
      // for (int i = 0; i < x.grad.rows; i++) {
      //   for (int j = 0; j < x.grad.cols; j++)
      //     MAT_AT(x.grad, i, j) += 3*MAT_AT(x.value, i, j)*MAT_AT(x.value, i, j)*MAT_AT(dLdy, i, j);
      // }
      break;

    case NODE_EXP:
      // upstream:
      //   dL/d(exp(x)) = dL/dy
      //
      // local:
      //   d(exp(x))/dx = exp(x) (element-wise)
      //
      // downstream:
      //   dL/dx = exp(x) ⊙ dL/dy (element-wise)
      // for (int i = 0; i < dLdy.rows*dLdy.cols; i++) {
      //   x.grad.elems[i] += fwd_out.value.elems[i]*dLdy.elems[i];
      // }
      break;

    case NODE_NEGATE:
      // upstream:
      //   dL/d(-x) = dL/dy
      //
      // local:
      //   d(-x)/dx = -1 (element-wise)
      //
      // downstream:
      //   dL/dx = (-1) ⊙ dL/dy = -dL/dy (element-wise)
      // for (int i = 0; i < dLdy.rows*dLdy.cols; i++) {
      //   x.grad.elems[i] += -dLdy.elems[i];
      // }
      break;

    case NODE_MULTIPLY:
      // upstream:
      //   dL/d(u*v) = dL/dy
      //
      // local:
      //   d(u*v)/du = v^T (matrix transpose)
      //   d(u*v)/dv = u^T (matrix transpose)
      //
      // downstream:
      //   dL/du = dL/dy * v^T (matrix mult: [N×M] * [M×P] = [N×P])
      //   dL/dv = u^T * dL/dy (matrix mult: [P×N] * [N×M] = [P×M])
      //
      // standard convention:
      //   u is [N×P], v is [P×M], dL/dy is [N×M]

      // dL/du = dL/dy * v^T
      mat_mult_A_and_B_transposed_acc(tape_node.input[0]->grad, dLdy, tape_node.input[1]->value);
      // dL/dv = u^T * dL/dy
      mat_mult_A_transposed_and_B_acc(tape_node.input[1]->grad, tape_node.input[0]->value, dLdy);
      break;

    case NODE_DIVIDE:
      // upstream:
      //   dL/d(u⊘v) = dL/dy
      //
      // local:
      //   d(u⊘v)/du = 1/v (element-wise)
      //   d(u⊘v)/dv = -u/v² (element-wise)
      //
      // downstream:
      //   dL/du = (1/v) ⊙ dL/dy (element-wise)
      //   dL/dv = (-u/v²) ⊙ dL/dy (element-wise)

      for (int i = 0; i < dLdy.rows*dLdy.cols; i++) {
        float inv_v = 1.0f / VEC_AT(tape_node.input[1]->value, i);

        VEC_AT(tape_node.input[0]->grad, i) += inv_v * VEC_AT(dLdy, i);
        VEC_AT(tape_node.input[1]->grad, i) -= VEC_AT(tape_node.input[0]->value, i) * inv_v * inv_v * VEC_AT(dLdy, i);
      }
      break;

    case NODE_ADD:
      // upstream:
      //   dL/d(u+v) = dL/dy
      //
      // local:
      //   d(u+v)/du = I (identity, element-wise)
      //   d(u+v)/dv = I (identity, element-wise)
      //
      // downstream:
      //   dL/du = I ⊙ dL/dy = dL/dy (element-wise)
      //   dL/dv = I ⊙ dL/dy = dL/dy (element-wise)

      mat_add(tape_node.input[0]->grad, tape_node.input[0]->grad, dLdy);
      mat_add(tape_node.input[1]->grad, tape_node.input[1]->grad, dLdy);
      break;

    case NODE_SUB:
      // upstream:
      //   dL/d(u-v) = dL/dy
      //
      // local:
      //   d(u-v)/du = I (identity, element-wise)
      //   d(u-v)/dv = -I (negative identity, element-wise)
      //
      // downstream:
      //   dL/du = I ⊙ dL/dy = dL/dy (element-wise)
      //   dL/dv = (-I) ⊙ dL/dy = -dL/dy (element-wise)

      mat_add(tape_node.input[0]->grad, tape_node.input[0]->grad, dLdy);
      mat_sub(tape_node.input[1]->grad, tape_node.input[1]->grad, dLdy);
      // }
      break;
  }

  node_backward(tape);
}

void update_grads_sgd(Optimizer* optimizer, size_t batch_size) {
  (void)batch_size;

  for (size_t i = 0; i < optimizer->tensors.count; i++) {
    Tensor* tensor = optimizer->tensors.elems[i];

    int elems = tensor->value.rows * tensor->value.cols;

    for (int j = 0; j < elems; j++) {
      float g = tensor->grad.elems[j]; // / batch_size;

      float adapt_lr = optimizer->learning_rate;
      tensor->value.elems[j] -= adapt_lr * g;

      tensor->grad.elems[j] = 0;
    }
  }
}

void update_grads_adam(Optimizer* optimizer, Tensor* emb, int32_t* tokens_in_batch) {
  float eps = 1e-7f;
  float beta1 = 0.900f;
  float beta2 = 0.999f;

  optimizer->updates += 1;

  float beta1_correct = 1.0f - powf(beta1, optimizer->updates);
  float beta2_correct = 1.0f - powf(beta2, optimizer->updates);

  for (size_t t = 0; t < optimizer->tensors.count; t++) {
    Tensor* tensor = optimizer->tensors.elems[t];
    NMatrix m = optimizer->history.elems[t];
    NMatrix v = optimizer->second.elems[t];

    for (int i = 0; i < tensor->value.rows; i++) {
      if (tensor == emb && i == 0) {
        mat_zero(mat_row(tensor->grad, i));
        continue;
      }
      if (tensor == emb && tokens_in_batch[i] == 0) {
        mat_zero(mat_row(tensor->grad, i));
        continue;
      }

      for (int j = 0; j < tensor->value.cols; j++) {
        float g = MAT_AT(tensor->grad, i, j);

        float m_value = beta1*MAT_AT(m, i, j) + (1.0f - beta1)*g;
        float v_value = beta2*MAT_AT(v, i, j) + (1.0f - beta2)*g*g;

        MAT_AT(m, i, j) = m_value;
        MAT_AT(v, i, j) = v_value;

        float m_hat = MAT_AT(m, i, j) / beta1_correct;
        float v_hat = MAT_AT(v, i, j) / beta2_correct;

        float adapt_lr = optimizer->learning_rate / (sqrtf(v_hat) + eps);

        MAT_AT(tensor->value, i, j) -= adapt_lr * m_hat;
      }

      mat_zero(mat_row(tensor->grad, i));
    }
  }
}

void update_grads_rms_prop(Optimizer* optimizer, size_t batch_size, Tensor* emb, int32_t* tokens_in_batch) {
  (void)batch_size;

  float eps = 1e-8f;
  float beta = 0.99f;
  float one_minus_beta = 1.0f - beta;

  for (size_t t = 0; t < optimizer->tensors.count; t++) {
    Tensor* tensor = optimizer->tensors.elems[t];
    NMatrix h = optimizer->history.elems[t];

    for (int i = 0; i < tensor->value.rows; i++) {
      if (tensor == emb && i == 0) {
        mat_zero(mat_row(tensor->grad, i));
        continue;
      }
      if (tensor == emb && tokens_in_batch[i] == 0) {
        mat_zero(mat_row(tensor->grad, i));
        continue;
      }

      for (int j = 0; j < tensor->value.cols; j++) {
        float g = MAT_AT(tensor->grad, i, j);

        MAT_AT(h, i, j) = (beta * MAT_AT(h, i, j)) + (one_minus_beta * g * g);

        float adapt_lr = optimizer->learning_rate / sqrtf(MAT_AT(h, i, j) + eps);

        MAT_AT(tensor->value, i, j) -= adapt_lr * g;

        MAT_AT(tensor->grad, i, j) = 0;
      }
    }
  }
}

void update_grads_ada_grad(Optimizer* optimizer, size_t batch_size) {
  (void)batch_size;

  float eps = 1e-8f;

  for (size_t i = 0; i < optimizer->tensors.count; i++) {
    Tensor* tensor = optimizer->tensors.elems[i];
    NMatrix ada_grad = optimizer->history.elems[i];

    int elems = tensor->value.rows * tensor->value.cols;

    for (int j = 0; j < elems; j++) {
      // average the gradient: ∇W = ∇W / B
      float g = tensor->grad.elems[j]; // / batch_size;

      // update AdaGrad: Gnew = Gold + ∇W^2
      ada_grad.elems[j] += g * g;

      // udpate the weight: Wnew = Wold - lr/sqrt(Gnew) * ∇W
      if (ada_grad.elems[j] > 0) {
        float adapt_lr = optimizer->learning_rate / (sqrtf(ada_grad.elems[j]) + eps);
        tensor->value.elems[j] -= adapt_lr * g;
      }

      tensor->grad.elems[j] = 0;
    }
  }
}

#endif // NODE_H
