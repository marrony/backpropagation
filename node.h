#ifndef NODE_H
#define NODE_H

#include "nn.h"

typedef enum {
  NODE_CONSTANT,
  NODE_VARIABLE,
  NODE_LINEAR,
  NODE_SIGMOID,
  NODE_SOFTMAX,
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
  STR(NODE_LINEAR),
  STR(NODE_SIGMOID),
  STR(NODE_SOFTMAX),
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
} Tensor;

typedef struct Node Node;
struct Node {
  Node_Type type;
  Node* input[MAX_INPUTS];   // u, v
  int input_size;
  Tensor params[3]; // x, w, b
};

#define X_SLOT 0
#define U_SLOT 0
#define V_SLOT 1
#define W_SLOT 1
#define B_SLOT 2
#define KERN_SLOT 1

#define THIS_VALUE(slot) (node->params[(slot)].value)
#define THIS_DELTA(slot) (node->params[(slot)].grad)
#define THIS_G_DELTA(slot) (node->params[(slot)].g_grad)

#define NULL_MATRIX (NMatrix) {0}

#define IN_VALUE(slot) (node->input[(slot)] ? node->input[(slot)]->params[0].value : NULL_MATRIX)
#define IN_DELTA(slot) (node->input[(slot)] ? node->input[(slot)]->params[0].grad : NULL_MATRIX)

#define FX_VALUE THIS_VALUE(X_SLOT)

#define FORWARD(slot) node_forward(node->input[(slot)])
#define BACKWARD(slot, dL) node_backward(node->input[(slot)], (dL))

NMatrix node_get_value(Node* node) {
  return THIS_VALUE(X_SLOT);
}

void node_set_value(Node* node, NMatrix value) {
  mat_copy(THIS_VALUE(X_SLOT), value);
}

int node_get_value_rows(Node* node) {
  return node_get_value(node).rows;
}

int node_get_value_cols(Node* node) {
  return node_get_value(node).cols;
}

int node_get_value_size(Node* node) {
  NMatrix value = node_get_value(node);
  return value.rows * value.cols;
}

void destroy_node(Node** node) {
  free(*node);
  *node = NULL;
}

Node* create_node(void) {
  Node* node = malloc(sizeof(Node));
  memset(node, 0, sizeof(Node));
  return node;
}

Node* create_constant(int in_size) {
  Node* node = create_node();
  node->type = NODE_CONSTANT;
  node->input_size = 0;
  THIS_VALUE(X_SLOT) = mat_alloc(1, in_size);
  THIS_DELTA(X_SLOT) = mat_alloc(1, in_size);
  THIS_G_DELTA(X_SLOT) = mat_alloc(1, in_size);
  
  return node;
}

Node* create_variable(NMatrix value) {
  Node* node = create_node();
  node->type = NODE_VARIABLE;
  node->input_size = 0;
  THIS_VALUE(X_SLOT) = mat_alloc(value.rows, value.cols);
  THIS_DELTA(X_SLOT) = mat_alloc(value.rows, value.cols);
  THIS_G_DELTA(X_SLOT) = mat_alloc(value.rows, value.cols);

  mat_copy(THIS_VALUE(X_SLOT), value);
  return node;
}

Node* create_linear(Node* input, int out_size) {
  Node* node = create_node();
  node->type = NODE_LINEAR;
  node->input[0] = input;
  node->input_size = 1;

  int in_size = node_get_value_size(input);

  THIS_VALUE(X_SLOT)   = mat_alloc(1, out_size);
  THIS_DELTA(X_SLOT)   = mat_alloc(1, out_size);
  THIS_G_DELTA(X_SLOT) = mat_alloc(1, out_size);

  THIS_VALUE(W_SLOT) = mat_alloc(out_size, in_size);
  THIS_DELTA(W_SLOT) = mat_alloc(out_size, in_size);
  THIS_G_DELTA(W_SLOT) = mat_alloc(out_size, in_size);

  THIS_VALUE(B_SLOT) = mat_alloc(1, out_size);
  THIS_DELTA(B_SLOT) = mat_alloc(1, out_size);
  THIS_G_DELTA(B_SLOT) = mat_alloc(1, out_size);

  // mat_rand(THIS_VALUE(W_SLOT));
  // mat_rand(THIS_VALUE(B_SLOT));

  return node;
}

Node* create_unary(Node_Type type, Node* input, int rows, int cols) {
  Node* node = create_node();
  node->type = type;
  node->input[0] = input;
  node->input_size = 1;
  THIS_VALUE(X_SLOT) = mat_alloc(rows, cols);
  THIS_DELTA(X_SLOT) = mat_alloc(rows, cols);
  THIS_G_DELTA(X_SLOT) = mat_alloc(rows, cols);
  return node;
}

Node* create_binary(Node_Type type, Node* input0, Node* input1, int rows, int cols) {
  Node* node = create_node();
  node->type = type;
  node->input[0] = input0;
  node->input[1] = input1;
  node->input_size = 2;
  THIS_VALUE(X_SLOT) = mat_alloc(rows, cols);
  THIS_DELTA(X_SLOT) = mat_alloc(rows, cols);
  THIS_G_DELTA(X_SLOT) = mat_alloc(rows, cols);
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
  return create_unary(NODE_SOFTMAX, input, rows, cols);
}

Node* create_relu(Node* input) {
  int rows = node_get_value_rows(input);
  int cols = node_get_value_cols(input);
  return create_unary(NODE_RELU, input, rows, cols);
}

Node* create_conv2d(Node* input, int img_size, int kern_size) {
  Node* node = create_node();
  node->type = NODE_CONV2D;
  node->input[0] = input;
  node->input_size = 1;

  int conv_size = (img_size - kern_size + 1);

  THIS_VALUE(X_SLOT) = mat_alloc(conv_size, conv_size);
  THIS_DELTA(X_SLOT) = mat_alloc(conv_size, conv_size);
  THIS_G_DELTA(X_SLOT) = mat_alloc(conv_size, conv_size);

  THIS_VALUE(KERN_SLOT) = mat_alloc(kern_size, kern_size);
  THIS_DELTA(KERN_SLOT) = mat_alloc(kern_size, kern_size);
  THIS_G_DELTA(KERN_SLOT) = mat_alloc(kern_size, kern_size);

  // mat_rand(THIS_VALUE(KERN_SLOT));

  return node;
}

Node* create_flatten(Node** inputs, int in_size) {
  Node* node = create_node();
  node->type = NODE_FLATTEN;

  node->input_size = in_size;

  int size = 0;
  for (int i = 0; i < in_size; i++) {
    node->input[i] = inputs[i];
    size += node_get_value_size(inputs[i]);
  }

  THIS_VALUE(X_SLOT) = mat_alloc(1, size);
  THIS_DELTA(X_SLOT) = mat_alloc(1, size);
  THIS_G_DELTA(X_SLOT) = mat_alloc(1, size);

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

void node_forward(Node* node) {
  NMatrix fx = FX_VALUE;
  NMatrix x = IN_VALUE(X_SLOT);
  NMatrix w = THIS_VALUE(W_SLOT);
  NMatrix b = THIS_VALUE(B_SLOT);
  NMatrix kern = THIS_VALUE(KERN_SLOT);
  NMatrix u = IN_VALUE(U_SLOT);
  NMatrix v = IN_VALUE(V_SLOT);

  switch (node->type) {
    case NODE_CONSTANT:
    case NODE_VARIABLE:
      break;

    case NODE_LINEAR:
      // f(x) = x*W.T + b
      FORWARD(X_SLOT);
      (void)b;
      // mat_mult_transpose_add(fx, x, w, b);
      mat_mult_transpose(fx, x, w);
      break;

    case NODE_SIGMOID:
      // f(x) = sigmoid(x)
      FORWARD(X_SLOT);
      sigmoid(fx, x);
      break;

    case NODE_SOFTMAX:
      // f(x) = softmax(x)
      FORWARD(X_SLOT);
      softmax(fx, x);
      break;

    case NODE_RELU:
      // f(x) = relu(x)
      FORWARD(X_SLOT);
      relu(fx, x);
      break;

    case NODE_FLATTEN:
      for (int i = 0; i < node->input_size; i++) {
        FORWARD(X_SLOT + i);

        Node* input = node->input[i];

        int rows = input->params[X_SLOT].value.rows;
        int cols = input->params[X_SLOT].value.cols;
        int size = node_get_value_size(input);

        NMatrix dst = mat_reshape(
            mat_row_slice(fx, i*size, size),
            rows,
            cols
        );

        mat_copy(dst, input->params[X_SLOT].value);
      }
      break;

    case NODE_CONV2D:
      // f(x) = conv2d(x, kernel)
      FORWARD(X_SLOT);

      x = mat_reshape(x,
          fx.rows + kern.rows - 1,
          fx.cols + kern.cols - 1
      );

      for (int i = 0; i < fx.rows; i++) {
        for (int j = 0; j < fx.cols; j++) {
          MAT_AT(fx, i, j) = 0; // bias

          for (int ki = 0; ki < kern.rows; ki++) {
            for (int kj = 0; kj < kern.cols; kj++) {
              float a = MAT_AT(x, i+ki, j+kj);
              float b = MAT_AT(kern, ki, kj);
              MAT_AT(fx, i, j) += a * b;
            }
          }
        }
      }
      break;

    case NODE_SQUARE:
      // f(x) = x^2
      FORWARD(X_SLOT);
      for (int i = 0; i < x.rows; i++) {
        for (int j = 0; j < x.cols; j++)
          MAT_AT(fx, i, j) = MAT_AT(x, i, j)*MAT_AT(x, i, j);
      }
      break;

    case NODE_CUBE:
      // f(x) = x^3
      FORWARD(X_SLOT);
      for (int i = 0; i < x.rows; i++) {
        for (int j = 0; j < x.cols; j++)
          MAT_AT(fx, i, j) = MAT_AT(x, i, j)*MAT_AT(x, i, j)*MAT_AT(x, i, j);
      }
      break;

    case NODE_EXP:
      // f(x) = exp(x)
      FORWARD(X_SLOT);
      for (int i = 0; i < x.rows; i++) {
        for (int j = 0; j < x.cols; j++)
          MAT_AT(fx, i, j) = expf(MAT_AT(x, i, j));
      }
      break;

    case NODE_NEGATE:
      // f(x) = -x
      FORWARD(X_SLOT);
      mat_scale(fx, x, -1);
      break;

    case NODE_MULTIPLY:
      // f(u, v) = u * v
      FORWARD(U_SLOT);
      FORWARD(V_SLOT);
      // mat_memberwise_mult(fx, u, v);
      mat_mult(fx, u, v);
      break;

    case NODE_DIVIDE:
      // f(u, v) = u / v
      FORWARD(U_SLOT);
      FORWARD(V_SLOT);
      mat_memberwise_div(fx, u, v);
      break;

    case NODE_ADD:
      // f(u, v) = u + v
      FORWARD(U_SLOT);
      FORWARD(V_SLOT);
      mat_add(fx, u, v);
      break;

    case NODE_SUB:
      // f(u, v) = u - v
      FORWARD(U_SLOT);
      FORWARD(V_SLOT);
      mat_sub(fx, u, v);
      break;
  }
}

void node_backward(Node* node, NMatrix dL) {
  // upstream
  NMatrix dL_df = THIS_DELTA(X_SLOT);
  // local
  NMatrix fx   = FX_VALUE;
  NMatrix x    = IN_VALUE(X_SLOT);
  NMatrix w    = THIS_VALUE(W_SLOT);
  NMatrix kern = THIS_VALUE(KERN_SLOT);
  NMatrix u    = IN_VALUE(U_SLOT);
  NMatrix v    = IN_VALUE(V_SLOT);
  // downstream
  NMatrix dL_dw    = THIS_DELTA(W_SLOT);
  NMatrix dL_db    = THIS_DELTA(B_SLOT);
  NMatrix dL_dkern = THIS_DELTA(KERN_SLOT);
  NMatrix dL_dx    = IN_DELTA(X_SLOT);
  NMatrix dL_du    = IN_DELTA(U_SLOT);
  NMatrix dL_dv    = IN_DELTA(V_SLOT);

  // printf("BACKWARD(%s) = [%dx%d]\n",
  //     Node_Type_Str[node->type],
  //     fx.rows, fx.cols
  // );

  switch (node->type) {
    case NODE_CONSTANT:
      // upstream:
      //   dL/d(c) = dL
      //
      // local:
      //   d(c)/dx = 0
      //
      // downstream:
      //   dL/dx = d(c)/dx * dL = 0*dL
      // mat_copy(dL_df, dL);
      mat_fill(dL_dx, 0);
      break;

    case NODE_VARIABLE:
      // upstream:
      //   dL/d(x) = dL
      //
      // local:
      //   d(x)/dx = 1
      //
      // downstream:
      //   dL/dx = d(x)/dx * dL = 1*dL
      mat_copy(dL_df, dL);
      // mat_copy(dL_dx, dL);
      break;

    case NODE_LINEAR:
      // upstream:
      //   dL/d(x*W.T + b) = dL
      //
      // local:
      //   d(x*W.T + b)/dw = x
      //   d(x*W.T + b)/db = I
      //   d(x*W.T + b)/dx = W
      //
      // downstream:
      //   dL/dw = dL * d(x*W.T + b)/dw = dL.T*x
      //   dL/db = dL * d(x*W.T + b)/db = I*dL
      //   dL/dx = dL * d(x*W.T + b)/dx = dL*W
      mat_copy(dL_df, dL);

      // dL_dw = dL.T * x
      mat_transpose_mult(dL_dw, dL, x);
      // dL_db = I * dL
      mat_copy(dL_db, dL);
      // dL_dx = dL * W
      mat_mult(dL_dx, dL, w);

      BACKWARD(X_SLOT, dL_dx);
      break;

    case NODE_SIGMOID:
      // upstream:
      //   dL/d(sigmoid(x)) = dL
      //
      // local:
      //   d(sigmoid(x))
      //
      // downstream:
      //   dL/dx = d(sigmoid(x)) * dL
      //         = sigmoid(x)*(1 - sigmoid(x)) * dL
      mat_copy(dL_df, dL);
      dsigmoid(dL_dx, fx, dL);
      BACKWARD(X_SLOT, dL_dx);
      break;

    case NODE_SOFTMAX:
      // upstream:
      //   dL/d(softmax(x)) = dL
      //
      // local:
      //   d(softmax(x))/dx
      //
      // downstream:
      //   dL/dx = d(softmax(x))/dx * dL
      //         = softmax(x) * [ dL - dot(softmax(x), dL) ]
      mat_copy(dL_df, dL);
      dsoftmax(dL_dx, fx, dL);
      BACKWARD(X_SLOT, dL_dx);
      break;

    case NODE_RELU:
      mat_copy(dL_df, dL);
      drelu(dL_dx, fx, dL);
      BACKWARD(X_SLOT, dL_dx);
      break;

    case NODE_FLATTEN:
      mat_copy(dL_df, dL);

      for (int i = 0; i < node->input_size; i++) {
        Node* input = node->input[i];

        int rows = input->params[X_SLOT].value.rows;
        int cols = input->params[X_SLOT].value.cols;
        int size = node_get_value_size(input);

        NMatrix ith_dL_dx = mat_reshape(
          mat_row_slice(dL, i*size, size),
          rows,
          cols
        );

        mat_copy(IN_DELTA(i), ith_dL_dx);

        BACKWARD(i, ith_dL_dx);
      }
      break;

    case NODE_CONV2D:
      // upstream:
      //   dL/d(conv2d(x, kern)) = dL
      //
      // local:
      //   d(conv2d(x, kern))/dx
      //   d(conv2d(x, kern))/dkern
      //
      // downstream:
      //   dL/dx = d(conv2d(x, kern))/dx * dL
      mat_copy(dL_df, dL);

      mat_fill(dL_dx, 0);
      mat_fill(dL_dkern, 0);

      for (int i = 0; i < dL.rows; i++) {
        for (int j = 0; j < dL.cols; j++) {
          NMatrix dL_dx_reshaped = mat_reshape(
              dL_dx,
              fx.rows + kern.rows - 1,
              fx.cols + kern.cols - 1
          );

          NMatrix x_reshaped = mat_reshape(
              x,
              fx.rows + kern.rows - 1,
              fx.cols + kern.cols - 1
          );

          float delta = MAT_AT(dL, i, j);
          // Bias gradient
          // grad_b[f] += delta;

          for(int u = 0; u < kern.rows; u++) {
            for(int v = 0; v < kern.cols; v++) {
              // Weight gradient
              MAT_AT(dL_dkern, u, v) += delta * MAT_AT(x_reshaped, i+u, j+v);

              // Input gradient
              MAT_AT(dL_dx_reshaped, i+u, j+v) += delta * MAT_AT(kern, u, v);
            }
          }
        }
      }

      BACKWARD(X_SLOT, dL_dx);
      break;

    case NODE_SQUARE:
      // upstream:
      //   dL/d(x^2) = dL
      //
      // local:
      //   d(x^2)/dx = 2*x
      //
      // downstream:
      //   dL/dx = d(x^2)/dx * dL = 2*x*dL
      mat_copy(dL_df, dL);

      for (int i = 0; i < dL_dx.rows; i++) {
        for (int j = 0; j < dL_dx.cols; j++)
          MAT_AT(dL_dx, i, j) = 2*MAT_AT(x, i, j)*MAT_AT(dL, i, j);
      }

      BACKWARD(X_SLOT, dL_dx);
      break;

    case NODE_CUBE:
      // upstream:
      //   dL/d(x^3) = dL
      //
      // local:
      //   d(x^3)/dx = 3*x^2
      //
      // downstream:
      //   dL/dx = d(x^3)/dx * dL = 3*x^2*dL
      mat_copy(dL_df, dL);

      for (int i = 0; i < dL_dx.rows; i++) {
        for (int j = 0; j < dL_dx.cols; j++)
          MAT_AT(dL_dx, i, j) = 3*MAT_AT(x, i, j)*MAT_AT(x, i, j)*MAT_AT(dL, i, j);
      }

      BACKWARD(X_SLOT, dL_dx);
      break;

    case NODE_EXP:
      // upstream:
      //   dL/d(exp(x)) = dL
      //
      // local:
      //   d(exp(x))/dx = exp(x)
      //
      // downstream:
      //   dL/dx = d(exp(x))/dx * dL = exp(x)*dL
      mat_copy(dL_df, dL);

      for (int i = 0; i < dL_dx.rows; i++) {
        for (int j = 0; j < dL_dx.cols; j++)
          MAT_AT(dL_dx, i, j) = MAT_AT(fx, i, j)*MAT_AT(dL, i, j);
      }

      BACKWARD(X_SLOT, dL_dx);
      break;

    case NODE_NEGATE:
      // upstream:
      //   dL/d(-x) = dL
      //
      // local:
      //   d(-x)/dx = -1
      //
      // downstream:
      //  dL/dx = d(-x)/dx * dL = -1*dL
      mat_copy(dL_df, dL);

      for (int i = 0; i < dL_dx.rows; i++) {
        for (int j = 0; j < dL_dx.cols; j++) {
          MAT_AT(dL_dx, i, j) = -MAT_AT(dL, i, j);
        }
      }

      BACKWARD(X_SLOT, dL_dx);
      break;

    case NODE_MULTIPLY:
      // upstream:
      //   dL/d(u*v) = dL
      //
      // local:
      //   d(u*v)/du = v
      //   d(u*v)/dv = u
      //
      // downstream:
      //   dL/du = dL * d(u*v)/du.T = dL * v.T
      //   dL/dv = d(u*v)/dv.T * dL = u.T * dL
      mat_copy(dL_df, dL);

      mat_mult_transpose(dL_du, dL, v);
      mat_transpose_mult(dL_dv, u, dL);

      BACKWARD(U_SLOT, dL_du);
      BACKWARD(V_SLOT, dL_dv);
      break;

    case NODE_DIVIDE:
      // upstream:
      //   dL/d(u/v) = dL
      //
      // local:
      //   d(u/v)/du = 1 / v
      //   d(u/v)/dv = -u / v^2
      //
      // downstream:
      //   dL/du = d(u/v)/du * dL = [1 / v]*dL
      //   dL/dv = d(u/v)/dv * dL = [-u / v^2]*dL
      mat_copy(dL_df, dL);

      for (int i = 0; i < dL_du.rows; i++) {
        for (int j = 0; j < dL_du.cols; j++) {
          float one_over_v = 1.0 / MAT_AT(v, i, j);

          MAT_AT(dL_du, i, j) = one_over_v * MAT_AT(dL, i, j);
          MAT_AT(dL_dv, i, j) = -MAT_AT(u, i, j) * one_over_v * one_over_v * MAT_AT(dL, i, j);
        }
      }

      BACKWARD(U_SLOT, dL_du);
      BACKWARD(V_SLOT, dL_dv);
      break;

    case NODE_ADD:
      // upstream:
      //   dL/d(u+v) = dL
      //
      // local:
      //   d(u+v)/du = 1
      //   d(u+v)/dv = 1
      //
      // downstream:
      //   dL/du = d(u+v)/du * dL = 1*dL
      //   dL/dv = d(u+v)/dv * dL = 1*dL
      mat_copy(dL_df, dL);
      mat_copy(dL_du, dL);
      mat_copy(dL_dv, dL);

      BACKWARD(U_SLOT, dL_du);
      BACKWARD(V_SLOT, dL_dv);
      break;

    case NODE_SUB:
      // upstream:
      //   dL/d(u-v) = dL
      //
      // local:
      //   d(u-v)/du = 1
      //   d(u-v)/dv = -1
      //
      // downstream:
      //   dL/du = d(u-v)/du * dL = +1*dL
      //   dL/dv = d(u-v)/dv * dL = -1*dL
      mat_copy(dL_df, dL);

      for (int i = 0; i < dL_du.rows; i++) {
        for (int j = 0; j < dL_du.cols; j++) {
          MAT_AT(dL_du, i, j) = +MAT_AT(dL, i, j);
          MAT_AT(dL_dv, i, j) = -MAT_AT(dL, i, j);
        }
      }

      BACKWARD(U_SLOT, dL_du);
      BACKWARD(V_SLOT, dL_dv);
      break;
  }
}

// accumulate and zero grads
void acc_grads(Node* node) {
  switch (node->type) {
    case NODE_CONSTANT:
      break;

    case NODE_VARIABLE:
      mat_add(THIS_G_DELTA(X_SLOT), THIS_G_DELTA(X_SLOT), THIS_DELTA(X_SLOT));
      break;

    case NODE_LINEAR:
      acc_grads(node->input[X_SLOT]);
      mat_add(THIS_G_DELTA(W_SLOT), THIS_G_DELTA(W_SLOT), THIS_DELTA(W_SLOT));
      mat_add(THIS_G_DELTA(B_SLOT), THIS_G_DELTA(B_SLOT), THIS_DELTA(B_SLOT));
      break;

    case NODE_CONV2D:
      acc_grads(node->input[X_SLOT]);
      mat_add(THIS_G_DELTA(KERN_SLOT), THIS_G_DELTA(KERN_SLOT), THIS_DELTA(KERN_SLOT));
      break;

    case NODE_FLATTEN:
      for (int i = 0; i < node->input_size; i++)
        acc_grads(node->input[i]);
      break;

    case NODE_SIGMOID:
    case NODE_SOFTMAX:
    case NODE_RELU:
    case NODE_SQUARE:
    case NODE_CUBE:
    case NODE_EXP:
    case NODE_NEGATE:
      acc_grads(node->input[X_SLOT]);
      break;

    case NODE_MULTIPLY:
    case NODE_DIVIDE:
    case NODE_ADD:
    case NODE_SUB:
      acc_grads(node->input[U_SLOT]);
      acc_grads(node->input[V_SLOT]);
      break;
  }
}

void update_grads(Node* node, float lr) {
  switch (node->type) {
    case NODE_CONSTANT:
      break;

    case NODE_VARIABLE:
      mat_weighted_add(THIS_VALUE(X_SLOT), THIS_VALUE(X_SLOT), THIS_G_DELTA(X_SLOT), -lr);
      break;

    case NODE_LINEAR:
      update_grads(node->input[X_SLOT], lr);
      mat_weighted_add(THIS_VALUE(W_SLOT), THIS_VALUE(W_SLOT), THIS_G_DELTA(W_SLOT), -lr);
      mat_weighted_add(THIS_VALUE(B_SLOT), THIS_VALUE(B_SLOT), THIS_G_DELTA(B_SLOT), -lr);
      break;

    case NODE_CONV2D:
      update_grads(node->input[X_SLOT], lr);
      mat_weighted_add(THIS_VALUE(KERN_SLOT), THIS_VALUE(KERN_SLOT), THIS_G_DELTA(KERN_SLOT), -lr);
      break;

    case NODE_FLATTEN:
      for (int i = 0; i < node->input_size; i++)
        update_grads(node->input[i], lr);
      break;

    case NODE_SIGMOID:
    case NODE_SOFTMAX:
    case NODE_RELU:
    case NODE_SQUARE:
    case NODE_CUBE:
    case NODE_EXP:
    case NODE_NEGATE:
      update_grads(node->input[X_SLOT], lr);
      break;

    case NODE_MULTIPLY:
    case NODE_DIVIDE:
    case NODE_ADD:
    case NODE_SUB:
      update_grads(node->input[U_SLOT], lr);
      update_grads(node->input[V_SLOT], lr);
      break;
  }
}

void zero_grads(Node* node) {
  switch (node->type) {
    case NODE_CONSTANT:
      break;

    case NODE_VARIABLE:
      mat_fill(THIS_G_DELTA(X_SLOT), 0);
      break;

    case NODE_LINEAR:
      zero_grads(node->input[X_SLOT]);
      mat_fill(THIS_G_DELTA(W_SLOT), 0);
      mat_fill(THIS_G_DELTA(B_SLOT), 0);
      break;

    case NODE_CONV2D:
      zero_grads(node->input[X_SLOT]);
      mat_fill(THIS_G_DELTA(KERN_SLOT), 0);
      break;

    case NODE_FLATTEN:
      for (int i = 0; i < node->input_size; i++)
        zero_grads(node->input[i]);
      break;

    case NODE_SIGMOID:
    case NODE_SOFTMAX:
    case NODE_RELU:
    case NODE_SQUARE:
    case NODE_CUBE:
    case NODE_EXP:
    case NODE_NEGATE:
      zero_grads(node->input[X_SLOT]);
      break;

    case NODE_MULTIPLY:
    case NODE_DIVIDE:
    case NODE_ADD:
    case NODE_SUB:
      zero_grads(node->input[U_SLOT]);
      zero_grads(node->input[V_SLOT]);
      break;
  }
}
#endif // NODE_H
