#include "nn.h"
#include <stdbool.h>
#include <stdio.h>

typedef enum {
  NODE_CONSTANT,
  NODE_VARIABLE,
  NODE_LINEAR,
  NODE_SIGMOID,
  NODE_SOFTMAX,
  NODE_RELU,
  NODE_CONV2D,
  NODE_DENSE,
  NODE_SQUARE,
  NODE_CUBE,
  NODE_EXP,
  NODE_NEGATE,

  // 
  NODE_MULTIPLY,
  NODE_DIVIDE,
  NODE_ADD,
  NODE_SUB,
} Node_Type;

typedef struct Node Node;
struct Node {
  Node_Type type;
  Node* input[16];   // u, v
  NMatrix output[3]; // x, w, b
  NMatrix delta[3];  // dx, dw, db
};

#define X_SLOT 0
#define U_SLOT 0
#define V_SLOT 1
#define W_SLOT 1
#define B_SLOT 2
#define KERN_SLOT 1

#define THIS_VALUE(slot) (node->output[(slot)])
#define THIS_DELTA(slot) (node->delta[(slot)])

#define NULL_MATRIX (NMatrix) {0}

#define IN_VALUE(slot) (node->input[(slot)] ? node->input[(slot)]->output[0] : NULL_MATRIX)
#define IN_DELTA(slot) (node->input[(slot)] ? node->input[(slot)]->delta[0] : NULL_MATRIX)

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

Node* create_constant(int in_size) {
  Node* node = malloc(sizeof(Node));
  node->type = NODE_CONSTANT;
  node->input[0] = NULL;
  node->input[1] = NULL;
  THIS_VALUE(X_SLOT) = mat_alloc(1, in_size);
  THIS_DELTA(X_SLOT) = mat_alloc(1, in_size);
  
  return node;
}

Node* create_variable(NMatrix value) {
  Node* node = malloc(sizeof(Node));
  node->type = NODE_VARIABLE;
  node->input[0] = NULL;
  node->input[1] = NULL;
  THIS_VALUE(X_SLOT) = mat_alloc(value.rows, value.cols);
  THIS_DELTA(X_SLOT) = mat_alloc(value.rows, value.cols);

  mat_copy(node->output[0], value);
  return node;
}

Node* create_linear(Node* input, int out_size) {
  Node* node = malloc(sizeof(Node));
  node->type = NODE_LINEAR;
  node->input[0] = input;
  node->input[1] = NULL;

  int in_size = node_get_value_size(input);

  THIS_VALUE(X_SLOT) = mat_alloc(1, out_size);
  THIS_DELTA(X_SLOT) = mat_alloc(1, out_size);

  THIS_VALUE(W_SLOT) = mat_alloc(out_size, in_size);
  THIS_DELTA(W_SLOT) = mat_alloc(out_size, in_size);

  THIS_DELTA(B_SLOT) = mat_alloc(1, out_size);
  THIS_VALUE(B_SLOT) = mat_alloc(1, out_size);

  mat_rand(THIS_VALUE(W_SLOT));
  mat_rand(THIS_VALUE(B_SLOT));

  return node;
}

Node* create_unary(Node_Type type, Node* input, int size) {
  Node* node = malloc(sizeof(Node));
  node->type = type;
  node->input[0] = input;
  node->input[1] = NULL;
  THIS_VALUE(X_SLOT) = mat_alloc(1, size);
  THIS_DELTA(X_SLOT) = mat_alloc(1, size);
  return node;
}

Node* create_binary(Node_Type type, Node* input0, Node* input1, int rows, int cols) {
  Node* node = malloc(sizeof(Node));
  node->type = type;
  node->input[0] = input0;
  node->input[1] = input1;
  THIS_VALUE(X_SLOT) = mat_alloc(rows, cols);
  THIS_DELTA(X_SLOT) = mat_alloc(rows, cols);
  return node;
}

Node* create_sigmoid(Node* input) {
  int in_size = node_get_value_size(input);
  return create_unary(NODE_SIGMOID, input, in_size);
}

Node* create_softmax(Node* input) {
  int in_size = node_get_value_size(input);
  return create_unary(NODE_SOFTMAX, input, in_size);
}

Node* create_relu(Node* input) {
  int in_size = node_get_value_size(input);
  return create_unary(NODE_RELU, input, in_size);
}

Node* create_conv2d(Node* input, int img_size, int kern_size) {
  Node* node = malloc(sizeof(Node));
  node->type = NODE_CONV2D;
  node->input[0] = input;
  node->input[1] = NULL;

  int conv_size = (img_size - kern_size + 1);

  THIS_VALUE(X_SLOT) = mat_alloc(conv_size, conv_size);
  THIS_DELTA(X_SLOT) = mat_alloc(conv_size, conv_size);

  THIS_VALUE(KERN_SLOT) = mat_alloc(kern_size, kern_size);
  THIS_DELTA(KERN_SLOT) = mat_alloc(kern_size, kern_size);

  mat_rand(THIS_VALUE(KERN_SLOT));

  return node;
}

Node* create_dense(Node* input[16]) {
  Node* node = malloc(sizeof(Node));
  node->type = NODE_DENSE;

  int size = 0;
  for (int i = 0; i < 16; i++) {
    node->input[i] = input[i];

    size += node_get_value_size(input[i]);
  }

  THIS_VALUE(X_SLOT) = mat_alloc(1, size);
  THIS_DELTA(X_SLOT) = mat_alloc(1, size);

  return node;
}

Node* create_square(Node* input, int size) {
  return create_unary(NODE_SQUARE, input, size);
}

Node* create_cube(Node* input, int size) {
  return create_unary(NODE_CUBE, input, size);
}

Node* create_exp(Node* input, int size) {
  return create_unary(NODE_EXP, input, size);
}

Node* create_negate(Node* input, int size) {
  return create_unary(NODE_NEGATE, input, size);
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
      mat_mult_transpose_add(fx, x, w, b);
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

    case NODE_DENSE:
      for (int i = 0; i < 16; i++) {
        FORWARD(X_SLOT + i);

        Node* input = node->input[i];

        int rows = input->output[X_SLOT].rows;
        int cols = input->output[X_SLOT].cols;
        int size = node_get_value_size(input);

        NMatrix dst = mat_reshape(
            mat_row_slice(fx, i*size, size),
            rows,
            cols
        );

        mat_copy(dst, input->output[X_SLOT]);
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
      mat_fill(dL_df, 0);
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
      // dL_db = I*dL
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

    case NODE_DENSE:
      mat_copy(dL_df, dL);

      for (int i = 0; i < 16; i++) {
        Node* input = node->input[i];

        int rows = input->output[X_SLOT].rows;
        int cols = input->output[X_SLOT].cols;
        int size = node_get_value_size(input);

        NMatrix dst = mat_reshape(
          mat_row_slice(dL_df, i*size, size),
          rows,
          cols
        );

        BACKWARD(X_SLOT+i, dst);
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

      dL_dx = mat_reshape(
          dL_dx,
          fx.rows + kern.rows - 1,
          fx.cols + kern.cols - 1
      );
      x = mat_reshape(
          x,
          fx.rows + kern.rows - 1,
          fx.cols + kern.cols - 1
      );

      for (int i = 0; i < dL_dkern.rows; i++) {
        for (int j = 0; j < dL_dkern.cols; j++) {
          float delta = MAT_AT(fx, i, j);
          // Bias gradient
          // grad_b[f] += delta;

          for(int u = 0; u < kern.rows; u++) {
            for(int v = 0; v < kern.cols; v++) {
              // Weight gradient
              MAT_AT(dL_dkern, u, v) += delta * MAT_AT(x, i+u, j+v);

              // Input gradient
              MAT_AT(dL_dx, i+u, j+v) += delta * MAT_AT(kern, u, v);
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

void update_grads(Node* node, float lr) {
  switch (node->type) {
    case NODE_CONSTANT:
      break;

    case NODE_VARIABLE:
      mat_weighted_add(THIS_VALUE(X_SLOT), THIS_VALUE(X_SLOT), THIS_DELTA(X_SLOT), -lr);
      break;

    case NODE_LINEAR:
      update_grads(node->input[X_SLOT], lr);
      mat_weighted_add(THIS_VALUE(W_SLOT), THIS_VALUE(W_SLOT), THIS_DELTA(W_SLOT), -lr);
      mat_weighted_add(THIS_VALUE(B_SLOT), THIS_VALUE(B_SLOT), THIS_DELTA(B_SLOT), -lr);
      break;

    case NODE_SIGMOID:
    case NODE_SOFTMAX:
    case NODE_RELU:
    case NODE_CONV2D:
    case NODE_DENSE:
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

void test_add(void) {
  int N = 2;

  NMatrix dL    = mat_alloc(1, N);
  NMatrix x_mat = mat_alloc(1, N);
  NMatrix y_mat = mat_alloc(1, N);

  mat_fill(x_mat, 2);
  mat_fill(y_mat, 1);
  mat_fill(dL, 2);

  Node* x = create_variable(x_mat);
  Node* y = create_variable(y_mat);
  Node* op = create_add(x, y);

  node_forward(op);
  node_backward(op, dL);

  printf("x       = "); mat_print(stdout, x->output[X_SLOT]);
  printf(" d(x)     = "); mat_println(stdout, x->delta[X_SLOT]);

  printf("y       = "); mat_print(stdout, y->output[X_SLOT]);
  printf(" d(y)     = "); mat_println(stdout, y->delta[X_SLOT]);

  printf("(x + y) = "); mat_print(stdout, op->output[X_SLOT]);
  printf(" d(x + y) = "); mat_println(stdout, op->delta[X_SLOT]);

  ASSERT_EQ(MAT_AT(op->output[X_SLOT], 0, 0), 3.0);
  ASSERT_EQ(MAT_AT(op->delta[X_SLOT], 0, 0),  MAT_AT(dL, 0, 0));
  ASSERT_EQ(MAT_AT(x->delta[X_SLOT], 0, 0), 2.0);
  ASSERT_EQ(MAT_AT(y->delta[X_SLOT], 0, 0), 2.0);

  destroy_node(&x);
  destroy_node(&y);
  destroy_node(&op);
}

void test_sub(void) {
  int N = 2;

  NMatrix dL = mat_alloc(1, N);
  NMatrix x_mat = mat_alloc(1, N);
  NMatrix y_mat = mat_alloc(1, N);

  mat_fill(x_mat, 2);
  mat_fill(y_mat, 1);
  mat_fill(dL, 2);

  Node* x = create_variable(x_mat);
  Node* y = create_variable(y_mat);
  Node* op = create_sub(x, y);

  node_forward(op);
  node_backward(op, dL);

  printf("x       = "); mat_print(stdout, x->output[X_SLOT]);
  printf(" d(x)     = "); mat_println(stdout, x->delta[X_SLOT]);

  printf("y       = "); mat_print(stdout, y->output[X_SLOT]);
  printf(" d(y)     = "); mat_println(stdout, y->delta[X_SLOT]);

  printf("(x - y) = "); mat_print(stdout, op->output[X_SLOT]);
  printf(" d(x - y) = "); mat_println(stdout, op->delta[X_SLOT]);

  ASSERT_EQ(MAT_AT(op->output[X_SLOT], 0, 0), 1.0);
  ASSERT_EQ(MAT_AT(op->delta[X_SLOT], 0, 0),  MAT_AT(dL, 0, 0));
  ASSERT_EQ(MAT_AT(x->delta[X_SLOT], 0, 0), +2.0);
  ASSERT_EQ(MAT_AT(y->delta[X_SLOT], 0, 0), -2.0);

  destroy_node(&x);
  destroy_node(&y);
  destroy_node(&op);
}

void test_mult(void) {
  int N = 1;
  int M = 2;
  int P = 2;

  NMatrix x_mat = mat_alloc(N, M);
  NMatrix y_mat = mat_alloc(M, P);
  NMatrix dL    = mat_alloc(N, P);

  mat_fill(x_mat, 2);
  mat_fill(y_mat, 3);
  mat_fill(dL, 2);

  Node* x = create_variable(x_mat);
  Node* y = create_variable(y_mat);
  Node* op = create_multiply(x, y);

  node_forward(op);
  node_backward(op, dL);

  printf("x         = "); mat_println(stdout, x->output[X_SLOT]);
  printf("d(x*y)/dx = "); mat_println(stdout, x->delta[X_SLOT]);

  printf("y         = "); mat_println(stdout, y->output[X_SLOT]);
  printf("d(x*y)/dy = "); mat_println(stdout, y->delta[X_SLOT]);

  printf("x*y       = "); mat_println(stdout, op->output[X_SLOT]);
  printf("dL/d(x*y) = "); mat_println(stdout, op->delta[X_SLOT]);

  //           u   *   v
  // [1x2] = [1x2] * [2x2]
  ASSERT_VEC_EQ(op->output[X_SLOT], ((float[]) {12, 12}));

  // [1x2] = [1x2] * [2x2].T
  // dL/du = dL * d(u*v)/du.T = dL * v.T
  ASSERT_VEC_EQ(op->delta[U_SLOT], ((float[]) {2, 2}));

  // [2x2] = [1x2].T * [1x2]
  // dL/dv = d(u*v)/dv.T * dL = u.T * dL
  ASSERT_VEC_EQ(op->delta[V_SLOT], ((float[]) {18, 18, 18, 18}));

  // [1x2]
  ASSERT_VEC_EQ(x->delta[X_SLOT], ((float[]) {12, 12}));

  // [2x2]
  ASSERT_VEC_EQ(y->delta[X_SLOT], ((float[]) {4, 4, 4, 4}));


  destroy_node(&x);
  destroy_node(&y);
  destroy_node(&op);
}

void test_div(void) {
  int N = 2;

  NMatrix dL = mat_alloc(1, N);
  NMatrix x_mat = mat_alloc(1, N);
  NMatrix y_mat = mat_alloc(1, N);

  mat_fill(x_mat, 2);
  mat_fill(y_mat, 3);
  mat_fill(dL, 2);

  Node* x = create_variable(x_mat);
  Node* y = create_variable(y_mat);
  Node* op = create_divide(x, y);

  node_forward(op);
  node_backward(op, dL);

  printf("x       = "); mat_print(stdout, x->output[X_SLOT]);
  printf(" d(x)     = "); mat_println(stdout, x->delta[X_SLOT]);

  printf("y       = "); mat_print(stdout, y->output[X_SLOT]);
  printf(" d(y)     = "); mat_println(stdout, y->delta[X_SLOT]);

  printf("(x * y) = "); mat_print(stdout, op->output[X_SLOT]);
  printf(" d(x / y) = "); mat_println(stdout, op->delta[X_SLOT]);

  ASSERT_EQ(MAT_AT(op->output[X_SLOT], 0, 0), +0.66666669f);
  // dL/du = d(u/v)/du * dL = [1 / v]*dL
  // dL/dv = d(u/v)/dv * dL = [-u / v^2]*dL
  ASSERT_EQ(MAT_AT(op->delta[X_SLOT], 0, 0),  MAT_AT(dL, 0, 0));
  ASSERT_EQ(MAT_AT(x->delta[X_SLOT], 0, 0),   +0.66666669f);
  ASSERT_EQ(MAT_AT(y->delta[X_SLOT], 0, 0),   -0.44444448f);

  destroy_node(&x);
  destroy_node(&y);
  destroy_node(&op);
}

void test_linear(void) {
  // y = xW.T + b
  NMatrix dL = mat_alloc(1, 3);
  NMatrix x_mat = mat_alloc(1, 2);
  NMatrix w_mat = mat_alloc(3, 2);
  NMatrix b_mat = mat_alloc(1, 3);

  mat_copy(x_mat, mat_init(1, 2, (float[]){1, 2}));
  mat_copy(w_mat, mat_init(3, 2, (float[]){1, 2, 3, 4, 5, 6}));
  mat_copy(b_mat, mat_init(1, 3, (float[]){1, 2, 3}));
  mat_copy(dL, mat_init(1, 3, (float[]){8, 22, 36}));

  Node* x = create_variable(x_mat);
  Node* op = create_linear(x, 3);
  mat_copy(op->output[W_SLOT], w_mat);
  mat_copy(op->output[B_SLOT], b_mat);

  node_forward(op);
  node_backward(op, dL);

  printf("x       = "); mat_print(stdout, x->output[X_SLOT]);
  printf(" d(x)     = "); mat_println(stdout, x->delta[X_SLOT]);

  printf("(x*w + b) = "); mat_print(stdout, op->output[X_SLOT]);
  printf(" d(x*w + b) = "); mat_println(stdout, op->delta[X_SLOT]);

  // f(x) = x*W.T + b
  ASSERT_VEC_EQ(op->output[X_SLOT], ((float[]) {6, 13, 20}));
  // dL/dw = dL * d(x*W.T + b)/dw = dL.T*x
  // dL/db = dL * d(x*W.T + b)/db = I*dL
  // dL/dx = dL * d(x*W.T + b)/dx = dL*W
  ASSERT_VEC_EQ(op->delta[X_SLOT],  dL.elems);
  ASSERT_VEC_EQ(op->delta[W_SLOT],  ((float[]) {8, 16, 22, 44, 36, 72}));
  ASSERT_VEC_EQ(op->delta[B_SLOT],  ((float[]) {8, 22, 36}));
  ASSERT_VEC_EQ(x->delta[X_SLOT],   ((float[]) {254, 320}));

  destroy_node(&x);
  destroy_node(&op);
}

void test_sigmoid(void) {
  int N = 2;

  NMatrix dL = mat_alloc(1, N);
  NMatrix x_mat = mat_alloc(1, N);

  mat_fill(x_mat, 0.5);
  mat_fill(dL, 2);

  Node* x = create_variable(x_mat);
  Node* op = create_sigmoid(x);

  node_forward(op);
  node_backward(op, dL);

  printf("x       = "); mat_print(stdout, x->output[X_SLOT]);
  printf(" d(x)     = "); mat_println(stdout, x->delta[X_SLOT]);

  printf("sigmoid(x) = "); mat_print(stdout, op->output[X_SLOT]);
  printf(" d(sigmoid(x)) = "); mat_println(stdout, op->delta[X_SLOT]);

  // sigmoid(x) = 1/(1+exp(-x))
  ASSERT_VEC_EQ(op->output[X_SLOT], ((float[]) {+0.62245935, +0.62245935}));
  // dL/dx = sigmoid(x)*(1 - sigmoid(x)) * dL
  ASSERT_VEC_EQ(op->delta[X_SLOT], ((float[]) {2, 2}));
  ASSERT_VEC_EQ(x->delta[X_SLOT], ((float[]) {+0.47000742, +0.47000742}));

  destroy_node(&x);
  destroy_node(&op);
}

void test_softmax(void) {
  int N = 2;

  NMatrix dL = mat_alloc(1, N);
  NMatrix x_mat = mat_alloc(1, N);

  VEC_AT(x_mat, 0) = 0.1;
  VEC_AT(x_mat, 1) = 0.5;
  VEC_AT(dL, 0) = 1;
  VEC_AT(dL, 1) = 2;

  Node* x = create_variable(x_mat);
  Node* op = create_softmax(x);

  node_forward(op);
  node_backward(op, dL);

  printf("x       = "); mat_print(stdout, x->output[X_SLOT]);
  printf(" d(x)     = "); mat_println(stdout, x->delta[X_SLOT]);

  printf("softmax(x) = "); mat_print(stdout, op->output[X_SLOT]);
  printf(" d(softmax(x)) = "); mat_println(stdout, op->delta[X_SLOT]);

  // softmax(x)
  ASSERT_VEC_EQ(op->output[X_SLOT], ((float[]) {0.4013123399, 0.5986876601}));
  // dL/dx = softmax(z) * [ dL/dh - dot(softmax(z), dL/dh) ]
  ASSERT_VEC_EQ(op->delta[X_SLOT], ((float[]) {1, 2}));
  ASSERT_VEC_EQ(x->delta[X_SLOT], ((float[]) {-0.5986876601*0.4013123399, 0.4013123399*0.5986876601}));

  destroy_node(&x);
  destroy_node(&op);
}

void test_nmist(void) {
  NMatrix train_data = read_idx("train-images-idx3-ubyte", .normalize = true);
  NMatrix train_labels = read_idx("train-labels-idx1-ubyte", .normalize = false);

  NMatrix test_data = read_idx("t10k-images-idx3-ubyte", .normalize = true);
  NMatrix test_labels = read_idx("t10k-labels-idx1-ubyte", .normalize = false);

  train_data.rows = train_labels.rows = 1;
  test_data.rows = test_labels.rows = 1;

  Node* x = create_constant(28*28);

  Node* layer_01_linear = create_linear(x, 20);
  Node* layer_01 = create_relu(layer_01_linear);

  Node* layer_02_linear = create_linear(layer_01, 10);
  Node* layer_02 = create_relu(layer_02_linear);

  Node* layer_03_linear = create_linear(layer_02, 10);
  Node* layer_03 = create_softmax(layer_03_linear);

  NMatrix dL = mat_alloc(1, 10);
  NMatrix target = mat_alloc(1, 10);

  for (int i = 0; i < 1; i++) {
    NMatrix label = mat_row(train_labels, i);

    mat_copy(x->output[X_SLOT], mat_row(train_data, i));
    node_forward(layer_03);

    mat_fill(target, 0);
    VEC_AT(target, (int)VEC_AT(label, 0)) = 1;

    mat_sub(dL, layer_03->output[X_SLOT], target);
    mat_scale(dL, dL, 2.0);

    node_backward(layer_03, dL);

    printf("target  = "); mat_println(stdout, target);
    printf("dL      = "); mat_println(stdout, dL);

    printf("layer 0 = "); mat_println(stdout, layer_01->output[X_SLOT]);
    printf("          "); mat_println(stdout, layer_01_linear->delta[X_SLOT]);

    printf("layer 1 = "); mat_println(stdout, layer_02->output[X_SLOT]);
    printf("          "); mat_println(stdout, layer_02_linear->delta[X_SLOT]);

    printf("layer 2 = "); mat_println(stdout, layer_03->output[X_SLOT]);
    printf("          "); mat_println(stdout, layer_03_linear->delta[X_SLOT]);

    // int arg = mat_row_max(layer_03->output[X_SLOT]);
    // printf("%d ", arg);
    // mat_print(mat_row(train_labels, i));
    // printf(" = ");
    // mat_println(layer_03->output[X_SLOT]);
  }

  // node_backward(layer_03, dL);
}

int main(void) {
  srand(0);

  test_add();
  test_sub();
  test_mult();
  test_div();
  test_linear();
  test_sigmoid();
  test_softmax();
  test_nmist();

  return 0;
}
