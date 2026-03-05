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
#define MAX_PARAMS 3

typedef struct {
  NMatrix value;
  NMatrix grad;
  NMatrix g_grad;
} Param;

typedef struct Node Node;
struct Node {
  Node_Type type;
  Node* input[MAX_INPUTS];
  int input_size;
  Param params[MAX_PARAMS];
  int param_size;
};

#define X_SLOT 0
#define U_SLOT 0
#define V_SLOT 1
#define W_SLOT 1
#define B_SLOT 2
#define KERN_SLOT 1

#define NULL_MATRIX (NMatrix) {0}
#define NULL_PARAM (Param) {0}

#define THIS_PARAM(node, slot) ((node)->params[(slot)])
#define IN_PARAM(node, slot) ((node)->input[(slot)] ? (node)->input[(slot)]->params[0] : NULL_PARAM)

#define FORWARD(slot) node_forward(node->input[(slot)])
#define BACKWARD(slot, dL) node_backward(node->input[(slot)], (dL))

void init_param(Param* param, int rows, int cols) {
  param->value  = mat_alloc(rows, cols);
  param->grad   = mat_alloc(rows, cols);
  param->g_grad = mat_alloc(rows, cols);
}

int node_get_value_rows(Node* node) {
  return THIS_PARAM(node, X_SLOT).value.rows;
}

int node_get_value_cols(Node* node) {
  return THIS_PARAM(node, X_SLOT).value.cols;
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

Node* create_constant(int rows, int cols) {
  Node* node = create_node();
  node->type = NODE_CONSTANT;
  node->input_size = 0;
  node->param_size = 1;
  init_param(&THIS_PARAM(node, X_SLOT), rows, cols);
  
  return node;
}

Node* create_variable(int rows, int cols) {
  Node* node = create_node();
  node->type = NODE_VARIABLE;
  node->input_size = 0;
  node->param_size = 1;
  init_param(&THIS_PARAM(node, X_SLOT), rows, cols);

  return node;
}

Node* create_linear(Node* input, int out_size) {
  Node* node = create_node();
  node->type = NODE_LINEAR;
  node->input[0] = input;
  node->input_size = 1;

  int rows = node_get_value_rows(input);
  int cols = node_get_value_cols(input);
  int in_size = rows * cols;

  node->param_size = 3;
  init_param(&THIS_PARAM(node, X_SLOT), 1, out_size);
  init_param(&THIS_PARAM(node, W_SLOT), out_size, in_size);
  init_param(&THIS_PARAM(node, B_SLOT), 1, out_size);

  // mat_rand(THIS_PARAM(W_SLOT));
  // mat_rand(THIS_PARAM(B_SLOT));

  return node;
}

Node* create_unary(Node_Type type, Node* input, int rows, int cols) {
  Node* node = create_node();
  node->type = type;
  node->input[0] = input;
  node->input_size = 1;
  node->param_size = 1;
  init_param(&THIS_PARAM(node, X_SLOT), rows, cols);
  return node;
}

Node* create_binary(Node_Type type, Node* input0, Node* input1, int rows, int cols) {
  Node* node = create_node();
  node->type = type;
  node->input[0] = input0;
  node->input[1] = input1;
  node->input_size = 2;
  node->param_size = 2;
  init_param(&THIS_PARAM(node, X_SLOT), rows, cols);
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

  node->param_size = 2;
  init_param(&THIS_PARAM(node, X_SLOT), conv_size, conv_size);
  init_param(&THIS_PARAM(node, KERN_SLOT), kern_size, kern_size);

  // mat_rand(THIS_PARAM(KERN_SLOT));

  return node;
}

Node* create_flatten(Node** inputs, int in_size) {
  Node* node = create_node();
  node->type = NODE_FLATTEN;

  node->input_size = in_size;

  int size = 0;
  for (int i = 0; i < in_size; i++) {
    node->input[i] = inputs[i];
    int rows = node_get_value_rows(inputs[i]);
    int cols = node_get_value_cols(inputs[i]);
    size += rows * cols;
  }

  node->param_size = 1;
  init_param(&THIS_PARAM(node, X_SLOT), 1, size);

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
  Param fx   = THIS_PARAM(node, X_SLOT);
  Param w    = THIS_PARAM(node, W_SLOT);
  Param b    = THIS_PARAM(node, B_SLOT);
  Param kern = THIS_PARAM(node, KERN_SLOT);

  Param x = IN_PARAM(node, X_SLOT);
  Param u = IN_PARAM(node, U_SLOT);
  Param v = IN_PARAM(node, V_SLOT);

  switch (node->type) {
    case NODE_CONSTANT:
    case NODE_VARIABLE:
      break;

    case NODE_LINEAR:
      // f(x) = x*W.T + b
      FORWARD(X_SLOT);
      mat_mult_transpose_add(fx.value, x.value, w.value, b.value);
      break;

    case NODE_SIGMOID:
      // f(x) = sigmoid(x)
      FORWARD(X_SLOT);
      sigmoid(fx.value, x.value);
      break;

    case NODE_SOFTMAX:
      // f(x) = softmax(x)
      FORWARD(X_SLOT);
      softmax(fx.value, x.value);
      break;

    case NODE_RELU:
      // f(x) = relu(x)
      FORWARD(X_SLOT);
      relu(fx.value, x.value);
      break;

    case NODE_FLATTEN:
      for (int i = 0; i < node->input_size; i++) {
        FORWARD(X_SLOT + i);

        Node* input = node->input[i];

        int rows = node_get_value_rows(input);
        int cols = node_get_value_cols(input);
        int size = rows * cols;

        NMatrix dst = mat_reshape(
            mat_row_slice(fx.value, i*size, size),
            rows,
            cols
        );

        mat_copy(dst, THIS_PARAM(input, X_SLOT).value);
      }
      break;

    case NODE_CONV2D:
    {
      // f(x) = conv2d(x, kernel)
      FORWARD(X_SLOT);

      NMatrix x_ = mat_reshape(x.value,
          fx.value.rows + kern.value.rows - 1,
          fx.value.cols + kern.value.cols - 1
      );

      for (int i = 0; i < fx.value.rows; i++) {
        for (int j = 0; j < fx.value.cols; j++) {
          MAT_AT(fx.value, i, j) = 0; // bias

          for (int ki = 0; ki < kern.value.rows; ki++) {
            for (int kj = 0; kj < kern.value.cols; kj++) {
              float a = MAT_AT(x_, i+ki, j+kj);
              float b = MAT_AT(kern.value, ki, kj);
              MAT_AT(fx.value, i, j) += a * b;
            }
          }
        }
      }
      break;
    }

    case NODE_SQUARE:
      // f(x) = x^2
      FORWARD(X_SLOT);
      for (int i = 0; i < x.value.rows; i++) {
        for (int j = 0; j < x.value.cols; j++)
          MAT_AT(fx.value, i, j) = MAT_AT(x.value, i, j)*MAT_AT(x.value, i, j);
      }
      break;

    case NODE_CUBE:
      // f(x) = x^3
      FORWARD(X_SLOT);
      for (int i = 0; i < x.value.rows; i++) {
        for (int j = 0; j < x.value.cols; j++)
          MAT_AT(fx.value, i, j) = MAT_AT(x.value, i, j)*MAT_AT(x.value, i, j)*MAT_AT(x.value, i, j);
      }
      break;

    case NODE_EXP:
      // f(x) = exp(x)
      FORWARD(X_SLOT);
      for (int i = 0; i < x.value.rows; i++) {
        for (int j = 0; j < x.value.cols; j++)
          MAT_AT(fx.value, i, j) = expf(MAT_AT(x.value, i, j));
      }
      break;

    case NODE_NEGATE:
      // f(x) = -x
      FORWARD(X_SLOT);
      mat_scale(fx.value, x.value, -1);
      break;

    case NODE_MULTIPLY:
      // f(u, v) = u * v
      FORWARD(U_SLOT);
      FORWARD(V_SLOT);
      // mat_memberwise_mult(fx, u, v);
      mat_mult(fx.value, u.value, v.value);
      break;

    case NODE_DIVIDE:
      // f(u, v) = u / v
      FORWARD(U_SLOT);
      FORWARD(V_SLOT);
      mat_memberwise_div(fx.value, u.value, v.value);
      break;

    case NODE_ADD:
      // f(u, v) = u + v
      FORWARD(U_SLOT);
      FORWARD(V_SLOT);
      mat_add(fx.value, u.value, v.value);
      break;

    case NODE_SUB:
      // f(u, v) = u - v
      FORWARD(U_SLOT);
      FORWARD(V_SLOT);
      mat_sub(fx.value, u.value, v.value);
      break;
  }
}

void node_backward(Node* node, NMatrix dL) {
  Param fx   = THIS_PARAM(node, X_SLOT);
  Param w    = THIS_PARAM(node, W_SLOT);
  Param b    = THIS_PARAM(node, B_SLOT);
  Param kern = THIS_PARAM(node, KERN_SLOT);

  Param x    = IN_PARAM(node, X_SLOT);
  Param u    = IN_PARAM(node, U_SLOT);
  Param v    = IN_PARAM(node, V_SLOT);

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
      mat_fill(x.grad, 0);
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
      mat_copy(fx.grad, dL);
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
      mat_copy(fx.grad, dL);

      // dL_dw = dL.T * x
      mat_transpose_mult(w.grad, dL, x.value);
      // dL_db = I * dL
      mat_copy(b.grad, dL);
      // dL_dx = dL * W
      mat_mult(x.grad, dL, w.value);

      BACKWARD(X_SLOT, x.grad);
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
      mat_copy(fx.grad, dL);
      dsigmoid(x.grad, fx.value, dL);
      BACKWARD(X_SLOT, x.grad);
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
      mat_copy(fx.grad, dL);
      dsoftmax(x.grad, fx.value, dL);
      BACKWARD(X_SLOT, x.grad);
      break;

    case NODE_RELU:
      mat_copy(fx.grad, dL);
      drelu(x.grad, fx.value, dL);
      BACKWARD(X_SLOT, x.grad);
      break;

    case NODE_FLATTEN:
      mat_copy(fx.grad, dL);

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

        mat_copy(IN_PARAM(node, i).grad, ith_dL_dx);

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
      mat_copy(fx.grad, dL);

      mat_fill(x.grad, 0);
      mat_fill(kern.grad, 0);

      for (int i = 0; i < dL.rows; i++) {
        for (int j = 0; j < dL.cols; j++) {
          NMatrix dL_dx_reshaped = mat_reshape(
              x.grad,
              fx.value.rows + kern.grad.rows - 1,
              fx.value.cols + kern.grad.cols - 1
          );

          NMatrix x_reshaped = mat_reshape(
              x.value,
              fx.value.rows + kern.grad.rows - 1,
              fx.value.cols + kern.grad.cols - 1
          );

          float delta = MAT_AT(dL, i, j);
          // Bias gradient
          // grad_b[f] += delta;

          for(int u = 0; u < kern.grad.rows; u++) {
            for(int v = 0; v < kern.grad.cols; v++) {
              // Weight gradient
              MAT_AT(kern.grad, u, v) += delta * MAT_AT(x_reshaped, i+u, j+v);

              // Input gradient
              MAT_AT(dL_dx_reshaped, i+u, j+v) += delta * MAT_AT(kern.value, u, v);
            }
          }
        }
      }

      BACKWARD(X_SLOT, x.grad);
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
      mat_copy(fx.grad, dL);

      for (int i = 0; i < x.grad.rows; i++) {
        for (int j = 0; j < x.grad.cols; j++)
          MAT_AT(x.grad, i, j) = 2*MAT_AT(x.value, i, j)*MAT_AT(dL, i, j);
      }

      BACKWARD(X_SLOT, x.grad);
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
      mat_copy(fx.grad, dL);

      for (int i = 0; i < x.grad.rows; i++) {
        for (int j = 0; j < x.grad.cols; j++)
          MAT_AT(x.grad, i, j) = 3*MAT_AT(x.value, i, j)*MAT_AT(x.value, i, j)*MAT_AT(dL, i, j);
      }

      BACKWARD(X_SLOT, x.grad);
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
      mat_copy(fx.grad, dL);

      for (int i = 0; i < x.grad.rows; i++) {
        for (int j = 0; j < x.grad.cols; j++)
          MAT_AT(x.grad, i, j) = MAT_AT(fx.value, i, j)*MAT_AT(dL, i, j);
      }

      BACKWARD(X_SLOT, x.grad);
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
      mat_copy(fx.grad, dL);

      for (int i = 0; i < x.grad.rows; i++) {
        for (int j = 0; j < x.grad.cols; j++) {
          MAT_AT(x.grad, i, j) = -MAT_AT(dL, i, j);
        }
      }

      BACKWARD(X_SLOT, x.grad);
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
      mat_copy(fx.grad, dL);

      mat_mult_transpose(u.grad, dL, v.value);
      mat_transpose_mult(v.grad, u.value, dL);

      BACKWARD(U_SLOT, u.grad);
      BACKWARD(V_SLOT, v.grad);
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
      mat_copy(fx.grad, dL);

      for (int i = 0; i < u.grad.rows; i++) {
        for (int j = 0; j < u.grad.cols; j++) {
          float one_over_v = 1.0 / MAT_AT(v.value, i, j);

          MAT_AT(u.grad, i, j) = one_over_v * MAT_AT(dL, i, j);
          MAT_AT(v.grad, i, j) = -MAT_AT(u.value, i, j) * one_over_v * one_over_v * MAT_AT(dL, i, j);
        }
      }

      BACKWARD(U_SLOT, u.grad);
      BACKWARD(V_SLOT, v.grad);
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
      mat_copy(fx.grad, dL);
      mat_copy(u.grad, dL);
      mat_copy(v.grad, dL);

      BACKWARD(U_SLOT, u.grad);
      BACKWARD(V_SLOT, v.grad);
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
      mat_copy(fx.grad, dL);

      for (int i = 0; i < u.grad.rows; i++) {
        for (int j = 0; j < u.grad.cols; j++) {
          MAT_AT(u.grad, i, j) = +MAT_AT(dL, i, j);
          MAT_AT(v.grad, i, j) = -MAT_AT(dL, i, j);
        }
      }

      BACKWARD(U_SLOT, u.grad);
      BACKWARD(V_SLOT, v.grad);
      break;
  }
}

// accumulate grads
void acc_grads(Node* node) {
  for (int i = 0; i < node->input_size; i++)
    acc_grads(node->input[i]);

  for (int i = 0; i < node->param_size; i++)
    mat_add(
        THIS_PARAM(node, i).g_grad,
        THIS_PARAM(node, i).g_grad,
        THIS_PARAM(node, i).grad
    );
}

void update_grads(Node* node, float lr) {
  for (int i = 0; i < node->input_size; i++)
    update_grads(node->input[i], lr);

  for (int i = 0; i < node->param_size; i++)
    mat_weighted_add(
        THIS_PARAM(node, i).value,
        THIS_PARAM(node, i).value,
        THIS_PARAM(node, i).g_grad,
        -lr
    );
}

void zero_grads(Node* node) {
  for (int i = 0; i < node->input_size; i++)
    zero_grads(node->input[i]);

  for (int i = 0; i < node->param_size; i++)
    mat_fill(THIS_PARAM(node, i).g_grad, 0);
}

#endif // NODE_H
