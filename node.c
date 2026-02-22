#include "nn.h"
#include <stdio.h>

typedef enum {
  NODE_CONSTANT,
  NODE_VARIABLE,
  NODE_LINEAR,
  NODE_SIGMOID,
  NODE_SOFTMAX,
  // NODE_RELU,
  // NODE_CONV2D,
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
  Node* input[2];   // u, v
  NMatrix output[3]; // x, w, b
  NMatrix delta[3];  // dx, dw, db
};

#define N 2
#define X_SLOT 0
#define U_SLOT 0
#define V_SLOT 1
#define W_SLOT 1
#define B_SLOT 2

#define THIS_VALUE(slot) (node->output[(slot)])
#define THIS_DELTA(slot) (node->delta[(slot)])

#define NULL_MATRIX (NMatrix) {0}

#define IN_VALUE(slot) (node->input[(slot)] ? node->input[(slot)]->output[0] : NULL_MATRIX)
#define IN_DELTA(slot) (node->input[(slot)] ? node->input[(slot)]->delta[0] : NULL_MATRIX)

#define FX_VALUE THIS_VALUE(X_SLOT)

#define FORWARD(slot) node_forward(node->input[(slot)])
#define BACKWARD(slot, dL) node_backward(node->input[(slot)], (dL))

void destroy_node(Node** node) {
  free(*node);
  *node = NULL;
}

Node* create_constant(NMatrix value) {
  Node* node = malloc(sizeof(Node));
  node->type = NODE_CONSTANT;
  node->input[0] = NULL;
  node->input[1] = NULL;
  THIS_VALUE(X_SLOT) = mat_alloc(1, N);
  THIS_DELTA(X_SLOT) = mat_alloc(1, N);
  
  mat_copy(node->output[0], value);
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

Node* create_linear(Node* input, NMatrix w, NMatrix b) {
  Node* node = malloc(sizeof(Node));
  node->type = NODE_LINEAR;
  node->input[0] = input;
  node->input[1] = NULL;

  NMatrix x = input->output[X_SLOT];

  THIS_VALUE(X_SLOT) = mat_alloc(x.rows, w.rows);
  THIS_VALUE(W_SLOT) = mat_alloc(w.rows, w.cols);
  THIS_VALUE(B_SLOT) = mat_alloc(b.rows, b.cols);
  THIS_DELTA(X_SLOT) = mat_alloc(x.rows, w.rows);
  THIS_DELTA(W_SLOT) = mat_alloc(w.rows, w.cols);
  THIS_DELTA(B_SLOT) = mat_alloc(b.rows, b.cols);

  mat_copy(node->output[W_SLOT], w);
  mat_copy(node->output[B_SLOT], b);
  return node;
}

Node* create_unary(Node_Type type, Node* input) {
  Node* node = malloc(sizeof(Node));
  node->type = type;
  node->input[0] = input;
  node->input[1] = NULL;
  THIS_VALUE(X_SLOT) = mat_alloc(1, N);
  THIS_DELTA(X_SLOT) = mat_alloc(1, N);
  return node;
}

Node* create_binary(Node_Type type, Node* input0, Node* input1) {
  Node* node = malloc(sizeof(Node));
  node->type = type;
  node->input[0] = input0;
  node->input[1] = input1;
  THIS_VALUE(X_SLOT) = mat_alloc(1, N);
  THIS_DELTA(X_SLOT) = mat_alloc(1, N);
  return node;
}

Node* create_sigmoid(Node* input) {
  return create_unary(NODE_SIGMOID, input);
}

Node* create_softmax(Node* input) {
  return create_unary(NODE_SOFTMAX, input);
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

void node_forward(Node* node) {
  NMatrix fx = FX_VALUE;
  NMatrix x = IN_VALUE(X_SLOT);
  NMatrix w = THIS_VALUE(W_SLOT);
  NMatrix b = THIS_VALUE(B_SLOT);
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
  NMatrix fx = FX_VALUE;
  NMatrix x = IN_VALUE(X_SLOT);
  NMatrix w = THIS_VALUE(W_SLOT);
  NMatrix u = IN_VALUE(U_SLOT);
  NMatrix v = IN_VALUE(V_SLOT);
  // downstream
  NMatrix dL_dw = THIS_DELTA(W_SLOT);
  NMatrix dL_db = THIS_DELTA(B_SLOT);
  NMatrix dL_dx = IN_DELTA(X_SLOT);
  NMatrix dL_du = IN_DELTA(U_SLOT);
  NMatrix dL_dv = IN_DELTA(V_SLOT);

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

    case NODE_LINEAR: {
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

      printf("dL = "); mat_println(dL);
      printf("x = "); mat_println(x);
      printf("w = "); mat_println(w);

      // dL_dw = dL.T * x
      mat_transpose_mult(dL_dw, dL, x);
      // dL_db = I*dL
      mat_copy(dL_db, dL);
      // dL_dx = dL * W
      mat_mult(dL_dx, dL, w);

      printf("dL/dw = "); mat_println(dL_dw);
      printf("dL/db = "); mat_println(dL_db);
      printf("dL/dx = "); mat_println(dL_dx);

      BACKWARD(X_SLOT, dL_dx);
      break;
    }

    case NODE_SIGMOID: {
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
    }

    case NODE_SOFTMAX: {
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
    }

    case NODE_SQUARE: {
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
    }

    case NODE_CUBE: {
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
    }

    case NODE_EXP: {
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
    }

    case NODE_NEGATE: {
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
    }

    case NODE_MULTIPLY: {
      // upstream:
      //   dL/d(u*v) = dL
      //
      // local:
      //   d(u*v)/du = v
      //   d(u*v)/dv = u
      //
      // downstream:
      //   dL/du = d(u*v)/du * dL = v*dL
      //   dL/dv = d(u*v)/dv * dL = u*dL
      mat_copy(dL_df, dL);

      for (int i = 0; i < dL_du.rows; i++) {
        for (int j = 0; j < dL_du.cols; j++) {
          MAT_AT(dL_du, i, j) = MAT_AT(v, i, j)*MAT_AT(dL, i, j);
          MAT_AT(dL_dv, i, j) = MAT_AT(u, i, j)*MAT_AT(dL, i, j);
        }
      }

      BACKWARD(U_SLOT, dL_du);
      BACKWARD(V_SLOT, dL_dv);
      break;
    }

    case NODE_DIVIDE: {
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
    }

    case NODE_ADD: {
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
    }

    case NODE_SUB: {
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

void expr0(void) {
  NMatrix image = mat_alloc(1, N);
  mat_rand(image);

  NMatrix dL = mat_alloc(1, N);
  NMatrix x_mat = mat_alloc(1, N);
  NMatrix one_mat = mat_alloc(1, N);

  mat_fill(x_mat, 2);
  mat_fill(one_mat, 1);

  Node* one = (Node*)create_constant(one_mat);
  (void)one;

  Node* x = (Node*)create_variable(x_mat);
#if 1
  //1/(1+exp(-x))
  Node* neg = (Node*)create_negate(x);
  Node* exp = (Node*)create_exp(neg);
  Node* sum = (Node*)create_add(one, exp);
  Node* output = (Node*)create_divide(one, sum);
#else
  Node* output = (Node*)create_sigmoid(x);
#endif

  struct {
    const char* str;
    Node* node;
  } nodes[] = {
    {"x   ", x},
    {"out ", output},
  };

  for (int iter = 0; iter < 1; iter++) {
    node_forward(output);

    mat_fill(dL, 1);
    printf("\nd(L) = "); mat_print(dL);

    node_backward(output, dL);

    for (size_t i = 0; i < ARRAY_LEN(nodes); i++) {
      printf("\n%s = ", nodes[i].str); mat_print(nodes[i].node->output[X_SLOT]);
      printf(" d(%s) = ", nodes[i].str); mat_print(nodes[i].node->delta[X_SLOT]);
    }

    update_grads(output, 0.05);

    printf("\n==================");
  }
}

void expr1(void) {
  NMatrix image = mat_alloc(1, N);
  mat_rand(image);

  NMatrix dL = mat_alloc(1, N);

  NMatrix x_mat = mat_alloc(1, N);
  NMatrix w_mat = mat_alloc(1, N);
  NMatrix b_mat = mat_alloc(1, N);
  NMatrix r3 = mat_alloc(1, N);
  mat_fill(x_mat, 0.5);
  mat_fill(w_mat, 0.1);
  mat_fill(b_mat, -0.01);
  mat_fill(r3, 2);

  Node* x = (Node*)create_variable(x_mat);
#if 1
  Node* output = (Node*)create_linear(x, w_mat, b_mat);
#else
  Node* w = (Node*)create_variable(w_mat);
  Node* b = (Node*)create_variable(b_mat);
  Node* mult = (Node*)create_multiply(x, w);
  Node* output = (Node*)create_add(mult, b);
#endif

  struct {
    const char* str;
    Node* node;
  } nodes[] = {
    {"x   ", x},
    {"out ", output},
  };

  for (int iter = 0; iter < 1; iter++) {
    node_forward(output);

    mat_fill(dL, 2*(MAT_AT(output->output[X_SLOT], 0, 0) - 2));
    printf("\nd(L) = "); mat_print(dL);

    node_backward(output, dL);

    for (size_t i = 0; i < ARRAY_LEN(nodes); i++) {
      printf("\n%s = ", nodes[i].str); mat_print(nodes[i].node->output[X_SLOT]);
      printf(" d(%s) = ", nodes[i].str); mat_print(nodes[i].node->delta[X_SLOT]);
    }

    update_grads(output, 0.05);

    printf("\n==================");
  }
}

void expr2(void) {
  Neuron_Layer layers[] = {
    create_layer(.inputs = 2, .outputs = 2, .forward = linear, .backward = dlinear),
  };

  // mat_copy(x_mat, mat_init(1, N, (float[]){5, 6}));
  // mat_copy(w_mat, mat_init(N, N, (float[]){1, 2, 3, 4}));
  // mat_copy(b_mat, mat_init(1, N, (float[]){7, 8}));
  // mat_copy(dL, mat_init(1, N, (float[]){5, 5}));

  Neuron_Network nn = neuron_create(layers, ARRAY_LEN(layers));
  Neuron_Network grad = neuron_clone(nn);
  NMatrix* outputs = create_outputs(nn);
  NMatrix* dL_dh = create_outputs(nn);
  NMatrix* dL_dz = create_outputs(nn);
  NMatrix target = mat_alloc(1, 2);
  NMatrix input = mat_alloc(1, 2);

  mat_copy(input, mat_init(1, N, (float[]){5, 6}));

  //( [x y] - [30 42] ) * 2 = [5 5]
  //[x y] - [30 42] = [5 5]/2
  //[x y] = [5 5]/2 + [30 42]
  mat_copy(target, mat_init(1, N, (float[]){30.0 - 5.0/2, 42.0 - 5.0/2}));

  mat_copy(nn.w[0], mat_init(N, N, (float[]){1, 2, 3, 4}));
  mat_copy(nn.b[0], mat_init(1, N, (float[]){7, 8}));

  for (int i = 0; i < 1; i++) {
    neuron_zero(&grad);
    forward(&nn, outputs, input);
    backward(&nn, outputs, &grad, dL_dh, dL_dz, input, target);
    neuron_weighted_add(&nn, &grad, -0.05);

    printf("\ndL/dh = "); mat_print(dL_dh[0]);
    printf("\ndL/dz = "); mat_print(dL_dz[0]);
    printf("\nh     = "); mat_print(outputs[0]);
    printf("\ndw    = "); mat_print(grad.w[0]);
    printf("\ndb    = "); mat_print(grad.b[0]);
  }
}

void assert_eq(const char* func, const char* file, int line, float a, float b) {
  float x = a - b;
  bool eq = (x*x) <= (0.001f*0.001f);
  if (!eq) {
    printf("\nAssertion Failed\n\n%s:%d (%s) => %+.8f != %+.8f\n", file, line, func, a, b);
    exit(1);
  }
}

void assert_vec_eq(const char* func, const char* file, int line, NMatrix va, const float* vb) {
  for (int i = 0; i < va.cols; i++) {
    float a = VEC_AT(va, i);
    float b = vb[i];
    float x = a - b;
    bool eq = (x*x) <= (0.001f*0.001f);
    if (!eq) {
      printf("\nAssertion Failed\n\n%s:%d (%s) => ", file, line, func);
      mat_print(va);
      printf(" != ");
      mat_println(mat_init(va.rows, va.cols, (float*)vb));
      exit(1);
    }
  }
}

#define ASSERT_EQ(a, b) assert_eq(__func__, __FILE__, __LINE__, (a), (b))

#define ASSERT_VEC_EQ(a, b) assert_vec_eq(__func__, __FILE__, __LINE__, (a), (b))

void test_add(void) {
  NMatrix dL = mat_alloc(1, N);
  NMatrix x_mat = mat_alloc(1, N);
  NMatrix y_mat = mat_alloc(1, N);

  mat_fill(x_mat, 2);
  mat_fill(y_mat, 1);
  mat_fill(dL, 2);

  Node* x = (Node*)create_variable(x_mat);
  Node* y = (Node*)create_variable(y_mat);
  Node* op = (Node*)create_add(x, y);

  node_forward(op);
  node_backward(op, dL);

  printf("x       = "); mat_print(x->output[X_SLOT]);
  printf(" d(x)     = "); mat_println(x->delta[X_SLOT]);

  printf("y       = "); mat_print(y->output[X_SLOT]);
  printf(" d(y)     = "); mat_println(y->delta[X_SLOT]);

  printf("(x + y) = "); mat_print(op->output[X_SLOT]);
  printf(" d(x + y) = "); mat_println(op->delta[X_SLOT]);

  ASSERT_EQ(MAT_AT(op->output[X_SLOT], 0, 0), 3.0);
  ASSERT_EQ(MAT_AT(op->delta[X_SLOT], 0, 0),  MAT_AT(dL, 0, 0));
  ASSERT_EQ(MAT_AT(x->delta[X_SLOT], 0, 0), 2.0);
  ASSERT_EQ(MAT_AT(y->delta[X_SLOT], 0, 0), 2.0);

  destroy_node(&x);
  destroy_node(&y);
  destroy_node(&op);
}

void test_sub(void) {
  NMatrix dL = mat_alloc(1, N);
  NMatrix x_mat = mat_alloc(1, N);
  NMatrix y_mat = mat_alloc(1, N);

  mat_fill(x_mat, 2);
  mat_fill(y_mat, 1);
  mat_fill(dL, 2);

  Node* x = (Node*)create_variable(x_mat);
  Node* y = (Node*)create_variable(y_mat);
  Node* op = (Node*)create_sub(x, y);

  node_forward(op);
  node_backward(op, dL);

  printf("x       = "); mat_print(x->output[X_SLOT]);
  printf(" d(x)     = "); mat_println(x->delta[X_SLOT]);

  printf("y       = "); mat_print(y->output[X_SLOT]);
  printf(" d(y)     = "); mat_println(y->delta[X_SLOT]);

  printf("(x - y) = "); mat_print(op->output[X_SLOT]);
  printf(" d(x - y) = "); mat_println(op->delta[X_SLOT]);

  ASSERT_EQ(MAT_AT(op->output[X_SLOT], 0, 0), 1.0);
  ASSERT_EQ(MAT_AT(op->delta[X_SLOT], 0, 0),  MAT_AT(dL, 0, 0));
  ASSERT_EQ(MAT_AT(x->delta[X_SLOT], 0, 0), +2.0);
  ASSERT_EQ(MAT_AT(y->delta[X_SLOT], 0, 0), -2.0);

  destroy_node(&x);
  destroy_node(&y);
  destroy_node(&op);
}

void test_mult(void) {
  NMatrix dL = mat_alloc(1, N);
  NMatrix x_mat = mat_alloc(1, N);
  NMatrix y_mat = mat_alloc(1, N);

  mat_fill(x_mat, 2);
  mat_fill(y_mat, 3);
  mat_fill(dL, 2);

  Node* x = (Node*)create_variable(x_mat);
  Node* y = (Node*)create_variable(y_mat);
  Node* op = (Node*)create_multiply(x, y);

  node_forward(op);
  node_backward(op, dL);

  printf("x       = "); mat_print(x->output[X_SLOT]);
  printf(" d(x)     = "); mat_println(x->delta[X_SLOT]);

  printf("y       = "); mat_print(y->output[X_SLOT]);
  printf(" d(y)     = "); mat_println(y->delta[X_SLOT]);

  printf("(x * y) = "); mat_print(op->output[X_SLOT]);
  printf(" d(x * y) = "); mat_println(op->delta[X_SLOT]);

  ASSERT_EQ(MAT_AT(op->output[X_SLOT], 0, 0), 6.0);
  // dL/du = d(u*v)/du * dL = v*dL
  // dL/dv = d(u*v)/dv * dL = u*dL
  ASSERT_EQ(MAT_AT(op->delta[X_SLOT], 0, 0),  MAT_AT(dL, 0, 0));
  ASSERT_EQ(MAT_AT(x->delta[X_SLOT], 0, 0), 6.0);
  ASSERT_EQ(MAT_AT(y->delta[X_SLOT], 0, 0), 4.0);

  destroy_node(&x);
  destroy_node(&y);
  destroy_node(&op);
}

void test_div(void) {
  NMatrix dL = mat_alloc(1, N);
  NMatrix x_mat = mat_alloc(1, N);
  NMatrix y_mat = mat_alloc(1, N);

  mat_fill(x_mat, 2);
  mat_fill(y_mat, 3);
  mat_fill(dL, 2);

  Node* x = (Node*)create_variable(x_mat);
  Node* y = (Node*)create_variable(y_mat);
  Node* op = (Node*)create_divide(x, y);

  node_forward(op);
  node_backward(op, dL);

  printf("x       = "); mat_print(x->output[X_SLOT]);
  printf(" d(x)     = "); mat_println(x->delta[X_SLOT]);

  printf("y       = "); mat_print(y->output[X_SLOT]);
  printf(" d(y)     = "); mat_println(y->delta[X_SLOT]);

  printf("(x * y) = "); mat_print(op->output[X_SLOT]);
  printf(" d(x / y) = "); mat_println(op->delta[X_SLOT]);

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

  Node* x = (Node*)create_variable(x_mat);
  Node* op = (Node*)create_linear(x, w_mat, b_mat);

  node_forward(op);
  node_backward(op, dL);

  printf("x       = "); mat_print(x->output[X_SLOT]);
  printf(" d(x)     = "); mat_println(x->delta[X_SLOT]);

  printf("(x*w + b) = "); mat_print(op->output[X_SLOT]);
  printf(" d(x*w + b) = "); mat_println(op->delta[X_SLOT]);

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
  NMatrix dL = mat_alloc(1, N);
  NMatrix x_mat = mat_alloc(1, N);

  mat_fill(x_mat, 0.5);
  mat_fill(dL, 2);

  Node* x = (Node*)create_variable(x_mat);
  Node* op = (Node*)create_sigmoid(x);

  node_forward(op);
  node_backward(op, dL);

  printf("x       = "); mat_print(x->output[X_SLOT]);
  printf(" d(x)     = "); mat_println(x->delta[X_SLOT]);

  printf("sigmoid(x) = "); mat_print(op->output[X_SLOT]);
  printf(" d(sigmoid(x)) = "); mat_println(op->delta[X_SLOT]);

  // sigmoid(x) = 1/(1+exp(-x))
  ASSERT_EQ(MAT_AT(op->output[X_SLOT], 0, 0), +0.62245935f);
  // dL/dx = sigmoid(x)*(1 - sigmoid(x)) * dL
  ASSERT_EQ(MAT_AT(op->delta[X_SLOT], 0, 0),  MAT_AT(dL, 0, 0));
  ASSERT_EQ(MAT_AT(x->delta[X_SLOT], 0, 0),   +0.47000742f);

  destroy_node(&x);
  destroy_node(&op);
}

void test_softmax(void) {
  NMatrix dL = mat_alloc(1, N);
  NMatrix x_mat = mat_alloc(1, N);

  mat_fill(x_mat, 0.5);
  mat_fill(dL, 2);

  Node* x = (Node*)create_variable(x_mat);
  Node* op = (Node*)create_softmax(x);

  node_forward(op);
  node_backward(op, dL);

  printf("x       = "); mat_print(x->output[X_SLOT]);
  printf(" d(x)     = "); mat_println(x->delta[X_SLOT]);

  printf("softmax(x) = "); mat_print(op->output[X_SLOT]);
  printf(" d(softmax(x)) = "); mat_println(op->delta[X_SLOT]);

  // softmax(x)
  ASSERT_EQ(MAT_AT(op->output[X_SLOT], 0, 0), +1.00000000f);
  // dL/dx = softmax(z) * [ dL/dh - dot(softmax(z), dL/dh) ]
  ASSERT_EQ(MAT_AT(op->delta[X_SLOT], 0, 0),  MAT_AT(dL, 0, 0));
  ASSERT_EQ(MAT_AT(x->delta[X_SLOT], 0, 0),   +0.00000000f);

  destroy_node(&x);
  destroy_node(&op);
}

int main(void) {
  srand(0);

  // expr0();
  // expr1();
  // expr2();

  printf("\n===================\n");

  // test_add();
  // test_sub();
  // test_mult();
  // test_div();
  test_linear();
  // test_sigmoid();
  // test_softmax();

  return 0;
}
