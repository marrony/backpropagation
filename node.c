#include "nn.h"

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

#define N 1
#define X_SLOT 0
#define U_SLOT 0
#define V_SLOT 1
#define W_SLOT 1
#define B_SLOT 2

#define THIS_VALUE(slot) (node->output[(slot)])
#define THIS_DELTA(slot) (node->delta[(slot)])

#define IN_VALUE(slot) (node->input[(slot)]->output[0])
#define IN_DELTA(slot) (node->input[(slot)]->delta[0])

#define FX_VALUE THIS_VALUE(X_SLOT)

#define FORWARD(slot) node_forward(node->input[(slot)])
#define BACKWARD(slot, dL) node_backward(node->input[(slot)], (dL))

Node* create_constant(NMatrix value) {
  Node* node = malloc(sizeof(Node));
  node->type = NODE_CONSTANT;
  THIS_VALUE(X_SLOT) = mat_alloc(1, N);
  THIS_DELTA(X_SLOT) = mat_alloc(1, N);
  
  mat_copy(node->output[0], value);
  return node;
}

Node* create_variable(NMatrix value) {
  Node* node = malloc(sizeof(Node));
  node->type = NODE_VARIABLE;
  THIS_VALUE(X_SLOT) = mat_alloc(1, N);
  THIS_DELTA(X_SLOT) = mat_alloc(1, N);

  mat_copy(node->output[0], value);
  return node;
}

Node* create_linear(Node* input, NMatrix w, NMatrix b) {
  Node* node = malloc(sizeof(Node));
  node->type = NODE_LINEAR;
  node->input[0] = input;
  THIS_VALUE(X_SLOT) = mat_alloc(1, N);
  THIS_VALUE(W_SLOT) = mat_alloc(1, N);
  THIS_VALUE(B_SLOT) = mat_alloc(1, N);
  THIS_DELTA(X_SLOT) = mat_alloc(1, N);
  THIS_DELTA(W_SLOT) = mat_alloc(1, N);
  THIS_DELTA(B_SLOT) = mat_alloc(1, N);

  mat_copy(node->output[W_SLOT], w);
  mat_copy(node->output[B_SLOT], b);
  return node;
}

Node* create_unary(Node_Type type, Node* input) {
  Node* node = malloc(sizeof(Node));
  node->type = type;
  node->input[0] = input;
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
  switch (node->type) {
    case NODE_CONSTANT:
    case NODE_VARIABLE:
      break;

    case NODE_LINEAR:
      // f(x) = x*w + b
      FORWARD(X_SLOT);
      mat_mult_add(FX_VALUE, IN_VALUE(X_SLOT), THIS_VALUE(W_SLOT), THIS_VALUE(B_SLOT));
      break;

    case NODE_SIGMOID:
      // f(x) = sigmoid(x)
      FORWARD(X_SLOT);
      sigmoid(FX_VALUE, IN_VALUE(X_SLOT));
      break;

    case NODE_SOFTMAX:
      // f(x) = softmax(x)
      FORWARD(X_SLOT);
      softmax(FX_VALUE, IN_VALUE(X_SLOT));
      break;

    case NODE_SQUARE: {
      // f(x) = x^2
      FORWARD(X_SLOT);
      NMatrix x = IN_VALUE(X_SLOT);
      for (int i = 0; i < x.rows; i++) {
        for (int j = 0; j < x.cols; j++)
          MAT_AT(FX_VALUE, i, j) = MAT_AT(x, i, j)*MAT_AT(x, i, j);
      }
      break;
    }

    case NODE_CUBE: {
      // f(x) = x^3
      FORWARD(X_SLOT);
      NMatrix x = IN_VALUE(X_SLOT);
      for (int i = 0; i < x.rows; i++) {
        for (int j = 0; j < x.cols; j++)
          MAT_AT(FX_VALUE, i, j) = MAT_AT(x, i, j)*MAT_AT(x, i, j)*MAT_AT(x, i, j);
      }
      break;
    }

    case NODE_EXP: {
      // f(x) = exp(x)
      FORWARD(X_SLOT);
      NMatrix x = IN_VALUE(X_SLOT);
      for (int i = 0; i < x.rows; i++) {
        for (int j = 0; j < x.cols; j++)
          MAT_AT(FX_VALUE, i, j) = expf(MAT_AT(x, i, j));
      }
      break;
    }

    case NODE_NEGATE: {
      // f(x) = -x
      FORWARD(X_SLOT);
      mat_scale(FX_VALUE, FX_VALUE, -1);
      break;
    }

    case NODE_MULTIPLY:
      // f(u, v) = u * v
      FORWARD(U_SLOT);
      FORWARD(V_SLOT);
      mat_memberwise_mult(FX_VALUE, IN_VALUE(U_SLOT), IN_VALUE(V_SLOT));
      break;

    case NODE_DIVIDE:
      // f(u, v) = u / v
      FORWARD(U_SLOT);
      FORWARD(V_SLOT);
      mat_memberwise_div(FX_VALUE, IN_VALUE(U_SLOT), IN_VALUE(V_SLOT));
      break;

    case NODE_ADD:
      // f(u, v) = u + v
      FORWARD(U_SLOT);
      FORWARD(V_SLOT);
      mat_add(FX_VALUE, IN_VALUE(U_SLOT), IN_VALUE(V_SLOT));
      break;

    case NODE_SUB:
      // f(u, v) = u - v
      FORWARD(U_SLOT);
      FORWARD(V_SLOT);
      mat_sub(FX_VALUE, IN_VALUE(U_SLOT), IN_VALUE(V_SLOT));
      break;
  }
}

void node_backward(Node* node, NMatrix dL) {
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
      mat_fill(THIS_DELTA(X_SLOT), 0);
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
      mat_copy(THIS_DELTA(X_SLOT), dL);
      break;

    case NODE_LINEAR: {
      // upstream:
      //   dL/d(x*w + b) = dL
      //
      // local:
      //   d(x*w + b)/dw = x
      //   d(x*w + b)/db = 1
      //   d(x*w + b)/dx = w
      //
      // downstream:
      //   dL/dw = d(x*w + b)/dw * dL = x*dL
      //   dL/db = d(x*w + b)/db * dL = 1*dL
      //   dL/dx = d(x*w + b)/dx * dL = w*dL
      NMatrix x = IN_VALUE(X_SLOT);
      NMatrix w = THIS_VALUE(W_SLOT);
      NMatrix dL_dx = IN_DELTA(X_SLOT);

      mat_copy(THIS_DELTA(X_SLOT), dL);

      for (int i = 0; i < dL_dx.rows; i++) {
        for (int j = 0; j < dL_dx.cols; j++) {
          MAT_AT(THIS_DELTA(W_SLOT), i, j) = MAT_AT(x, i, j) * MAT_AT(dL, i, j);
          MAT_AT(THIS_DELTA(B_SLOT), i, j) = 1.0             * MAT_AT(dL, i, j);
          MAT_AT(dL_dx, i, j)              = MAT_AT(w, i, j) * MAT_AT(dL, i, j);
        }
      }

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
      NMatrix dx_dL = IN_DELTA(X_SLOT);

      mat_copy(THIS_DELTA(X_SLOT), dL);
      dsigmoid(dx_dL, FX_VALUE, dL);
      BACKWARD(X_SLOT, dx_dL);
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
      NMatrix dx_dL = IN_DELTA(X_SLOT);

      mat_copy(THIS_DELTA(X_SLOT), dL);
      dsoftmax(dx_dL, FX_VALUE, dL);
      BACKWARD(X_SLOT, dx_dL);
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
      NMatrix x = IN_VALUE(X_SLOT);
      NMatrix dL_dx = IN_DELTA(X_SLOT);

      mat_copy(THIS_DELTA(X_SLOT), dL);

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
      NMatrix x = IN_VALUE(X_SLOT);
      NMatrix dL_dx = IN_DELTA(X_SLOT);

      mat_copy(THIS_DELTA(X_SLOT), dL);

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
      NMatrix dL_dx = IN_DELTA(X_SLOT);

      mat_copy(THIS_DELTA(X_SLOT), dL);

      for (int i = 0; i < dL_dx.rows; i++) {
        for (int j = 0; j < dL_dx.cols; j++)
          MAT_AT(dL_dx, i, j) = MAT_AT(FX_VALUE, i, j)*MAT_AT(dL, i, j);
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
      NMatrix dL_dx = IN_DELTA(X_SLOT);

      mat_copy(THIS_DELTA(X_SLOT), dL);

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
      NMatrix u = IN_VALUE(U_SLOT);
      NMatrix v = IN_VALUE(V_SLOT);
      NMatrix dL_du = IN_DELTA(U_SLOT);
      NMatrix dL_dv = IN_DELTA(V_SLOT);

      mat_copy(THIS_DELTA(X_SLOT), dL);

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
      NMatrix u = IN_VALUE(U_SLOT);
      NMatrix v = IN_VALUE(V_SLOT);
      NMatrix dL_du = IN_DELTA(U_SLOT);
      NMatrix dL_dv = IN_DELTA(V_SLOT);

      mat_copy(THIS_DELTA(X_SLOT), dL);

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
      //   dL+du = d(u+v)/du * dL = 1*dL
      //   dL+dv = d(u+v)/dv * dL = 1*dL
      NMatrix dL_du = IN_DELTA(U_SLOT);
      NMatrix dL_dv = IN_DELTA(V_SLOT);

      mat_copy(THIS_DELTA(X_SLOT), dL);
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
      //   dL-du = d(u-v)/du * dL = +1*dL
      //   dL-dv = d(u-v)/dv * dL = -1*dL
      NMatrix dL_du = IN_DELTA(U_SLOT);
      NMatrix dL_dv = IN_DELTA(V_SLOT);

      mat_copy(THIS_DELTA(X_SLOT), dL);

      for (int i = 0; i < dL_du.rows; i++) {
        for (int j = 0; j < dL_du.cols; j++) {
          MAT_AT(dL_du, i, j) = +MAT_AT(dL, i, j);
          MAT_AT(dL_du, i, j) = -MAT_AT(dL, i, j);
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
    create_layer(.inputs = 1, .outputs = 1, .forward = linear, .backward = dlinear),
  };

  Neuron_Network nn = neuron_create(layers, ARRAY_LEN(layers));
  Neuron_Network grad = neuron_clone(nn);
  NMatrix* outputs = create_outputs(nn);
  NMatrix* dL_dh = create_outputs(nn);
  NMatrix* dL_dz = create_outputs(nn);
  NMatrix target = mat_alloc(1, 1);
  NMatrix input = mat_alloc(1, 1);

  MAT_AT(input, 0, 0) = 0.5;
  MAT_AT(target, 0, 0) = 2;

  MAT_AT(nn.w[0], 0, 0) = 0.1;
  MAT_AT(nn.b[0], 0, 0) = -0.01;

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

int main(void) {
  srand(0);

  // expr0();
  expr1();
  expr2();

  return 0;
}
