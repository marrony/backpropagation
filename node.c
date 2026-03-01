#include "nn.h"
#include "node.h"
#include <stdbool.h>
#include <stdio.h>

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
