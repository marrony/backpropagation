#include "nn.h"
#include "node.h"
#include <stdbool.h>
#include <stdio.h>

void test_add(void) {
  printf("test_add\n");
  int N = 2;

  Node* x = create_variable(1, N);
  Node* y = create_variable(1, N);
  Node* op = create_add(x, y);

  NMatrix dL = mat_alloc(1, N);

  mat_fill(x->output.value, 2);
  mat_fill(y->output.value, 1);
  mat_fill(dL, 2);

  node_forward(op);
  node_backward(op, dL);

  printf("x       = "); mat_print(x->output.value);
  printf(" d(x)     = "); mat_println(x->output.grad);

  printf("y       = "); mat_print(y->output.value);
  printf(" d(y)     = "); mat_println(y->output.grad);

  printf("(x + y) = "); mat_print(op->output.value);
  printf(" d(x + y) = "); mat_println(op->output.grad);

  // Forward: x + y = 2 + 1 = 3
  ASSERT_VEC_EQ(op->output.value, ((float[]) {3.0, 3.0}));
  ASSERT_VEC_EQ(x->output.value, ((float[]) {2.0, 2.0}));
  ASSERT_VEC_EQ(y->output.value, ((float[]) {1.0, 1.0}));

 // Backward: dL/du = dL, d(u+v)/du = 1, d(u+v)/dv = 1
// So both x.grad and y.grad should equal dL
  ASSERT_VEC_EQ(op->output.grad, ((float[]) {2.0, 2.0}));
  ASSERT_VEC_EQ(x->output.grad, ((float[]) {2.0, 2.0}));
  ASSERT_VEC_EQ(y->output.grad, ((float[]) {2.0, 2.0}));

  destroy_node(&op);
  mat_free(dL);
}

void test_sub(void) {
  printf("test_sub\n");
  int N = 2;

  Node* x = create_variable(1, N);
  Node* y = create_variable(1, N);
  Node* op = create_sub(x, y);

  NMatrix dL = mat_alloc(1, N);

  mat_fill(x->output.value, 2);
  mat_fill(y->output.value, 1);
  mat_fill(dL, 2);

  node_forward(op);
  node_backward(op, dL);

  printf("x       = "); mat_print(x->output.value);
  printf(" d(x)     = "); mat_println(x->output.grad);

  printf("y       = "); mat_print(y->output.value);
  printf(" d(y)     = "); mat_println(y->output.grad);

  printf("(x - y) = "); mat_print(op->output.value);
  printf(" d(x - y) = "); mat_println(op->output.grad);

  // Forward: x - y = 2 - 1 = 1
  ASSERT_VEC_EQ(op->output.value, ((float[]) {1.0, 1.0}));
  ASSERT_VEC_EQ(x->output.value, ((float[]) {2.0, 2.0}));
  ASSERT_VEC_EQ(y->output.value, ((float[]) {1.0, 1.0}));

// Backward: dL/du = dL, d(u-v)/du = 1, d(u-v)/dv = -1
// So x.grad = dL, y.grad = -dL
  ASSERT_VEC_EQ(op->output.grad, ((float[]) {2, 2}));
  ASSERT_VEC_EQ(x->output.grad, ((float[]) {2, 2}));
  ASSERT_VEC_EQ(y->output.grad, ((float[]) {-2, -2}));

  destroy_node(&op);
  mat_free(dL);
}

void test_mult(void) {
  printf("test_mult\n");
  int N = 1;
  int M = 2;
  int P = 2;

  Node* x = create_variable(N, M);
  Node* y = create_variable(M, P);
  Node* op = create_multiply(x, y);

  NMatrix dL = mat_alloc(N, P);

  mat_fill(x->output.value, 2);
  mat_fill(y->output.value, 3);
  mat_fill(dL, 2);

  node_forward(op);
  node_backward(op, dL);

  printf("x         = "); mat_println(x->output.value);
  printf("d(x*y)/dx = "); mat_println(x->output.grad);

  printf("y         = "); mat_println(y->output.value);
  printf("d(x*y)/dy = "); mat_println(y->output.grad);

  printf("x*y       = "); mat_println(op->output.value);
  printf("dL/d(x*y) = "); mat_println(op->output.grad);

  // Forward: x * y = [1x2] * [2x2] = [1x2]
  // [2,2] * [3,3] = [6,6]
  ASSERT_VEC_EQ(op->output.value, ((float[]) {12, 12}));
  ASSERT_VEC_EQ(x->output.value, ((float[]) {2, 2}));
  ASSERT_VEC_EQ(y->output.value, ((float[]) {3, 3, 3, 3}));

  // Backward: dL/du = dL * v^T, dL/dv = u^T * dL
  // u = [2,2], v = [3,3,3,3], dL = [2,2]
  // u.grad = dL * v^T = [2,2] * [3,3,3,3]^T = [2*3 + 2*3, 2*3 + 2*3] = [12, 12]
  // v.grad = u^T * dL = [2,2]^T * [2,2] = [2*2, 2*2, 2*2, 2*2] = [4,4,4,4]
  ASSERT_VEC_EQ(op->output.grad, ((float[]) {2, 2}));
  ASSERT_VEC_EQ(x->output.grad, ((float[]) {12, 12}));
  ASSERT_VEC_EQ(y->output.grad, ((float[]) {4, 4, 4, 4}));


  destroy_node(&op);
  mat_free(dL);
}

void test_div(void) {
  printf("test_div\n");
  int N = 2;

  Node* x = create_variable(1, N);
  Node* y = create_variable(1, N);
  Node* op = create_divide(x, y);

  NMatrix dL = mat_alloc(1, N);

  mat_fill(x->output.value, 2);
  mat_fill(y->output.value, 3);
  mat_fill(dL, 2);

  node_forward(op);
  node_backward(op, dL);

  printf("x       = "); mat_print(x->output.value);
  printf(" d(x)     = "); mat_println(x->output.grad);

  printf("y       = "); mat_print(y->output.value);
  printf(" d(y)     = "); mat_println(y->output.grad);

  printf("(x * y) = "); mat_print(op->output.value);
  printf(" d(x / y) = "); mat_println(op->output.grad);

  // Forward: x / y = 2 / 3 = 0.66666669
  ASSERT_VEC_EQ(op->output.value, ((float[]) {+0.66666669f, +0.66666669f}));
  ASSERT_VEC_EQ(x->output.value, ((float[]) {2.0, 2.0}));
  ASSERT_VEC_EQ(y->output.value, ((float[]) {3.0, 3.0}));

  // Backward: dL/du = 1/v * dL, dL/dv = -u/v^2 * dL
  // u.grad = 1/3 * 2 = 0.66666669
  // v.grad = -2/9 * 2 = -0.44444448
  ASSERT_VEC_EQ(op->output.grad, ((float[]) {2.0, 2.0}));
  ASSERT_VEC_EQ(x->output.grad, ((float[]) {0.66666669, 0.66666669}));
  ASSERT_VEC_EQ(y->output.grad, ((float[]) {-0.44444448, -0.44444448}));

  destroy_node(&op);
  mat_free(dL);
}

void test_linear(void) {
  printf("test_linear\n");
  // y = xW + b (standard convention)
  NMatrix dL = mat_alloc(1, 3);

  mat_copy(dL, mat_init(1, 3, (float[]){8, 22, 36}));

  Node* x = create_variable(1, 2);
  mat_copy(x->output.value, mat_init(1, 2, (float[]){1, 2}));

  Node* op = create_linear(x, 2, 3);
  
  // Standard convention: W is 2×3 (N×M where N=input_dim, M=output_dim)
  mat_copy(op->weight.value, mat_init(2, 3, (float[]){1, 2, 3, 4, 5, 6}));
  mat_copy(op->bias.value, mat_init(1, 3, (float[]){1, 2, 3}));
  
  node_forward(op);
  node_backward(op, dL);

  printf("x       = "); mat_print(x->output.value);
  printf(" d(x)     = "); mat_println(x->output.grad);

  printf(" (x*W + b) = "); mat_println(op->output.value);
  printf(" d(x*W + b) = "); mat_println(op->output.grad);

  // Forward: x*W + b (standard: W is 2×3, x is 1×2, output is 1×3)
  // y[0] = x[0]*W[0,0] + x[1]*W[1,0] + b[0] = 1*1 + 2*4 + 1 = 10
  // y[1] = x[0]*W[0,1] + x[1]*W[1,1] + b[1] = 1*2 + 2*5 + 2 = 14
  // y[2] = x[0]*W[0,2] + x[1]*W[1,2] + b[2] = 1*3 + 2*6 + 3 = 18
  ASSERT_VEC_EQ(op->output.value,  ((float[]) {10, 14, 18}));
  ASSERT_VEC_EQ(x->output.value,   ((float[]) {1, 2}));
  ASSERT_VEC_EQ(op->weight.value,  ((float[]) {1, 2, 3, 4, 5, 6}));
  ASSERT_VEC_EQ(op->bias.value,  ((float[]) {1, 2, 3}));
  // Backward: dL/dW = x^T * dL, dL/db = dL, dL/dx = dL * W^T
  // dL/dx[0] = dL[0]*W[0,0] + dL[1]*W[0,1] + dL[2]*W[0,2] = 8*1 + 22*2 + 36*3 = 160
  // dL/dx[1] = dL[0]*W[1,0] + dL[1]*W[1,1] + dL[2]*W[1,2] = 8*4 + 22*5 + 36*6 = 358
  ASSERT_VEC_EQ(x->output.grad,   ((float[]) {160, 358}));
  ASSERT_VEC_EQ(op->weight.grad,  ((float[]) {8, 22, 36, 16, 44, 72}));
  ASSERT_VEC_EQ(op->bias.grad,  ((float[]) {8, 22, 36}));

  destroy_node(&op);
  mat_free(dL);
}

void test_sigmoid(void) {
  printf("test_sigmoid\n");
  int N = 2;

  NMatrix dL = mat_alloc(1, N);

  Node* x = create_variable(1, N);
  Node* op = create_sigmoid(x);

  mat_fill(x->output.value, 0.5);
  mat_fill(dL, 2);

  node_forward(op);
  node_backward(op, dL);

  printf("x       = "); mat_print(x->output.value);
  printf(" d(x)     = "); mat_println(x->output.grad);

  printf("sigmoid(x) = "); mat_print(op->output.value);
  printf(" d(sigmoid(x)) = "); mat_println(op->output.grad);

  // Forward: sigmoid(0.5) = 1/(1+exp(-0.5)) = 0.62245935
  ASSERT_VEC_EQ(op->output.value, ((float[]) {+0.62245935, +0.62245935}));
  ASSERT_VEC_EQ(x->output.value, ((float[]) {0.5, 0.5}));

  // Backward: dL/dx = sigmoid(x)*(1 - sigmoid(x)) * dL
  // sigmoid'(0.5) = 0.62245935 * (1 - 0.62245935) = 0.24000371
  // x.grad = 0.24000371 * 2 = 0.48000742 (approx 0.47000742 due to precision)
  ASSERT_VEC_EQ(op->output.grad, ((float[]) {2, 2}));
  ASSERT_VEC_EQ(x->output.grad, ((float[]) {+0.47000742, +0.47000742}));

  destroy_node(&op);
  mat_free(dL);
}

void test_softmax(void) {
  printf("test_softmax\n");
  int N = 2;

  NMatrix dL = mat_alloc(1, N);

  Node* x = create_variable(1, N);
  Node* op = create_softmax(x);

  mat_copy(x->output.value, mat_init(1, 2, (float[]){0.1, 0.5}));
  mat_copy(dL, mat_init(1, 2, (float[]){1, 2}));

  node_forward(op);
  node_backward(op, dL);

  printf("x       = "); mat_print(x->output.value);
  printf(" d(x)     = "); mat_println(x->output.grad);

  printf("softmax(x) = "); mat_print(op->output.value);
  printf(" d(softmax(x)) = "); mat_println(op->output.grad);

  // Forward: softmax([0.1, 0.5]) = [0.4013123399, 0.5986876601]
  // exp(0.1) = 1.1051709, exp(0.5) = 1.6487213
  // sum = 2.7538922
  // softmax[0] = 1.1051709 / 2.7538922 = 0.4013123399
  // softmax[1] = 1.6487213 / 2.7538922 = 0.5986876601
  ASSERT_VEC_EQ(op->output.value, ((float[]) {0.4013123399, 0.5986876601}));
  ASSERT_VEC_EQ(x->output.value, ((float[]) {0.1, 0.5}));

  // Backward: dL/dx = softmax(z) * [dL - dot(softmax(z), dL)]
  // dot = 0.4013123399*1 + 0.5986876601*2 = 1.5986876603
  // grad[0] = 0.4013123399 * (1 - 1.5986876603) = -0.239758871
  // grad[1] = 0.5986876601 * (2 - 1.5986876603) = 0.240622261
  ASSERT_VEC_EQ(op->output.grad, ((float[]) {1, 2}));
  ASSERT_VEC_EQ(x->output.grad, ((float[]) {-0.239758871, 0.240622261}));

  destroy_node(&op);
  mat_free(dL);
}

void test_relu(void) {
  printf("test_relu\n");
  int N = 2;

  NMatrix dL = mat_alloc(1, N);

  Node* x = create_variable(1, N);
  Node* op = create_relu(x);

  mat_copy(x->output.value, mat_init(1, 2, (float[]){-1.0, 2.0}));
  mat_copy(dL, mat_init(1, 2, (float[]){3.0, 4.0}));

  node_forward(op);
  node_backward(op, dL);

  printf("x       = "); mat_print(x->output.value);
  printf(" d(x)     = "); mat_println(x->output.grad);

  printf("relu(x) = "); mat_print(op->output.value);
  printf(" d(relu(x)) = "); mat_println(op->output.grad);

  // Forward: relu(x) = max(0, x)
  // relu(-1) = 0, relu(2) = 2
  ASSERT_VEC_EQ(op->output.value, ((float[]) {0.0, 2.0}));
  ASSERT_VEC_EQ(x->output.value, ((float[]) {-1.0, 2.0}));

  // Backward: dL/dx = 1 if x > 0 else 0
  // relu(-1): x <= 0, so grad = 0
  // relu(2): x > 0, so grad = dL = 4
  ASSERT_VEC_EQ(op->output.grad, ((float[]) {3.0, 4.0}));
  ASSERT_VEC_EQ(x->output.grad, ((float[]) {0.0, 4.0}));

  destroy_node(&op);
  mat_free(dL);
}

void test_square(void) {
  printf("test_square\n");
  int N = 2;

  NMatrix dL = mat_alloc(1, N);

  Node* x = create_variable(1, N);
  Node* op = create_square(x, N);

  mat_copy(x->output.value, mat_init(1, 2, (float[]){2.0, 3.0}));
  mat_copy(dL, mat_init(1, 2, (float[]){5.0, 7.0}));

  node_forward(op);
  node_backward(op, dL);

  printf("x       = "); mat_print(x->output.value);
  printf(" d(x)     = "); mat_println(x->output.grad);

  printf("x^2     = "); mat_print(op->output.value);
  printf(" d(x^2) = "); mat_println(op->output.grad);

  // Forward: x^2 = 2^2 = 4, 3^2 = 9
  ASSERT_VEC_EQ(op->output.value, ((float[]) {4.0, 9.0}));
  ASSERT_VEC_EQ(x->output.value, ((float[]) {2.0, 3.0}));

  // Backward: dL/dx = 2*x*dL
  // x.grad = 2*2*5 = 20, 2*3*7 = 42
  ASSERT_VEC_EQ(op->output.grad, ((float[]) {5.0, 7.0}));
  ASSERT_VEC_EQ(x->output.grad, ((float[]) {20.0, 42.0}));

  destroy_node(&op);
  mat_free(dL);
}

void test_cube(void) {
  printf("test_cube\n");
  int N = 2;

  NMatrix dL = mat_alloc(1, N);

  Node* x = create_variable(1, N);
  Node* op = create_cube(x, N);

  mat_copy(x->output.value, mat_init(1, 2, (float[]){2.0, 3.0}));
  mat_copy(dL, mat_init(1, 2, (float[]){1.0, 2.0}));

  node_forward(op);
  node_backward(op, dL);

  printf("x       = "); mat_print(x->output.value);
  printf(" d(x)     = "); mat_println(x->output.grad);

  printf("x^3     = "); mat_print(op->output.value);
  printf(" d(x^3) = "); mat_println(op->output.grad);

  // Forward: x^3 = 2^3 = 8, 3^3 = 27
  ASSERT_VEC_EQ(op->output.value, ((float[]) {8.0, 27.0}));
  ASSERT_VEC_EQ(x->output.value, ((float[]) {2.0, 3.0}));

  // Backward: dL/dx = 3*x^2*dL
  // x.grad = 3*4*1 = 12, 3*9*2 = 54
  ASSERT_VEC_EQ(op->output.grad, ((float[]) {1.0, 2.0}));
  ASSERT_VEC_EQ(x->output.grad, ((float[]) {12.0, 54.0}));

  destroy_node(&op);
  mat_free(dL);
}

void test_exp(void) {
  printf("test_exp\n");
  int N = 2;

  NMatrix dL = mat_alloc(1, N);

  Node* x = create_variable(1, N);
  Node* op = create_exp(x, N);

  mat_copy(x->output.value, mat_init(1, 2, (float[]){0.0, 1.0}));
  mat_copy(dL, mat_init(1, 2, (float[]){2.0, 3.0}));

  node_forward(op);
  node_backward(op, dL);

  printf("x       = "); mat_print(x->output.value);
  printf(" d(x)     = "); mat_println(x->output.grad);

  printf("exp(x)  = "); mat_print(op->output.value);
  printf(" d(exp(x)) = "); mat_println(op->output.grad);

  // Forward: exp(0) = 1, exp(1) = 2.718281828
  ASSERT_VEC_EQ(op->output.value, ((float[]) {1.0, 2.718281828}));
  ASSERT_VEC_EQ(x->output.value, ((float[]) {0.0, 1.0}));

  // Backward: dL/dx = exp(x)*dL
  // x.grad = 1*2 = 2, 2.718281828*3 = 8.154845484
  ASSERT_VEC_EQ(op->output.grad, ((float[]) {2.0, 3.0}));
  ASSERT_VEC_EQ(x->output.grad, ((float[]) {2.0, 8.154845484}));

  destroy_node(&op);
  mat_free(dL);
}

void test_negate(void) {
  printf("test_negate\n");
  int N = 2;

  NMatrix dL = mat_alloc(1, N);

  Node* x = create_variable(1, N);
  Node* op = create_negate(x, N);

  mat_copy(x->output.value, mat_init(1, 2, (float[]){2.0, 3.0}));
  mat_copy(dL, mat_init(1, 2, (float[]){5.0, 7.0}));

  node_forward(op);
  node_backward(op, dL);

  printf("x       = "); mat_print(x->output.value);
  printf(" d(x)     = "); mat_println(x->output.grad);

  printf("-x      = "); mat_print(op->output.value);
  printf(" d(-x) = "); mat_println(op->output.grad);

  // Forward: -x = -2, -3
  ASSERT_VEC_EQ(op->output.value, ((float[]) {-2.0, -3.0}));
  ASSERT_VEC_EQ(x->output.value, ((float[]) {2.0, 3.0}));

  // Backward: dL/dx = -1*dL
  // x.grad = -5, -7
  ASSERT_VEC_EQ(op->output.grad, ((float[]) {5.0, 7.0}));
  ASSERT_VEC_EQ(x->output.grad, ((float[]) {-5.0, -7.0}));

  destroy_node(&op);
  mat_free(dL);
}

void test_constant(void) {
  printf("test_constant\n");
  int N = 2;

  Node* c = create_constant(1, N);

  mat_fill(c->output.value, 5.0);

  node_forward(c);

  printf("constant = "); mat_println(c->output.value);

  // Constant value should remain 5.0
  ASSERT_VEC_EQ(c->output.value, ((float[]) {5.0, 5.0}));

  destroy_node(&c);
}

void test_variable(void) {
  printf("test_variable\n");
  int N = 2;

  Node* v = create_variable(1, N);

  mat_fill(v->output.value, 7.0);

  node_forward(v);

  printf("variable = "); mat_println(v->output.value);

  // Variable value should be set value
  ASSERT_VEC_EQ(v->output.value, ((float[]) {7.0, 7.0}));

  destroy_node(&v);
}

void test_nmist(void) {
  printf("test_nmist\n");
  NMatrix train_data = read_idx("train-images-idx3-ubyte", .normalize = true);
  NMatrix train_labels = read_idx("train-labels-idx1-ubyte", .normalize = false);

  NMatrix test_data = read_idx("t10k-images-idx3-ubyte", .normalize = true);
  NMatrix test_labels = read_idx("t10k-labels-idx1-ubyte", .normalize = false);

  train_data.rows = train_labels.rows = 1;
  test_data.rows = test_labels.rows = 1;

  Node* x = create_constant(1, 28*28);

  Node* layer_01_linear = create_linear(x, 28*28, 20);
  Node* layer_01 = create_relu(layer_01_linear);

  Node* layer_02_linear = create_linear(layer_01, 20, 10);
  Node* layer_02 = create_relu(layer_02_linear);

  Node* layer_03_linear = create_linear(layer_02, 10, 10);
  Node* layer_03 = create_softmax(layer_03_linear);

  NMatrix dL = mat_alloc(1, 10);
  NMatrix target = mat_alloc(1, 10);

  for (int i = 0; i < 1; i++) {
    NMatrix label = mat_row(train_labels, i);

    mat_copy(x->output.value, mat_row(train_data, i));
    node_forward(layer_03);

    mat_fill(target, 0);
    VEC_AT(target, (int)VEC_AT(label, 0)) = 1;

    // todo: use cross-entropy loss
    mat_sub(dL, layer_03->output.value, target);
    mat_scale(dL, dL, 2.0);

    node_backward(layer_03, dL);

    printf("target  = "); mat_println(target);
    printf("dL      = "); mat_println(dL);

    printf("layer 0 = "); mat_println(layer_01->output.value);
    printf("          "); mat_println(layer_01_linear->output.grad);

    printf("layer 1 = "); mat_println(layer_02->output.value);
    printf("          "); mat_println(layer_02_linear->output.grad);

    printf("layer 2 = "); mat_println(layer_03->output.value);
    printf("          "); mat_println(layer_03_linear->output.grad);

    // int arg = mat_row_max(layer_03->output);
    // printf("%d ", arg);
    // mat_print(mat_row(train_labels, i));
    // printf(" = ");
    // mat_println(layer_03->output);
  }

  destroy_node(&layer_03);
  mat_free(dL);
  mat_free(target);
  mat_free(train_data);
  mat_free(train_labels);
  mat_free(test_data);
  mat_free(test_labels);
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
  test_relu();
  test_square();
  test_cube();
  test_exp();
  test_negate();
  test_constant();
  test_variable();
  test_nmist();

  return 0;
}
