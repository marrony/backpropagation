#define BYTEBUFFER_IMPLEMENTATION
#define ALLOCATOR_IMPLEMENATION

#include "bytebuffer.h"
#include "allocator.h"
#include "node.h"
#include "transformer.h"
#include <math.h>
#include <stdbool.h>
#include <stdio.h>

#define PRECISION 5

Malloc_Allocator mallocator = MALLOC_CREATE();

void test_relu(int times) {
  Arena_Allocator arena = ARENA_CREATE(&mallocator.alloc, 1024*1024);

  Node* input =  create_variable(1, 3);
  Node* relu = create_relu(input, 0);

  Tape_Node_Array tape = ARRAY_CREATE(&mallocator.alloc);

  mat_copy(input->as_variable.value.value, mat_init(1, 3, (float[]){0.4967, -0.1383, 0.6477}));

  for (int i = 0; i < times; i++) {
    Tensor* out = node_forward(&arena, relu, &tape);

    ASSERT_VEC_EQ(out->value, ((float[]) {0.4967, 0.0000, 0.6477}));

    mat_copy(out->grad, mat_init(1, 3, (float[]){-0.2509, 0.9014, 0.4640}));
  }

  node_backward(&tape);

  ASSERT_VEC_EQ(input->as_variable.value.grad, ((float[]){-0.2509*times, 0.0000*times, 0.4640*times}));
}

void test_linear(int times) {
  Arena_Allocator arena = ARENA_CREATE(&mallocator.alloc, 1024*1024);

  Optimizer optimizer = {
    .tensors = ARRAY_CREATE(&mallocator.alloc),
    .history = ARRAY_CREATE(&mallocator.alloc),
    .second = ARRAY_CREATE(&mallocator.alloc),
    .learning_rate = 0.01f,
    .updates = 0,
  };

  Node* input =  create_variable(1, 3);
  Node* linear = create_linear(&optimizer, input, 3, 2);

  Tape_Node_Array tape = ARRAY_CREATE(&mallocator.alloc);

  mat_copy(input->as_variable.value.value, mat_init(1, 3, (float[]){0.4967, -0.1383, 0.6477}));
  mat_copy(linear->as_linear.weight.value, mat_init(3, 2, (float[]){
        1.5230, -0.2342,
        -0.2341, 1.5792,
        0.7674, -0.4695
  }));
  mat_copy(linear->as_linear.bias.value, mat_init(1, 2, (float[]){0.5426, -0.4634}));

  for (int i = 0; i < times; i++) {
    Tensor* out = node_forward(&arena, linear, &tape);

    ASSERT_VEC_EQ(out->value, ((float[]) {1.8285, -1.1022}));

    mat_copy(out->grad, mat_init(1, 2, (float[]){-0.4657, 0.2420}));
  }
  node_backward(&tape);

  ASSERT_VEC_EQ(input->as_variable.value.grad, ((float[]) {-0.7659377289*times, 0.4911870575*times, -0.4709969330*times}));
  ASSERT_VEC_EQ(mat_row(linear->as_linear.weight.grad, 0), ((float[]) {-0.2313132095*times, 0.1202013493*times}));
  ASSERT_VEC_EQ(mat_row(linear->as_linear.weight.grad, 1), ((float[]) {0.0644063616*times, -0.0334685545*times}));
  ASSERT_VEC_EQ(mat_row(linear->as_linear.weight.grad, 2), ((float[]) {-0.3016338539*times, 0.1567432785*times}));
  ASSERT_VEC_EQ(linear->as_linear.bias.grad, ((float[]) {-0.4657*times, 0.2420*times}));
}

void test_softmax(int times) {
  Arena_Allocator arena = ARENA_CREATE(&mallocator.alloc, 1024*1024);

  Node* input =  create_variable(1, 3);
  Node* softmax = create_softmax(input);

  Tape_Node_Array tape = ARRAY_CREATE(&mallocator.alloc);

  mat_copy(input->as_variable.value.value, mat_init(1, 3, (float[]){0.3, 0.5, 0.9}));

  for (int i = 0; i < times; i++) {
    Tensor* out = node_forward(&arena, softmax, &tape);

    ASSERT_VEC_EQ(out->value, ((float[]) {+0.24731, +0.30206, +0.45063}));

    mat_copy(out->grad, mat_init(1, 3, (float[]){0, 0, 1}));
  }
  node_backward(&tape);

  ASSERT_VEC_EQ(input->as_variable.value.grad, ((float[]){-0.11144412*times, -0.13611814*times, 0.24756226*times}));
}

void test_softmax_cross_entropy(int times) {
  Arena_Allocator arena = ARENA_CREATE(&mallocator.alloc, 1024*1024);

  Node* input =  create_variable(1, 3);
  NMatrix target =  mat_alloc(1, 3);
  Node* softmax = create_softmax_cross_entropy(input, target);

  Tape_Node_Array tape = ARRAY_CREATE(&mallocator.alloc);

  mat_copy(input->as_variable.value.value, mat_init(1, 3, (float[]){0.3, 0.5, 0.9}));
  mat_copy(target, mat_init(1, 3, (float[]){0, 0, 1}));

  for (int i = 0; i < times; i++) {
    Tensor* out = node_forward(&arena, softmax, &tape);

    ASSERT_VEC_EQ(out->value, ((float[]) {0.7971}));

    mat_copy(out->grad, mat_init(1, 1, (float[]){1}));
  }

  node_backward(&tape);

  ASSERT_VEC_EQ(input->as_variable.value.grad, ((float[]){0.247309184*times, 0.302064109*times, -0.549373198*times}));
}

void test_embeddings(int times) {
  Arena_Allocator arena = ARENA_CREATE(&mallocator.alloc, 1024*1024);
  Tape_Node_Array tape = ARRAY_CREATE(&mallocator.alloc);

  Optimizer optimizer = {
    .tensors = ARRAY_CREATE(&mallocator.alloc),
    .history = ARRAY_CREATE(&mallocator.alloc),
    .second = ARRAY_CREATE(&mallocator.alloc),
    .learning_rate = 0.01f,
    .updates = 0,
  };

  // [2x3]
  Node* embeddings = create_embeddings(&optimizer, 3, 3, 4);
  embeddings->as_embedding.context[0] = 0;
  embeddings->as_embedding.context[1] = 2;
  embeddings->as_embedding.context[2] = 1;
  embeddings->as_embedding.context[3] = 2;
  mat_copy(embeddings->as_embedding.embeddings.value, mat_init(3, 3, (float[]){
        0.4967, -0.1383, 0.6477,
        -0.2509, 0.9014, 0.4640,
        0.2472, 0.3021, -0.5494,
  }));

  for (int i = 0; i < times; i++) {
    Tensor* out = node_forward(&arena, embeddings, &tape);

    ASSERT_VEC_EQ(out->value, ((float[]) {
          0.4967, -0.1383, 0.6477, // 0
          0.2472, 0.3021, -0.5494, // 2
          -0.2509, 0.9014, 0.4640, // 1
          0.2472, 0.3021, -0.5494, // 2
    }));

    mat_copy(out->grad, mat_init(1, 12, (float[]){
          1, 2, 3,    // 0
          4, 5, 6,    // 2
          7, 8, 9,    // 1
          10, 11, 12, // 2
    }));
  }
  node_backward(&tape);

  ASSERT_VEC_EQ(mat_row(embeddings->as_embedding.embeddings.grad, 0), ((float[]) {0, 0, 0}));
  ASSERT_VEC_EQ(mat_row(embeddings->as_embedding.embeddings.grad, 1), ((float[]) {7*times, 8*times, 9*times}));
  ASSERT_VEC_EQ(mat_row(embeddings->as_embedding.embeddings.grad, 2), ((float[]) {(4+10)*times, (5+11)*times, (6+12)*times}));
}

void test_composed(int times) {
  Arena_Allocator arena = ARENA_CREATE(&mallocator.alloc, 1024*1024);

  Optimizer optimizer = {
    .tensors = ARRAY_CREATE(&mallocator.alloc),
    .history = ARRAY_CREATE(&mallocator.alloc),
    .second = ARRAY_CREATE(&mallocator.alloc),
    .learning_rate = 0.01f,
    .updates = 0,
  };

  Node* input =  create_variable(1, 3);
  NMatrix target = mat_alloc(1, 2);
  Node* linear = create_linear(&optimizer, input, 3, 2);
  Node* relu = create_relu(linear, 0);
  Node* softmax = create_softmax_cross_entropy(relu, target);

  Tape_Node_Array tape = ARRAY_CREATE(&mallocator.alloc);

  mat_copy(input->as_variable.value.value, mat_init(1, 3, (float[]){0.4967, -0.1383, 0.6477}));
  mat_copy(linear->as_linear.weight.value, mat_init(3, 2, (float[]){
        1.5230, -0.2342,
        -0.2341, 1.5792,
        0.7674, -0.4695
  }));
  mat_copy(linear->as_linear.bias.value, mat_init(1, 2, (float[]){0.5426, -0.4634}));
  mat_copy(target, mat_init(1, 2, (float[]){0, 1}));

  for (int i = 0; i < times; i++) {
    Tensor* out = node_forward(&arena, softmax, &tape);

    ASSERT_VEC_EQ(out->value, ((float[]) {1.9775}));

    mat_copy(out->grad, mat_init(1, 1, (float[]){1}));
  }

  node_backward(&tape);

  ASSERT_VEC_EQ(input->as_variable.value.grad, ((float[]) {1.3121888733*times, -0.2016964149*times, 0.6611785126*times}));
  ASSERT_VEC_EQ(mat_row(linear->as_linear.weight.grad, 0), ((float[]) {+0.4279479599*times, 0.0000}));
  ASSERT_VEC_EQ(mat_row(linear->as_linear.weight.grad, 1), ((float[]) {-0.1191568375*times, 0.0000}));
  ASSERT_VEC_EQ(mat_row(linear->as_linear.weight.grad, 2), ((float[]) {+0.5580473709*times, 0.0000}));
  ASSERT_VEC_EQ(linear->as_linear.bias.grad, ((float[]) {+0.8615821838*times, 0.0000}));
}

void test_multiply(int times) {
  Arena_Allocator arena = ARENA_CREATE(&mallocator.alloc, 1024*1024);

  Node* u =  create_variable(2, 2);
  Node* v =  create_variable(2, 1);
  Node* multiply = create_multiply(u, v);

  Tape_Node_Array tape = ARRAY_CREATE(&mallocator.alloc);

  mat_copy(u->as_variable.value.value, mat_init(2, 2, (float[]){
        0.4967, -0.1383,
        0.6477, +0.9014,
  }));
  mat_copy(v->as_variable.value.value, mat_init(2, 1, (float[]){
        2,
        2,
  }));

  for (int i = 0; i < times; i++) {
    Tensor* out = node_forward(&arena, multiply, &tape);

    ASSERT_VEC_EQ(mat_row(out->value, 0), ((float[]) {+0.71679997}));
    ASSERT_VEC_EQ(mat_row(out->value, 1), ((float[]) {+3.09820008}));

    mat_copy(out->grad, mat_init(2, 1, (float[]){
          -0.2509,
          0.4640,
    }));
  }

  node_backward(&tape);

  ASSERT_VEC_EQ(mat_row(u->as_variable.value.grad, 0), ((float[]){-0.5018*times, -0.5018*times}));
  ASSERT_VEC_EQ(mat_row(u->as_variable.value.grad, 1), ((float[]){+0.9280*times, +0.9280*times}));

  ASSERT_VEC_EQ(mat_row(v->as_variable.value.grad, 0), ((float[]){0.1759108925*times}));
  ASSERT_VEC_EQ(mat_row(v->as_variable.value.grad, 1), ((float[]){0.4529494476*times}));
}

void test_multiply_add(int times) {
  Arena_Allocator arena = ARENA_CREATE(&mallocator.alloc, 1024*1024);

  Node* u =  create_variable(2, 2);
  Node* v =  create_variable(2, 1);
  Node* multiply = create_multiply(u, v);
  Node* add = create_add(multiply, v);

  Tape_Node_Array tape = ARRAY_CREATE(&mallocator.alloc);

  mat_copy(u->as_variable.value.value, mat_init(2, 2, (float[]){
        0.4967, -0.1383,
        0.6477, +0.9014,
  }));
  mat_copy(v->as_variable.value.value, mat_init(2, 1, (float[]){
        2,
        2,
  }));

  for (int i = 0; i < times; i++) {
    Tensor* out = node_forward(&arena, add, &tape);

    ASSERT_VEC_EQ(mat_row(out->value, 0), ((float[]) {+0.71679997 + 2}));
    ASSERT_VEC_EQ(mat_row(out->value, 1), ((float[]) {+3.09820008 + 2}));

    mat_copy(out->grad, mat_init(2, 1, (float[]){
          -0.2509,
          0.4640,
    }));
  }

  node_backward(&tape);

  ASSERT_VEC_EQ(mat_row(u->as_variable.value.grad, 0), ((float[]){-0.5018*times, -0.5018*times}));
  ASSERT_VEC_EQ(mat_row(u->as_variable.value.grad, 1), ((float[]){+0.9280*times, +0.9280*times}));

  ASSERT_VEC_EQ(mat_row(v->as_variable.value.grad, 0), ((float[]){(0.1759108925 - 0.2509)*times}));
  ASSERT_VEC_EQ(mat_row(v->as_variable.value.grad, 1), ((float[]){(0.4529494476 + 0.4640)*times}));
}

void test_layer_norm_forward(void) {
  Arena_Allocator arena = ARENA_CREATE(&mallocator.alloc, 1024*1024);

  int N = 4;
  int D = 4;
  // int H = 2;

  NMatrix out      = mat_alloc2(&arena.alloc, N, D);
  NMatrix in       = mat_alloc2(&arena.alloc, N, D);
  NMatrix gamma    = mat_alloc2(&arena.alloc, 1, D);
  NMatrix beta     = mat_alloc2(&arena.alloc, 1, D);
  NMatrix mean     = mat_alloc2(&arena.alloc, 1, N);
  NMatrix variance = mat_alloc2(&arena.alloc, 1, N);
  NMatrix x_hat    = mat_alloc2(&arena.alloc, N, D);

  mat_copy(mat_row(in, 0), mat_init(1, D, (float[]){-0.36380788683891296, -0.4134192168712616, -0.9900033473968506, 0.2764623463153839}));
  mat_copy(mat_row(in, 1), mat_init(1, D, (float[]){0.16764935851097107, 0.6411328315734863, -1.7130550146102905, 2.114539384841919}));
  mat_copy(mat_row(in, 2), mat_init(1, D, (float[]){-0.49599435925483704, -0.07817786931991577, -0.8475110530853271, 1.954899787902832}));
  mat_copy(mat_row(in, 3), mat_init(1, D, (float[]){0.7176350355148315, 1.2813740968704224, 0.34164267778396606, -0.45972520112991333}));

  mat_copy(gamma, mat_init(1, D, (float[]){1.0020004510879517, 1.0019868612289429, 0.9979996681213379, 0.9979987740516663}));
  mat_copy(beta,  mat_init(1, D, (float[]){-0.00200053327716887, 0.0019987025298178196, 0.0020008590072393417, -0.002001350512728095}));

  Layer_Norm ln_in = {
    .gamma = tensor(gamma, NULL_MATRIX),
    .beta  = tensor(beta, NULL_MATRIX),
  };

  Layer_Norm_Output ln_out = {
    .var  = variance,
    .mean = mean,
    .xhat = x_hat,
    .out  = tensor(out, NULL_MATRIX),
  };

  layer_norm_forward(
      .ln_out = &ln_out,
      .ln_in  = &ln_in,
      .in     = tensor(in, NULL_MATRIX),
  );

  ASSERT_VEC_EQ(mat_row(out, 0), ((float[]){+0.01785211, -0.08900940, -1.37194133, +1.44281244}));
  ASSERT_VEC_EQ(mat_row(out, 1), ((float[]){-0.10086682, +0.25009388, -1.46913266, +1.32049477}));
  ASSERT_VEC_EQ(mat_row(out, 2), ((float[]){-0.58241475, -0.19305260, -0.89901215, +1.67138207}));
  ASSERT_VEC_EQ(mat_row(out, 3), ((float[]){+0.38989305, +1.28685272, -0.20087424, -1.46919501}));
  ASSERT_VEC_EQ(mean,     ((float[]){-0.3726920187473297, 0.30256664752960205, 0.13330411911010742, 0.47023165225982666}));
  ASSERT_VEC_EQ(variance, ((float[]){0.20105306804180145, 1.8697013854980469, 1.1802376508712769, 0.4001288414001465}));
  ASSERT_VEC_EQ(mat_row(x_hat, 0), ((float[]){0.01981295272707939, -0.09082769602537155, -1.3766961097717285, 1.4477108716964722}));
  ASSERT_VEC_EQ(mat_row(x_hat, 1), ((float[]){-0.0986689031124115, 0.24760322272777557, -1.4740822315216064, 1.3251479864120483}));
  ASSERT_VEC_EQ(mat_row(x_hat, 2), ((float[]){-0.5792554616928101, -0.19466453790664673, -0.9028188586235046, 1.6767388582229614}));
  ASSERT_VEC_EQ(mat_row(x_hat, 3), ((float[]){0.39111122488975525, 1.2823063135147095, -0.20328174531459808, -1.4701358079910278}));
}

void test_layer_norm_forward_mean_variance(void) {
  Arena_Allocator arena = ARENA_CREATE(&mallocator.alloc, 1024*1024);

  NMatrix out      = mat_alloc2(&arena.alloc, 3, 2);
  NMatrix in       = mat_alloc2(&arena.alloc, 3, 2);
  NMatrix gamma    = mat_alloc2(&arena.alloc, 1, 2);
  NMatrix beta     = mat_alloc2(&arena.alloc, 1, 2);
  NMatrix mean     = mat_alloc2(&arena.alloc, 1, 3);
  NMatrix variance = mat_alloc2(&arena.alloc, 1, 3);
  NMatrix x_hat    = mat_alloc2(&arena.alloc, 3, 2);

  mat_copy(mat_row(in, 0), mat_init(1, 2, (float[]){1, 2}));
  mat_copy(mat_row(in, 1), mat_init(1, 2, (float[]){2, 4}));
  mat_copy(mat_row(in, 2), mat_init(1, 2, (float[]){3, 6}));

  mat_fill(gamma, 1);
  mat_fill(beta, 0);

  Layer_Norm ln_in = {
    .gamma = tensor(gamma, NULL_MATRIX),
    .beta  = tensor(beta, NULL_MATRIX),
  };

  Layer_Norm_Output ln_out = {
    .var  = variance,
    .mean = mean,
    .xhat = x_hat,
    .out  = tensor(out, NULL_MATRIX),
  };

  layer_norm_forward(
      .ln_out = &ln_out,
      .ln_in  = &ln_in,
      .in     = tensor(in, NULL_MATRIX),
  );

  // run again with the normlized value, it should produce:
  // mean = 0 and variance = 1
  mat_copy(in, out);

  layer_norm_forward(
      .ln_out = &ln_out,
      .ln_in  = &ln_in,
      .in     = tensor(in, NULL_MATRIX),
  );

  ASSERT_VEC_EQ(mean,     ((float[]){0.00000, 0.00000, 0.00000}));
  ASSERT_VEC_EQ(variance, ((float[]){1.00000, 1.00000, 1.00000}));
}

void test_layer_norm_forward_gamma_beta(void) {
  Arena_Allocator arena = ARENA_CREATE(&mallocator.alloc, 1024*1024);

  NMatrix out      = mat_alloc2(&arena.alloc, 3, 2);
  NMatrix in       = mat_alloc2(&arena.alloc, 3, 2);
  NMatrix gamma    = mat_alloc2(&arena.alloc, 1, 2);
  NMatrix beta     = mat_alloc2(&arena.alloc, 1, 2);
  NMatrix mean     = mat_alloc2(&arena.alloc, 1, 3);
  NMatrix variance = mat_alloc2(&arena.alloc, 1, 3);
  NMatrix x_hat    = mat_alloc2(&arena.alloc, 3, 2);

  mat_copy(mat_row(in, 0), mat_init(1, 2, (float[]){1, 2}));
  mat_copy(mat_row(in, 1), mat_init(1, 2, (float[]){2, 4}));
  mat_copy(mat_row(in, 2), mat_init(1, 2, (float[]){3, 6}));

  mat_fill(gamma, 2);
  mat_fill(beta, 1);

  Layer_Norm ln_in = {
    .gamma = tensor(gamma, NULL_MATRIX),
    .beta  = tensor(beta, NULL_MATRIX),
  };

  Layer_Norm_Output ln_out = {
    .var  = variance,
    .mean = mean,
    .xhat = x_hat,
    .out  = tensor(out, NULL_MATRIX),
  };

  layer_norm_forward(
      .ln_out = &ln_out,
      .ln_in  = &ln_in,
      .in     = tensor(in, NULL_MATRIX),
  );

  // output should be scaled by gamma and shifted by beta
  ASSERT_VEC_EQ(mat_row(out, 0), ((float[]){-0.99998*2 + 1, +0.99998*2 + 1}));
  ASSERT_VEC_EQ(mat_row(out, 1), ((float[]){-0.99999*2 + 1, +0.99999*2 + 1}));
  ASSERT_VEC_EQ(mat_row(out, 2), ((float[]){-1.00000*2 + 1, +1.00000*2 + 1}));

  ASSERT_VEC_EQ(mat_row(x_hat, 0), ((float[]){-0.99998, +0.99998}));
  ASSERT_VEC_EQ(mat_row(x_hat, 1), ((float[]){-0.99999, +0.99999}));
  ASSERT_VEC_EQ(mat_row(x_hat, 2), ((float[]){-1.00000, +1.00000}));

  ASSERT_VEC_EQ(mean,     ((float[]){+1.50000, +3.00000, +4.50000}));
  ASSERT_VEC_EQ(variance, ((float[]){+0.25000, +1.00000, +2.25000}));
}

void test_layer_norm_backward(void) {
  Arena_Allocator arena = ARENA_CREATE(&mallocator.alloc, 1024*1024);

  int N = 4;
  int D = 4;
  // int H = 2;

  NMatrix dx     = mat_alloc2(&arena.alloc, N, D);
  NMatrix dgamma = mat_alloc2(&arena.alloc, 1, D);
  NMatrix dbeta  = mat_alloc2(&arena.alloc, 1, D);
  NMatrix dL     = mat_alloc2(&arena.alloc, N, D);
  NMatrix gamma  = mat_alloc2(&arena.alloc, 1, D);
  NMatrix var    = mat_alloc2(&arena.alloc, 1, N);
  NMatrix xhat   = mat_alloc2(&arena.alloc, N, D);

  mat_copy(mat_row(dL, 0), mat_init(1, D, (float[]){0.0002737737959250808, 0.00021247370750643313, 0.00026379668270237744, -0.0010246539022773504}));
  mat_copy(mat_row(dL, 1), mat_init(1, D, (float[]){0.0004571757453959435, 0.00035481053055264056, 0.00044051490840502083, -0.0017110728658735752}));
  mat_copy(mat_row(dL, 2), mat_init(1, D, (float[]){-0.00029440654907375574, -0.00022848662047181278, -0.00028367750928737223, 0.0011018761433660984}));
  mat_copy(mat_row(dL, 3), mat_init(1, D, (float[]){-0.000466999743366614, -0.00036243483191356063, -0.00044998087105341256, 0.0017478411318734288}));

  mat_copy(gamma, mat_init(1, D, (float[]){1.0010000467300415, 1.0010000467300415, 1.0010000467300415, 1.0010000467300415}));
  mat_copy(var, mat_init(1, D, (float[]){1.82850182056427, 4.095080852508545, 4.043269634246826, 0.24859601259231567}));
  mat_copy(mat_row(xhat, 0), mat_init(1, D, (float[]){-0.5863534808158875, -0.9618210196495056, -0.10131143778562546, 1.6494859457015991}));
  mat_copy(mat_row(xhat, 1), mat_init(1, D, (float[]){-0.5718590021133423, -0.3308160603046417, -0.8048704862594604, 1.707545518875122}));
  mat_copy(mat_row(xhat, 2), mat_init(1, D, (float[]){-0.7022160291671753, -0.6498586535453796, -0.3657192289829254, 1.7177939414978027}));
  mat_copy(mat_row(xhat, 3), mat_init(1, D, (float[]){-1.3712605237960815, -0.2520727217197418, 0.20407409965991974, 1.4192591905593872}));

  mat_zero(dgamma);
  mat_zero(dbeta);
  mat_zero(dx);

  Layer_Norm ln_in = {
    .gamma = tensor(gamma, dgamma),
    .beta  = tensor(NULL_MATRIX, dbeta),
  };

  Layer_Norm_Output ln_out = {
    .out  = tensor(NULL_MATRIX, dL),
    .xhat = xhat,
    .mean = NULL_MATRIX,
    .var  = var,
  };

  layer_norm_backward(
      .in     = tensor(NULL_MATRIX, dx),
      .ln_in  = &ln_in,
      .ln_out = &ln_out,
  );

  ASSERT_VEC_EQ(mat_row(dx, 0), ((float[]){0.000027584579584072344, -0.00016244733706116676, 0.00020706774375867099, -0.00007220498082460836}));
  ASSERT_VEC_EQ(mat_row(dx, 1), ((float[]){0.00002437009970890358, 0.00008268712554126978, -0.0000891934905666858, -0.000017863745597423986}));
  ASSERT_VEC_EQ(mat_row(dx, 2), ((float[]){0.00002221674185420852, 0.00003970838224631734, -0.00007092984742484987, 0.000009004715138871688}));
  ASSERT_VEC_EQ(mat_row(dx, 3), ((float[]){0.0009750236058607697, -0.0005679313326254487, -0.0014581095892935991, 0.001051017316058278}));

  ASSERT_VEC_EQ(dgamma, ((float[]){+0.0004251470, -0.0000818948, -0.0003693662, -0.0002384512}));
  ASSERT_VEC_EQ(dbeta,  ((float[]){-0.0000304567, -0.0000236372, -0.0000293468, +0.0001139905}));
}

void test_block_forward(void) {
  srand(0);

  Arena_Allocator arena = ARENA_CREATE(&mallocator.alloc, 1024*1024);

  Block block = {0};
  Block_Output block_out = {0};

  size_t N = 4;
  size_t D = 4;
  size_t H = 2;
  size_t F = 2;

  NMatrix in     = mat_alloc2(&arena.alloc, N, D);
  NMatrix scores = mat_alloc2(&arena.alloc, N, N);

  init_block(&arena.alloc, &block, D, F);
  init_block_output(&arena, &block_out, N, D, H, F);

  mat_copy(mat_reshape(in, 1, N*D), mat_init(1, N*D, (float[]){
    -0.36380788683891296, -0.4134192168712616, -0.9900033473968506, 0.2764623463153839,
    0.16764935851097107, 0.6411328315734863, -1.7130550146102905, 2.114539384841919,
    -0.49599435925483704, -0.07817786931991577, -0.8475110530853271, 1.954899787902832,
    0.7176350355148315, 1.2813740968704224, 0.34164267778396606, -0.45972520112991333
  }));

  mat_copy(block.ln1.gamma.value, mat_init(1, D, (float[]){1.0020004510879517, 1.0019868612289429, 0.9979996681213379, 0.9979987740516663}));
  mat_copy(block.ln1.beta.value, mat_init(1, D, (float[]){-0.00200053327716887, 0.0019987025298178196, 0.0020008590072393417, -0.002001350512728095}));
  mat_copy(block.ln2.gamma.value, mat_init(1, D, (float[]){1.0020002126693726, 1.0019965171813965, 1.0020012855529785, 1.0019992589950562}));
  mat_copy(block.ln2.beta.value, mat_init(1, D, (float[]){-0.001999453641474247, -0.00199936144053936, -0.001999504631385207, 0.0019992943853139877}));

  mat_copy(mat_reshape(block.attn.Q.weight.value, 1, N*D), mat_init(1, N*D, (float[]){
    -0.28038862347602844, 0.16468968987464905, -0.3046887516975403, 0.3993356227874756,
    -0.3527085781097412, -0.08901438117027283, 0.5962929725646973, 0.3398438096046448,
    0.8536521196365356, 0.6778435707092285, -0.1199512779712677, 0.07637869566679001,
    -0.35546043515205383, -0.689300000667572, 0.3387598395347595, -0.325349897146225
  }));
  mat_copy(block.attn.Q.bias.value, mat_init(1, D, (float[]){
    0.002000355627387762, -0.0020008657593280077, 0.0019862623885273933, 0.0019769964274019003
  }));

  mat_copy(mat_reshape(block.attn.K.weight.value, 1, N*D), mat_init(1, N*D, (float[]){
    0.4932657778263092, 0.7030916810035706, -0.7018579840660095, -0.04064561426639557,
    0.5556344389915466, -0.6388959288597107, 0.8208088874816895, -0.5550593137741089,
    0.36074310541152954, 0.3478701710700989, -0.47750362753868103, -0.14894138276576996,
    -0.5809427499771118, 0.2658921182155609, -0.2861584424972534, -0.5095506310462952
  }));
  mat_copy(block.attn.K.bias.value, mat_init(1, D, (float[]){
        0.0003560356271918863, 0.00010378981096437201, 0.0010042828507721424, 0.00023625933681614697
  }));

  mat_copy(mat_reshape(block.attn.V.weight.value, 1, N*D), mat_init(1, N*D, (float[]){
    0.22526857256889343, 0.6680930852890015, -0.30272331833839417, 0.5084951519966125,
    -0.18741565942764282, -0.6364452242851257, 0.6319502592086792, -0.22766847908496857,
    -0.8475242257118225, 0.40922805666923523, 0.23228542506694794, 0.6531195044517517,
    0.7049607038497925, -0.5020469427108765, 0.18201522529125214, -0.020633606240153313
  }));
  mat_copy(block.attn.V.bias.value, mat_init(1, D, (float[]){
    -0.0020010783337056637, 0.002000225242227316, 0.001998402876779437, -0.0020013086032122374
  }));

  mat_copy(mat_reshape(block.attn.O.weight.value, 1, N*D), mat_init(1, N*D, (float[]){
    -0.549282968044281, -0.7817981839179993, 0.3540251553058624, 0.3589060306549072,
    -0.359149694442749, -0.39087215065956116, -0.28101611137390137, -0.8136096596717834,
    -0.4854240119457245, 0.8568031191825867, 0.12333469092845917, 0.8442171216011047,
    -0.5802040100097656, -0.5721324682235718, 0.10759422183036804, -0.3573928773403168
  }));
  mat_copy(block.attn.O.bias.value, mat_init(1, D, (float[]){
    -0.0019959742203354836, 0.002000064356252551, -0.0020011686719954014, -0.0020013207104057074
  }));

  mat_copy(mat_reshape(block.ff1.weight.value, 1, D*F), mat_init(1, D*F, (float[]){
    -0.21268746256828308, -0.13047446310520172,
    -0.16528503596782684, 0.3716326355934143,
    -0.20497411489486694, -0.3281082212924957,
    0.7932820320129395, -0.34734752774238586
  }));
  mat_copy(block.ff1.bias.value, mat_init(1, F, (float[]){0.001999274827539921, -0.0007235786179080606}));

  mat_copy(mat_reshape(block.ff2.weight.value, 1, F*D), mat_init(1, D*F, (float[]){
    -0.011471477337181568, 0.22373977303504944, 0.09072287380695343, 0.176594540476799,
    -0.5310277342796326, -0.3497594892978668, -0.4011905789375305, -0.3585017919540405
  }));
  mat_copy(block.ff2.bias.value, mat_init(1, D, (float[]){
    -0.001996306236833334, 0.0020001570228487253, -0.0020012909080833197, -0.002001351211220026
  }));

  mat_zero(scores);

  block_forward(
      .block_out = &block_out,
      .block_in  = &block,
      .in        = tensor(in, NULL_MATRIX),
      .scores    = scores,
  );

  ////////////////
  ASSERT_VEC_EQ(mat_row(block_out.ln1.out.value, 0), ((float[]){0.01785205490887165, -0.08900945633649826, -1.3719414472579956, 1.4428123235702515}));
  ASSERT_VEC_EQ(mat_row(block_out.ln1.out.value, 1), ((float[]){-0.10086681693792343, 0.25009387731552124, -1.4691327810287476, 1.3204946517944336}));
  ASSERT_VEC_EQ(mat_row(block_out.ln1.out.value, 2), ((float[]){-0.5824147462844849, -0.19305260479450226, -0.89901202917099, 1.671381950378418}));
  ASSERT_VEC_EQ(mat_row(block_out.ln1.out.value, 3), ((float[]){0.3898930847644806, 1.2868527173995972, -0.20087425410747528, -1.469195008277893}));

  ////////////////
  ASSERT_VEC_EQ(block_out.ln1.mean, ((float[]){-0.3726920187473297, 0.30256664752960205, 0.13330411911010742, 0.47023165225982666}));
  ASSERT_VEC_EQ(block_out.ln1.var,  ((float[]){0.20105306804180145, 1.8697013854980469, 1.1802376508712769, 0.4001288414001465}));
  
  ////////////////
  ASSERT_VEC_EQ(mat_row(block_out.ln1.xhat, 0), ((float[]){0.01981295272707939, -0.09082769602537155, -1.3766961097717285, 1.4477108716964722}));
  ASSERT_VEC_EQ(mat_row(block_out.ln1.xhat, 1), ((float[]){-0.0986689031124115, 0.24760322272777557, -1.4740822315216064, 1.3251479864120483}));
  ASSERT_VEC_EQ(mat_row(block_out.ln1.xhat, 2), ((float[]){-0.5792554616928101, -0.19466453790664673, -0.9028188586235046, 1.6767388582229614}));
  ASSERT_VEC_EQ(mat_row(block_out.ln1.xhat, 3), ((float[]){0.39111122488975525, 1.2823063135147095, -0.20328174531459808, -1.4701358079910278}));
  
  ////////////////
  ASSERT_VEC_EQ(mat_row(block_out.attn.Q.value, 0), ((float[]){-1.6556341648101807, -1.9156298637390137, 0.5968042016029358, -0.5953493118286133}));
  ASSERT_VEC_EQ(mat_row(block_out.attn.Q.value, 1), ((float[]){-1.7814399003982544, -1.946933627128601, 0.8054033517837524, -0.4951431155204773}));
  ASSERT_VEC_EQ(mat_row(block_out.attn.Q.value, 2), ((float[]){-1.1281596422195435, -1.8422071933746338, 0.7383602857589722, -0.908659040927887}));
  ASSERT_VEC_EQ(mat_row(block_out.attn.Q.value, 3), ((float[]){-0.21044126152992249, 0.824216902256012, 0.17692232131958008, 1.0576640367507935}));

  ASSERT_VEC_EQ(mat_row(block_out.attn.K.value, 0), ((float[]){-1.373404622077942, -0.02410189062356949, 0.15764902532100677, -0.4819308817386627}));
  ASSERT_VEC_EQ(mat_row(block_out.attn.K.value, 1), ((float[]){-1.2075486183166504, -0.39055711030960083, 0.6007232666015625, -0.5885250568389893}));
  ASSERT_VEC_EQ(mat_row(block_out.attn.K.value, 2), ((float[]){-1.6894854307174683, -0.15437887609004974, 0.20231886208057404, -0.5866891145706177}));
  ASSERT_VEC_EQ(mat_row(block_out.attn.K.value, 3), ((float[]){1.6887508630752563, -1.0084561109542847, 1.2999557256698608, 0.04865695536136627}));

  ASSERT_VEC_EQ(mat_row(block_out.attn.V.value, 0), ((float[]){2.1985819339752197, -1.2152197360992432, -0.11572356522083282, -0.8984711170196533}));
  ASSERT_VEC_EQ(mat_row(block_out.attn.V.value, 1), ((float[]){2.1044278144836426, -1.4887199401855469, 0.0896720290184021, -1.0969959497451782}));
  ASSERT_VEC_EQ(mat_row(block_out.attn.V.value, 2), ((float[]){1.8431733846664429, -1.4712527990341187, 0.15169885754585266, -0.8758533000946045}));
  ASSERT_VEC_EQ(mat_row(block_out.attn.V.value, 3), ((float[]){-1.020825743675232, 0.09887530654668808, 0.38311952352523804, -0.19759847223758698}));

  ////////////////
  ASSERT_VEC_EQ(mat_row(scores, 0), ((float[]){+0.26940968, -INFINITY,   -INFINITY,   -INFINITY}));
  ASSERT_VEC_EQ(mat_row(scores, 1), ((float[]){+0.25851526, +0.54816943, -INFINITY,   -INFINITY}));
  ASSERT_VEC_EQ(mat_row(scores, 2), ((float[]){+0.39195826, +0.69177591, +0.48258954, -INFINITY}));
  ASSERT_VEC_EQ(mat_row(scores, 3), ((float[]){-0.34070485, -0.36499476, -0.41346323, +0.19901795}));

  ////////////////
  ASSERT_VEC_EQ(mat_row(block_out.attn.weights.value, 0), ((float[]){
                      1,                   0, 0,                                    0,
    0.42664042115211487,0.573359608650207500, 0,                                    0,
    0.25401076674461365, 0.35867854952812195, 0.3873106837272644,                   0,
    0.32084584236145020, 0.25282740592956543, 0.3117084503173828, 0.11461830884218216,
  }));
  ASSERT_VEC_EQ(mat_row(block_out.attn.weights.value, 1), ((float[]){
                      1,                   0, 0,                                     0,
    0.42808854579925537, 0.57191145420074460, 0,                                     0,
    0.29031977057456970, 0.39181923866271970, 0.31786099076271057,                   0,
    0.21638654172420502, 0.21119387447834015, 0.20120172202587128, 0.37121787667274475
  }));

  ////////////////
  ASSERT_VEC_EQ(mat_row(block_out.attn.out, 0), ((float[]){-0.195722296833992, -0.8269596099853516, 1.0069055557250977, 1.999208927154541}));
  ASSERT_VEC_EQ(mat_row(block_out.attn.out, 1), ((float[]){-0.10089627653360367, -0.5578543543815613, 1.0341328382492065, 2.2471654415130615}));
  ASSERT_VEC_EQ(mat_row(block_out.attn.out, 2), ((float[]){-0.07008278369903564, -0.43366020917892456, 1.0144646167755127, 2.2631113529205322}));
  ASSERT_VEC_EQ(mat_row(block_out.attn.out, 3), ((float[]){-0.18604084849357605, -0.31944382190704346, 0.886951208114624, 1.975862741470337}));

  ////////////////
  ASSERT_VEC_EQ(mat_row(block_out.x1, 0), ((float[]){-0.5595301985740662, -1.2403788566589355, 0.01690220832824707,  2.2756712436676025}));
  ASSERT_VEC_EQ(mat_row(block_out.x1, 1), ((float[]){ 0.0667530819773674, 0.08327847719192505, -0.678922176361084,  4.3617048263549805}));
  ASSERT_VEC_EQ(mat_row(block_out.x1, 2), ((float[]){-0.5660771131515503, -0.5118380784988403, 0.16695356369018555,   4.218010902404785}));
  ASSERT_VEC_EQ(mat_row(block_out.x1, 3), ((float[]){0.5315941572189331,  0.9619302749633789, 1.2285938262939453,  1.5161375999450684}));
  
  ////////////////
  ASSERT_VEC_EQ(mat_row(block_out.ln2.out.value, 0), ((float[]){-0.5202155113220215, -1.0370250940322876, -0.08266159892082214, 1.635905385017395}));
  ASSERT_VEC_EQ(mat_row(block_out.ln2.out.value, 1), ((float[]){-0.4510899782180786, -0.44276317954063416, -0.826743483543396, 1.7165967226028442}));
  ASSERT_VEC_EQ(mat_row(block_out.ln2.out.value, 2), ((float[]){-0.7071709632873535, -0.6797080636024475, -0.3360500931739807, 1.7189306020736694}));
  ASSERT_VEC_EQ(mat_row(block_out.ln2.out.value, 3), ((float[]){-1.4617561101913452, -0.2719407379627228, 0.4653429687023163, 1.2643550634384155}));
  
  ////////////////
  ASSERT_VEC_EQ(mat_row(block_out.ln2.xhat, 0), ((float[]){-0.5171815752983093, -1.0329633951187134, -0.08050099015235901, 1.630645990371704}));
  ASSERT_VEC_EQ(mat_row(block_out.ln2.xhat, 1), ((float[]){-0.4481940269470215, -0.43988555669784546, -0.823096752166748, 1.7111762762069702}));
  ASSERT_VEC_EQ(mat_row(block_out.ln2.xhat, 2), ((float[]){-0.7037638425827026, -0.6763583421707153, -0.33338338136672974, 1.713505506515503}));
  ASSERT_VEC_EQ(mat_row(block_out.ln2.xhat, 3), ((float[]){-1.4568426609039307, -0.26940351724624634, 0.4664090573787689, 1.259837031364441}));
  
  ////////////////
  ASSERT_VEC_EQ(block_out.ln2.mean, ((float[]){0.12316609919071198, 0.9582035541534424, 0.826762318611145, 1.0595639944076538}));
  ASSERT_VEC_EQ(block_out.ln2.var,  ((float[]){1.742474913597107, 3.956044912338257, 3.9169418811798096, 0.1313287615776062}));

  ////////////////
  ASSERT_VEC_EQ(mat_row(block_out.ff1_out.value, 0), ((float[]){1.5987250804901123, -0.859346866607666}));
  ASSERT_VEC_EQ(mat_row(block_out.ff1_out.value, 1), ((float[]){1.7023289203643799, -0.4314073920249939}));
  ASSERT_VEC_EQ(mat_row(block_out.ff1_out.value, 2), ((float[]){1.6972295045852661, -0.6478630304336548}));
  ASSERT_VEC_EQ(mat_row(block_out.ff1_out.value, 3), ((float[]){1.2654510736465454, -0.5029172897338867}));

  ////////////////
  ASSERT_VEC_EQ(mat_row(block_out.relu_out.value, 0), ((float[]){1.5987250804901123, 0}));
  ASSERT_VEC_EQ(mat_row(block_out.relu_out.value, 1), ((float[]){1.7023289203643799, 0}));
  ASSERT_VEC_EQ(mat_row(block_out.relu_out.value, 2), ((float[]){1.6972295045852661, 0}));
  ASSERT_VEC_EQ(mat_row(block_out.relu_out.value, 3), ((float[]){1.2654510736465454, 0}));

  ////////////////
  ASSERT_VEC_EQ(mat_row(block_out.ff2_out, 0), ((float[]){-0.02033604495227337, 0.3596985340118408, 0.14303964376449585, 0.280324786901474}));
  ASSERT_VEC_EQ(mat_row(block_out.ff2_out, 1), ((float[]){-0.02152453362941742, 0.382878839969635, 0.15243887901306152, 0.29862064123153687}));
  ASSERT_VEC_EQ(mat_row(block_out.ff2_out, 2), ((float[]){-0.02146603725850582, 0.3817378878593445, 0.15197625756263733, 0.2977201044559479}));
  ASSERT_VEC_EQ(mat_row(block_out.ff2_out, 3), ((float[]){-0.016512898728251457, 0.28513190150260925, 0.11280406266450882, 0.2214704006910324}));

  ////////////////
  ASSERT_VEC_EQ(mat_row(block_out.out.value, 0), ((float[]){-0.5798662304878235,  -0.8806803226470947, 0.15994185209274292,   2.5559959411621094}));
  ASSERT_VEC_EQ(mat_row(block_out.out.value, 1), ((float[]){0.04522854834794998,  0.46615731716156006, -0.5264832973480225,    4.660325527191162}));
  ASSERT_VEC_EQ(mat_row(block_out.out.value, 2), ((float[]){-0.5875431299209595, -0.13010019063949585, 0.3189298212528229,    4.515730857849121}));
  ASSERT_VEC_EQ(mat_row(block_out.out.value, 3), ((float[]){0.5150812864303589,   1.2470622062683105, 1.3413978815078735,   1.7376079559326172}));
}

void test_block_backward(void) {
  srand(0);

  Arena_Allocator arena = ARENA_CREATE(&mallocator.alloc, 1024*1024);

  Block block = {0};
  Block_Output block_out = {0};

  size_t N = 4;
  size_t D = 4;
  size_t H = 2;
  size_t F = 2;

  init_block(&arena.alloc, &block, D, F);
  init_block_output(&arena, &block_out, N, D, H, F);

  NMatrix dL   = mat_alloc2(&arena.alloc, N, D);
  NMatrix dLdx = mat_alloc2(&arena.alloc, N, D);

  mat_copy(mat_reshape(dL, 1, N*D), mat_init(1, N*D, (float[]){
  -0.022073691710829735,
  0.012354745529592037,
  0.01235650759190321,
  -0.0026375618763267994,
  -0.004216896370053291,
  -0.0037513868883252144,
  0.0070995124988257885,
  0.0008687709341757,
  -0.009847737848758698,
  0.011655228212475777,
  -0.0008701930055394769,
  -0.0009372981730848551,
  -0.0020905770361423492,
  0.029328832402825356,
  -0.027361374348402023,
  0.0001231197384186089
  }));

  mat_copy(mat_reshape(block_out.ln1.out.value, 1, N*D), mat_init(1, N*D, (float[]){
    -0.15357095003128052,
    -0.06366109848022461,
    -1.29647696018219,
    1.5137089490890503,
    0.15024594962596893,
    0.44921544194221497,
    -1.6406080722808838,
    1.0411466360092163,
    -0.5992678999900818,
    -0.19097451865673065,
    -0.8878705501556396,
    1.6781129837036133,
    0.21429391205310822,
    1.443113088607788,
    -0.3297770321369171,
    -1.3276299238204956
  }));
  mat_copy(mat_reshape(block_out.ln1.mean, 1, N), mat_init(1, N, (float[]){
    -0.27813759446144104,
    0.31815388798713684,
    0.15311959385871887,
    0.4716077446937561
  }));
  mat_copy(mat_reshape(block_out.ln1.var, 1, N), mat_init(1, N, (float[]){
    0.20431946218013763,
    0.7912920117378235,
    1.3592743873596191,
    0.26948219537734985
  }));
  mat_copy(mat_reshape(block_out.ln1.xhat, 1, N*D), mat_init(1, N*D, (float[]){
    -0.15357095003128052,
    -0.06366109848022461,
    -1.29647696018219,
    1.5137089490890503,
    0.15024594962596893,
    0.44921544194221497,
    -1.6406080722808838,
    1.0411466360092163,
    -0.5992678999900818,
    -0.19097451865673065,
    -0.8878705501556396,
    1.6781129837036133,
    0.21429391205310822,
    1.443113088607788,
    -0.3297770321369171,
    -1.3276299238204956
  }));
  mat_copy(mat_reshape(block_out.ln2.out.value, 1, N*D), mat_init(1, N*D, (float[]){
    -0.596727192401886,
    -0.9439001083374023,
    -0.11461052298545837,
    1.6552377939224243,
    -0.4174764156341553,
    -0.5497881770133972,
    -0.7524152994155884,
    1.719679832458496,
    -0.7060611844062805,
    -0.6559946537017822,
    -0.3542356491088867,
    1.7162914276123047,
    -1.324567437171936,
    -0.301631361246109,
    0.16806165874004364,
    1.4581371545791626
  }));
  mat_copy(mat_reshape(block_out.ln2.mean, 1, N), mat_init(1, N, (float[]){
    0.34020599722862244,
    0.9940856695175171,
    0.8652784824371338,
    1.1544139385223389
  }));
  mat_copy(mat_reshape(block_out.ln2.var, 1, N), mat_init(1, N, (float[]){
    1.8715219497680664,
    2.1398470401763916,
    4.2232818603515625,
    0.30564209818840027
  }));
  mat_copy(mat_reshape(block_out.ln2.xhat, 1, N*D), mat_init(1, N*D, (float[]){
    -0.596727192401886,
    -0.9439001083374023,
    -0.11461052298545837,
    1.6552377939224243,
    -0.4174764156341553,
    -0.5497881770133972,
    -0.7524152994155884,
    1.719679832458496,
    -0.7060611844062805,
    -0.6559946537017822,
    -0.3542356491088867,
    1.7162914276123047,
    -1.324567437171936,
    -0.301631361246109,
    0.16806165874004364,
    1.4581371545791626
  }));
  mat_copy(mat_reshape(block.ff1.weight.value, 1, D*F), mat_init(1, D*F, (float[]){
    -0.21068733930587769,
    -0.13117828965187073,
    -0.1632888913154602,
    0.37227359414100647,
    -0.20297282934188843,
    -0.32775840163230896,
    0.7912827730178833,
    -0.345366507768631
  }));
  mat_copy(mat_reshape(block.ff2.weight.value, 1, F*D), mat_init(1, F*D, (float[]){
    -0.009474903345108032,
    0.22173966467380524,
    0.09272415190935135,
    0.1785958856344223,
    -0.5291860103607178,
    -0.3492687940597534,
    -0.40243953466415405,
    -0.35886868834495544
  }));
  mat_copy(mat_reshape(block.attn.Q.weight.value, 1, D*D), mat_init(1, D*D, (float[]){
    -0.28238996863365173,
    0.16669102013111115,
    -0.30664610862731934,
    0.39735668897628784,
    -0.3547089695930481,
    -0.08701349049806595,
    0.5943112373352051,
    0.33786284923553467,
    0.8556529879570007,
    0.6758430004119873,
    -0.11796743422746658,
    0.07831144332885742,
    -0.3534604609012604,
    -0.6913005709648132,
    0.3407316505908966,
    -0.32336416840553284
  }));
  mat_copy(mat_reshape(block.attn.K.weight.value, 1, D*D), mat_init(1, D*D, (float[]){
    0.4952670931816101,
    0.7010917067527771,
    -0.703833818435669,
    -0.042622655630111694,
    0.5576351284980774,
    -0.6408947110176086,
    0.8188208341598511,
    -0.5570364594459534,
    0.3627428114414215,
    0.3458753526210785,
    -0.4794626235961914,
    -0.1509275734424591,
    -0.5829433798789978,
    0.2678900361061096,
    -0.2841755747795105,
    -0.5075703263282776
  }));
  mat_copy(mat_reshape(block.attn.V.weight.value, 1, D*D), mat_init(1, D*D, (float[]){
    0.22326721251010895,
    0.670093834400177,
    -0.3007221817970276,
    0.5064989328384399,
    -0.1854233294725418,
    -0.6384463906288147,
    0.6299506425857544,
    -0.22566822171211243,
    -0.8495252132415771,
    0.4112277030944824,
    0.23428380489349365,
    0.6511183381080627,
    0.7069616317749023,
    -0.5040462017059326,
    0.18001645803451538,
    -0.01863263174891472
  }));
  mat_copy(mat_reshape(block.attn.O.weight.value, 1, D*D), mat_init(1, D*D, (float[]){
    -0.5472874641418457,
    -0.7837979197502136,
    0.35602614283561707,
    0.3609073758125305,
    -0.3611462414264679,
    -0.3888719379901886,
    -0.2830173075199127,
    -0.815610945224762,
    -0.48342329263687134,
    0.854802131652832,
    0.12531675398349762,
    0.8462181091308594,
    -0.5822008848190308,
    -0.5701321363449097,
    0.1055930033326149,
    -0.3593941330909729
  }));

  mat_copy(mat_reshape(block_out.ff1_out.value, 1, N*F), mat_init(1, N*F, (float[]){
    1.61287522315979,
    -0.8072105646133423,
    1.6912041902542114,
    -0.49721717834472656,
    1.6858469247817993,
    -0.6282354593276978,
    1.4480094909667969,
    -0.4972102642059326
  }));
  mat_copy(mat_reshape(block_out.ff2_out, 1, N*D), mat_init(1, N*D, (float[]){
    -0.01528183650225401,
    0.357638418674469,
    0.14955249428749084,
    0.28805288672447205,
    -0.016023995354771614,
    0.3750070631504059,
    0.15681546926498413,
    0.3020420968532562,
    -0.01597323641180992,
    0.37381914258003235,
    0.15631872415542603,
    0.30108532309532166,
    -0.013719749636948109,
    0.32108113169670105,
    0.1342654526233673,
    0.25860854983329773
  }));
  mat_copy(mat_reshape(block_out.relu_out.value, 1, N*F), mat_init(1, N*F, (float[]){
    1.61287522315979,
    0,
    1.6912041902542114,
    0,
    1.6858469247817993,
    0,
    1.4480094909667969,
    0
  }));

  mat_copy(mat_reshape(mat_row(block_out.attn.weights.value, 0), 1, N*N), mat_init(1, N*N, (float[]){
                      1,                   0,
                      0,                   0,
     0.5862705111503601,  0.4137295186519623,
                      0,                   0,
     0.3040033280849457,  0.3063565790653229,
    0.38964009284973145,                   0,
    0.32824310660362244, 0.23965591192245483,
    0.34360602498054504, 0.08849496394395828,
  }));
  mat_copy(mat_reshape(mat_row(block_out.attn.weights.value, 1), 1, N*N), mat_init(1, N*N, (float[]){
                      1,                   0,
                      0,                   0,
     0.4314057528972626,  0.5685942769050598,
                      0,                   0,
    0.30029696226119995,  0.3938947021961212,
    0.30580833554267883,                   0,
    0.19906039535999298, 0.22681744396686554,
    0.18992385268211365, 0.38419830799102783
  }));

  mat_copy(mat_reshape(block_out.attn.Q.value, 1, N*D), mat_init(1, N*D, (float[]){
    -1.5784225463867188,  -1.942702293395996,
     0.6779680252075195,  -0.673539400100708,
    -1.9735641479492188,  -1.842581868171692,
     0.7691913843154907, -0.2536734640598297,
     -1.115887999534607, -1.8434168100357056,
      0.746790885925293, -0.9148183465003967,
     -0.385309636592865,  0.6050643920898438,
     0.3784833252429962,  0.9762080311775208
  }));
  mat_copy(mat_reshape(block_out.attn.K.value, 1, N*D), mat_init(1, N*D, (float[]){
    -1.4642525911331177,
    -0.10977914184331894,
    0.2474145144224167,
    -0.5306324362754822,
    -0.8771381378173828,
    -0.4710967242717743,
    0.7528205513954163,
    -0.5374754071235657,
    -1.7036052942276,
    -0.15528999269008636,
    0.2142331451177597,
    -0.5858340263366699,
    1.5651720762252808,
    -1.2443643808364868,
    1.566219449043274,
    -0.08936238288879395
  }));
  mat_copy(mat_reshape(block_out.attn.V.value, 1, N*D), mat_init(1, N*D, (float[]){
    2.149040937423706,
    -1.3583892583847046,
    -0.025172177702188492,
    -0.9357815384864807,
    2.080038547515869,
    -1.3855706453323364,
    0.04085690155625343,
    -1.1129034757614136,
    1.8422441482543945,
    -1.4906022548675537,
    0.15398290753364563,
    -0.8698081374168396,
    -0.8781715631484985,
    -0.24417991936206818,
    0.5283904671669006,
    -0.4071117639541626
  }));

  mat_copy(mat_reshape(block_out.attn.vals.value, 1, N*D), mat_init(1, N*D, (float[]){
    +2.14904093742370605, -1.35838925838470459, -0.02517217770218849, -0.93578153848648071,
    +2.12049269676208496, -1.36963498592376709, +0.01237157825380564, -1.03649210929870605,
    +2.00836133956909180, -1.41823196411132812, +0.05562344565987587, -0.98537373542785645,
    +1.75919389724731445, -1.31173074245452881, +0.23650802671909332, -0.76031190156936646,
  }));
  // mat_copy(mat_reshape(block_out.attn_out, 1, N*D), mat_init(1, N*D, (float[]){
  //   -0.1285843402147293, -0.6441724300384521, 1.0475959777832031, 2.198535203933716,
  //   -0.06841472536325455, -0.527912437915802, 1.0346852540969849, 2.265368938446045,
  //   -0.040166087448596954, -0.413298636674881, 1.0193352699279785, 2.2827649116516113,
  //   -0.1607373207807541, -0.23311124742031097, 0.9469163417816162, 2.178156852722168
  // }));

  NMatrix dscores = mat_alloc2(&arena.alloc, N, N);

  block_backward(
      .in        = tensor(NULL_MATRIX, dLdx),
      .dout      = dL,
      .dscores   = dscores,
      .block_in  = &block,
      .block_out = &block_out,
  );

  ////////////////
  ASSERT_VEC_EQ(block.ff2.bias.grad, ((float[]){ -0.03822890296578407, 0.049587421119213104, -0.00877554714679718, -0.0025829693768173456 }));
  ASSERT_VEC_EQ(mat_row(block.ff2.weight.grad, 0), ((float[]){ -0.06236269697546959, 0.07569965720176697, -0.009150313213467598, -0.004186651669442654}));
  ASSERT_VEC_EQ(mat_row(block.ff2.weight.grad, 1), ((float[]){ 0, 0, 0, 0 }));
  
  ////////////////
  // ASSERT_VEC_EQ(mat_row(block_out.relu_out.grad, 0), ((float[]){+2.33857942, -1.65599322}));
  // ASSERT_VEC_EQ(mat_row(block_out.relu_out.grad, 1), ((float[]){+4.67715883, -3.31198645}));
  // ASSERT_VEC_EQ(mat_row(block_out.relu_out.grad, 2), ((float[]){+7.01573849, -4.96797991}));
  
  ////////////////
  // ASSERT_VEC_EQ(mat_row(block_out.ff1_out.grad, 0), ((float[]){+2.33857942, +0.00000000}));
  // ASSERT_VEC_EQ(mat_row(block_out.ff1_out.grad, 1), ((float[]){+4.67715883, +0.00000000}));
  // ASSERT_VEC_EQ(mat_row(block_out.ff1_out.grad, 2), ((float[]){+7.01573849, +0.00000000}));

  ////////////////
  ASSERT_VEC_EQ(block.ff1.bias.grad, ((float[]){0.010082699358463287, 0}));
  ASSERT_VEC_EQ(mat_row(block.ff1.weight.grad, 0), ((float[]){ -0.009195653721690178, 0, }));
  ASSERT_VEC_EQ(mat_row(block.ff1.weight.grad, 1), ((float[]){ -0.0062347701750695705, 0, }));
  ASSERT_VEC_EQ(mat_row(block.ff1.weight.grad, 2), ((float[]){ -0.0006185720558278263, 0, }));
  ASSERT_VEC_EQ(mat_row(block.ff1.weight.grad, 3), ((float[]){ 0.016048995777964592, 0 }));

  ////////////////
  // ASSERT_VEC_EQ(mat_row(block_out.ln2_out.grad, 0), ((float[]){+1.34989548, +1.13798153}));
  // ASSERT_VEC_EQ(mat_row(block_out.ln2_out.grad, 1), ((float[]){+2.69979095, +2.27596307}));
  // ASSERT_VEC_EQ(mat_row(block_out.ln2_out.grad, 2), ((float[]){+4.04968643, +3.41394472}));

  ////////////////
  ASSERT_VEC_EQ(block.ln2.gamma.grad, ((float[]){ 0.0019374078838154674, 0.0010180686367675662, 0.00012555332796182483, 0.012699292972683907 }));
  ASSERT_VEC_EQ(block.ln2.beta.grad, ((float[]){ -0.0021242971997708082, -0.0016463929787278175, -0.0020465143024921417, 0.007978267036378384 }));
  // ASSERT_VEC_EQ(block_out.dln2_in, ((float[]){-0.0021242971997708082, -0.0016463929787278175, -0.0020465143024921417, 0.007978267036378384}));

  ////////////////
  ASSERT_VEC_EQ(block.attn.O.bias.grad, ((float[]){-0.03669314086437225, 0.04931109398603439, -0.011759607121348381, -0.0008583473972976208}));
  ASSERT_VEC_EQ(mat_row(block.attn.O.weight.grad, 0), ((float[]){-0.07714809477329254, 0.09329966455698013, -0.013770516030490398, -0.0023810570128262043,    }));
  ASSERT_VEC_EQ(mat_row(block.attn.O.weight.grad, 1), ((float[]){0.050453223288059235, -0.06631115823984146, 0.014569934457540512, 0.001288002822548151,      }));
  ASSERT_VEC_EQ(mat_row(block.attn.O.weight.grad, 2), ((float[]){-0.00016594539920333773, 0.007035775110125542, -0.007280029822140932, 0.00041020038770511746,}));
  ASSERT_VEC_EQ(mat_row(block.attn.O.weight.grad, 3), ((float[]){0.035154957324266434, -0.04133879765868187, 0.005133924074470997, 0.0010499146301299334      }));

  ////////////////
  ASSERT_VEC_EQ(block.attn.V.bias.grad, ((float[]){-0.0230647511780262, -0.0018958615837618709, 0.05768952518701553, -0.007684307638555765}));
  ASSERT_VEC_EQ(mat_row(block.attn.V.weight.grad, 0), ((float[]){0.006453856360167265, 0.0005073369247838855, -0.006205786019563675, -0.00041125540155917406}));
  ASSERT_VEC_EQ(mat_row(block.attn.V.weight.grad, 1), ((float[]){-0.004111418500542641, -0.0006411970825865865, 0.013708511367440224, -0.011379019357264042}));
  ASSERT_VEC_EQ(mat_row(block.attn.V.weight.grad, 2), ((float[]){0.023909728974103928, 0.0018679029308259487, -0.06674480438232422, 0.0016815270064398646}));
  ASSERT_VEC_EQ(mat_row(block.attn.V.weight.grad, 3), ((float[]){-0.02625216729938984, -0.0017340427730232477, 0.059242065995931625, 0.010108746588230133}));

  ////////////////
  ASSERT_VEC_EQ(block.attn.Q.bias.grad, ((float[]){0.014217209070920944, -0.005024487618356943, -0.00004900237399851903, -0.00007141266542021185}));
  ASSERT_VEC_EQ(mat_row(block.attn.Q.weight.grad, 0), ((float[]){0.0031238358933478594, -0.0010918822372332215, 0.000014283838936535176, -0.000003138887677778257}));
  ASSERT_VEC_EQ(mat_row(block.attn.Q.weight.grad, 1), ((float[]){0.0207213144749403, -0.007311682216823101, 0.00004437602183315903, -0.00007950782310217619}));
  ASSERT_VEC_EQ(mat_row(block.attn.Q.weight.grad, 2), ((float[]){-0.004563435912132263, 0.0016021659830585122, 0.00012878859706688672, 0.00003060022572753951}));
  ASSERT_VEC_EQ(mat_row(block.attn.Q.weight.grad, 3), ((float[]){-0.019281713292002678, 0.006801398005336523, -0.0001874484441941604, 0.00005204648550716229}));

  ////////////////
  ASSERT_VEC_EQ(block.attn.K.bias.grad, ((float[]){5.820766091346741e-10, 6.984919309616089e-10, 9.458744898438454e-11, -3.637978807091713e-11}));
  ASSERT_VEC_EQ(mat_row(block.attn.K.weight.grad, 0), ((float[]){-0.00041284883627668023, 0.0012005791068077087, -0.00010981442756019533, 0.00019411204266361892}));
  ASSERT_VEC_EQ(mat_row(block.attn.K.weight.grad, 1), ((float[]){-0.0024427655152976513, 0.0043050264939665794, -0.00008626104681752622, 0.00009305080311605707}));
  ASSERT_VEC_EQ(mat_row(block.attn.K.weight.grad, 2), ((float[]){-0.002210317412391305, 0.0029166946187615395, 0.00004100789374206215, -0.0003594428417272866}));
  ASSERT_VEC_EQ(mat_row(block.attn.K.weight.grad, 3), ((float[]){0.005065931472927332, -0.008422299288213253, 0.00015506762429140508, 0.00007227998139569536}));

  ////////////////
  ASSERT_VEC_EQ(mat_row(dLdx, 0), ((float[]){-0.05296050384640694, 0.03916819393634796, 0.01528068445622921, -0.00148837361484766,      }));
  ASSERT_VEC_EQ(mat_row(dLdx, 1), ((float[]){-0.012503748759627342, 0.0062066419050097466, 0.007655058987438679, -0.0013579517835751176,}));
  ASSERT_VEC_EQ(mat_row(dLdx, 2), ((float[]){-0.020350612699985504, 0.018972961232066154, 0.0030805973801761866, -0.0017029473092406988,}));
  ASSERT_VEC_EQ(mat_row(dLdx, 3), ((float[]){-0.020278366282582283, 0.03355841338634491, -0.013059146702289581, -0.00022090179845690727 }));
}

void test_transformer_forward(void) {
  srand(0);

  Arena_Allocator arena = ARENA_CREATE(&mallocator.alloc, 1024*1024);

  size_t C = 4;
  size_t D = 4;
  size_t H = 2;
  size_t F = 2;
  size_t V = 6;

  Transformer trans_in = {0};
  Transformer_Output trans_out = {0};

  init_transformer(
      .alloc = &arena.alloc,
      .trans = &trans_in,
      .context_size = C,
      .vocab_size = V,
      .emb_size = D,
      .ff_size = F,
  );

  TokenID_Array tokens = ARRAY_CREATE(&arena.alloc);
  array_append(&tokens, 1);
  array_append(&tokens, 2);
  array_append(&tokens, 3);

  size_t N = tokens.count;
  init_transformer_output(
      .arena = &arena,
      .trans_out = &trans_out,
      .vocab_size = trans_in.vocab_size,
      .emb_size = trans_in.emb_size,
      .ff_size = trans_in.ff_size,
      .sequence_size = N,
      .heads_count = H,
  );

  transformer_forward(tokens, &trans_out, &trans_in);

  TokenID_Array targets = ARRAY_CREATE(&arena.alloc);
  array_append(&targets, 2);
  array_append(&targets, 3);
  array_append(&targets, 4);

  printf("\n");

  transformer_backward(tokens, targets, &trans_out, &trans_in);

  printf("tok_emb = "); mat_println(trans_in.tok_emb.grad, 4);
  printf("pos_emb = "); mat_println(trans_in.pos_emb.grad, 4);

  ASSERT_VEC_EQ(mat_row(trans_in.tok_emb.grad, 0), ((float[]){+0.0000000000, +0.0000000000, +0.0000000000, +0.0000000000}));
  ASSERT_VEC_EQ(mat_row(trans_in.tok_emb.grad, 1), ((float[]){+0.0818862766, -0.0046921317, -0.1202818751, +0.0430877060}));
  ASSERT_VEC_EQ(mat_row(trans_in.tok_emb.grad, 2), ((float[]){-0.2411776930, +0.1224838421, -0.0354717784, +0.1541656554}));
  ASSERT_VEC_EQ(mat_row(trans_in.tok_emb.grad, 3), ((float[]){+0.0191762932, -0.0155209899, +0.0288937502, -0.0325490534}));
  ASSERT_VEC_EQ(mat_row(trans_in.tok_emb.grad, 4), ((float[]){+0.0000000000, +0.0000000000, +0.0000000000, +0.0000000000}));
  ASSERT_VEC_EQ(mat_row(trans_in.tok_emb.grad, 5), ((float[]){+0.0000000000, +0.0000000000, +0.0000000000, +0.0000000000}));

  ASSERT_VEC_EQ(mat_row(trans_in.pos_emb.grad, 0), ((float[]){+0.0818862766, -0.0046921317, -0.1202818751, +0.0430877060}));
  ASSERT_VEC_EQ(mat_row(trans_in.pos_emb.grad, 1), ((float[]){-0.2411776930, +0.1224838421, -0.0354717784, +0.1541656554}));
  ASSERT_VEC_EQ(mat_row(trans_in.pos_emb.grad, 2), ((float[]){+0.0191762932, -0.0155209899, +0.0288937502, -0.0325490534}));
  ASSERT_VEC_EQ(mat_row(trans_in.pos_emb.grad, 3), ((float[]){+0.0000000000, +0.0000000000, +0.0000000000, +0.0000000000}));
}

int main(void) {
  srand(0);

  test_relu(100);
  test_linear(100);
  test_softmax(100);
  test_softmax_cross_entropy(100);
  test_embeddings(100);
  test_composed(100);
  test_multiply(100);
  test_multiply_add(100);
  test_layer_norm_forward();
  test_layer_norm_forward_mean_variance();
  test_layer_norm_forward_gamma_beta();
  test_layer_norm_backward();
  test_block_forward();
  test_block_backward();
  test_transformer_forward();

  printf("All tests passed\n");

  return 0;
}

