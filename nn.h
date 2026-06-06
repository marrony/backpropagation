#ifndef NN_H
#define NN_H

#include <math.h>
#include <float.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <assert.h>
#include <unistd.h>

#include <pthread.h>
#include <sys/semaphore.h>

#include <raylib.h>

#define ARRAY_LEN(a) (sizeof(a) / sizeof(*(a)))

typedef struct {
  float* elems;
  int32_t rows;
  int32_t cols;
} NMatrix;

#define MAT_AT(m, row, col) ((m).elems[((m).cols)*(row) + (col)])
#define VEC_AT(v, idx) MAT_AT((v), 0, (idx))

NMatrix mat_alloc(int rows, int cols) {
  return (NMatrix) {
    .elems = malloc(sizeof(float) * rows * cols),
    .rows = rows,
    .cols = cols,
  };
}

NMatrix mat_init(int rows, int cols, float* data) {
  return (NMatrix) {
    .elems = data,
    .rows = rows,
    .cols = cols,
  };
}

void mat_fprint(FILE* fp, NMatrix m) {
  fprintf(fp, "[");
  for (int i = 0; i < m.rows; i++) {
    if (i > 0) fprintf(fp, " ");

    if (m.rows > 1) fprintf(fp, "[");
    for (int j = 0; j < m.cols; j++) {
      if (j > 0) fprintf(fp, " ");
      fprintf(fp, "%+.8f", MAT_AT(m, i, j));
    }
    if (m.rows > 1) fprintf(fp, "]");
  }
  fprintf(fp, "]");
}

void mat_fprintln(FILE* fp, NMatrix m) {
  mat_fprint(fp, m);
  fprintf(fp, "\n");
}

void mat_print(NMatrix m) {
  mat_fprint(stdout, m);
}

void mat_println(NMatrix m) {
  mat_fprintln(stdout, m);
}

void assert_fmt(
    bool pred,
    const char* pred_str,
    const char* func,
    const char* file,
    int line,
    const char* fmt,
    ...
) {
  if (!pred) {
    va_list args;
    va_start(args, fmt);
    fprintf(stderr,
        "\nAssertion failed: (%s), function %s, file %s, line %d.\n",
        pred_str, func, file, line);
    vfprintf(stderr, fmt, args);
    fprintf(stderr, "\n");
    va_end(args);
    exit(1);
  }
}

void assert_eq(const char* func, const char* file, int line, float a, float b) {
  float x = a - b;
  bool eq = (x*x) <= (0.001f*0.001f);
  if (!eq) {
    fprintf(stderr,
        "\nAssertion failed: (%+.8f == %+.8f), function %s, file %s, line %d.\n",
        a, b, func, file, line);
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
      fprintf(stderr, "\nAssertion failed: (");
      mat_fprint(stderr, va);
      fprintf(stderr, " != ");
      mat_fprint(stderr, mat_init(va.rows, va.cols, (float*)vb));
      fprintf(stderr, "), function %s, file %s, line %d.\n", func, file, line);
      exit(1);
    }
  }
}

#define ASSERT_FMT(pred, fmt, ...) assert_fmt((pred), #pred, __func__, __FILE__, __LINE__, fmt, __VA_ARGS__)
#define ASSERT_EQ(a, b) assert_eq(__func__, __FILE__, __LINE__, (a), (b))
#define ASSERT_VEC_EQ(a, b) assert_vec_eq(__func__, __FILE__, __LINE__, (a), (b))

void mat_print_sizes(size_t n, ...) {
  va_list args;

  va_start(args, n);

  for (size_t i = 0; i < n; i++) {
    NMatrix mat = va_arg(args, NMatrix);
    printf("(%dx%d), ", mat.rows, mat.cols);
  }

  va_end(args);
}

// Reshape matrix to new dimensions. This is a VIEW operation, not a copy.
// The returned matrix points to the SAME data buffer as the input.
// IMPORTANT: Changing shape does NOT change data layout. This is safe only if
// the new dimensions produce the same memory layout (same row-major order).
// If you need a different layout, use mat_copy with reshape.
NMatrix mat_reshape(NMatrix x, int rows, int cols) {
  assert(x.rows*x.cols == rows*cols);

  return (NMatrix) {
    .elems = x.elems,
    .rows = rows,
    .cols = cols,
  };
}

void mat_fill(NMatrix dst, float k) {
  for (int i = 0; i < dst.rows; i++) {
    for (int j = 0; j < dst.cols; j++) {
      MAT_AT(dst, i, j) = k;
    }
  }
}

void mat_copy(NMatrix dst, NMatrix src) {
  ASSERT_FMT(
      dst.cols == src.cols,
      "[%dx%d] != [%dx%d]",
      dst.rows, dst.cols, src.rows, src.cols);
  ASSERT_FMT(
      dst.rows == src.rows,
      "[%dx%d] != [%dx%d]",
      dst.rows, dst.cols, src.rows, src.cols);

  for (int i = 0; i < dst.rows; i++) {
    for (int j = 0; j < dst.cols; j++) {
      MAT_AT(dst, i, j) = MAT_AT(src, i, j);
    }
  }
}

void mat_transpose(NMatrix dst, NMatrix src) {
  assert(dst.rows == src.cols);
  assert(dst.cols == src.rows);

  for (int i = 0; i < dst.rows; i++) {
    for (int j = 0; j < dst.cols; j++) {
      MAT_AT(dst, i, j) = MAT_AT(src, j, i);
    }
  }
}

float mat_dot_row_col(NMatrix a, NMatrix b, int a_row, int b_col) {
  float dot = 0;

  for (int k = 0; k < a.cols; k++) {
    dot += MAT_AT(a, a_row, k) * MAT_AT(b, k, b_col);
  }

  return dot;
}

float mat_dot_col_col(NMatrix a, NMatrix b, int a_col, int b_col) {
  float dot = 0;

  for (int k = 0; k < a.rows; k++) {
    dot += MAT_AT(a, k, a_col) * MAT_AT(b, k, b_col);
  }

  return dot;
}

float mat_dot_row_row(NMatrix a, NMatrix b, int a_row, int b_row) {
  float dot = 0;

  for (int k = 0; k < a.cols; k++) {
    dot += MAT_AT(a, a_row, k) * MAT_AT(b, b_row, k);
  }

  return dot;
}

void mat_sum_row(NMatrix dst, NMatrix src) {
  assert(dst.cols == 1);
  assert(dst.rows == src.rows);

  for (int i = 0; i < src.rows; i++) {
    MAT_AT(dst, i, 0) = 0;
    for (int j = 0; j < src.cols; j++) {
      MAT_AT(dst, i, 0) += MAT_AT(src, i, j);
    }
  }
}

// dst = A x B
void mat_mult(NMatrix dst, NMatrix a, NMatrix b) {
  ASSERT_FMT(
      a.cols == b.rows,
      "[%dx%d] * [%dx%d] = [%dx%d]",
      a.rows, a.cols, b.rows, b.cols, dst.rows, dst.cols);
  ASSERT_FMT(
      dst.rows == a.rows,
      "[%dx%d] * [%dx%d] = [%dx%d]",
      a.rows, a.cols, b.rows, b.cols, dst.rows, dst.cols);
  ASSERT_FMT(
      dst.cols == b.cols,
      "[%dx%d] * [%dx%d] = [%dx%d]",
      a.rows, a.cols, b.rows, b.cols, dst.rows, dst.cols);

#if 1
  for (int i = 0; i < dst.rows; i++) {
    for (int j = 0; j < dst.cols; j++) {
      MAT_AT(dst, i, j) = mat_dot_row_col(a, b, i, j);
    }
  }
#else
  memset(dst.elems, 0, sizeof(float)*dst.rows*dst.cols);
  for (int i = 0; i < a.rows; i++) {
    for (int j = 0; j < a.cols; j++) {
      float v = MAT_AT(a, i, j);

      for (int k = 0; k < b.cols; k++) {
        MAT_AT(dst, i, k) +=  v * MAT_AT(b, j, k);
      }
    }
  }
#endif
}

// dst = A x B.T
void mat_mult_transpose(NMatrix dst, NMatrix a, NMatrix b) {
  ASSERT_FMT(
      a.cols == b.cols,
      "[%dx%d] * [%dx%d] = [%dx%d]",
      a.rows, a.cols, b.cols, b.rows, dst.rows, dst.cols);
  ASSERT_FMT(
      dst.rows == a.rows,
      "[%dx%d] * [%dx%d] = [%dx%d]",
      a.rows, a.cols, b.cols, b.rows, dst.rows, dst.cols);
  ASSERT_FMT(
      dst.cols == b.rows,
      "[%dx%d] * [%dx%d] = [%dx%d]",
      a.rows, a.cols, b.cols, b.rows, dst.rows, dst.cols);

  for (int i = 0; i < dst.rows; i++) {
    for (int j = 0; j < dst.cols; j++) {
      MAT_AT(dst, i, j) = mat_dot_row_row(a, b, i, j);
    }
  }
}

// dst = A x B.T + C
void mat_mult_transpose_add(NMatrix dst, NMatrix a, NMatrix b, NMatrix c) {
  ASSERT_FMT(
      a.cols == b.cols,
      "[%dx%d] * [%dx%d] = [%dx%d]",
      a.rows, a.cols, b.cols, b.rows, dst.rows, dst.cols);
  ASSERT_FMT(
      dst.rows == a.rows,
      "[%dx%d] * [%dx%d] = [%dx%d]",
      a.rows, a.cols, b.cols, b.rows, dst.rows, dst.cols);
  ASSERT_FMT(
      dst.cols == b.rows,
      "[%dx%d] * [%dx%d] = [%dx%d]",
      a.rows, a.cols, b.cols, b.rows, dst.rows, dst.cols);

  for (int i = 0; i < dst.rows; i++) {
    for (int j = 0; j < dst.cols; j++) {
      MAT_AT(dst, i, j) = mat_dot_row_row(a, b, i, j) + MAT_AT(c, i, j);
    }
  }
}

// dst = A.T x B
void mat_transpose_mult(NMatrix dst, NMatrix a, NMatrix b) {
  // assert(a.rows == b.rows);
  // assert(dst.rows == a.cols);
  // assert(dst.cols == b.cols);

  ASSERT_FMT(
      a.rows == b.rows,
      "[%dx%d] * [%dx%d] = [%dx%d]",
      a.cols, a.rows, b.rows, b.cols, dst.rows, dst.cols);
  ASSERT_FMT(
      dst.rows == a.cols,
      "[%dx%d] * [%dx%d] = [%dx%d]",
      a.cols, a.rows, b.rows, b.cols, dst.rows, dst.cols);
  ASSERT_FMT(
      dst.cols == b.cols,
      "[%dx%d] * [%dx%d] = [%dx%d]",
      a.cols, a.rows, b.rows, b.cols, dst.rows, dst.cols);

  for (int i = 0; i < dst.rows; i++) {
    for (int j = 0; j < dst.cols; j++) {
      MAT_AT(dst, i, j) = mat_dot_col_col(a, b, i, j);
    }
  }
}

// dst = A.T x B + C
void mat_transpose_mult_add(NMatrix dst, NMatrix a, NMatrix b, NMatrix c) {
  assert(a.rows == b.rows);
  assert(dst.rows == a.cols);
  assert(dst.cols == b.cols);

  for (int i = 0; i < dst.rows; i++) {
    for (int j = 0; j < dst.cols; j++) {
      MAT_AT(dst, i, j) = mat_dot_col_col(a, b, i, j) + MAT_AT(c, i, j);
    }
  }
}

// dst = A x B + C
void mat_mult_add(NMatrix dst, NMatrix a, NMatrix b, NMatrix c) {
  assert(a.cols == b.rows);
  assert(dst.rows == a.rows);
  assert(dst.cols == b.cols);
  assert(dst.rows == c.rows);
  assert(dst.cols == c.cols);

#if 1
  for (int i = 0; i < dst.rows; i++) {
    for (int j = 0; j < dst.cols; j++) {
      MAT_AT(dst, i, j) = mat_dot_row_col(a, b, i, j) + MAT_AT(c, i, j);
    }
  }
#else
  mat_copy(dst, c);

  for (int i = 0; i < a.rows; i++) {
    for (int j = 0; j < a.cols; j++) {
      float v = MAT_AT(a, i, j);

      for (int k = 0; k < b.cols; k++) {
        MAT_AT(dst, i, k) +=  v * MAT_AT(b, j, k);
      }
    }
  }
#endif
}

float mat_dot(NMatrix a, NMatrix b) {
  assert(a.rows == 1);
  assert(b.rows == 1);
  assert(a.cols == b.cols);

  float dot = 0;
  for (int n = 0; n < a.cols; n++) {
    dot += VEC_AT(a, n) * VEC_AT(b, n);
  }
  return dot;
}

void mat_add(NMatrix dst, NMatrix a, NMatrix b) {
  assert(dst.cols == a.cols);
  assert(dst.rows == a.rows);
  assert(dst.cols == b.cols);
  assert(dst.rows == b.rows);

  for (int i = 0; i < dst.rows; i++) {
    for (int j = 0; j < dst.cols; j++) {
      assert(!isnan(MAT_AT(a, i, j)));
      assert(!isnan(MAT_AT(b, i, j)));

      MAT_AT(dst, i, j) = MAT_AT(a, i, j) + MAT_AT(b, i, j);
    }
  }
}

void mat_weighted_add(NMatrix dst, NMatrix a, NMatrix b, float k) {
  assert(dst.cols == a.cols);
  assert(dst.rows == a.rows);
  assert(dst.cols == b.cols);
  assert(dst.rows == b.rows);

  for (int i = 0; i < dst.rows; i++) {
    for (int j = 0; j < dst.cols; j++) {
      assert(!isnan(MAT_AT(a, i, j)));
      assert(!isnan(MAT_AT(b, i, j)));

      MAT_AT(dst, i, j) = MAT_AT(a, i, j) + MAT_AT(b, i, j) * k;
    }
  }
}

void mat_sub(NMatrix dst, NMatrix a, NMatrix b) {
  assert(dst.cols == a.cols);
  assert(dst.rows == a.rows);
  assert(dst.cols == b.cols);
  assert(dst.rows == b.rows);

  for (int i = 0; i < dst.rows; i++) {
    for (int j = 0; j < dst.cols; j++) {
      MAT_AT(dst, i, j) = MAT_AT(a, i, j) - MAT_AT(b, i, j);
    }
  }
}

void mat_memberwise_mult(NMatrix dst, NMatrix a, NMatrix b) {
  assert(dst.cols == a.cols);
  assert(dst.rows == a.rows);
  assert(dst.cols == b.cols);
  assert(dst.rows == b.rows);

  for (int i = 0; i < dst.rows; i++) {
    for (int j = 0; j < dst.cols; j++) {
      MAT_AT(dst, i, j) = MAT_AT(a, i, j) * MAT_AT(b, i, j);
    }
  }
}

void mat_memberwise_div(NMatrix dst, NMatrix a, NMatrix b) {
  assert(dst.cols == a.cols);
  assert(dst.rows == a.rows);
  assert(dst.cols == b.cols);
  assert(dst.rows == b.rows);

  for (int i = 0; i < dst.rows; i++) {
    for (int j = 0; j < dst.cols; j++) {
      MAT_AT(dst, i, j) = MAT_AT(a, i, j) / MAT_AT(b, i, j);
    }
  }
}

void mat_square(NMatrix dst, NMatrix src) {
  assert(dst.cols == src.cols);
  assert(dst.rows == src.rows);

  for (int i = 0; i < dst.rows; i++) {
    for (int j = 0; j < dst.cols; j++) {
      MAT_AT(dst, i, j) = MAT_AT(src, i, j) * MAT_AT(src, i, j);
    }
  }
}

void mat_scale(NMatrix dst, NMatrix src, float k) {
  assert(dst.cols == src.cols);
  assert(dst.rows == src.rows);

  for (int i = 0; i < dst.rows; i++) {
    for (int j = 0; j < dst.cols; j++) {
      MAT_AT(dst, i, j) = MAT_AT(src, i, j) * k;
    }
  }
}

NMatrix mat_row_slice(NMatrix m, int start, int size) {
  assert(m.rows == 1);
  assert(start+size <= m.cols);

  return (NMatrix) {
    .elems = &(VEC_AT(m, start)),
    .rows = 1,
    .cols = size,
  };
}

NMatrix mat_slice(NMatrix m, int start, int size) {
  return (NMatrix) {
    .elems = &(MAT_AT(m, start, 0)),
    .rows = size,
    .cols = m.cols,
  };
}

NMatrix mat_row(NMatrix m, int row) {
  return mat_slice(m, row, 1);
}

int mat_row_max(NMatrix row) {
  assert(row.rows == 1);
  assert(row.cols >= 1);

  int max_index = 0;
  float max_value = MAT_AT(row, 0, 0);

  for (int i = 1; i < row.cols; i++) {
    float v = VEC_AT(row, i);

    if (v > max_value) {
      max_value = v;
      max_index = i;
    }
  }

  return max_index;
}

void mat_ident(NMatrix m) {
  assert(m.rows == m.cols);

  memset(m.elems, 0, sizeof(float) * m.rows * m.cols);

  for (int i = 0; i < m.rows; i++) {
    MAT_AT(m, i, i) = 1.0f;
  }
}

void mat_rand(NMatrix m) {
  for (int i = 0; i < m.rows; i++) {
    for (int j = 0; j < m.cols; j++) {
      float r = rand() / (float)RAND_MAX;
      MAT_AT(m, i, j) = r * 2.0f - 1.0f;
    }
  }
}

/*
 * Calculates the forward function on x.
 *
 * y = f(x)
 */
typedef void (*Forward_Fn)(NMatrix y, NMatrix x);

/*
 * Calculates dL/dx.
 *
 * y = f(x)
 * L = loss(y)
 *
 * dL   dL   dy
 * -- = -- * --
 * dx   dy   dx
 *
 * dy
 * -- = local derivative
 * dx
 *
 * dL
 * -- = gradient passed from the next layer to the current layer
 * dy
 */
typedef void (*Backward_Fn)(NMatrix dL_dx, NMatrix y, NMatrix dL_dy);

/*
 * x = [ x1, x2, ... ]
 *
 *        n1  n2
 * w = | w11 w12 ... |
 *     | w21 w22 ... |
 *     | ...         |
 *
 */
typedef struct {
  NMatrix* w;
  NMatrix* b;
  Forward_Fn* forward;
  Backward_Fn* backward;
  int layers;
} Neuron_Network;

static inline float sigmoidf(float x) {
  return 1.0f / (1.0f + expf(-x));
}

// Derivative of sigmoid function, where x is the sigmoid output (not input)
// Returns: sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
// IMPORTANT: x must be the output of sigmoid() function (values in range 0-1)
static inline float dsigmoidf(float x) {
  return x * (1.0f - x);
}

// Sigmoid function
//
//              1
// returns -----------
//         1 + exp(-x)
void sigmoid(NMatrix dst, NMatrix x) {
  assert(x.rows == 1);
  assert(dst.rows == 1);

  for (int j = 0; j < x.cols; j++) {
    VEC_AT(dst, j) = sigmoidf(VEC_AT(x, j));
  }
}

void dsigmoid(NMatrix dst, NMatrix h, NMatrix dL_dh) {
  assert(h.rows == 1);
  assert(dst.rows == 1);

  for (int j = 0; j < h.cols; j++) {
    VEC_AT(dst, j) = dsigmoidf(VEC_AT(h, j)) * VEC_AT(dL_dh, j);
  }
}

static inline float reluf(float x) {
  return x > 0 ? x : 0;
}

static inline float dreluf(float x) {
  return x > 0 ? 1 : 0;
}

// Relu function
//
// returns if x > 0  : x
//         if x <= 0 : 0
void relu(NMatrix dst, NMatrix x) {
  for (int i = 0; i < x.rows; i++) {
    for (int j = 0; j < x.cols; j++) {
      MAT_AT(dst, i, j) = reluf(MAT_AT(x, i, j));
    }
  }
}

void drelu(NMatrix dst, NMatrix h, NMatrix dL_dh) {
  for (int i = 0; i < h.rows; i++) {
    for (int j = 0; j < h.cols; j++) {
      MAT_AT(dst, i, j) = dreluf(MAT_AT(h, i, j)) * MAT_AT(dL_dh, i, j);
    }
  }
}

// Softmax function
//
//           exp(x[i])
// returns --------------
//         sum(exp(x[j]))
void softmax(NMatrix dst, NMatrix x) {
  assert(x.rows == 1);
  assert(dst.rows == 1);

  float m = VEC_AT(x, 0);

  for (int j = 0; j < x.cols; j++) {
    if (VEC_AT(x, j) > m) m = VEC_AT(x, j);
  }

  float sum = 0;
  for (int j = 0; j < x.cols; j++) {
    float e = expf(VEC_AT(x, j) - m);
    // Check for overflow (expf returns infinity for very large values)
    if (isinf(e)) {
      e = FLT_MAX; // Treat overflow as very large value
    }
    VEC_AT(dst, j) = e;
    sum += e;
  }

  for (int j = 0; j < x.cols; j++) {
    VEC_AT(dst, j) /= sum;

    assert(!isnan(VEC_AT(dst, j)));
  }
}

void dsoftmax(NMatrix dst, NMatrix h, NMatrix dL_dh) {
  assert(h.rows == 1);
  assert(dst.rows == 1);

#if 0
  for (int j = 0; j < h.cols; j++) {
    float sj = VEC_AT(h, j);

    float dot = 0;
    for (int i = 0; i < h.cols; i++) {
      float si = VEC_AT(h, i);

      float dij = i == j ? 1 : 0;

      float s = dij*si - (si * sj);

      dot += VEC_AT(dL_dh, i) * s;
    }

    VEC_AT(dst, j) = dot;
  }
#else
  // dL/dz = Softmax(z) * [ dL/dh - dot(Softmax(z), dL/dh) ]
  float dot = 0;
  for (int i = 0; i < h.cols; i++)
    dot += VEC_AT(h, i) * VEC_AT(dL_dh, i);

  for (int i = 0; i < h.cols; i++)
    VEC_AT(dst, i) = VEC_AT(h, i) * (VEC_AT(dL_dh, i) - dot);
#endif
}

void linear(NMatrix h, NMatrix z) {
  mat_copy(h, z);
}

void dlinear(NMatrix dst, NMatrix h, NMatrix dL_dh) {
  (void)h;
  // given h = z, dh/dz = 1 then dL/dz = dL/dh
  mat_copy(dst, dL_dh);
  
  // Check for NaN/Inf propagation
  for (int i = 0; i < dst.rows; i++) {
    for (int j = 0; j < dst.cols; j++) {
      float v = MAT_AT(dst, i, j);
      if (isnan(v) || isinf(v)) {
        assert(false); // NaN/Inf detected in gradient
      }
    }
  }
}

// Forward pass through the network.
// IMPORTANT: The 'h' array is reused as both input and output buffer for each layer.
// For layer i: h[i] receives input from h[i-1] (or 'input' for layer 0).
// The same h[i] buffer is first used to compute z = x*W.T + b, then activated.
// This is an in-place operation pattern: h[i] = activation(h[i-1] * w[i] + b[i])
NMatrix forward(Neuron_Network* nn, NMatrix* h, NMatrix input) {
  for (int i = 0; i < nn->layers; i++) {
    NMatrix in = i == 0 ? input : h[i-1];
    
    // f = x*W.T + b (compute pre-activation in h[i])
    mat_mult_transpose_add(h[i], in, nn->w[i], nn->b[i]);

    // Apply activation function (in-place: h[i] = activation(h[i]))
    nn->forward[i](h[i], h[i]);
  }

  return h[nn->layers-1];
}

float backward_sigmoid_only(
    Neuron_Network* nn,
    NMatrix* activations,
    Neuron_Network* grad,
    NMatrix* dL_dw,
    NMatrix inputs,
    NMatrix targets
) {
  // error[N-1] = 2(y - h[N-1])
  //
  // for (i from N-1 to 0) {
  //   delta = error[i] * sigmoid'( h[i] )
  //
  //   db[i] += delta
  //   dw[i] += h[i-1].T * delta
  //
  //   error[i-1] += delta * w[i].T
  // }

  int N = grad->layers;

  // dL = h[N-1] - y
  mat_sub(dL_dw[N-1], activations[N-1], targets);

  float L = mat_dot(dL_dw[N-1], dL_dw[N-1]);

  // dL = 2( h[N-1] - y )
  mat_scale(dL_dw[N-1], dL_dw[N-1], 2);

  for (int i = N-1; i >= 0; i--) {
    NMatrix act = i == 0 ? inputs : activations[i-1];

    for (int j = 0; j < grad->w[i].cols; j++) {
      // delta_i = sigmoid'(h[i]) * dL[i]
      float delta =  dsigmoidf(VEC_AT(activations[i], j)) * VEC_AT(dL_dw[i], j);
      assert(!isnan(delta));

      // db[i] = delta_i
      VEC_AT(grad->b[i], j) = delta;

      // dw[i] = h[i-1].T * delta_i
      for (int k = 0; k < grad->w[i].rows; k++) {
        MAT_AT(grad->w[i], k, j) = VEC_AT(act, k) * delta;
      }

      if (i > 0) {
        // dL[i-1] = delta_i * w[i].T
        for (int k = 0; k < nn->w[i].rows; k++) {
          VEC_AT(dL_dw[i-1], k) = delta * MAT_AT(nn->w[i], k, j);
        }
      }
    }
  }

  return L;
}

// Backward pass: computes gradients for all layers.
// For MSE loss L = 0.5 * (y - h)^2, the derivative is dL/dh = -(y - h),
// and the chain rule gives the 2x factor: dL/dh = 2 * (h - y).
// This function accumulates gradients into 'grad' and dL_dh.
// IMPORTANT: The 2x scaling is included here to match standard MSE loss
// formulation L = (y - h)^2 (without the 1/2 factor).
// When using neuron_weighted_add with learning rate lr, the effective update
// becomes: w = w - lr * 2 * (h - y). This is equivalent to using lr/2
// with the standard 0.5*MSE formulation.
float backward(
    Neuron_Network* nn,
    NMatrix* activations,
    Neuron_Network* grad,
    NMatrix* dL_dh,
    NMatrix* delta, // dL/dz
    NMatrix inputs,
    NMatrix targets
) {
  //   z = h[-1] * w + b
  //   h = forward(z)
  //
  //   dz
  //   -- = h[-1]
  //   dw
  //
  //   dz
  //   -- = 1
  //   db
  //
  //     dz
  //   ------ = w
  //   dh[-1]
  //
  //   ================================================
  //
  //   dL   dh   dL
  //   -- = -- * -- = delta
  //   dz   dz   dh
  //
  //   ================================================
  //
  //   dL   dz   dh   dL    dz
  //   -- = -- * -- * -- => -- * delta => h[-1] * delta
  //   dw   dw   dz   dh    dw
  //
  //   ================================================
  //
  //   dL   dz   dh   dL    dz
  //   -- = -- * -- * -- => -- * delta => 1 * delta
  //   db   db   dz   dh    db
  //
  //   ================================================
  //
  //    dL[-1]     dz     dh   dL      dz
  //   ------- = ------ * -- * -- => ------ * delta => w * delta
  //   dh[-1]    dh[-1]   dz   dh    dh[-1]
  //
  //
  // dL[N-1] = 2( h[N-1] - y )
  //
  // for (i from N-1 to 0) {
  //   delta = forward'( h[i] ) x dL[i]
  //
  //   dw[i] += h[i-1].T * delta
  //   db[i] += delta
  //
  //   dL[i-1] += delta * w[i].T
  // }

  int N = nn->layers;

  // dL = h[N-1] - y
  mat_sub(dL_dh[N-1], activations[N-1], targets);

  float L = mat_dot(dL_dh[N-1], dL_dh[N-1]);

  // dL = 2( h[N-1] - y )
  // Note: The 2x factor is retained to match MSE loss L = (y-h)^2 formulation.
  // This ensures gradients are 2x larger than with L = 0.5*(y-h)^2,
  // which compensates for removing the 1/2 factor in the loss definition.
  mat_scale(dL_dh[N-1], dL_dh[N-1], 2.0f);

  for (int i = N-1; i >= 0; i--) {
    NMatrix act = i == 0 ? inputs : activations[i-1];

    // delta_i = forward'(h[i]) * dL[i]
    nn->backward[i](delta[i], activations[i], dL_dh[i]);

    // db[i] = delta_i
    mat_copy(grad->b[i], delta[i]);

    // dw[i] = delta_i.T * h[i-1]
    mat_transpose_mult(grad->w[i], delta[i], act);

    if (i > 0) {
      // dL[i-1] = delta_i * w[i]
      mat_mult(dL_dh[i-1], delta[i], nn->w[i]);
    }
  }

  return L;
}

NMatrix* create_outputs(Neuron_Network nn) {
  NMatrix* h = malloc(sizeof(NMatrix) * nn.layers);

  for (int i = 0; i < nn.layers; i++) {
    h[i] = mat_alloc(1, nn.w[i].rows);
    mat_fill(h[i], 0);
  }

  return h;
}

typedef struct {
  int32_t inputs;
  int32_t outputs;
  bool randomize;
  Forward_Fn forward;
  Backward_Fn backward;
} Neuron_Layer;

#define create_layer(...) ((Neuron_Layer) { .randomize = true, .forward = sigmoid, .backward = dsigmoid, __VA_ARGS__ })

Neuron_Network neuron_create(Neuron_Layer* layers, size_t layers_count) {
  NMatrix* w = malloc(sizeof(NMatrix) * layers_count);
  NMatrix* b = malloc(sizeof(NMatrix) * layers_count);
  Forward_Fn* forward = malloc(sizeof(Forward_Fn) * layers_count);
  Backward_Fn* backward = malloc(sizeof(Backward_Fn) * layers_count);

  for (size_t i = 0; i < layers_count; i++) {
    Neuron_Layer layer = layers[i];

    forward[i] = layer.forward;
    backward[i] = layer.backward;

    // NxM
    w[i] = mat_alloc(layer.outputs, layer.inputs);

    // 1xM
    b[i] = mat_alloc(1, layer.outputs);

    if (layer.randomize) {
      mat_rand(w[i]);
      mat_rand(b[i]);
    } else {
      mat_fill(w[i], 0);
      mat_fill(b[i], 0);
    }
  }

  return (Neuron_Network) {
      .w = w,
      .b = b,
      .forward = forward,
      .backward = backward,
      .layers = layers_count,
  };
}

Neuron_Network neuron_clone(Neuron_Network nn) {
  NMatrix* w = malloc(sizeof(NMatrix) * nn.layers);
  NMatrix* b = malloc(sizeof(NMatrix) * nn.layers);

  for (int i = 0; i < nn.layers; i++) {
    w[i] = mat_alloc(nn.w[i].rows, nn.w[i].cols);
    b[i] = mat_alloc(nn.b[i].rows, nn.b[i].cols);
  }

  return (Neuron_Network) { .w = w, .b = b, .layers = nn.layers, };
}

void neuron_weighted_add(Neuron_Network* dst, Neuron_Network* src, float k) {
  for (int32_t j = 0; j < dst->layers; j++) {
    mat_weighted_add(dst->w[j], dst->w[j], src->w[j], k);
    mat_weighted_add(dst->b[j], dst->b[j], src->b[j], k);
  }
}

void neuron_add(Neuron_Network* dst, Neuron_Network* src) {
  for (int32_t j = 0; j < dst->layers; j++) {
    mat_add(dst->w[j], dst->w[j], src->w[j]);
    mat_add(dst->b[j], dst->b[j], src->b[j]);
  }
}

void neuron_zero(Neuron_Network* nn) {
  for (int32_t j = 0; j < nn->layers; j++) {
    mat_fill(nn->w[j], 0);
    mat_fill(nn->b[j], 0);
  }
}

void image_to_pixels(NMatrix image, unsigned char* pixels) {
  float min = +FLT_MAX;
  float max = -FLT_MAX;

  for (int i = 0; i < image.cols; i++) {
    float v = MAT_AT(image, 0, i);
    if (v > max) max = v;
    if (v < min) min = v;
  }

  float diff = max - min;

  for (int i = 0; i < image.cols; i++) {
    float v;
    if (diff == 0) {
      v = 0.5f; // All pixels same value, set to middle gray
    } else {
      v = (MAT_AT(image, 0, i) - min) / diff;
    }

    if (!isnan(v)) {
      unsigned char ch = (unsigned char)(v * 255);

      pixels[i] = ch;
    }
  }
}

void weights_to_pixels(NMatrix weights, int neuron, unsigned char* pixels) {
  float min = +FLT_MAX;
  float max = -FLT_MAX;

  for (int i = 0; i < weights.rows; i++) {
    float v = MAT_AT(weights, i, neuron);
    if (v > max) max = v;
    if (v < min) min = v;
  }

  float diff = max - min;

  for (int i = 0; i < weights.rows; i++) {
    float v;
    if (diff == 0) {
      v = 0.5f; // All weights same value, set to middle gray
    } else {
      v = (MAT_AT(weights, i, neuron) - min) / diff;
    }

    if (!isnan(v)) {
      unsigned char ch = (unsigned char)(v * 255);

      pixels[i] = ch;
    }
  }
}

typedef struct {
  bool normalize;
} Read_Idx_Opts;

NMatrix read_idx_opt(const char* filename, Read_Idx_Opts opts) {
  FILE* fp = fopen(filename, "rb");

  uint32_t magic;
  if (fread(&magic, sizeof(magic), 1, fp) != 1) {
    fclose(fp);
    fprintf(stderr, "Could not read magic number from %s\n", filename);
    exit(1);
  }

  uint32_t data_type = (magic >> 16) & 0xff;
  uint32_t dimension = (magic >> 24) & 0xff;

  // 0x08: unsigned byte 
  // 0x09: signed byte 
  // 0x0B: short (2 bytes) 
  // 0x0C: int (4 bytes) 
  // 0x0D: float (4 bytes) 
  // 0x0E: double (8 bytes)
  assert(data_type == 0x08);

  NMatrix output = {0};

  uint32_t rows = 0;
  uint32_t cols = 0;

  if (dimension == 1) {
    fread(&rows, sizeof(rows), 1, fp);

    rows = ntohl(rows);
    cols = 1;
  }

  if (dimension == 2) {
    fread(&rows, sizeof(rows), 1, fp);
    fread(&cols, sizeof(cols), 1, fp);

    rows = ntohl(rows);
    cols = ntohl(cols);
  }

  if (dimension == 3) {
    uint32_t width, height;
    fread(&rows, sizeof(rows), 1, fp);
    fread(&width, sizeof(width), 1, fp);
    fread(&height, sizeof(height), 1, fp);

    rows = ntohl(rows);
    width = ntohl(width);
    height = ntohl(height);

    cols = width * height;
  }

  size_t size = rows * cols;
  uint8_t *data = malloc(size * sizeof(uint8_t));
  if (fread(data, sizeof(uint8_t), size, fp) != size) {
    fclose(fp);
    fprintf(stderr, "Could not read the elements\n");
    exit(1);
  }

  output = mat_alloc(rows, cols);

  for (size_t i = 0; i < size; i++) {
    output.elems[i] = (float)data[i];

    if (opts.normalize)
      output.elems[i] /= 255.0f;
  }

  free(data);

  fclose(fp);
  return output;
}

#define read_idx(filename, ...) read_idx_opt((filename), (Read_Idx_Opts){ __VA_ARGS__ })

#define MICRO_TO_NS 1000ull
#define MILLI_TO_NS 1000000ull
#define SEC_TO_NS 1000000000ull
#define SEC_TO_US 1000000ull

int64_t get_system_micros(void) {
  struct timespec ts = {0, 0};
  timespec_get(&ts, TIME_UTC);
  return (int64_t)(ts.tv_sec * SEC_TO_US) + (int64_t)(ts.tv_nsec / MICRO_TO_NS);
}

int infer(Neuron_Network nn, NMatrix* outputs, NMatrix input) {
  forward(&nn, outputs, input);
  NMatrix output = outputs[nn.layers-1];

  return mat_row_max(output);
}

#endif //NN_H
