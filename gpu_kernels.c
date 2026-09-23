#ifdef OPENCL_VIM_HACK
#define TS 16
#define kernel
#define global
#define local
#define get_global_id(x) (x)
#define get_local_id(x) (x)
#define get_group_id(x) (x)
#define get_local_size(x) (x)
#define barrier(x)
#define exp(x) (x)
#define sqrt(x) (x)
#define pow(x) (x)
#define cos(x) (x)
#define sin(x) (x)
#define sincos(x, y) _sincos((x), (y))
#define bool int
#define uint int
#define size_t int
#define float2 Vector2
#define true 1
#define false 0
#define CLK_LOCAL_MEM_FENCE 1
#define INFINITY 0

typedef struct { float x, y; } Vector2;

float _sincos(float phi, float *cos_phi) {
  *cos_phi = cos(phi);
  return sin(phi);
}
#endif

typedef struct __attribute__((packed)) {
  uint rows;               // 4 bytes
  uint cols;               // 4 bytes
  uint stride;             // 4 bytes
  uint offset;             // 4 bytes
} MatrixData;             // 16 bytes

typedef struct __attribute__((packed)) {
  float v[32];
} InvFreq;

// void ERROR_branch_not_eliminated(void);
// if (__builtin_constant_p(is_acc)) {
//   ERROR_branch_not_eliminated();
// }

// 1D:  linear = id0
// 2D:  linear = id1 * size0 + id0
// 3D:  linear = id2 * size1 * size0 + id1 * size0 + id0
//
// for (d2 = 0; d2 < P; d2++) {
//   for (d1 = 0; d1 < M; d1++) {
//     for (d0 = 0; d0 < N; d0++) {
//       work_item(d0, d1, d2);
//     }
//   }
// }

// [NxM] = [NxP] x [PxM]
__attribute__((always_inline))
static inline
void matrix_mul_stride0(
  // outputs
  global       float* restrict buffer_c, MatrixData mat_c,
  // inputs
  global const float* restrict buffer_a, MatrixData mat_a,
  global const float* restrict buffer_b, MatrixData mat_b,
  local float Asub[TS][TS],
  local float Bsub[TS][TS],
  const bool is_acc
) {
  size_t globalCol = get_global_id(0); // Cols of B
  size_t globalRow = get_global_id(1); // Rows of A
  size_t localCol  = get_local_id(0);
  size_t localRow  = get_local_id(1);

  global const float* mat_a_ptr = buffer_a + mat_a.offset + globalRow * mat_a.stride;
  global       float* mat_c_ptr = buffer_c + mat_c.offset + globalRow * mat_c.stride;

  float acc = 0.0f;
  size_t numTiles = (mat_a.cols + TS - 1) / TS;

  for (size_t t = 0; t < numTiles; t++) {
    size_t tiledCol = t * TS + localCol;

    // Load A utilizing explicit A_stride spacing calculation
    if (globalRow < mat_a.rows && tiledCol < mat_a.cols) {
      Asub[localRow][localCol] = mat_a_ptr[tiledCol];
    } else {
      Asub[localRow][localCol] = 0.0f;
    }

    size_t tiledRow = t * TS + localRow;

    // Load B utilizing explicit B_stride spacing calculation
    if (tiledRow < mat_b.rows && globalCol < mat_b.cols) {
      global const float* mat_b_ptr = buffer_b + mat_b.offset + tiledRow * mat_b.stride;
      Bsub[localRow][localCol] = mat_b_ptr[globalCol];
    } else {
      Bsub[localRow][localCol] = 0.0f;
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    for (size_t k = 0; k < TS; k++) {
      acc += Asub[localRow][k] * Bsub[k][localCol];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // Store back into target slot safely using C_stride boundaries
  if (globalRow < mat_a.rows && globalCol < mat_b.cols) {
    if (is_acc) {
      mat_c_ptr[globalCol] += acc;
    } else {
      mat_c_ptr[globalCol] = acc;
    }
  }
}

kernel void matrix_mul_stride(
  // inputs
  global const float* restrict buffer_a, MatrixData mat_a,
  global const float* restrict buffer_b, MatrixData mat_b,
  // outputs
  global       float* restrict buffer_c, MatrixData mat_c
) {
  local float Asub[TS][TS];
  local float Bsub[TS][TS];

  matrix_mul_stride0(
      // outputs
      buffer_c, mat_c,
      // inputs
      buffer_a, mat_a,
      buffer_b, mat_b,
      Asub, Bsub,
      false
  );
}

kernel void matrix_mul_acc_stride(
  // inputs
  global const float* restrict buffer_a, MatrixData mat_a,
  global const float* restrict buffer_b, MatrixData mat_b,
  // outputs
  global       float* restrict buffer_c, MatrixData mat_c
) {
  local float Asub[TS][TS];
  local float Bsub[TS][TS];

  matrix_mul_stride0(
      // outputs
      buffer_c, mat_c,
      // inputs
      buffer_a, mat_a,
      buffer_b, mat_b,
      Asub, Bsub,
      true
  );
}

// [NxM] = [PxN]^T x [PxM]
__attribute__((always_inline))
static inline
void matrix_mul_At_stride0(
  // outputs
  global       float* restrict buffer_c, MatrixData mat_c,
  // inputs
  global const float* restrict buffer_a, MatrixData mat_a,
  global const float* restrict buffer_b, MatrixData mat_b,
  local float Asub[TS][TS],
  local float Bsub[TS][TS],
  const bool is_acc
) {
  size_t globalRow = get_global_id(0); // N (Rows of A^T and C)
  size_t globalCol = get_global_id(1); // M (Cols of B and C)
  size_t localRow  = get_local_id(0);
  size_t localCol  = get_local_id(1);

  global float* mat_c_ptr = buffer_c + mat_c.offset + globalRow * mat_c.stride;

  float acc = 0.0f;
  size_t numTiles = (mat_a.rows + TS - 1) / TS; // K dimension is A_rows in physical layout

  for (size_t t = 0; t < numTiles; t++) {
    size_t tiledCol = t * TS + localCol; // K index

    // Transposed Load: Physical rows and columns are swapped
    if (globalRow < mat_a.cols && tiledCol < mat_a.rows) {
      size_t mat_a_offset = mat_a.offset + tiledCol * mat_a.stride;
      Asub[localRow][localCol] = buffer_a[mat_a_offset + globalRow];
    } else {
      Asub[localRow][localCol] = 0.0f;
    }

    size_t tiledRow = t * TS + localRow; // K index
    if (tiledRow < mat_b.rows && globalCol < mat_b.cols) {
      size_t mat_b_offset = mat_b.offset + tiledRow * mat_b.stride;
      Bsub[localRow][localCol] = buffer_b[mat_b_offset + globalCol];
    } else {
      Bsub[localRow][localCol] = 0.0f;
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    for (size_t k = 0; k < TS; k++) {
      acc += Asub[localRow][k] * Bsub[k][localCol];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (globalRow < mat_a.cols && globalCol < mat_b.cols) {
    if (is_acc) {
      mat_c_ptr[globalCol] += acc;
    } else {
      mat_c_ptr[globalCol] = acc;
    }
  }
}


kernel void matrix_mul_At_stride(
  global const float* restrict buffer_a, MatrixData mat_a,
  global const float* restrict buffer_b, MatrixData mat_b,
  global       float* restrict buffer_c, MatrixData mat_c
) {
  local float Asub[TS][TS];
  local float Bsub[TS][TS];

  matrix_mul_At_stride0(
      // outputs
      buffer_c, mat_c,
      // inputs
      buffer_a, mat_a,
      buffer_b, mat_b,
      Asub, Bsub,
      false
  );
}

kernel void matrix_mul_At_acc_stride(
  global const float* restrict buffer_a, MatrixData mat_a,
  global const float* restrict buffer_b, MatrixData mat_b,
  global       float* restrict buffer_c, MatrixData mat_c
) {
  local float Asub[TS][TS];
  local float Bsub[TS][TS];

  matrix_mul_At_stride0(
      // outputs
      buffer_c, mat_c,
      // inputs
      buffer_a, mat_a,
      buffer_b, mat_b,
      Asub, Bsub,
      true
  );
}

// [NxM] = [NxP] x [MxP]^T
__attribute__((always_inline))
static inline
void matrix_mul_Bt_stride0(
  // outputs
  global       float* restrict buffer_c, MatrixData mat_c,
  // inputs
  global const float* restrict buffer_a, MatrixData mat_a,
  global const float* restrict buffer_b, MatrixData mat_b,
  local float Asub[TS][TS],
  local float Bsub[TS][TS],
  const bool is_acc
) {
  size_t globalCol = get_global_id(0); // Rows of B, which are columns of B^T
  size_t globalRow = get_global_id(1); // Rows of A and C
  size_t localCol  = get_local_id(0);
  size_t localRow  = get_local_id(1);

  global const float* mat_a_ptr = buffer_a + mat_a.offset + globalRow * mat_a.stride;
  global const float* mat_b_ptr = buffer_b + mat_b.offset + globalCol * mat_b.stride;
  global       float* mat_c_ptr = buffer_c + mat_c.offset + globalRow * mat_c.stride;

  float acc = 0.0f;
  size_t numTiles = (mat_a.cols + TS - 1) / TS; // K dimension

  for (size_t t = 0; t < numTiles; t++) {
    size_t tiledCol = t * TS + localCol; // K index
    if (globalRow < mat_a.rows && tiledCol < mat_a.cols) {
      Asub[localRow][localCol] = mat_a_ptr[tiledCol];
    } else {
      Asub[localRow][localCol] = 0.0f;
    }

    size_t tiledRow = t * TS + localRow; // K index

    // Transposed Load: Access B along rows instead of columns
    if (tiledRow < mat_b.cols && globalCol < mat_b.rows) {
      Bsub[localRow][localCol] = mat_b_ptr[tiledRow];
    } else {
      Bsub[localRow][localCol] = 0.0f;
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    for (size_t k = 0; k < TS; k++) {
      acc += Asub[localRow][k] * Bsub[k][localCol];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (globalRow < mat_a.rows && globalCol < mat_b.rows) {
    if (is_acc) {
      mat_c_ptr[globalCol] += acc;
    } else {
      mat_c_ptr[globalCol] = acc;
    }
  }
}

kernel void matrix_mul_Bt_stride(
  global const float* restrict buffer_a, MatrixData mat_a,
  global const float* restrict buffer_b, MatrixData mat_b,
  global       float* restrict buffer_c, MatrixData mat_c
) {
  local float Asub[TS][TS];
  local float Bsub[TS][TS];

  matrix_mul_Bt_stride0(
      // outputs
      buffer_c, mat_c,
      // inputs
      buffer_a, mat_a,
      buffer_b, mat_b,
      Asub, Bsub,
      false
  );
}

kernel void matrix_mul_Bt_acc_stride(
  global const float* restrict buffer_a, MatrixData mat_a,
  global const float* restrict buffer_b, MatrixData mat_b,
  global       float* restrict buffer_c, MatrixData mat_c
) {
  local float Asub[TS][TS];
  local float Bsub[TS][TS];

  matrix_mul_Bt_stride0(
      // outputs
      buffer_c, mat_c,
      // inputs
      buffer_a, mat_a,
      buffer_b, mat_b,
      Asub, Bsub,
      true
  );
}


kernel void matrix_add_kernel(
  // outputs
  global       float* restrict buffer_out, MatrixData mat_out,
  // inputs
  global const float* restrict buffer_a,   MatrixData mat_a,
  global const float* restrict buffer_b,   MatrixData mat_b
) {
  size_t r = get_group_id(1);
  if (r >= mat_out.rows) return;

  size_t local_col  = get_local_id(0);
  size_t local_size = get_local_size(0);

  global float* out_ptr = buffer_out + mat_out.offset + r * mat_out.stride;
  global const float* a_ptr = buffer_a + mat_a.offset + r * mat_a.stride;
  global const float* b_ptr = buffer_b + mat_b.offset + r * mat_b.stride;

  for (size_t d = local_col; d < mat_out.cols; d += local_size) {
    out_ptr[d] = a_ptr[d] + b_ptr[d];
  }
}

// h = softmax(x)
//
// local  = {256, 1}
// global = {256, N}
// group  = { [0..0], [0..N-1] }
kernel void softmax_temperature_kernel(
  // outputs
  global       float* restrict buffer_out, MatrixData mat_out,
  // inputs
  global const float* restrict buffer_x,   MatrixData mat_x,
  float temperature,
  // scratch
  local        float* local_scratch
) {
  size_t r = get_group_id(1);
  if (r >= mat_out.rows) return;

  size_t local_col  = get_local_id(0);
  size_t local_size = get_local_size(0);
  float inv_t = 1.0f / temperature;

  size_t x_offset   = mat_x.offset   + r * mat_x.stride;
  size_t out_offset = mat_out.offset + r * mat_out.stride;

  // 1. Find the maximum scaled logit for numerical stability
  float max_logit_thread = -INFINITY;
  for (size_t i = local_col; i < mat_x.cols; i += local_size) {
    float logit = buffer_x[x_offset + i] * inv_t;
    if (logit > max_logit_thread) max_logit_thread = logit;
  }

  local_scratch[local_col] = max_logit_thread;
  barrier(CLK_LOCAL_MEM_FENCE);

  // max-reduce
  for (size_t stride = local_size / 2; stride > 0; stride /= 2) {
    if (local_col < stride) {
      if (local_scratch[local_col + stride] > local_scratch[local_col])
        local_scratch[local_col] = local_scratch[local_col + stride];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  float max_logit = local_scratch[0];
  barrier(CLK_LOCAL_MEM_FENCE);

  // 2. Compute the sum of exponentials (shifted)
  float sum_exp_thread = 0.0f;
  for (size_t i = local_col; i < mat_x.cols; i += local_size) {
    float e = exp(buffer_x[x_offset + i] * inv_t - max_logit);
    buffer_out[out_offset + i] = e;
    sum_exp_thread += e;
  }

  local_scratch[local_col] = sum_exp_thread;
  barrier(CLK_LOCAL_MEM_FENCE);

  // sum-reduce
  for (size_t stride = local_size / 2; stride > 0; stride /= 2) {
    if (local_col < stride) {
      local_scratch[local_col] += local_scratch[local_col + stride];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  float sum_exp = local_scratch[0];

  // 3. Calculate Forward Softmax Probabilities
  float inv_sum = 1.0f / sum_exp;
  for (size_t i = local_col; i < mat_x.cols; i += local_size) {
    buffer_out[out_offset + i] *= inv_sum;
  }
}

kernel void softmax_with_masking_kernel(
  // outputs
  global       float* restrict buffer_out, MatrixData mat_out,
  // inputs
  global       float* restrict buffer_x,   MatrixData mat_x,
  float temperature, float scale, int base, int total,
  // scratch
  local        float* local_scratch
) {
  size_t r = get_group_id(1);
  if (r >= mat_out.rows) return;

  size_t local_col  = get_local_id(0);
  size_t local_size = get_local_size(0);

  global float* x_ptr = buffer_x + mat_x.offset + r * mat_x.stride;

  for (size_t i = local_col; i < mat_x.cols; i += local_size) {
    if (i >= (size_t)base+r+1 && i < (size_t)total)
      x_ptr[i] = -INFINITY;
    else
      x_ptr[i] *= scale;
  }

  softmax_temperature_kernel(
      buffer_out, mat_out,
      buffer_x, mat_x,
      temperature,
      local_scratch
  );
}


// h  = softmax(x)
// dz = 1/t * h * [ dL - dot(h, dL) ]
//
// local  = {256, 1}
// global = {256, N}
// group  = { [0..0], [0..N-1] }
kernel void dsoftmax_temperature_kernel(
  // outputs
  global       float* restrict buffer_dz, MatrixData mat_dz,
  // inputs
  global const float* restrict buffer_h,  MatrixData mat_h,
  global const float* restrict buffer_dL, MatrixData mat_dL,
  local        float* local_scratch,
  float temperature
) {
  size_t r = get_group_id(1);
  if (r >= mat_dz.rows) return;

  size_t local_col  = get_local_id(0);
  size_t local_size = get_local_size(0);

  size_t h_offset  = mat_h.offset  + r * mat_h.stride;
  size_t dz_offset = mat_dz.offset + r * mat_dz.stride;
  size_t dL_offset = mat_dL.offset + r * mat_dL.stride;

  // 1. dot product
  float dot_thread = 0.0f;
  for (size_t i = local_col; i < mat_dz.cols; i += local_size) {
    dot_thread += buffer_h[h_offset + i] * buffer_dL[dL_offset + i];
  }

  local_scratch[local_col] = dot_thread;
  barrier(CLK_LOCAL_MEM_FENCE);

  // sum-reduce
  for (size_t stride = local_size / 2; stride > 0; stride /= 2) {
    if (local_col < stride) {
      local_scratch[local_col] += local_scratch[local_col + stride];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  float dot = local_scratch[0];

  float inv_t = 1.0f / temperature;

  for (size_t i = local_col; i < mat_dz.cols; i += local_size) {
    buffer_dz[dz_offset + i] = inv_t * buffer_h[h_offset + i] * (buffer_dL[dL_offset + i] - dot);
  }
}

// local  = {256, 1}
// global = {256, N}
// group  = { [0..0], [0..N-1] }
kernel void layer_norm_forward_kernel(
  // outputs
  global float* restrict buffer_out, MatrixData mat_out,
  global float* restrict buffer_mean, MatrixData mat_mean,
  global float* restrict buffer_rstd, MatrixData mat_rstd,
  global float* restrict buffer_xhat, MatrixData mat_xhat,
  // inputs
  global const float* restrict buffer_in, MatrixData mat_in,
  global const float* restrict buffer_gamma, MatrixData mat_gamma,
  global const float* restrict buffer_beta, MatrixData mat_beta,
  local        float* local_scratch
) {
  size_t r = get_group_id(1);
  if (r >= mat_out.rows) return;

  size_t local_col  = get_local_id(0);
  size_t local_size = get_local_size(0);

  // [NxD]
  size_t in_offset    = mat_in.offset    + r * mat_in.stride;
  size_t out_offset   = mat_out.offset   + r * mat_out.stride;
  size_t xhat_offset  = mat_xhat.offset  + r * mat_xhat.stride;

  // [1xN]
  size_t mean_offset  = mat_mean.offset + r;
  size_t rstd_offset  = mat_rstd.offset + r;

  // [1xD]
  size_t gamma_offset = mat_gamma.offset;
  size_t beta_offset  = mat_beta.offset;

  // mean
  float mean_thread = 0.0f;
  for (size_t d = local_col; d < mat_out.cols; d += local_size) {
    mean_thread += buffer_in[in_offset + d];
  }

  local_scratch[local_col] = mean_thread;
  barrier(CLK_LOCAL_MEM_FENCE);

  // sum-reduce
  for (size_t stride = local_size / 2; stride > 0; stride /= 2) {
    if (local_col < stride) {
      local_scratch[local_col] += local_scratch[local_col + stride];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  float mean = local_scratch[0] / mat_out.cols;
  barrier(CLK_LOCAL_MEM_FENCE);

  // var
  float var_thread = 0.0f;
  for (size_t d = local_col; d < mat_out.cols; d += local_size) {
    float diff = buffer_in[in_offset + d] - mean;
    var_thread += diff * diff;
  }

  local_scratch[local_col] = var_thread;
  barrier(CLK_LOCAL_MEM_FENCE);

  // sum-reduce
  for (size_t stride = local_size / 2; stride > 0; stride /= 2) {
    if (local_col < stride) {
      local_scratch[local_col] += local_scratch[local_col + stride];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  float var = local_scratch[0] / mat_out.cols;
  float rstd = 1.0f / sqrt(var + 1e-5f);

  if (local_col == 0) {
    buffer_mean[mean_offset] = mean;
    buffer_rstd[rstd_offset] = rstd;
  }

  for (size_t d = local_col; d < mat_out.cols; d += local_size) {
    float hat = (buffer_in[in_offset + d] - mean) * rstd;
    buffer_xhat[xhat_offset + d] = hat;
    buffer_out[out_offset + d] = buffer_gamma[gamma_offset + d] * hat + buffer_beta[beta_offset + d];
  }
}


// local  = {256, 1}
// global = {256, N}
// group  = { [0..0], [0..N-1] }
kernel void layer_norm_backward_dx_kernel(
  // outpus
  global float* restrict buffer_din, MatrixData mat_din,
  // inputs
  global const float* restrict buffer_dout, MatrixData mat_dout,
  global const float* restrict buffer_gamma, MatrixData mat_gamma,
  global const float* restrict buffer_rstd, MatrixData mat_rstd,
  global const float* restrict buffer_xhat, MatrixData mat_xhat,
  local        float* local_scratch0,
  local        float* local_scratch1
) {
  size_t r = get_group_id(1);
  if (r >= mat_din.rows) return;

  size_t local_col  = get_local_id(0);
  size_t local_size = get_local_size(0);

  // [NxD]
  size_t din_offset   = mat_din.offset  + r * mat_din.stride;
  size_t dout_offset  = mat_dout.offset + r * mat_dout.stride;
  size_t xhat_offset  = mat_xhat.offset + r * mat_xhat.stride;

  // [1xN]
  size_t rstd_offset  = mat_rstd.offset + r;

  // [1xD]
  size_t gamma_offset  = mat_gamma.offset;

  float sum_dx_hat_thread = 0;
  float sum_dx_hat_x_hat_thread = 0;
  for (size_t d = local_col; d < mat_din.cols; d += local_size) {
    float dx_hat = buffer_dout[dout_offset + d] * buffer_gamma[gamma_offset + d];
    sum_dx_hat_thread += dx_hat;
    sum_dx_hat_x_hat_thread += dx_hat * buffer_xhat[xhat_offset + d];
  }

  local_scratch0[local_col] = sum_dx_hat_thread;
  local_scratch1[local_col] = sum_dx_hat_x_hat_thread;
  barrier(CLK_LOCAL_MEM_FENCE);

  // sum-reduce
  for (size_t stride = local_size / 2; stride > 0; stride /= 2) {
    if (local_col < stride) {
      local_scratch0[local_col] += local_scratch0[local_col + stride];
      local_scratch1[local_col] += local_scratch1[local_col + stride];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  float sum_dx_hat = local_scratch0[0];
  float sum_dx_hat_x_hat = local_scratch1[0];
  barrier(CLK_LOCAL_MEM_FENCE);

  float rstd = buffer_rstd[rstd_offset];

  for (size_t d = local_col; d < mat_din.cols; d += local_size) {
    float dx_hat = buffer_dout[dout_offset + d] * buffer_gamma[gamma_offset + d];
    float dx = dx_hat - sum_dx_hat/mat_din.cols - buffer_xhat[xhat_offset + d] * sum_dx_hat_x_hat/mat_din.cols;
    
    buffer_din[din_offset + d] += dx * rstd;
  }
}

// local  = {256, 1}
// global = {256, D}
// group  = { [0..0], [0..D-1] }
kernel void layer_norm_backward_dgamma_dbeta_kernel(
  // outpus
  global float* restrict buffer_dgamma, MatrixData mat_dgamma,
  global float* restrict buffer_dbeta, MatrixData mat_dbeta,
  // inputs
  global const float* restrict buffer_dout, MatrixData mat_dout,
  global const float* restrict buffer_xhat, MatrixData mat_xhat
) {
  size_t d = get_group_id(0);
  if (d >= mat_dout.cols) return;

  float dgamma = 0;
  float dbeta = 0;

  for (size_t r = 0; r < mat_dout.rows; r++) {
    size_t dout_offset = mat_dout.offset + r * mat_dout.stride + d;
    size_t xhat_offset = mat_xhat.offset + r * mat_xhat.stride + d;

    dgamma += buffer_dout[dout_offset] * buffer_xhat[xhat_offset];
    dbeta += buffer_dout[dout_offset];
  }

  size_t dgamma_offset = mat_dgamma.offset + d;
  size_t dbeta_offset = mat_dbeta.offset + d;

  buffer_dgamma[dgamma_offset] = dgamma;
  buffer_dbeta[dbeta_offset] = dbeta;
}

kernel void rope_kernel(
  // outputs
  global float* restrict buffer_qk, MatrixData mat_qk,
  // inputs
  int base, int d_head,
  InvFreq inv_freq
) {
  size_t p = get_global_id(0);
  size_t r = get_global_id(1);
  if (r >= mat_qk.rows || p >= (mat_qk.cols >> 1)) return;

  size_t i = p << 1;
  size_t local_i = i % d_head;

  size_t qk_offset = mat_qk.offset + r * mat_qk.stride;

  float inv_d = -1.0f / (float)d_head;
  const float log_base = 9.21034037197f;

  float theta = exp((float)local_i * inv_d * log_base);
  // float theta = inv_freq.v[local_i >> 1];
  float phi = (float)(base + r) * theta;

  global float2* qk_ptr = (global float2*)(buffer_qk + qk_offset + i);
  float2 qk = *qk_ptr;

  float cos_phi;
  float sin_phi = sincos(phi, &cos_phi);

  float2 qk_rot;
  qk_rot.x = qk.x*cos_phi - qk.y*sin_phi;
  qk_rot.y = qk.x*sin_phi + qk.y*cos_phi;

  *qk_ptr = qk_rot;
}

kernel void drope_kernel(
  // outputs
  global float* restrict buffer_dqk, MatrixData mat_dqk,
  // inputs
  global const float* restrict buffer_dout, MatrixData mat_dout,
  int base, int d_head,
  InvFreq inv_freq
) {
  size_t p = get_global_id(0);
  size_t r = get_global_id(1);
  if (r >= mat_dqk.rows || p >= (mat_dqk.cols >> 1)) return;

  size_t i = p << 1;
  size_t local_i = i % d_head;

  size_t dqk_offset  = mat_dqk.offset  + r * mat_dqk.stride;
  size_t dout_offset = mat_dout.offset + r * mat_dout.stride;

  float theta = inv_freq.v[local_i >> 1];
  float phi = (float)(base + r) * theta;

  global float2* dqk_ptr  = (global float2*)(buffer_dqk + dqk_offset + i);
  global float2* dout_ptr = (global float2*)(buffer_dout + dout_offset + i);

  float2 dout = *dout_ptr;

  float cos_phi;
  float sin_phi = sincos(phi, &cos_phi);

  float2 dqk_rot;
  dqk_rot.x = dout.x*cos_phi + dout.y*sin_phi;
  dqk_rot.y = dout.y*cos_phi - dout.x*sin_phi;

  *dqk_ptr = dqk_rot;
}

kernel void relu_kernel(
  // outputs
  global       float* restrict buffer_dst, MatrixData mat_dst,
  // inputs
  global const float* restrict buffer_x, MatrixData mat_x,
  float a
) {
  size_t d = get_global_id(0);
  size_t r = get_global_id(1);
  if (r >= mat_dst.rows || d >= mat_dst.cols) return;

  size_t dst_offset = mat_dst.offset + r * mat_dst.stride + d;
  size_t x_offset   = mat_x.offset   + r * mat_x.stride   + d;

  float x = buffer_x[x_offset];
  buffer_dst[dst_offset] = x > 0 ? x : x*a;
}

kernel void drelu_kernel(
  // outputs
  global       float* restrict buffer_dst, MatrixData mat_dst,
  // inputs
  global const float* restrict buffer_x, MatrixData mat_x,
  float a
) {
  size_t d = get_global_id(0);
  size_t r = get_global_id(1);
  if (r >= mat_dst.rows || d >= mat_dst.cols) return;

  size_t dst_offset = mat_dst.offset + r * mat_dst.stride + d;
  size_t x_offset   = mat_x.offset   + r * mat_x.stride   + d;

  float x = buffer_x[x_offset];
  buffer_dst[dst_offset] = x > 0 ? 1 : a;
}

// [NxD] = [NxD] x [DxD] + [1xD]
kernel void broadcast_bias_kernel(
  // outputs
  global       float* restrict buffer_out, MatrixData mat_out,
  // inputs
  global const float* restrict buffer_b, MatrixData mat_b
) {
  size_t d = get_global_id(0);
  size_t r = get_global_id(1);
  if (r >= mat_out.rows || d >= mat_out.cols) return;

  size_t local_col = get_local_id(0);
  size_t local_size = get_local_size(0);

  size_t out_offset = mat_out.offset + r * mat_out.stride;
  size_t b_offset = mat_b.offset + d;

  buffer_out[out_offset] += buffer_b[b_offset];
}

kernel void dproject_db_kernel(
  // outputs
  global       float* restrict buffer_dx, MatrixData mat_dx,
  global       float* restrict buffer_dW, MatrixData mat_dW,
  global       float* restrict buffer_db, MatrixData mat_db,
  // inputs
  global const float* restrict buffer_x, MatrixData mat_x,
  global const float* restrict buffer_W, MatrixData mat_W,
  global const float* restrict buffer_dout, MatrixData mat_dout,
  local        float* local_scratch
) {
  // db += dout
  size_t d = get_group_id(0);
  if (d >= mat_db.cols) return;

  size_t local_row = get_local_id(1);
  size_t local_size = get_local_size(1);

  float db_thread = 0.0f;

  for (size_t r = local_row; r < mat_dout.rows; r += local_size) {
    size_t dout_offset = mat_dout.offset + r * mat_dout.stride;
    db_thread += buffer_dout[dout_offset + d];
  }

  local_scratch[local_row] = db_thread;
  barrier(CLK_LOCAL_MEM_FENCE);

  // sum-reduce
  for (size_t stride = local_size / 2; stride > 0; stride /= 2) {
    if (local_row < stride) {
      local_scratch[local_row] += local_scratch[local_row + stride];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  float db = local_scratch[0];
  barrier(CLK_LOCAL_MEM_FENCE);

  if (local_row == 0) {
    size_t db_offset = mat_db.offset + d;
    buffer_db[db_offset] = db;
  }
}

kernel void embed_gatter_forward_kernel(
  // outputs
  global       float* restrict buffer_out, MatrixData mat_out,
  // inputs
  global const float* restrict buffer_emb, MatrixData mat_emb,
  global const int* restrict tokens,
  int token_base
) {
  size_t d = get_global_id(0);
  size_t r = get_global_id(1);
  if (r >= mat_out.rows || d >= mat_out.cols) return;

  size_t token = tokens[token_base + r];
  size_t out_offset = mat_out.offset + r * mat_out.stride     + d;
  size_t emb_offset = mat_emb.offset + token * mat_emb.stride + d;

  buffer_out[out_offset] = buffer_emb[emb_offset];
}

kernel void row_argmax_kernel(
  // outputs
  global int* tokens_buffer,
  global float* probs_buffer,
  global int* done_buffer,
  int tokens_base,
  // inputs
  global const float* buffer_probs, MatrixData mat_probs,
  int last_token,
  int eos_token,
  // scracth
  local float* local_scratch0,
  local int* local_scratch1
) {
  size_t local_col  = get_local_id(0);
  size_t local_size = get_local_size(0);

  global const float* probs_ptr = buffer_probs + mat_probs.offset + last_token * mat_probs.stride;

  float max_prob_thread = -INFINITY;
  int max_index_thread = 0;
  for (size_t i = local_col; i < mat_probs.cols; i += local_size) {
    float prob = probs_ptr[i];
    if (prob > max_prob_thread) {
      max_prob_thread = prob;
      max_index_thread = i;
    }
  }

  local_scratch0[local_col] = max_prob_thread;
  local_scratch1[local_col] = max_index_thread;
  barrier(CLK_LOCAL_MEM_FENCE);

  // max-reduce
  for (size_t stride = local_size / 2; stride > 0; stride /= 2) {
    if (local_col < stride) {
      float v0 = local_scratch0[local_col];
      int   j0 = local_scratch1[local_col];

      float v1 = local_scratch0[local_col + stride];
      int   j1 = local_scratch1[local_col + stride];

      if (v1 > v0 || (v1 == v0 && j1 < j0)) {
        local_scratch0[local_col] = v1;
        local_scratch1[local_col] = j1;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  float max_prob = local_scratch0[0];
  size_t index = local_scratch1[0];

  if (local_col == 0) {
    tokens_buffer[tokens_base] = index;
    probs_buffer[tokens_base] = max_prob;

    if (index == eos_token)
      *done_buffer = 1;
  }
}

kernel void gemm_nn_debug_kernel(
  // output
  global float *C, MatrixData mc,
  // input
  global const float *A, MatrixData ma,
  global const float *B, MatrixData mb
) {
  size_t col = get_global_id(0);
  size_t row = get_global_id(1);

  if (row >= mc.rows || col >= mc.cols) return;

  float sum = 0.0f;
  for (size_t k = 0; k < ma.cols; k++)
    sum += A[ma.offset + row*ma.stride + k] * B[mb.offset + k*mb.stride + col];

  C[mc.offset + row*mc.stride + col] = sum;
}
