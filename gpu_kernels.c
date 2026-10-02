typedef struct __attribute__((packed)) {
  uint rows;               // 4 bytes
  uint cols;               // 4 bytes
  uint stride;             // 4 bytes
  uint offset;             // 4 bytes
} MatrixData;             // 16 bytes

// [NxM] = [NxP] x [PxM]
kernel void matrix_mul_stride(
  // inputs
  global const float* restrict buffer_a, MatrixData mat_a,
  global const float* restrict buffer_b, MatrixData mat_b,
  // outputs
  global       float* restrict buffer_c, MatrixData mat_c
) {
  local float Asub[TS][TS];
  local float Bsub[TS][TS];

  int globalRow = get_global_id(0); // N (Rows of A)
  int globalCol = get_global_id(1); // M (Cols of B)
  int localRow  = get_local_id(0);
  int localCol  = get_local_id(1);

  int mat_a_offset = mat_a.offset + globalRow * mat_a.stride;
  int mat_c_offset = mat_c.offset + globalRow * mat_c.stride;

  float acc = 0.0f;
  int numTiles = (mat_a.cols + TS - 1) / TS;

  for (int t = 0; t < numTiles; t++) {
    int tiledCol = t * TS + localCol;

    // Load A utilizing explicit A_stride spacing calculation
    if (globalRow < mat_a.rows && tiledCol < mat_a.cols) {
      Asub[localRow][localCol] = buffer_a[mat_a_offset + tiledCol];
    } else {
      Asub[localRow][localCol] = 0.0f;
    }

    int tiledRow = t * TS + localRow;

    // Load B utilizing explicit B_stride spacing calculation
    if (tiledRow < mat_b.rows && globalCol < mat_b.cols) {
      int mat_b_offset = mat_b.offset + tiledRow * mat_b.stride;
      Bsub[localRow][localCol] = buffer_b[mat_b_offset + globalCol];
    } else {
      Bsub[localRow][localCol] = 0.0f;
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    for (int k = 0; k < TS; k++) {
      acc += Asub[localRow][k] * Bsub[k][localCol];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // Store back into target slot safely using C_stride boundaries
  if (globalRow < mat_a.rows && globalCol < mat_b.cols) {
    buffer_c[mat_c_offset + globalCol] = acc;
  }
}

// [NxM] = [PxN]^T x [PxM]
kernel void matrix_mul_At_stride(
  global const float* restrict buffer_a, MatrixData mat_a,
  global const float* restrict buffer_b, MatrixData mat_b,
  global       float* restrict buffer_c, MatrixData mat_c
) {
  local float Asub[TS][TS];
  local float Bsub[TS][TS];

  int globalRow = get_global_id(0); // N (Rows of A^T and C)
  int globalCol = get_global_id(1); // M (Cols of B and C)
  int localRow  = get_local_id(0);
  int localCol  = get_local_id(1);

  int mat_c_offset = mat_c.offset + globalRow * mat_c.stride;

  float acc = 0.0f;
  int numTiles = (mat_a.rows + TS - 1) / TS; // K dimension is A_rows in physical layout

  for (int t = 0; t < numTiles; t++) {
    int tiledCol = t * TS + localCol; // K index

    // Transposed Load: Physical rows and columns are swapped
    if (globalRow < mat_a.cols && tiledCol < mat_a.rows) {
      int mat_a_offset = mat_a.offset + tiledCol * mat_a.stride;
      Asub[localRow][localCol] = buffer_a[mat_a_offset + globalRow];
    } else {
      Asub[localRow][localCol] = 0.0f;
    }

    int tiledRow = t * TS + localRow; // K index
    if (tiledRow < mat_b.rows && globalCol < mat_b.cols) {
      int mat_b_offset = mat_b.offset + tiledRow * mat_b.stride;
      Bsub[localRow][localCol] = buffer_b[mat_b_offset + globalCol];
    } else {
      Bsub[localRow][localCol] = 0.0f;
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    for (int k = 0; k < TS; k++) {
      acc += Asub[localRow][k] * Bsub[k][localCol];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (globalRow < mat_a.cols && globalCol < mat_b.cols) {
    buffer_c[mat_c_offset + globalCol] = acc;
  }
}

// [NxM] = [NxP] x [MxP]^T
kernel void matrix_mul_Bt_stride(
  global const float* restrict buffer_a, MatrixData mat_a,
  global const float* restrict buffer_b, MatrixData mat_b,
  global       float* restrict buffer_c, MatrixData mat_c
) {
  local float Asub[TS][TS];
  local float Bsub[TS][TS];

  int globalRow = get_global_id(0); // N (Rows of A and C)
  int globalCol = get_global_id(1); // M (Rows of B, which are columns of B^T)
  int localRow  = get_local_id(0);
  int localCol  = get_local_id(1);

  int mat_a_offset = mat_a.offset + globalRow * mat_a.stride;
  int mat_b_offset = mat_b.offset + globalCol * mat_b.stride;
  int mat_c_offset = mat_c.offset + globalRow * mat_c.stride;

  float acc = 0.0f;
  int numTiles = (mat_a.cols + TS - 1) / TS; // K dimension

  for (int t = 0; t < numTiles; t++) {
    int tiledCol = t * TS + localCol; // K index
    if (globalRow < mat_a.rows && tiledCol < mat_a.cols) {
      Asub[localRow][localCol] = buffer_a[mat_a_offset + tiledCol];
    } else {
      Asub[localRow][localCol] = 0.0f;
    }

    int tiledRow = t * TS + localRow; // K index

    // Transposed Load: Access B along rows instead of columns
    if (tiledRow < mat_b.cols && globalCol < mat_b.rows) {
      Bsub[localRow][localCol] = buffer_b[mat_b_offset + tiledRow];
    } else {
      Bsub[localRow][localCol] = 0.0f;
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    for (int k = 0; k < TS; k++) {
      acc += Asub[localRow][k] * Bsub[k][localCol];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (globalRow < mat_a.rows && globalCol < mat_b.rows) {
    buffer_c[mat_c_offset + globalCol] = acc;
  }
}

kernel void matrix_add_kernel(
  // outputs
  global       float* restrict buffer_out, MatrixData mat_out,
  // inputs
  global const float* restrict buffer_a,   MatrixData mat_a,
  global const float* restrict buffer_b,   MatrixData mat_b
) {
  int r = get_group_id(1);
  if (r >= mat_out.rows) return;

  int local_col  = get_local_id(0);
  int local_size = get_local_size(0);

  global float* out_ptr = buffer_out + mat_out.offset + r * mat_out.stride;
  global const float* a_ptr = buffer_a + mat_a.offset + r * mat_a.stride;
  global const float* b_ptr = buffer_b + mat_b.offset + r * mat_b.stride;

  for (int d = local_col; d < mat_out.cols; d += local_size) {
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
  int r = get_group_id(1);
  if (r >= mat_out.rows) return;

  int local_col  = get_local_id(0);
  int local_size = get_local_size(0);
  float inv_t = 1.0f / temperature;

  int x_offset   = mat_x.offset   + r * mat_x.stride;
  int out_offset = mat_out.offset + r * mat_out.stride;

  // 1. Find the maximum scaled logit for numerical stability
  float max_logit_thread = -INFINITY;
  for (int i = local_col; i < mat_x.cols; i += local_size) {
    float logit = buffer_x[x_offset + i] * inv_t;
    if (logit > max_logit_thread) max_logit_thread = logit;
  }

  local_scratch[local_col] = max_logit_thread;
  barrier(CLK_LOCAL_MEM_FENCE);

  // max-reduce
  for (int stride = local_size / 2; stride > 0; stride /= 2) {
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
  for (int i = local_col; i < mat_x.cols; i += local_size) {
    float e = exp(buffer_x[x_offset + i] * inv_t - max_logit);
    buffer_out[out_offset + i] = e;
    sum_exp_thread += e;
  }

  local_scratch[local_col] = sum_exp_thread;
  barrier(CLK_LOCAL_MEM_FENCE);

  // sum-reduce
  for (int stride = local_size / 2; stride > 0; stride /= 2) {
    if (local_col < stride) {
      local_scratch[local_col] += local_scratch[local_col + stride];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  float sum_exp = local_scratch[0];

  // 3. Calculate Forward Softmax Probabilities
  float inv_sum = 1.0f / sum_exp;
  for (int i = local_col; i < mat_x.cols; i += local_size) {
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
  int r = get_group_id(1);
  if (r >= mat_out.rows) return;

  int local_col  = get_local_id(0);
  int local_size = get_local_size(0);

  global float* x_ptr = buffer_x + mat_x.offset + r * mat_x.stride;

  for (int i = local_col; i < mat_x.cols; i += local_size) {
    if (i >= base+r+1 && i < total)
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
  int r = get_group_id(1);
  if (r >= mat_dz.rows) return;

  int local_col  = get_local_id(0);
  int local_size = get_local_size(0);

  int h_offset  = mat_h.offset  + r * mat_h.stride;
  int dz_offset = mat_dz.offset + r * mat_dz.stride;
  int dL_offset = mat_dL.offset + r * mat_dL.stride;

  // 1. dot product
  float dot_thread = 0.0f;
  for (int i = local_col; i < mat_dz.cols; i += local_size) {
    dot_thread += buffer_h[h_offset + i] * buffer_dL[dL_offset + i];
  }

  local_scratch[local_col] = dot_thread;
  barrier(CLK_LOCAL_MEM_FENCE);

  // sum-reduce
  for (int stride = local_size / 2; stride > 0; stride /= 2) {
    if (local_col < stride) {
      local_scratch[local_col] += local_scratch[local_col + stride];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  float dot = local_scratch[0];

  float inv_t = 1.0f / temperature;

  for (int i = local_col; i < mat_dz.cols; i += local_size) {
    buffer_dz[dz_offset + i] += inv_t * buffer_h[h_offset + i] * (buffer_dL[dL_offset + i] - dot);
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
  int r = get_group_id(1);
  if (r >= mat_out.rows) return;

  int local_col  = get_local_id(0);
  int local_size = get_local_size(0);

  // [NxD]
  int in_offset    = mat_in.offset    + r * mat_in.stride;
  int out_offset   = mat_out.offset   + r * mat_out.stride;
  int xhat_offset  = mat_xhat.offset  + r * mat_xhat.stride;

  // [1xN]
  int mean_offset  = mat_mean.offset + r;
  int rstd_offset  = mat_rstd.offset + r;

  // [1xD]
  int gamma_offset = mat_gamma.offset;
  int beta_offset  = mat_beta.offset;

  // mean
  float mean_thread = 0.0f;
  for (int d = local_col; d < mat_out.cols; d += local_size) {
    mean_thread += buffer_in[in_offset + d];
  }

  local_scratch[local_col] = mean_thread;
  barrier(CLK_LOCAL_MEM_FENCE);

  // sum-reduce
  for (int stride = local_size / 2; stride > 0; stride /= 2) {
    if (local_col < stride) {
      local_scratch[local_col] += local_scratch[local_col + stride];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  float mean = local_scratch[0] / mat_out.cols;
  buffer_mean[mean_offset] = mean;

  // var
  float var_thread = 0.0f;
  for (int d = local_col; d < mat_out.cols; d += local_size) {
    float diff = buffer_in[in_offset + d] - mean;
    var_thread += diff * diff;
  }

  local_scratch[local_col] = var_thread;
  barrier(CLK_LOCAL_MEM_FENCE);

  // sum-reduce
  for (int stride = local_size / 2; stride > 0; stride /= 2) {
    if (local_col < stride) {
      local_scratch[local_col] += local_scratch[local_col + stride];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  float var = local_scratch[0] / mat_out.cols;
  float rstd = 1.0f / sqrt(var);

  buffer_rstd[rstd_offset] = rstd;

  for (int d = local_col; d < mat_out.cols; d += local_size) {
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
  int r = get_group_id(1);
  if (r >= mat_din.rows) return;

  int local_col  = get_local_id(0);
  int local_size = get_local_size(0);

  // [NxD]
  int din_offset   = mat_din.offset  + r * mat_din.stride;
  int dout_offset  = mat_dout.offset + r * mat_dout.stride;
  int xhat_offset  = mat_xhat.offset + r * mat_xhat.stride;

  // [1xN]
  int rstd_offset  = mat_rstd.offset + r;

  // [1xD]
  int gamma_offset  = mat_gamma.offset;

  float sum_dx_hat_thread = 0;
  float sum_dx_hat_x_hat_thread = 0;
  for (int d = local_col; d < mat_din.cols; d += local_size) {
    float dx_hat = buffer_dout[dout_offset + d] * buffer_gamma[gamma_offset + d];
    sum_dx_hat_thread += dx_hat;
    sum_dx_hat_x_hat_thread += dx_hat * buffer_xhat[xhat_offset + d];
  }

  local_scratch0[local_col] = sum_dx_hat_thread;
  local_scratch1[local_col] = sum_dx_hat_x_hat_thread;
  barrier(CLK_LOCAL_MEM_FENCE);

  // sum-reduce
  for (int stride = local_size / 2; stride > 0; stride /= 2) {
    if (local_col < stride) {
      local_scratch0[local_col] += local_scratch0[local_col + stride];
      local_scratch1[local_col] += local_scratch1[local_col + stride];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  float sum_dx_hat = local_scratch0[0];
  float sum_dx_hat_x_hat = local_scratch1[0];

  float rstd = buffer_rstd[rstd_offset];

  for (int d = local_col; d < mat_din.cols; d += local_size) {
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
  int d = get_group_id(0);
  if (d >= mat_dout.cols) return;

  float dgamma = 0;
  float dbeta = 0;

  for (int r = 0; r < mat_dout.rows; r++) {
    int dout_offset = mat_dout.offset + r * mat_dout.stride + d;
    int xhat_offset = mat_xhat.offset + r * mat_xhat.stride + d;

    dgamma += buffer_dout[dout_offset] * buffer_xhat[xhat_offset];
    dbeta += buffer_dout[dout_offset];
  }

  int dgamma_offset = mat_dgamma.offset + d;
  int dbeta_offset = mat_dbeta.offset + d;

  buffer_dgamma[dgamma_offset] = dgamma;
  buffer_dbeta[dbeta_offset] = dbeta;
}

typedef struct { float x, y; } Vector2;
float _sincos(float phi, float *cos_phi) {
  *cos_phi = cos(phi);
  return sin(phi);
}

kernel void rope_kernel(
  // outputs
  global float* restrict buffer_qk, MatrixData mat_qk,
  // inputs
  int base, int d_head
) {
  int r = get_global_id(1);
  if (r >= mat_qk.rows) return;

  int local_col = get_local_id(0);
  int local_size = get_local_size(0);

  // [NxD]
  int qk_offset = mat_qk.offset + r * mat_qk.stride;

  float inv_d = -1.0f / (float)d_head;

  // ln(10000.0f)
  const float log_base = 9.21034037197f;

  for (int i = local_col*2; i < mat_qk.cols; i += local_size*2) {
    if (i + 1 >= mat_qk.cols) break;

    int local_i = i % d_head;
    float theta = exp((float)local_i * inv_d * log_base);
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
}

kernel void drope_kernel(
  // outputs
  global float* restrict buffer_dqk, MatrixData mat_dqk,
  // inputs
  global const float* restrict buffer_dout, MatrixData mat_dout,
  int base, int d_head
) {
  int r = get_global_id(1);
  if (r >= mat_dqk.rows) return;

  int local_col = get_local_id(0);
  int local_size = get_local_size(0);

  // [NxD]
  int dqk_offset  = mat_dqk.offset  + r * mat_dqk.stride;
  int dout_offset = mat_dout.offset + r * mat_dout.stride;

  float inv_d = -1.0f / (float)d_head;

  // ln(10000.0f)
  const float log_base = 9.21034037197f;

  for (int i = local_col*2; i < mat_dqk.cols; i += local_size*2) {
    if (i + 1 >= mat_dqk.cols) break;

    int local_i = i % d_head;
    float theta = exp((float)local_i * inv_d * log_base);
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
}

kernel void relu_kernel(
  // outputs
  global       float* restrict buffer_dst, MatrixData mat_dst,
  // inputs
  global const float* restrict buffer_x, MatrixData mat_x,
  float a
) {
  int r = get_global_id(1);
  if (r >= mat_dst.rows) return;

  int local_col = get_local_id(0);
  int local_size = get_local_size(0);

  // [NxD]
  int dst_offset = mat_dst.offset + r * mat_dst.stride;
  int x_offset   = mat_x.offset   + r * mat_x.stride;

  for (int d = local_col; d < mat_dst.cols; d += local_size) {
    float x = buffer_x[x_offset + d];
    buffer_dst[dst_offset + d] = x > 0 ? x : x*a;
  }
}

kernel void drelu_kernel(
  // outputs
  global       float* restrict buffer_dst, MatrixData mat_dst,
  // inputs
  global const float* restrict buffer_x, MatrixData mat_x,
  float a
) {
  int r = get_global_id(1);
  if (r >= mat_dst.rows) return;

  int local_col = get_local_id(0);
  int local_size = get_local_size(0);

  // [NxD]
  int dst_offset = mat_dst.offset + r * mat_dst.stride;
  int x_offset   = mat_x.offset   + r * mat_x.stride;

  for (int d = local_col; d < mat_dst.cols; d += local_size) {
    float x = buffer_x[x_offset + d];
    buffer_dst[dst_offset + d] = x > 0 ? 1 : a;
  }
}

// [NxD] = [NxD] x [DxD] + [1xD]
kernel void project_kernel(
  // outputs
  global       float* restrict buffer_out, MatrixData mat_out,
  // inputs
  global const float* restrict buffer_x, MatrixData mat_x,
  global const float* restrict buffer_W, MatrixData mat_W,
  global const float* restrict buffer_b, MatrixData mat_b,
  int transpose_W
) {
  // if (transpose_W != 0) {
  //   matrix_mul_Bt_stride(
  //       buffer_x, mat_x,
  //       buffer_W, mat_W,
  //       buffer_out, mat_out
  //   );
  // } else {
  //   matrix_mul_stride(
  //       buffer_x, mat_x,
  //       buffer_W, mat_W,
  //       buffer_out, mat_out
  //   );
  // }

  int r = get_global_id(1);
  if (r >= mat_out.rows) return;

  int local_col = get_local_id(0);
  int local_size = get_local_size(0);

  int out_offset = mat_out.offset + r * mat_out.stride;

  for (int d = local_col; d < mat_out.cols; d += local_size) {
    int b_offset = mat_b.offset;
    buffer_out[out_offset + d] += buffer_b[b_offset + d];
  }
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
  int transpose_W,
  local        float* local_scratch
) {
  // db += dout
  int d = get_group_id(0);
  if (d >= mat_db.cols) return;

  int local_row = get_local_id(1);
  int local_size = get_local_size(1);

  float db_thread = 0.0f;

  for (int r = local_row; r < mat_dout.rows; r += local_size) {
    int dout_offset = mat_dout.offset + r * mat_dout.stride;
    db_thread += buffer_dout[dout_offset + d];
  }

  local_scratch[local_row] = db_thread;
  barrier(CLK_LOCAL_MEM_FENCE);

  // sum-reduce
  for (int stride = local_size / 2; stride > 0; stride /= 2) {
    if (local_row < stride) {
      local_scratch[local_row] += local_scratch[local_row + stride];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  float db = local_scratch[0];

  int db_offset = mat_db.offset + d;
  buffer_db[db_offset] = db;
}

kernel void dproject_dW_kernel(
  // outputs
  global       float* restrict buffer_dx, MatrixData mat_dx,
  global       float* restrict buffer_dW, MatrixData mat_dW,
  global       float* restrict buffer_db, MatrixData mat_db,
  // inputs
  global const float* restrict buffer_x, MatrixData mat_x,
  global const float* restrict buffer_W, MatrixData mat_W,
  global const float* restrict buffer_dout, MatrixData mat_dout,
  int transpose_W
) {
  // if (transpose_W)
  //   dW += dout^T * x
  // else
  //   dW += x^T * dout
  if (transpose_W != 0) {
    matrix_mul_At_stride(
        buffer_dout, mat_dout,
        buffer_x,    mat_x,
        buffer_dW,   mat_dW
    );
  } else {
    matrix_mul_At_stride(
        buffer_x,    mat_x,
        buffer_dout, mat_dout,
        buffer_dW,   mat_dW
    );
  }
}

kernel void dproject_dx_kernel(
  // outputs
  global       float* restrict buffer_dx, MatrixData mat_dx,
  global       float* restrict buffer_dW, MatrixData mat_dW,
  global       float* restrict buffer_db, MatrixData mat_db,
  // inputs
  global const float* restrict buffer_x, MatrixData mat_x,
  global const float* restrict buffer_W, MatrixData mat_W,
  global const float* restrict buffer_dout, MatrixData mat_dout,
  int transpose_W
) {
  // if (transpose_W)
  //   dx += dout * W
  // else
  //   dx += dout * W^T
  if (transpose_W != 0) {
    matrix_mul_stride(
        buffer_dout, mat_dout,
        buffer_W,    mat_W,
        buffer_dx,   mat_dx
    );
  } else {
    matrix_mul_Bt_stride(
        buffer_dout, mat_dout,
        buffer_W,    mat_W,
        buffer_dx,   mat_dx
    );
  }
}

kernel void embed_gatter_forward_kernel(
  // outputs
  global       float* restrict buffer_out, MatrixData mat_out,
  // inputs
  global const float* restrict buffer_emb, MatrixData mat_emb,
  global const int* restrict tokens
) {
  int r = get_group_id(1);
  if (r >= mat_out.rows) return;

  int local_col  = get_local_id(0);
  int local_size = get_local_size(0);

  int out_offset = mat_out.offset + r * mat_out.stride;

  for (int i = local_col; i < mat_out.cols; i += local_size) {
    int token = tokens[r];
    int emb_offset = mat_emb.offset + token * mat_emb.stride;
    buffer_out[out_offset + i] = buffer_emb[emb_offset + i];
  }
}

