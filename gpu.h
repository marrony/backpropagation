#ifndef GPU_H
#define GPU_H

#if GPU_COMPUTATION
#include "bytebuffer.h"
#include "nn.h"

void from_matrix2(cl_mem buffer, NMatrix* mat, size_t* offset) {
  assert(mat->stride == mat->cols);

  MatrixData data = {
    .offset = *offset,
    .rows   = mat->rows,
    .cols   = mat->cols,
    .stride = mat->stride,
  };

  mat->data = data;
  mat->buffer = buffer;

  *offset += mat->rows*mat->stride;
}

void from_matrix3(cl_mem buffer, NMatrix* mat) {
  size_t offset = 0;
  from_matrix2(buffer, mat, &offset);
}

const char gpu_kernels[] = {
  #embed "gpu_kernels.c"
  , 0
};

cl_device_id device_id;             // compute device id 
cl_context context;                 // compute context
cl_command_queue commands;          // compute command queue
cl_program program;                 // compute program
cl_kernel matrix_mul_stride_kernel;
cl_kernel matrix_mul_At_stride_kernel;
cl_kernel matrix_mul_Bt_stride_kernel;
cl_kernel matrix_add_kernel;
cl_kernel softmax_temperature_kernel;
cl_kernel softmax_with_masking_kernel;
cl_kernel dsoftmax_temperature_kernel;
cl_kernel layer_norm_forward_kernel;
cl_kernel layer_norm_backward_dx_kernel;
cl_kernel layer_norm_backward_dgamma_dbeta_kernel;
cl_kernel rope_kernel;
cl_kernel drope_kernel;
cl_kernel relu_kernel;
cl_kernel drelu_kernel;
cl_kernel project_kernel;
cl_kernel dproject_db_kernel;
cl_kernel dproject_dW_kernel;
cl_kernel dproject_dx_kernel;
cl_kernel embed_gatter_forward_kernel;

// buffer_offset = buffer_origin[2] × buffer_slice_pitch + buffer_origin[1] × buffer_row_pitch + buffer_origin[0].
// host_offset   = host_origin[2]   × host_slice_pitch   + host_origin[1]   × host_row_pitch   + host_origin[0]
cl_int copy_gpu_to_cpu(NMatrix matrix, cl_event* event) {
  size_t host_size = matrix.cols*sizeof(float)*matrix.rows;

  bool copy_rect = true;

  if (matrix.data.offset == 0) {
    size_t buffer_size = 0;

    clGetMemObjectInfo(
        matrix.buffer,
        CL_MEM_SIZE,
        sizeof(buffer_size),
        &buffer_size,
        NULL
    );

    copy_rect = buffer_size != host_size;
  }

  if (copy_rect) {
    size_t buffer_origin[3] = {matrix.data.offset*sizeof(float), 0, 0}; // offset relative to buffer
    size_t host_origin[3] = {0*sizeof(float), 0, 0}; // offset relative to NMatrix
    size_t region[3] = {matrix.cols*sizeof(float), matrix.rows, 1};

    return clEnqueueReadBufferRect(
        commands,
        matrix.buffer,
        CL_FALSE,
        buffer_origin,
        host_origin,
        region,
        matrix.stride * sizeof(float), // buffer_row_pitch
        0, // buffer_slice_pitch
        matrix.stride * sizeof(float), // host_row_pitch
        0, // host_slice_pitch
        matrix.elems,
        0,
        NULL,
        event
    );
  } else {
    return clEnqueueReadBuffer(
        commands,
        matrix.buffer,
        CL_FALSE,
        0,
        host_size,
        matrix.elems,
        0,
        NULL,
        event
    );
  }
}

cl_int copy_to_gpu(Byte_Buffer bytes, cl_mem buffer, cl_event* event) {
  return clEnqueueWriteBuffer(
      commands,
      buffer,
      CL_FALSE,
      0,
      bytes.len,
      bytes.ptr,
      0,
      NULL,
      event
  );
}

cl_int copy_cpu_to_gpu(NMatrix matrix, cl_event* event) {
  size_t host_size = matrix.cols*sizeof(float)*matrix.rows;

  bool copy_rect = true;

  if (matrix.data.offset == 0) {
    size_t buffer_size = 0;

    clGetMemObjectInfo(
        matrix.buffer,
        CL_MEM_SIZE,
        sizeof(buffer_size),
        &buffer_size,
        NULL
    );

    copy_rect = buffer_size != host_size;
  }

  if (copy_rect) {
    size_t buffer_origin[3] = {matrix.data.offset*sizeof(float), 0, 0}; // offset relative to buffer
    size_t host_origin[3] = {0*sizeof(float), 0, 0}; // offset relative to NMatrix
    size_t region[3] = {matrix.cols*sizeof(float), matrix.rows, 1};

    return clEnqueueWriteBufferRect(
        commands,
        matrix.buffer,
        CL_FALSE,
        buffer_origin,
        host_origin,
        region,
        matrix.stride * sizeof(float), // buffer_row_pitch
        0, // buffer_slice_pitch
        matrix.stride * sizeof(float), // host_row_pitch
        0, // host_slice_pitch
        matrix.elems,
        0,
        NULL,
        event
    );
  } else {
    return clEnqueueWriteBuffer(
        commands,
        matrix.buffer,
        CL_FALSE,
        0,
        host_size,
        matrix.elems,
        0,
        NULL,
        event
    );
  }
}

void copy_gpu_to_cpu_sync(NMatrix matrix) {
  cl_event copy_event[1];
  assert(copy_gpu_to_cpu(matrix, &copy_event[0]) == CL_SUCCESS);
  clWaitForEvents(1, copy_event);
  clReleaseEvent(copy_event[0]);
}

void copy_cpu_to_gpu_sync(NMatrix matrix) {
  cl_event copy_event[1];
  assert(copy_cpu_to_gpu(matrix, &copy_event[0]) == CL_SUCCESS);
  clWaitForEvents(1, copy_event);
  clReleaseEvent(copy_event[0]);
}

bool init_opencl(void) {
  if (program) return true;

  int err;

  err = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
  if (err != CL_SUCCESS) {
      printf("Error: Failed to create a device group!\n");
      return false;
  }

  context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
  if (!context) {
      printf("Error: Failed to create a compute context!\n");
      return false;
  }

  commands = clCreateCommandQueue(context, device_id, 0, &err);
  if (!commands) {
      printf("Error: Failed to create a command commands!\n");
      return false;
  }

  const char* program_source[] = {
    gpu_kernels,
    NULL
  };

  program = clCreateProgramWithSource(context, 1, program_source, NULL, &err);
  if (!program) {
      printf("Error: Failed to create compute program!\n");
      return false;
  }

  err = clBuildProgram(program, 0, NULL, "-DTS=16", NULL, NULL);
  if (err != CL_SUCCESS) {
      size_t len = 0;
      printf("Error: Failed to build program executable!\n");
      clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);

      char* buffer = malloc(len);
      clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, len, buffer, NULL);
      printf("%.*s\n", (int)len, buffer);
      free(buffer);
      return false;
  }

  size_t binary_size = 0;
  clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &binary_size, NULL);
  printf("Program size = %zu\n", binary_size);
  void* program_binary = malloc(binary_size);
  clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(unsigned char*), &program_binary, NULL);
  free(program_binary);

  matrix_mul_stride_kernel = clCreateKernel(program, "matrix_mul_stride", &err);
  if (!matrix_mul_stride_kernel || err != CL_SUCCESS) {
      printf("Error: Failed to create compute kernel!\n");
      return false;
  }

  matrix_mul_At_stride_kernel = clCreateKernel(program, "matrix_mul_At_stride", &err);
  if (!matrix_mul_At_stride_kernel || err != CL_SUCCESS) {
      printf("Error: Failed to create compute kernel!\n");
      return false;
  }

  matrix_mul_Bt_stride_kernel = clCreateKernel(program, "matrix_mul_Bt_stride", &err);
  if (!matrix_mul_Bt_stride_kernel || err != CL_SUCCESS) {
      printf("Error: Failed to create compute kernel!\n");
      return false;
  }

  matrix_add_kernel = clCreateKernel(program, "matrix_add_kernel", &err);
  if (!matrix_add_kernel || err != CL_SUCCESS) {
      printf("Error: Failed to create compute kernel!\n");
      return false;
  }

  softmax_temperature_kernel = clCreateKernel(program, "softmax_temperature_kernel", &err);
  if (!softmax_temperature_kernel || err != CL_SUCCESS) {
      printf("Error: Failed to create compute kernel!\n");
      return false;
  }

  softmax_with_masking_kernel = clCreateKernel(program, "softmax_with_masking_kernel", &err);
  if (!softmax_with_masking_kernel || err != CL_SUCCESS) {
      printf("Error: Failed to create compute kernel!\n");
      return false;
  }

  dsoftmax_temperature_kernel = clCreateKernel(program, "dsoftmax_temperature_kernel", &err);
  if (!dsoftmax_temperature_kernel || err != CL_SUCCESS) {
      printf("Error: Failed to create compute kernel!\n");
      return false;
  }

  layer_norm_forward_kernel = clCreateKernel(program, "layer_norm_forward_kernel", &err);
  if (!layer_norm_forward_kernel || err != CL_SUCCESS) {
      printf("Error: Failed to create compute kernel!\n");
      return false;
  }

  layer_norm_backward_dx_kernel = clCreateKernel(program, "layer_norm_backward_dx_kernel", &err);
  if (!layer_norm_backward_dx_kernel || err != CL_SUCCESS) {
      printf("Error: Failed to create compute kernel!\n");
      return false;
  }

  layer_norm_backward_dgamma_dbeta_kernel = clCreateKernel(program, "layer_norm_backward_dgamma_dbeta_kernel", &err);
  if (!layer_norm_backward_dgamma_dbeta_kernel || err != CL_SUCCESS) {
      printf("Error: Failed to create compute kernel!\n");
      return false;
  }

  rope_kernel = clCreateKernel(program, "rope_kernel", &err);
  if (!rope_kernel || err != CL_SUCCESS) {
      printf("Error: Failed to create compute kernel!\n");
      return false;
  }

  drope_kernel = clCreateKernel(program, "drope_kernel", &err);
  if (!drope_kernel || err != CL_SUCCESS) {
      printf("Error: Failed to create compute kernel!\n");
      return false;
  }

  relu_kernel = clCreateKernel(program, "relu_kernel", &err);
  if (!relu_kernel || err != CL_SUCCESS) {
      printf("Error: Failed to create compute kernel!\n");
      return false;
  }

  drelu_kernel = clCreateKernel(program, "drelu_kernel", &err);
  if (!drelu_kernel || err != CL_SUCCESS) {
      printf("Error: Failed to create compute kernel!\n");
      return false;
  }

  project_kernel = clCreateKernel(program, "project_kernel", &err);
  if (!project_kernel || err != CL_SUCCESS) {
      printf("Error: Failed to create compute kernel!\n");
      return false;
  }

  dproject_db_kernel = clCreateKernel(program, "dproject_db_kernel", &err);
  if (!dproject_db_kernel || err != CL_SUCCESS) {
      printf("Error: Failed to create compute kernel!\n");
      return false;
  }

  dproject_dW_kernel = clCreateKernel(program, "dproject_dW_kernel", &err);
  if (!dproject_dW_kernel || err != CL_SUCCESS) {
      printf("Error: Failed to create compute kernel!\n");
      return false;
  }

  dproject_dx_kernel = clCreateKernel(program, "dproject_dx_kernel", &err);
  if (!dproject_dx_kernel || err != CL_SUCCESS) {
      printf("Error: Failed to create compute kernel!\n");
      return false;
  }

  embed_gatter_forward_kernel = clCreateKernel(program, "embed_gatter_forward_kernel", &err);
  if (!embed_gatter_forward_kernel || err != CL_SUCCESS) {
      printf("Error: Failed to create compute kernel!\n");
      return false;
  }

  return true;
}

void destoy_opencl(void) {
  clFlush(commands);
  clFinish(commands);

  clReleaseKernel(matrix_mul_stride_kernel);
  clReleaseKernel(matrix_mul_At_stride_kernel);
  clReleaseKernel(matrix_mul_Bt_stride_kernel);
  clReleaseKernel(matrix_add_kernel);
  clReleaseKernel(softmax_temperature_kernel);
  clReleaseKernel(softmax_with_masking_kernel);
  clReleaseKernel(dsoftmax_temperature_kernel);
  clReleaseKernel(layer_norm_forward_kernel);
  clReleaseKernel(layer_norm_backward_dx_kernel);
  clReleaseKernel(layer_norm_backward_dgamma_dbeta_kernel);
  clReleaseKernel(rope_kernel);
  clReleaseKernel(drope_kernel);
  clReleaseKernel(relu_kernel);
  clReleaseKernel(drelu_kernel);
  clReleaseKernel(project_kernel);
  clReleaseKernel(dproject_db_kernel);
  clReleaseKernel(dproject_dW_kernel);
  clReleaseKernel(dproject_dx_kernel);
  clReleaseKernel(embed_gatter_forward_kernel);

  clReleaseProgram(program);
  clReleaseCommandQueue(commands);

  clReleaseContext(context);
}

#define ROUND_UP(size, tile) (((size) + (tile) - 1) / (tile) * (tile))

cl_int call_matrix_mul_stride_kernel(
  // outputs
  NMatrix mat_output,
  // inputs
  NMatrix mat_x,
  NMatrix mat_W
) {
  cl_int err = 0;

  // inputs
  err |= clSetKernelArg(matrix_mul_stride_kernel, 0, sizeof(cl_mem), &mat_x.buffer);
  err |= clSetKernelArg(matrix_mul_stride_kernel, 1, sizeof(MatrixData), &mat_x.data);
  err |= clSetKernelArg(matrix_mul_stride_kernel, 2, sizeof(cl_mem), &mat_W.buffer);
  err |= clSetKernelArg(matrix_mul_stride_kernel, 3, sizeof(MatrixData), &mat_W.data);
  assert(err == CL_SUCCESS);
  // outputs
  err  = clSetKernelArg(matrix_mul_stride_kernel, 4, sizeof(cl_mem), &mat_output.buffer);
  err |= clSetKernelArg(matrix_mul_stride_kernel, 5, sizeof(MatrixData), &mat_output.data);
  assert(err == CL_SUCCESS);

  size_t local_ws[2]  = {16, 16};
  size_t global_ws[2] = {
    ROUND_UP(mat_x.data.rows, 16), // N
    ROUND_UP(mat_W.data.cols, 16), // D
  };
  return clEnqueueNDRangeKernel(
      commands,
      matrix_mul_stride_kernel,
      2,
      NULL,
      global_ws,
      local_ws,
      0, NULL, NULL
  );
}

cl_int call_matrix_mul_Bt_stride_kernel(
  // outputs
  NMatrix mat_output,
  // inputs
  NMatrix mat_x,
  NMatrix mat_W
) {
  cl_int err = 0;

  // inputs
  err |= clSetKernelArg(matrix_mul_Bt_stride_kernel, 0, sizeof(cl_mem), &mat_x.buffer);
  err |= clSetKernelArg(matrix_mul_Bt_stride_kernel, 1, sizeof(MatrixData), &mat_x.data);
  err |= clSetKernelArg(matrix_mul_Bt_stride_kernel, 2, sizeof(cl_mem), &mat_W.buffer);
  err |= clSetKernelArg(matrix_mul_Bt_stride_kernel, 3, sizeof(MatrixData), &mat_W.data);
  assert(err == CL_SUCCESS);
  // outputs
  err  = clSetKernelArg(matrix_mul_Bt_stride_kernel, 4, sizeof(cl_mem), &mat_output.buffer);
  err |= clSetKernelArg(matrix_mul_Bt_stride_kernel, 5, sizeof(MatrixData), &mat_output.data);
  assert(err == CL_SUCCESS);

  size_t local_ws[2]  = {16, 16};
  size_t global_ws[2] = {
    ROUND_UP(mat_x.data.rows, 16), // N
    ROUND_UP(mat_W.data.rows, 16), // D
  };
  return clEnqueueNDRangeKernel(
      commands,
      matrix_mul_Bt_stride_kernel,
      2,
      NULL,
      global_ws,
      local_ws,
      0, NULL, NULL
  );
}

cl_int call_matrix_add_kernel(
  // outputs
  NMatrix mat_out,
  // inputs
  NMatrix mat_a,
  NMatrix mat_b
) {
  cl_int err = 0;

  // outputs
  err |= clSetKernelArg(matrix_add_kernel, 0, sizeof(cl_mem), &mat_out.buffer);
  err |= clSetKernelArg(matrix_add_kernel, 1, sizeof(MatrixData), &mat_out.data);
  assert(err == CL_SUCCESS);
  // inputs
  err |= clSetKernelArg(matrix_add_kernel, 2, sizeof(cl_mem), &mat_a.buffer);
  err |= clSetKernelArg(matrix_add_kernel, 3, sizeof(MatrixData), &mat_a.data);
  err  = clSetKernelArg(matrix_add_kernel, 4, sizeof(cl_mem), &mat_b.buffer);
  err |= clSetKernelArg(matrix_add_kernel, 5, sizeof(MatrixData), &mat_b.data);
  assert(err == CL_SUCCESS);

  size_t local_ws[2]  = { 256, 1 };
  size_t global_ws[2] = { 256, mat_out.data.rows };
  return clEnqueueNDRangeKernel(
      commands,
      matrix_add_kernel,
      2,
      NULL,
      global_ws,
      local_ws,
      0, NULL, NULL
  );
}

cl_int call_project_kernel(
  // outputs
  NMatrix mat_output,
  // inputs
  NMatrix mat_x,
  NMatrix mat_W,
  NMatrix mat_b,
  cl_int transpose_W
) {
  cl_int err = 0;

  if (transpose_W) {
    assert(call_matrix_mul_Bt_stride_kernel(mat_output, mat_x, mat_W) == CL_SUCCESS);
  } else {
    assert(call_matrix_mul_stride_kernel(mat_output, mat_x, mat_W) == CL_SUCCESS);
  }

  // outputs
  err  = clSetKernelArg(project_kernel, 0, sizeof(cl_mem), &mat_output.buffer);
  err |= clSetKernelArg(project_kernel, 1, sizeof(MatrixData), &mat_output.data);
  assert(err == CL_SUCCESS);
  // inputs
  err |= clSetKernelArg(project_kernel, 2, sizeof(cl_mem), &mat_x.buffer);
  err |= clSetKernelArg(project_kernel, 3, sizeof(MatrixData), &mat_x.data);
  err |= clSetKernelArg(project_kernel, 4, sizeof(cl_mem), &mat_W.buffer);
  err |= clSetKernelArg(project_kernel, 5, sizeof(MatrixData), &mat_W.data);
  err |= clSetKernelArg(project_kernel, 6, sizeof(cl_mem), &mat_b.buffer);
  err |= clSetKernelArg(project_kernel, 7, sizeof(MatrixData), &mat_b.data);
  err |= clSetKernelArg(project_kernel, 8, sizeof(cl_int), &transpose_W);
  assert(err == CL_SUCCESS);

  size_t local_ws[2]  = { 256, 1 };
  size_t global_ws[2] = { 256, mat_output.data.rows };
  return clEnqueueNDRangeKernel(
      commands,
      project_kernel,
      2,
      NULL,
      global_ws,
      local_ws,
      0, NULL, NULL
  );
}

cl_int call_dproject_dx_kernel(
  // outputs
  NMatrix mat_dx,
  NMatrix mat_dW,
  NMatrix mat_db,
  // inputs
  NMatrix mat_x,
  NMatrix mat_W,
  NMatrix mat_dout,
  cl_int transpose_W
) {
  cl_int err = 0;

  // outputs
  err  = clSetKernelArg(dproject_dx_kernel, 0, sizeof(cl_mem), &mat_dx.buffer);
  err |= clSetKernelArg(dproject_dx_kernel, 1, sizeof(MatrixData), &mat_dx.data);
  err |= clSetKernelArg(dproject_dx_kernel, 2, sizeof(cl_mem), &mat_dW.buffer);
  err |= clSetKernelArg(dproject_dx_kernel, 3, sizeof(MatrixData), &mat_dW.data);
  err |= clSetKernelArg(dproject_dx_kernel, 4, sizeof(cl_mem), &mat_db.buffer);
  err |= clSetKernelArg(dproject_dx_kernel, 5, sizeof(MatrixData), &mat_db.data);
  assert(err == CL_SUCCESS);
  // inputs
  err |= clSetKernelArg(dproject_dx_kernel, 6, sizeof(cl_mem), &mat_x.buffer);
  err |= clSetKernelArg(dproject_dx_kernel, 7, sizeof(MatrixData), &mat_x.data);
  err |= clSetKernelArg(dproject_dx_kernel, 8, sizeof(cl_mem), &mat_W.buffer);
  err |= clSetKernelArg(dproject_dx_kernel, 9, sizeof(MatrixData), &mat_W.data);
  err |= clSetKernelArg(dproject_dx_kernel, 10, sizeof(cl_mem), &mat_dout.buffer);
  err |= clSetKernelArg(dproject_dx_kernel, 11, sizeof(MatrixData), &mat_dout.data);
  err |= clSetKernelArg(dproject_dx_kernel, 12, sizeof(cl_int), &transpose_W);
  assert(err == CL_SUCCESS);

  size_t local_ws[2]  = {16, 16};
  size_t global_ws[2] = {
    ROUND_UP(mat_dx.data.rows, 16), // N
    ROUND_UP(mat_dx.data.cols, 16), // D
  };
  return clEnqueueNDRangeKernel(
      commands,
      dproject_dx_kernel,
      2,
      NULL,
      global_ws,
      local_ws,
      0, NULL, NULL
  );
}

cl_int call_dproject_dW_kernel(
  // outputs
  NMatrix mat_dx,
  NMatrix mat_dW,
  NMatrix mat_db,
  // inputs
  NMatrix mat_x,
  NMatrix mat_W,
  NMatrix mat_dout,
  cl_int transpose_W
) {
  cl_int err = 0;

  // outputs
  err  = clSetKernelArg(dproject_dW_kernel, 0, sizeof(cl_mem), &mat_dx.buffer);
  err |= clSetKernelArg(dproject_dW_kernel, 1, sizeof(MatrixData), &mat_dx.data);
  err |= clSetKernelArg(dproject_dW_kernel, 2, sizeof(cl_mem), &mat_dW.buffer);
  err |= clSetKernelArg(dproject_dW_kernel, 3, sizeof(MatrixData), &mat_dW.data);
  err |= clSetKernelArg(dproject_dW_kernel, 4, sizeof(cl_mem), &mat_db.buffer);
  err |= clSetKernelArg(dproject_dW_kernel, 5, sizeof(MatrixData), &mat_db.data);
  assert(err == CL_SUCCESS);
  // inputs
  err |= clSetKernelArg(dproject_dW_kernel, 6, sizeof(cl_mem), &mat_x.buffer);
  err |= clSetKernelArg(dproject_dW_kernel, 7, sizeof(MatrixData), &mat_x.data);
  err |= clSetKernelArg(dproject_dW_kernel, 8, sizeof(cl_mem), &mat_W.buffer);
  err |= clSetKernelArg(dproject_dW_kernel, 9, sizeof(MatrixData), &mat_W.data);
  err |= clSetKernelArg(dproject_dW_kernel, 10, sizeof(cl_mem), &mat_dout.buffer);
  err |= clSetKernelArg(dproject_dW_kernel, 11, sizeof(MatrixData), &mat_dout.data);
  err |= clSetKernelArg(dproject_dW_kernel, 12, sizeof(cl_int), &transpose_W);
  assert(err == CL_SUCCESS);

  size_t local_ws[2]  = {16, 16};
  size_t global_ws[2] = {
    ROUND_UP(mat_W.data.rows, 16), // D
    ROUND_UP(mat_W.data.cols, 16), // D
  };
  return clEnqueueNDRangeKernel(
      commands,
      dproject_dW_kernel,
      2,
      NULL,
      global_ws,
      local_ws,
      0, NULL, NULL
  );
}

cl_int call_dproject_db_kernel(
  // outputs
  NMatrix mat_dx,
  NMatrix mat_dW,
  NMatrix mat_db,
  // inputs
  NMatrix mat_x,
  NMatrix mat_W,
  NMatrix mat_dout,
  cl_int transpose_W
) {
  size_t local_ws[2]  = { 1, 128 };
  size_t global_ws[2] = { mat_db.data.cols, 128 };

  cl_int err = 0;

  // outputs
  err  = clSetKernelArg(dproject_db_kernel, 0, sizeof(cl_mem), &mat_dx.buffer);
  err |= clSetKernelArg(dproject_db_kernel, 1, sizeof(MatrixData), &mat_dx.data);
  err |= clSetKernelArg(dproject_db_kernel, 2, sizeof(cl_mem), &mat_dW.buffer);
  err |= clSetKernelArg(dproject_db_kernel, 3, sizeof(MatrixData), &mat_dW.data);
  err |= clSetKernelArg(dproject_db_kernel, 4, sizeof(cl_mem), &mat_db.buffer);
  err |= clSetKernelArg(dproject_db_kernel, 5, sizeof(MatrixData), &mat_db.data);
  assert(err == CL_SUCCESS);

  // inputs
  err |= clSetKernelArg(dproject_db_kernel, 6, sizeof(cl_mem), &mat_x.buffer);
  err |= clSetKernelArg(dproject_db_kernel, 7, sizeof(MatrixData), &mat_x.data);
  err |= clSetKernelArg(dproject_db_kernel, 8, sizeof(cl_mem), &mat_W.buffer);
  err |= clSetKernelArg(dproject_db_kernel, 9, sizeof(MatrixData), &mat_W.data);
  err |= clSetKernelArg(dproject_db_kernel, 10, sizeof(cl_mem), &mat_dout.buffer);
  err |= clSetKernelArg(dproject_db_kernel, 11, sizeof(MatrixData), &mat_dout.data);
  err |= clSetKernelArg(dproject_db_kernel, 12, sizeof(cl_int), &transpose_W);
  err |= clSetKernelArg(dproject_db_kernel, 13, local_ws[1] * sizeof(float), NULL);
  assert(err == CL_SUCCESS);

  return clEnqueueNDRangeKernel(
      commands,
      dproject_db_kernel,
      2,
      NULL,
      global_ws,
      local_ws,
      0, NULL, NULL
  );
}

cl_int call_embed_gatter_forward_kernel(
    NMatrix x0,
    NMatrix tok_emb,
    cl_mem opencl_tokens_buffer
) {
  cl_int err = 0;

  // outputs
  err |= clSetKernelArg(embed_gatter_forward_kernel, 0, sizeof(cl_mem), &x0.buffer);
  err |= clSetKernelArg(embed_gatter_forward_kernel, 1, sizeof(MatrixData), &x0.data);
  assert(err == CL_SUCCESS);
  // inputs
  err |= clSetKernelArg(embed_gatter_forward_kernel, 2, sizeof(cl_mem), &tok_emb.buffer);
  err |= clSetKernelArg(embed_gatter_forward_kernel, 3, sizeof(MatrixData), &tok_emb.data);
  err |= clSetKernelArg(embed_gatter_forward_kernel, 4, sizeof(cl_mem), &opencl_tokens_buffer);
  assert(err == CL_SUCCESS);

  size_t local_ws[2]  = {256, 1};
  size_t global_ws[2] = {256, x0.data.rows};
  return clEnqueueNDRangeKernel(
      commands,
      embed_gatter_forward_kernel,
      2,
      NULL,
      global_ws,
      local_ws,
      0, NULL, NULL
  );
}

cl_int call_layer_norm_forward_kernel(
    // outputs
    NMatrix mat_out,
    NMatrix mat_mean,
    NMatrix mat_rstd,
    NMatrix mat_xhat,
    // inputs
    NMatrix mat_in,
    NMatrix mat_gamma,
    NMatrix mat_beta
) {
  /////////////////////////////////

  cl_int err = 0;

  // outputs
  err |= clSetKernelArg(layer_norm_forward_kernel, 0, sizeof(cl_mem), &mat_out.buffer);
  err |= clSetKernelArg(layer_norm_forward_kernel, 1, sizeof(MatrixData), &mat_out.data);
  err |= clSetKernelArg(layer_norm_forward_kernel, 2, sizeof(cl_mem), &mat_mean.buffer);
  err |= clSetKernelArg(layer_norm_forward_kernel, 3, sizeof(MatrixData), &mat_mean.data);
  err |= clSetKernelArg(layer_norm_forward_kernel, 4, sizeof(cl_mem), &mat_rstd.buffer);
  err |= clSetKernelArg(layer_norm_forward_kernel, 5, sizeof(MatrixData), &mat_rstd.data);
  err |= clSetKernelArg(layer_norm_forward_kernel, 6, sizeof(cl_mem), &mat_xhat.buffer);
  err |= clSetKernelArg(layer_norm_forward_kernel, 7, sizeof(MatrixData), &mat_xhat.data);
  assert(err == CL_SUCCESS);

  // inputs
  err |= clSetKernelArg(layer_norm_forward_kernel, 8, sizeof(cl_mem), &mat_in.buffer);
  err |= clSetKernelArg(layer_norm_forward_kernel, 9, sizeof(MatrixData), &mat_in.data);
  err |= clSetKernelArg(layer_norm_forward_kernel, 10, sizeof(cl_mem), &mat_gamma.buffer);
  err |= clSetKernelArg(layer_norm_forward_kernel, 11, sizeof(MatrixData), &mat_gamma.data);
  err |= clSetKernelArg(layer_norm_forward_kernel, 12, sizeof(cl_mem), &mat_beta.buffer);
  err |= clSetKernelArg(layer_norm_forward_kernel, 13, sizeof(MatrixData), &mat_beta.data);
  assert(err == CL_SUCCESS);

  size_t local_ws[2]  = { 256, 1 }; // 256 threads per row, 1 row deep per group
  size_t global_ws[2] = { 256, mat_out.data.rows }; // Scale Y to equal total row count

  // scratch
  err |= clSetKernelArg(layer_norm_forward_kernel, 14, local_ws[0] * sizeof(float), NULL);
  assert(err == CL_SUCCESS);

  return clEnqueueNDRangeKernel(
      commands,
      layer_norm_forward_kernel,
      2,              // work_dim = 2D Grid
      NULL,
      global_ws,      // Total threads {256, rows}
      local_ws,       // Local layout {256, 1}
      0, NULL, NULL
  );
}

cl_int call_softmax_temperature_kernel(
  // outputs
  NMatrix out,
  // inputs
  NMatrix x,
  float temperature
) {
  size_t local_ws[2]  = { 256, 1 }; // 256 threads per row, 1 row deep per group
  size_t global_ws[2] = { 256, out.data.rows }; // Scale Y to equal total row count

  // outputs
  clSetKernelArg(softmax_temperature_kernel, 0, sizeof(cl_mem), &out.buffer);
  clSetKernelArg(softmax_temperature_kernel, 1, sizeof(MatrixData), &out.data);
  // inputs
  clSetKernelArg(softmax_temperature_kernel, 2, sizeof(cl_mem), &x.buffer);
  clSetKernelArg(softmax_temperature_kernel, 3, sizeof(MatrixData), &x.data);
  clSetKernelArg(softmax_temperature_kernel, 4, sizeof(float), &temperature);
  // scratch
  clSetKernelArg(softmax_temperature_kernel, 5, local_ws[0] * sizeof(float), NULL);

  return clEnqueueNDRangeKernel(
      commands,
      softmax_temperature_kernel,
      2,              // work_dim = 2D Grid
      NULL,
      global_ws,      // Total threads {256, rows}
      local_ws,       // Local layout {256, 1}
      0, NULL, NULL
  );
}

cl_int call_softmax_with_masking_kernel(
  // outputs
  NMatrix out,
  // inputs
  NMatrix x,
  float temperature,
  float scale,
  cl_int base,
  cl_int total
) {
  size_t local_ws[2]  = { 256, 1 }; // 256 threads per row, 1 row deep per group
  size_t global_ws[2] = { 256, out.data.rows }; // Scale Y to equal total row count

  // outputs
  clSetKernelArg(softmax_with_masking_kernel, 0, sizeof(cl_mem), &out.buffer);
  clSetKernelArg(softmax_with_masking_kernel, 1, sizeof(MatrixData), &out.data);
  // inputs
  clSetKernelArg(softmax_with_masking_kernel, 2, sizeof(cl_mem), &x.buffer);
  clSetKernelArg(softmax_with_masking_kernel, 3, sizeof(MatrixData), &x.data);
  clSetKernelArg(softmax_with_masking_kernel, 4, sizeof(float), &temperature);
  clSetKernelArg(softmax_with_masking_kernel, 5, sizeof(float), &scale);
  clSetKernelArg(softmax_with_masking_kernel, 6, sizeof(cl_int), &base);
  clSetKernelArg(softmax_with_masking_kernel, 7, sizeof(cl_int), &total);
  // scratch
  clSetKernelArg(softmax_with_masking_kernel, 8, local_ws[0] * sizeof(float), NULL);

  return clEnqueueNDRangeKernel(
      commands,
      softmax_with_masking_kernel,
      2,              // work_dim = 2D Grid
      NULL,
      global_ws,      // Total threads {256, rows}
      local_ws,       // Local layout {256, 1}
      0, NULL, NULL
  );
}

cl_int call_relu_kernel(
  // outputs
  NMatrix out,
  // inputs
  NMatrix x,
  float a
) {
  size_t local_ws[2]  = { 256, 1 }; // 256 threads per row, 1 row deep per group
  size_t global_ws[2] = { 256, out.data.rows }; // Scale Y to equal total row count

  // outputs
  clSetKernelArg(relu_kernel, 0, sizeof(cl_mem), &out.buffer);
  clSetKernelArg(relu_kernel, 1, sizeof(MatrixData), &out.data);
  // inputs
  clSetKernelArg(relu_kernel, 2, sizeof(cl_mem), &x.buffer);
  clSetKernelArg(relu_kernel, 3, sizeof(MatrixData), &x.data);
  clSetKernelArg(relu_kernel, 4, sizeof(float), &a);

  return clEnqueueNDRangeKernel(
      commands,
      relu_kernel,
      2,              // work_dim = 2D Grid
      NULL,
      global_ws,      // Total threads {256, rows}
      local_ws,       // Local layout {256, 1}
      0, NULL, NULL
  );
}

cl_int call_rope_kernel(
  // input/output
  NMatrix qk,
  // inputs
  cl_int base,
  cl_int d_head
) {
  size_t local_ws[2]  = { 256, 1 }; // 256 threads per row, 1 row deep per group
  size_t global_ws[2] = { 256, qk.data.rows }; // Scale Y to equal total row count

  // input/output
  clSetKernelArg(rope_kernel, 0, sizeof(cl_mem), &qk.buffer);
  clSetKernelArg(rope_kernel, 1, sizeof(MatrixData), &qk.data);
  // inputs
  clSetKernelArg(rope_kernel, 2, sizeof(cl_int), &base);
  clSetKernelArg(rope_kernel, 3, sizeof(cl_int), &d_head);

  return clEnqueueNDRangeKernel(
      commands,
      rope_kernel,
      2,              // work_dim = 2D Grid
      NULL,
      global_ws,      // Total threads {256, rows}
      local_ws,       // Local layout {256, 1}
      0, NULL, NULL
  );
}

#endif

#endif // GPU_H
