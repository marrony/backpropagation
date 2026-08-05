#include "nn.h"
#include <OpenCL/cl.h>
#include <stdio.h>

#include <OpenCL/opencl.h>

#define STRINGFY(x) #x

const char* matrix_mul_stride = STRINGFY(
__kernel void matrix_mul_stride(
  __global const float* A, 
  __global const float* B, 
  __global float* C,
  const int A_rows,
  const int A_cols,
  const int A_stride,
  const int B_rows,
  const int B_cols,
  const int B_stride,
  const int C_stride
) {
  const int TS = 16;

  __local float Asub[TS][TS];
  __local float Bsub[TS][TS];

  int globalRow = get_global_id(1); 
  int globalCol = get_global_id(0); 
  int localRow  = get_local_id(1);
  int localCol  = get_local_id(0);

  float acc = 0.0f;
  int numTiles = (A_cols + TS - 1) / TS;

  for (int t = 0; t < numTiles; t++) {
    int tiledCol = t * TS + localCol;
    
    // Load A utilizing explicit A_stride spacing calculation
    if (globalRow < A_rows && tiledCol < A_cols) {
      Asub[localRow][localCol] = A[globalRow * A_stride + tiledCol];
    } else {
      Asub[localRow][localCol] = 0.0f;
    }
    
    int tiledRow = t * TS + localRow;
    
    // Load B utilizing explicit B_stride spacing calculation
    if (tiledRow < B_rows && globalCol < B_cols) {
      Bsub[localRow][localCol] = B[tiledRow * B_stride + globalCol];
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
  if (globalRow < A_rows && globalCol < B_cols) {
    C[globalRow * C_stride + globalCol] += acc;
  }
}
);

const char* matrix_mul_At_stride = STRINGFY(
__kernel void matrix_mul_At_stride(
  __global const float* A, 
  __global const float* B, 
  __global float* C,
  const int A_rows,
  const int A_cols,
  const int A_stride,
  const int B_rows,
  const int B_cols,
  const int B_stride,
  const int C_stride
) {
  const int TS = 16;
  __local float Asub[TS][TS];
  __local float Bsub[TS][TS];
  
  int globalRow = get_global_id(1); // M (Rows of A^T and C)
  int globalCol = get_global_id(0); // N (Cols of B and C)
  int localRow  = get_local_id(1);
  int localCol  = get_local_id(0);
  
  float acc = 0.0f;
  int numTiles = (A_rows + TS - 1) / TS; // K dimension is A_rows in physical layout
  
  for (int t = 0; t < numTiles; t++) {
    int tiledCol = t * TS + localCol; // K index
    
    // Transposed Load: Physical rows and columns are swapped
    if (globalRow < A_cols && tiledCol < A_rows) {
      Asub[localRow][localCol] = A[tiledCol * A_stride + globalRow];
    } else {
      Asub[localRow][localCol] = 0.0f;
    }
    
    int tiledRow = t * TS + localRow; // K index
    if (tiledRow < B_rows && globalCol < B_cols) {
      Bsub[localRow][localCol] = B[tiledRow * B_stride + globalCol];
    } else {
      Bsub[localRow][localCol] = 0.0f;
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int k = 0; k < TS; k++) {
      acc += Asub[localRow][k] * Bsub[k][localCol];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  
  if (globalRow < A_cols && globalCol < B_cols) {
    C[globalRow * C_stride + globalCol] += acc;
  }
}
);

const char* matrix_mul_Bt_stride = STRINGFY(
__kernel void matrix_mul_Bt_stride(
  __global const float* A, 
  __global const float* B, 
  __global float* C, 
  const int A_rows,
  const int A_cols,
  const int A_stride,
  const int B_rows,
  const int B_cols,
  const int B_stride,
  const int C_stride
) {
  const int TS = 16;
  __local float Asub[TS][TS];
  __local float Bsub[TS][TS];
  
  int globalRow = get_global_id(1); // M (Rows of A and C)
  int globalCol = get_global_id(0); // N (Rows of B, which are columns of B^T)
  int localRow  = get_local_id(1);
  int localCol  = get_local_id(0);
  
  float acc = 0.0f;
  int numTiles = (A_cols + TS - 1) / TS; // K dimension
  
  for (int t = 0; t < numTiles; t++) {
    int tiledCol = t * TS + localCol; // K index
    if (globalRow < A_rows && tiledCol < A_cols) {
      Asub[localRow][localCol] = A[globalRow * A_stride + tiledCol];
    } else {
      Asub[localRow][localCol] = 0.0f;
    }
    
    int tiledRow = t * TS + localRow; // K index
    
    // Transposed Load: Access B along rows instead of columns
    if (tiledRow < B_cols && globalCol < B_rows) {
      Bsub[localRow][localCol] = B[globalCol * B_stride + tiledRow];
    } else {
      Bsub[localRow][localCol] = 0.0f;
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int k = 0; k < TS; k++) {
      acc += Asub[localRow][k] * Bsub[k][localCol];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  
  if (globalRow < A_rows && globalCol < B_rows) {
    C[globalRow * C_stride + globalCol] += acc;
  }
}
);

cl_device_id device_id;             // compute device id 
cl_context context;                 // compute context
cl_command_queue commands;          // compute command queue
cl_program program;                 // compute program
cl_kernel matrix_mul_stride_kernel;
cl_kernel matrix_mul_At_stride_kernel;
cl_kernel matrix_mul_Bt_stride_kernel;

cl_mem A;                       // device memory used for the input array
cl_mem B;                       // device memory used for the output array
cl_mem C;                       //

size_t max_buffer_size = 0;

bool init_opencl(void) {
  if (program) return true;

  int err;                            // error code returned from api calls

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
    matrix_mul_stride,
    matrix_mul_At_stride,
    matrix_mul_Bt_stride,
  };

  program = clCreateProgramWithSource(context, 3, program_source, NULL, &err);
  if (!program) {
      printf("Error: Failed to create compute program!\n");
      return false;
  }

  err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  if (err != CL_SUCCESS) {
      size_t len = 0;
      char buffer[2048];

      printf("Error: Failed to build program executable!\n");
      clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
      buffer[len] = 0;
      printf("%s\n", buffer);
      return false;
  }

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
  if (!matrix_mul_At_stride_kernel || err != CL_SUCCESS) {
      printf("Error: Failed to create compute kernel!\n");
      return false;
  }

  return true;
}

bool ensure_buffer_size(size_t buffer_size) {
  if (buffer_size > max_buffer_size) {
    clReleaseMemObject(A);
    clReleaseMemObject(B);
    clReleaseMemObject(C);
  }

  max_buffer_size = buffer_size;
  A = clCreateBuffer(context, CL_MEM_READ_ONLY, buffer_size, NULL, NULL);
  B = clCreateBuffer(context, CL_MEM_READ_ONLY, buffer_size, NULL, NULL);
  C = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, NULL, NULL);
  if (!A || !B || !C) {
      printf("Error: Failed to allocate device memory!\n");
      return false;
  }

  return true;
}

int execute_kernel(
    cl_command_queue commands,
    cl_kernel kernel,
    NMatrix A_,
    NMatrix B_,
    NMatrix C_,
    bool At,
    bool Bt
) {
  int err;

  size_t A_size = A_.rows * A_.stride * sizeof(float);
  size_t B_size = B_.rows * B_.stride * sizeof(float);
  size_t C_size = C_.rows * C_.stride * sizeof(float);

  cl_int A_rows   = A_.rows;
  cl_int A_cols   = A_.cols;
  cl_int A_stride = A_.stride;
  cl_int B_rows   = B_.rows;
  cl_int B_cols   = B_.cols;
  cl_int B_stride = B_.stride;
  cl_int C_stride = C_.stride;

  err = 0;
  err |= clSetKernelArg(kernel, 0, sizeof(cl_mem), &A);
  err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &B);
  err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &C);
  if (err != CL_SUCCESS) {
      printf("Error: Failed to set kernel buffers! %d\n", err);
      return err;
  }

  err |= clSetKernelArg(kernel, 3, sizeof(cl_int), &A_rows);   // A_rows
  err |= clSetKernelArg(kernel, 4, sizeof(cl_int), &A_cols);   // A_cols
  err |= clSetKernelArg(kernel, 5, sizeof(cl_int), &A_stride); // A_stride
  if (err != CL_SUCCESS) {
      printf("Error: Failed to set kernel A parameters! %d\n", err);
      return err;
  }

  err |= clSetKernelArg(kernel, 6, sizeof(cl_int), &B_rows);   // B_rows
  err |= clSetKernelArg(kernel, 7, sizeof(cl_int), &B_cols);   // B_cols
  err |= clSetKernelArg(kernel, 8, sizeof(cl_int), &B_stride); // B_stride
  if (err != CL_SUCCESS) {
      printf("Error: Failed to set kernel B parameters! %d\n", err);
      return err;
  }

  err |= clSetKernelArg(kernel, 9, sizeof(cl_int), &C_stride); // C_stride
  if (err != CL_SUCCESS) {
      printf("Error: Failed to set kernel C parameters! %d\n", err);
      return err;
  }

  err = clEnqueueWriteBuffer(commands, A, CL_FALSE, 0, A_size, A_.elems, 0, NULL, NULL);
  if (err != CL_SUCCESS) {
      printf("Error: Failed to write to source array!\n");
      exit(1);
  }

  err = clEnqueueWriteBuffer(commands, B, CL_FALSE, 0, B_size, B_.elems, 0, NULL, NULL);
  if (err != CL_SUCCESS) {
      printf("Error: Failed to write to source array!\n");
      exit(1);
  }

  err = clEnqueueWriteBuffer(commands, C, CL_FALSE, 0, C_size, C_.elems, 0, NULL, NULL);
  if (err != CL_SUCCESS) {
      printf("Error: Failed to write to source array!\n");
      exit(1);
  }

  size_t local[2]  = {16, 16};
  size_t global[2] = {
      ((size_t)B_cols + 15) / 16 * 16, // Map columns to Dim 0 (X axis)
      ((size_t)A_rows + 15) / 16 * 16  // Map rows to Dim 1 (Y axis)      
  };

  if (At) {
    global[0] = ((size_t)B_cols + 15) / 16 * 16;
    global[1] = ((size_t)A_cols + 15) / 16 * 16;
  }

  if (Bt) {
    global[0] = ((size_t)B_rows + 15) / 16 * 16;
    global[1] = ((size_t)A_rows + 15) / 16 * 16;
  }

  err = clEnqueueNDRangeKernel(commands, kernel, 2, NULL, global, local, 0, NULL, NULL);
  if (err != CL_SUCCESS) {
      printf("Error: Failed to execute kernel!\n");
      return err;
  }

  err = clEnqueueReadBuffer(commands, C, CL_TRUE, 0, C_size, C_.elems, 0, NULL, NULL );  
  if (err != CL_SUCCESS) {
      printf("Error: Failed to read output array! %d\n", err);
      exit(1);
  }

  // clFinish(commands);

  return err;
}
