#include <stdio.h>
#include <string.h>

#include "nn.h"

#define IMG_SIZE 28
#define KERN_SIZE 3
#define CONV_OUT (IMG_SIZE - KERN_SIZE + 1)
#define FEATURES 2
#define DENSE_UNITS 1
#define DENSE_IN (FEATURES*CONV_OUT*CONV_OUT)

typedef struct {
    float weight[FEATURES][KERN_SIZE][KERN_SIZE];
    float grad_weight[FEATURES][KERN_SIZE][KERN_SIZE];
} ConvLayer;

typedef struct {
    float weight[DENSE_UNITS][DENSE_IN];
    float grad_weight[DENSE_UNITS][DENSE_IN];
} DenseLayer;

void dense_backward(
    float delta_out[DENSE_UNITS],
    float flat[DENSE_IN],
    float dense_w[DENSE_UNITS][DENSE_IN],
    float grad_w[DENSE_UNITS][DENSE_IN],
    //float grad_b[DENSE_UNITS],
    float delta_in[DENSE_IN]
) {
    // Initialize input gradient to zero
    for(int v = 0; v < DENSE_IN; v++)
        delta_in[v] = 0.0f;

    for(int u = 0; u < DENSE_UNITS; u++) {

        // Bias gradient
        //grad_b[u] = delta_out[u];

        for(int v = 0; v < DENSE_IN; v++) {

            // Weight gradient
            grad_w[u][v] = delta_out[u] * flat[v];

            // Input gradient accumulation
            delta_in[v] += delta_out[u] * dense_w[u][v];
        }
    }
}


void conv_backward(
    float delta_out[FEATURES][CONV_OUT][CONV_OUT],
    NMatrix image,  // float image[IMG_SIZE][IMG_SIZE],
    float conv_w[FEATURES][KERN_SIZE][KERN_SIZE],
    float grad_w[FEATURES][KERN_SIZE][KERN_SIZE],
    //float grad_b[FEATURES],
    float delta_in[IMG_SIZE][IMG_SIZE]
) {

    // Initialize gradients to zero
    for (int f=0; f<FEATURES; f++) {
        //grad_b[f] = 0.0f;
        for(int u=0; u<KERN_SIZE; u++)
        for(int v=0; v<KERN_SIZE; v++)
            grad_w[f][u][v] = 0.0f;
    }

    for (int i=0; i<IMG_SIZE; i++)
    for (int j=0; j<IMG_SIZE; j++)
        delta_in[i][j] = 0.0f;

    // Compute gradients
    for (int f=0; f<FEATURES; f++) {

        for(int i=0; i<CONV_OUT; i++) {
        for(int j=0; j<CONV_OUT; j++) {

            float delta = delta_out[f][i][j];

            // Bias gradient
            // grad_b[f] += delta;

            for(int u=0; u<KERN_SIZE; u++) {
            for(int v=0; v<KERN_SIZE; v++) {
                // Weight gradient
                grad_w[f][u][v] += delta * MAT_AT(image, i+u, j+v);

                // Input gradient
                delta_in[i+u][j+v] += delta * conv_w[f][u][v];
            }}
        }}
    }
}

float rand_float(void) {
  return (rand() / (float)RAND_MAX) * 2.0 - 1.0;
}

int main(void) {
    // --- DATA ---
    NMatrix train_data = read_idx("train-images-idx3-ubyte");

    NMatrix input = mat_row(train_data, 7);
    input.cols = IMG_SIZE;
    input.rows = IMG_SIZE;

    // mat_scale(input, input, 1.0/255.0);
    // output = (input - mean) / std.
    // Normalize((0.1307,), (0.3081,)),
    for (int i = 0; i < IMG_SIZE*IMG_SIZE; i++) {
      input.elems[i] /= 255.0;

      // int r = i / IMG_SIZE;
      // int c = i % IMG_SIZE;
      // //MAT_AT(input, r, c) = (MAT_AT(input, r, c)/255.0 - 0.1307) / 0.3081;
      // MAT_AT(input, r, c) = MAT_AT(input, r, c)/255.0;
      // //MAT_AT(input, r, c) *= MAT_AT(input, r, c);
    }

    float target = 50.0f;

    // --- INITIALIZE LAYERS ---
    ConvLayer conv = {0};
    DenseLayer dense = {0};

    srand(0);

    for (int feature = 0; feature < FEATURES; feature++) {
      for (int i = 0; i < KERN_SIZE*KERN_SIZE; i++) {
          int r = i / KERN_SIZE;
          int c = i % KERN_SIZE;
          conv.weight[feature][r][c] = rand_float() * 0.5;
      }
    }

    for (int d = 0; d < DENSE_UNITS; d++) {
      for (int i = 0; i < DENSE_IN; i++)
        dense.weight[d][i] = rand_float() - 0.05;
    }

    float lr = 0.001;

    for (int iter = 0; iter < 1000; iter++) {
        // --- FORWARD PASS ---
        
      // 1. Conv Layer: conv_out[f] = image * kernel[f]
      float conv_out[FEATURES][CONV_OUT][CONV_OUT];
      for (int feature = 0; feature < FEATURES; feature++) {
        for (int i = 0; i < CONV_OUT; i++) {
            for (int j = 0; j < CONV_OUT; j++) {
                conv_out[feature][i][j] = 0;
                for (int ki = 0; ki < KERN_SIZE; ki++)
                    for (int kj = 0; kj < KERN_SIZE; kj++)
                        conv_out[feature][i][j] += MAT_AT(input, i+ki, j+kj) * conv.weight[feature][ki][kj];
            }
        }
      }

      // 2. ReLU Activation (Element-wise): relu_out[f] = conv_out[f]
      float relu_out[FEATURES][CONV_OUT][CONV_OUT];
      for (int feature = 0; feature < FEATURES; feature++) {
        for (int i = 0; i < CONV_OUT; i++) {
            for (int j = 0; j < CONV_OUT; j++)
                relu_out[feature][i][j] = conv_out[feature][i][j] > 0 ? conv_out[feature][i][j] : 0;
        }
      }

      // 3. Dense Layer (Flatten ReLU output and dot product)
      float flatten[DENSE_IN] = {0};
      int flatten_count = 0;
      for (int feature = 0; feature < FEATURES; feature++) {
        for (int i = 0; i < CONV_OUT; i++) {
            for (int j = 0; j < CONV_OUT; j++)
              flatten[flatten_count++] = relu_out[feature][i][j];
        }
      }

      float final_out[DENSE_UNITS] = {0};
      for (int d = 0; d < DENSE_UNITS; d++) {
        float sum = 0;
        for (int i = 0; i < DENSE_IN; i++) {
          sum += flatten[i] * dense.weight[d][i];
        }
        final_out[d] = sum;
      }

      printf("target %f\n", target);
      for (int d = 0; d < DENSE_UNITS; d++) {
        printf("final_out[%d] = %f\n", d, final_out[d]);
      }

      // --- BACKWARD PASS (The "grad" Chain) ---

      // 1. Loss Gradient (dL/dfinal_out) - MSE Loss: (out - target)^2
      // float grad_loss = 2 * (final_out - target); 
      // printf("Initial Loss Grad: %f\n", grad_loss);
      float delta_out_dense[DENSE_UNITS] = {0};
      float delta_in_dense[DENSE_IN] = {0};

      for (int i = 0; i < DENSE_UNITS; i++)
        delta_out_dense[i] = 2 * (final_out[i] - target);

      dense_backward(
        delta_out_dense,   // float delta_out[DENSE_UNITS],
        flatten,           // float flat[DENSE_IN],
        dense.weight,      // float dense_w[DENSE_UNITS][DENSE_IN],
        dense.grad_weight, // float grad_w[DENSE_UNITS][DENSE_IN],
        // float grad_b[DENSE_UNITS],
        delta_in_dense     // float delta_in[DENSE_IN]
      );

      float delta_out_conv[FEATURES][CONV_OUT][CONV_OUT];
      float delta_in_conv[IMG_SIZE][IMG_SIZE];

      // unflatten + relu
      int unflatten_count = 0;
      for (int feature = 0; feature < FEATURES; feature++) {
        for (int i = 0; i < CONV_OUT; i++) {
            for (int j = 0; j < CONV_OUT; j++) {
              float grad = delta_in_dense[unflatten_count++];
              delta_out_conv[feature][i][j] = grad * (conv_out[feature][i][j] > 0);
            }
        }
      }

      conv_backward(
         delta_out_conv,   // float delta_out[FEATURES][CONV_OUT][CONV_OUT],
         input,            // float image[IMG_SIZE][IMG_SIZE],
         conv.weight,      // float conv_w[FEATURES][KERN_SIZE][KERN_SIZE],
         conv.grad_weight, // float grad_w[FEATURES][KERN_SIZE][KERN_SIZE],
          // float grad_b[FEATURES],
         delta_in_conv     // float delta_in[IMG_SIZE][IMG_SIZE]
      );

#if 0
      for (int i = 0; i < IMG_SIZE; i++) {
          for (int j = 0; j < IMG_SIZE; j++) {
#if 0
            printf("%+.2f ", delta_in_conv[i][j]);
#else
            float x = delta_in_conv[i][j];
            printf("%d ", x*x >= 0.0000001 ? 1 : 0);
#endif
          }
          printf("\n");
      }
      printf("\n");
#endif

      for (int d = 0; d < DENSE_UNITS; d++) {
        for (int i = 0; i < DENSE_IN; i++) {
            dense.weight[d][i] -= lr * dense.grad_weight[d][i];
        }
      }

      for (int feature = 0; feature < FEATURES; feature++) {
        for (int i = 0; i < KERN_SIZE*KERN_SIZE; i++) {
            int r = i / KERN_SIZE;
            int c = i % KERN_SIZE;
            conv.weight[feature][r][c] -= lr * conv.grad_weight[feature][r][c];
        }

        printf("Conv Kernel: %+0.5f %+0.5f %+0.5f\n", conv.weight[feature][0][0], conv.weight[feature][0][1], conv.weight[feature][0][1]);
        printf("Conv Kernel: %+0.5f %+0.5f %+0.5f\n", conv.weight[feature][1][0], conv.weight[feature][1][1], conv.weight[feature][1][1]);
        printf("Conv Kernel: %+0.5f %+0.5f %+0.5f\n", conv.weight[feature][2][0], conv.weight[feature][2][1], conv.weight[feature][2][1]);
      }
#if 0
      // 1. Loss Gradient (dL/dfinal_out) - MSE Loss: (out - target)^2
      float grad_loss = 2 * (final_out - target); 
      printf("Initial Loss Grad: %f\n", grad_loss);

      // 2. Backprop into Dense Layer
      // PyTorch: self: grad * input
      float grad_to_relu[CONV_OUT*CONV_OUT]; 
      for (int i = 0; i < CONV_OUT*CONV_OUT; i++) {
          int r = i / CONV_OUT;
          int c = i % CONV_OUT;
          dense.grad_weight[feature][i] = grad_loss * relu_out[r][c];  // Update for weights
          grad_to_relu[i]      = grad_loss * dense.weight[feature][i]; // 'grad' for the next layer back
      }

      // 3. Backprop into ReLU
      // PyTorch: self: grad * (result > 0)
      float grad_to_conv[CONV_OUT][CONV_OUT];
      for (int i = 0; i < CONV_OUT*CONV_OUT; i++) {
          int r = i / CONV_OUT;
          int c = i % CONV_OUT;
          float grad = grad_to_relu[i];
          grad_to_conv[r][c] = grad * (conv_out[r][c] > 0);
      }

      // 4. Backprop into Conv Layer
      memset(conv.grad_weight, 0, sizeof(conv.grad_weight));
      for (int i = 0; i < CONV_OUT; i++) {
          for (int j = 0; j < CONV_OUT; j++) {
              float grad = grad_to_conv[i][j]; // This is 'grad' in derivatives.yaml
              for (int ki = 0; ki < KERN_SIZE; ki++) {
                  for (int kj = 0; kj < KERN_SIZE; kj++) {
                    conv.grad_weight[feature][ki][kj] += grad * MAT_AT(input, i+ki, j+kj);
                  }
              }
          }
      }

      float lr = 0.00001;

      for (int i = 0; i < CONV_OUT*CONV_OUT; i++) {
          dense.weight[feature][i] -= lr * dense.grad_weight[feature][i];
      }

      for (int i = 0; i < KERN_SIZE*KERN_SIZE; i++) {
          int r = i / KERN_SIZE;
          int c = i % KERN_SIZE;
          conv.weight[feature][r][c] -= lr * conv.grad_weight[feature][r][c];
      }

      printf("Conv Kernel: %+0.5f %+0.5f %+0.5f\n", conv.weight[feature][0][0], conv.weight[feature][0][1], conv.weight[feature][0][1]);
      printf("Conv Kernel: %+0.5f %+0.5f %+0.5f\n", conv.weight[feature][1][0], conv.weight[feature][1][1], conv.weight[feature][1][1]);
      printf("Conv Kernel: %+0.5f %+0.5f %+0.5f\n", conv.weight[feature][2][0], conv.weight[feature][2][1], conv.weight[feature][2][1]);
#endif
    }


#if 0

    for (int feature = 0; feature < DENSE_UNITS; feature++) {
      for (int i = 0; i < CONV_OUT; i++) {
          for (int j = 0; j < CONV_OUT; j++) {
            int k = i*CONV_OUT + j;
#if 1
            printf("%.2f ", dense.weight[feature][k]);
#else
            printf("%d ", dense.weight[feature][k] >= 0.05 ? 1 : 0);
#endif
          }
          printf("\n");
      }
      printf("\n");
    }

#if 0
    for (int i = 0; i < IMG_SIZE; i++) {
        for (int j = 0; j < IMG_SIZE; j++) {
          int k = i*IMG_SIZE + j;
          printf("%d ", input.elems[k] >= 0.5 ? 1 : 0);
        }
        printf("\n");
    }
#endif

#endif

    return 0;
}

