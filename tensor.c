#include <stdio.h>
#include <string.h>

#include "nn.h"
#include "raylib.h"

#define IMG_SIZE 28
#define KERN_SIZE 5
#define CONV_OUT (IMG_SIZE - KERN_SIZE + 1)
#define SCALE 8
#define SCALE_FILTER 20
#define FF 4
#define FEATURES (FF*FF)
#define DENSE_UNITS 1
#define DENSE_IN (FEATURES*CONV_OUT*CONV_OUT)

typedef struct {
    float weight[FEATURES][KERN_SIZE][KERN_SIZE];
    float grad_weight[FEATURES][KERN_SIZE][KERN_SIZE];
} ConvLayer;

typedef struct {
    float conv_out[FEATURES][CONV_OUT][CONV_OUT];
} ConvLayer_Output;

typedef struct {
    float weight[DENSE_UNITS][DENSE_IN];
    float grad_weight[DENSE_UNITS][DENSE_IN];
} DenseLayer;

typedef struct {
      float final_out[DENSE_UNITS];
} DenseLayer_Output;

typedef struct {
      float relu_out[FEATURES][CONV_OUT][CONV_OUT];
} ReLULayer_Output;

typedef struct {
      float flatten_out[DENSE_IN];
} Flatten_Output;

void relu_forward(
    float conv_out[FEATURES][CONV_OUT][CONV_OUT],
    float relu_out[FEATURES][CONV_OUT][CONV_OUT]
) {
  for (int feature = 0; feature < FEATURES; feature++) {
    for (int i = 0; i < CONV_OUT; i++) {
      for (int j = 0; j < CONV_OUT; j++)
        relu_out[feature][i][j] = conv_out[feature][i][j] > 0 ? conv_out[feature][i][j] : 0;
    }
  }
}

void relu_backward(
    float delta_out_relu[FEATURES][CONV_OUT][CONV_OUT],
    float conv_out[FEATURES][CONV_OUT][CONV_OUT],
    float delta_out_conv[FEATURES][CONV_OUT][CONV_OUT]
) {
  for (int feature = 0; feature < FEATURES; feature++) {
    for (int i = 0; i < CONV_OUT; i++) {
      for (int j = 0; j < CONV_OUT; j++) {
        float grad = delta_out_relu[feature][i][j];
        delta_out_conv[feature][i][j] = grad * (conv_out[feature][i][j] > 0);
      }
    }
  }
}

void flatten_forwared(
    float relu_out[FEATURES][CONV_OUT][CONV_OUT],
    float flatten_out[DENSE_IN]
) {
  int flatten_count = 0;
  for (int feature = 0; feature < FEATURES; feature++) {
    for (int i = 0; i < CONV_OUT; i++) {
      for (int j = 0; j < CONV_OUT; j++)
        flatten_out[flatten_count++] = relu_out[feature][i][j];
    }
  }
}

void flatten_backward(
    float delta_in_dense[DENSE_IN],
    float delta_out_relu[FEATURES][CONV_OUT][CONV_OUT]
) {
  int unflatten_count = 0;
  for (int feature = 0; feature < FEATURES; feature++) {
    for (int i = 0; i < CONV_OUT; i++) {
      for (int j = 0; j < CONV_OUT; j++) {
        delta_out_relu[feature][i][j] = delta_in_dense[unflatten_count++];
      }
    }
  }
}

void dense_forward(
    float flatten_out[DENSE_IN],
    float dense_w[DENSE_UNITS][DENSE_IN],
    float final_out[DENSE_UNITS]
) {
  for (int d = 0; d < DENSE_UNITS; d++) {
    float sum = 0;
    for (int i = 0; i < DENSE_IN; i++) {
      sum += flatten_out[i] * dense_w[d][i];
    }
    final_out[d] = sum;
  }
}

void dense_backward(
    float delta_out[DENSE_UNITS],
    float flat[DENSE_IN],
    float dense_w[DENSE_UNITS][DENSE_IN],
    float grad_w[DENSE_UNITS][DENSE_IN],
    //float grad_b[DENSE_UNITS],
    float delta_in[DENSE_IN]
) {
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

void conv_forward(
    float input[IMG_SIZE][IMG_SIZE],
    float conv_w[FEATURES][KERN_SIZE][KERN_SIZE],
    float conv_out[FEATURES][CONV_OUT][CONV_OUT]
) {
  for (int feature = 0; feature < FEATURES; feature++) {
    for (int i = 0; i < CONV_OUT; i++) {
      for (int j = 0; j < CONV_OUT; j++) {
        conv_out[feature][i][j] = 0;  //bias[feature]
                                      //
        for (int ki = 0; ki < KERN_SIZE; ki++) {
          for (int kj = 0; kj < KERN_SIZE; kj++)
            conv_out[feature][i][j] += input[i+ki][j+kj] * conv_w[feature][ki][kj];
        }
      }
    }
  }
}

void conv_backward(
    float delta_out[FEATURES][CONV_OUT][CONV_OUT],
    float image[IMG_SIZE][IMG_SIZE],
    float conv_w[FEATURES][KERN_SIZE][KERN_SIZE],
    float grad_w[FEATURES][KERN_SIZE][KERN_SIZE],
    //float grad_b[FEATURES],
    float delta_in[IMG_SIZE][IMG_SIZE]
) {
  // Initialize gradients to zero
  for (int f = 0; f < FEATURES; f++) {
    //grad_b[f] = 0.0f;
    for(int u = 0; u < KERN_SIZE; u++)
      for(int v = 0; v < KERN_SIZE; v++)
        grad_w[f][u][v] = 0.0f;
  }

  for (int i = 0; i < IMG_SIZE; i++) {
    for (int j = 0; j < IMG_SIZE; j++)
        delta_in[i][j] = 0.0f;
  }

  // Compute gradients
  for (int f = 0; f < FEATURES; f++) {
    for (int i = 0; i < CONV_OUT; i++) {
      for (int j = 0; j < CONV_OUT; j++) {
        float delta = delta_out[f][i][j];

        // Bias gradient
        // grad_b[f] += delta;

        for(int u = 0; u < KERN_SIZE; u++) {
          for(int v = 0; v < KERN_SIZE; v++) {
            // Weight gradient
            grad_w[f][u][v] += delta * image[i+u][j+v];

            // Input gradient
            delta_in[i+u][j+v] += delta * conv_w[f][u][v];
          }
        }
      }
    }
  }
}

float rand_float(void) {
  return (rand() / (float)RAND_MAX) * 2.0 - 1.0;
}

float clampf(float x) {
  if (x < 0) return 0;
  if (x > 1) return 1;
  return x;
}

void conv_to_pixels(float image[CONV_OUT][CONV_OUT], unsigned char* pixels) {
  float min = +FLT_MAX;
  float max = -FLT_MAX;

  for (int i = 0; i < CONV_OUT; i++) {
    for (int j = 0; j < CONV_OUT; j++) {
      float v = image[i][j];
      if (v > max) max = v;
      if (v < min) min = v;
    }
  }

  float diff = max - min;

  for (int i = 0; i < CONV_OUT; i++) {
    for (int j = 0; j < CONV_OUT; j++) {
      float v = (image[i][j] - min) / diff;
      pixels[i*IMG_SIZE + j] = v * 255;
    }
  }
}

int main(void) {
  int padding = 20;
  int marging = 20;
  int offset_filter = FF*(SCALE*IMG_SIZE + padding);
  int WindowWidth = offset_filter + marging + FF*(SCALE_FILTER*KERN_SIZE + padding);
  int WindowHeight = offset_filter + marging;
  InitWindow(WindowWidth, WindowHeight, "Tensors");
  SetWindowPosition(0, 0);

  Font font = LoadFontEx("fonts/MonacoNerdFont-Regular.ttf", 128, NULL, 95);
  unsigned char pixels[IMG_SIZE*IMG_SIZE] = {0};
  unsigned char pixels_filter[KERN_SIZE*KERN_SIZE] = {0};

  Image image = {
    .data = pixels,
    .width = IMG_SIZE,
    .height = IMG_SIZE,
    .format = PIXELFORMAT_UNCOMPRESSED_GRAYSCALE,
    .mipmaps = 1,
  };

  Image image_filter = {
    .data = pixels_filter,
    .width = KERN_SIZE,
    .height = KERN_SIZE,
    .format = PIXELFORMAT_UNCOMPRESSED_GRAYSCALE,
    .mipmaps = 1,
  };

  Texture2D textures[FEATURES];
  Texture2D textures_filter[FEATURES];
  for (int feature = 0; feature < FEATURES; feature++) {
    textures[feature] = LoadTextureFromImage(image);
    textures_filter[feature] = LoadTextureFromImage(image_filter);
  }

  // --- DATA ---
  NMatrix train_data = read_idx("train-images-idx3-ubyte");

  NMatrix input = mat_row(train_data, 4);
  // NMatrix input_texture = input;
  input.cols = IMG_SIZE;
  input.rows = IMG_SIZE;

  // mat_scale(input, input, 1.0/255.0);
  // output = (input - mean) / std.
  // Normalize((0.1307,), (0.3081,)),
  for (int i = 0; i < IMG_SIZE*IMG_SIZE; i++) {
    input.elems[i] /= 255.0;
  }

  float target = 1.0f;

  ConvLayer conv = {0};
  DenseLayer dense = {0};

  srand(time(0));
  // srand(0);

  for (int feature = 0; feature < FEATURES; feature++) {
    for (int i = 0; i < KERN_SIZE*KERN_SIZE; i++) {
      int r = i / KERN_SIZE;
      int c = i % KERN_SIZE;
      conv.weight[feature][r][c] = rand_float() * 0.5;
    }
  }

  for (int d = 0; d < DENSE_UNITS; d++) {
    for (int i = 0; i < DENSE_IN; i++)
      dense.weight[d][i] = rand_float() * 0.05;
  }

  float input_image[IMG_SIZE][IMG_SIZE];
  for (int i = 0; i < IMG_SIZE; i++) {
      for (int j = 0; j < IMG_SIZE; j++) {
        input_image[i][j] = MAT_AT(input, i, j);
      }
  }

  float lr = 0.0005;

  while (!WindowShouldClose()) {
    // --- FORWARD PASS ---
      
    // 1. Conv Layer
    ConvLayer_Output conv_output;
    conv_forward(input_image, conv.weight, conv_output.conv_out);

    // 2. ReLU Activation
    ReLULayer_Output relu_output;
    relu_forward(conv_output.conv_out, relu_output.relu_out);

    // 3. Flatten ReLU
    Flatten_Output flatten_output;
    flatten_forwared(relu_output.relu_out, flatten_output.flatten_out);

    // 4. Dense Layer
    DenseLayer_Output dense_output;
    dense_forward(flatten_output.flatten_out, dense.weight, dense_output.final_out);

    BeginDrawing();
    ClearBackground(BEIGE);

    for (int feature = 0; feature < FEATURES; feature++) {
      int feature_x = feature % FF;
      int feature_y = feature / FF;

      // draw feature maps
      memset(pixels, 0, sizeof(pixels));
      conv_to_pixels(conv_output.conv_out[feature], pixels);
      UpdateTexture(textures[feature], pixels);
      DrawTextureEx(
          textures[feature],
          (Vector2) {.x = feature_x*(SCALE*IMG_SIZE + padding) + marging, .y = feature_y*(SCALE*IMG_SIZE + padding) + marging},
          0,
          SCALE,
          WHITE
      );

      // draw filters
      memset(pixels_filter, 0, sizeof(pixels_filter));
      for (int i = 0; i < KERN_SIZE; i++) {
        for (int j = 0; j < KERN_SIZE; j++) {
          pixels_filter[i*KERN_SIZE + j] = conv.weight[feature][i][j] * 255;
        }
      }
      UpdateTexture(textures_filter[feature], pixels_filter);
      int offset = offset_filter + padding;
      DrawTextureEx(
          textures_filter[feature],
          (Vector2) {.x = offset + feature_x*(SCALE_FILTER*KERN_SIZE + padding), .y = feature_y*(SCALE*IMG_SIZE + padding) + marging},
          0,
          SCALE_FILTER,
          WHITE
      );
    }

    EndDrawing();

    // --- BACKWARD PASS (The "grad" Chain) ---

    // 1. Loss Gradient (dL/dfinal_out) - MSE Loss: (out - target)^2
    float delta_loss[DENSE_UNITS];
    for (int i = 0; i < DENSE_UNITS; i++)
      delta_loss[i] = 2 * (dense_output.final_out[i] - target);

    // 2. Dense
    float delta_dense[DENSE_IN];
    dense_backward(
      delta_loss,                 // float delta_out[DENSE_UNITS],
      flatten_output.flatten_out, // float flat[DENSE_IN],
      dense.weight,               // float dense_w[DENSE_UNITS][DENSE_IN],
      dense.grad_weight,          // float grad_w[DENSE_UNITS][DENSE_IN],
                                  // float grad_b[DENSE_UNITS],
      delta_dense                 // float delta_in[DENSE_IN]
    );

    // 3. Unflatten
    float delta_unflatten[FEATURES][CONV_OUT][CONV_OUT];
    flatten_backward(
      delta_dense,
      delta_unflatten
    );
    
    // 4. Relu
    float delta_relu[FEATURES][CONV_OUT][CONV_OUT];
    relu_backward(
        delta_unflatten,
        conv_output.conv_out,
        delta_relu
    );
    
    // 5. Conv
    float delta_conv[IMG_SIZE][IMG_SIZE];
    conv_backward(
       delta_relu,       // float delta_out[FEATURES][CONV_OUT][CONV_OUT],
       input_image,      // float image[IMG_SIZE][IMG_SIZE],
       conv.weight,      // float conv_w[FEATURES][KERN_SIZE][KERN_SIZE],
       conv.grad_weight, // float grad_w[FEATURES][KERN_SIZE][KERN_SIZE],
                         // float grad_b[FEATURES],
       delta_conv        // float delta_in[IMG_SIZE][IMG_SIZE]
    );

    for (int d = 0; d < DENSE_UNITS; d++) {
      for (int i = 0; i < DENSE_IN; i++) {
        dense.weight[d][i] -= lr * dense.grad_weight[d][i];
      }
    }

    for (int feature = 0; feature < FEATURES; feature++) {
      for (int i = 0; i < KERN_SIZE; i++) {
        for (int j = 0; j < KERN_SIZE; j++) {
          conv.weight[feature][i][j] -= lr * conv.grad_weight[feature][i][j];
        }
      }
    }
  }

  UnloadFont(font);
  CloseWindow();

  return 0;
}

