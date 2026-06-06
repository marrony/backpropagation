#include <stdio.h>
#include <string.h>
#include <math.h>

#include "nn.h"
#include "raylib.h"

#define IMG_SIZE 28
#define KERN_SIZE 5
#define CONV_OUT (IMG_SIZE - KERN_SIZE + 1)
#define SCALE 6
#define SCALE_FILTER 20
#define FF 3
#define FEATURES (FF*FF)
#define DENSE_UNITS 10
#define DENSE_IN (FEATURES*CONV_OUT*CONV_OUT)

#define HUGE_FLOAT 10000000000.0f

#define CHECK_FLOAT(x) \
  do { \
    assert(!isnan(x)); \
    assert(!isinf(x)); \
    assert((x) < HUGE_FLOAT); \
  } while (0);

#define CHECK_ARRAY(arr, y) \
  for (int j = 0; j < y; j++)  \
      CHECK_FLOAT((arr)[j]);

#define CHECK_MATRIX(arr, x, y)  \
  for (int i = 0; i < x; i++) \
    CHECK_ARRAY(arr[i], y);

typedef float ConvLayer_W[FEATURES][KERN_SIZE][KERN_SIZE];
typedef float DenseLayer_W[DENSE_UNITS][DENSE_IN];

typedef struct {
    float conv_out[FEATURES][CONV_OUT][CONV_OUT];
} ConvLayer_Output;

typedef struct {
      float final_out[DENSE_UNITS];
} DenseLayer_Output;

typedef struct {
      float relu_out[FEATURES][CONV_OUT][CONV_OUT];
} ReLULayer_Output;

typedef struct {
      float flatten_out[DENSE_IN];
} Flatten_Output;

typedef struct {
      float softmax_out[DENSE_UNITS];
} Softmax_Output;

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
        // CHECK_FLOAT(grad);
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
  assert(unflatten_count == DENSE_IN);
}

void dense_forward(
    float flatten_out[DENSE_IN],
    float dense_w[DENSE_UNITS][DENSE_IN],
    float final_out[DENSE_UNITS]
) {
  for (int d = 0; d < DENSE_UNITS; d++) {
    float sum = 0;
    for (int i = 0; i < DENSE_IN; i++) {
      CHECK_FLOAT(flatten_out[i]);
      CHECK_FLOAT(dense_w[d][i]);
      sum += flatten_out[i] * dense_w[d][i];
    }
    final_out[d] = sum;
  }
}

void dense_backward(
    // inputs
    float delta_out[DENSE_UNITS],
    float flat[DENSE_IN],
    float dense_w[DENSE_UNITS][DENSE_IN],
    // outputs
    float grad_w[DENSE_UNITS][DENSE_IN],
    //float grad_b[DENSE_UNITS],
    float delta_in[DENSE_IN]
) {
  for(int v = 0; v < DENSE_IN; v++)
    delta_in[v] = 0.0f;

  for(int u = 0; u < DENSE_UNITS; u++) {
    float delta = delta_out[u];

    // Bias gradient
    //grad_b[u] = delta;

    for(int v = 0; v < DENSE_IN; v++) {
      // Weight gradient

      CHECK_FLOAT(delta);
      CHECK_FLOAT(flat[v]);

      //printf("%f %f\n", delta, flat[v]);

      grad_w[u][v] = delta * flat[v];

      // Input gradient accumulation
      delta_in[v] += delta * dense_w[u][v];
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
    // inputs
    float delta_out[FEATURES][CONV_OUT][CONV_OUT],
    float image[IMG_SIZE][IMG_SIZE],
    float conv_w[FEATURES][KERN_SIZE][KERN_SIZE],
    // outputs
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

void softmax_forward(
    float dense_output[DENSE_UNITS],
    float softmax_out[DENSE_UNITS]
) {
  softmax(
      mat_init(1, DENSE_UNITS, softmax_out),
      mat_init(1, DENSE_UNITS, dense_output)
  );
}

void softmax_backward(
    float delta_loss[DENSE_UNITS],
    float dense_output[DENSE_UNITS],
    float delta_softmax[DENSE_UNITS]
) {
  dsoftmax(
      mat_init(1, DENSE_UNITS, delta_softmax),
      mat_init(1, DENSE_UNITS, dense_output),
      mat_init(1, DENSE_UNITS, delta_loss)
  );
}

// Uniform random in [0, 1)
float rand_uniform(void) {
  return rand() / (float)RAND_MAX;
}

// Gaussian random using Marsaglia Polar Method
// Generates two independent normal samples at once (Box-Muller style)
// Uses rejection sampling: samples u,v ~ U[-1,1], accepts if u²+v² < 1
float random_normal(float mean, float stddev) {
  static int hasSpare = 0;
  static float spare;

  if (hasSpare) {
    hasSpare = 0;
    return mean + stddev * spare;
  }

  hasSpare = 1;

  float u, v, s;

  do {
    u = rand_uniform() * 2.0f - 1.0f;
    v = rand_uniform() * 2.0f - 1.0f;
    s = u*u + v*v;
  } while (s >= 1.0f || s == 0.0f);

  // Marsaglia formula: z = u * sqrt(-2*ln(s)/s)
  s = sqrtf(-2.0f * logf(s) / s);

  spare = v * s;
  return mean + stddev * (u * s);
}

float clampf(float x) {
  if (x < 0) return 0;
  if (x > 1) return 1;
  return x;
}

typedef struct {
  unsigned char intensity;
  unsigned char alpha;
} Pixel_Alpha;

typedef struct {
  unsigned char intensity;
} Pixel;

// Convert convolution output image to pixel buffer
// Each pixel is scaled to 0-255 range
// IMPORTANT: The pixels buffer can be larger than CONV_OUT x CONV_OUT.
// If pixels is larger, the border regions contain default values (from previous
// normalization). This allows the convolution output to be displayed with
// implicit border padding without needing to explicitly fill the buffer.
void conv_to_pixels(float image[CONV_OUT][CONV_OUT], Pixel_Alpha* pixels) {
  float min = image[0][0];
  float max = image[0][0];

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
      int index = i*IMG_SIZE + j;
      pixels[index].intensity = v * 255;
      pixels[index].alpha = 255;
    }
  }
}

// Convert kernel image to pixel buffer
// Each pixel is scaled to 0-255 range
// IMPORTANT: The pixels buffer can be larger than KERN_SIZE x KERN_SIZE.
// If pixels is larger, the border regions contain default values (from previous
// normalization). This allows the kernel to be displayed with implicit border
// padding without needing to explicitly fill the buffer.
void filter_to_pixels(float image[KERN_SIZE][KERN_SIZE], Pixel* pixels) {
  float min = image[0][0];
  float max = image[0][0];

  for (int i = 0; i < KERN_SIZE; i++) {
    for (int j = 0; j < KERN_SIZE; j++) {
      float v = image[i][j];
      if (v > max) max = v;
      if (v < min) min = v;
    }
  }

  float diff = max - min;

  for (int i = 0; i < KERN_SIZE; i++) {
    for (int j = 0; j < KERN_SIZE; j++) {
      float v = (image[i][j] - min) / diff;
      pixels[i*KERN_SIZE + j].intensity = v * 255;
    }
  }
}

int main(void) {
  int padding = 20;
  int marging = 20;
  int offset_filter = FF*(SCALE*IMG_SIZE + padding);
  int WindowWidth = offset_filter + marging + FF*(SCALE_FILTER*KERN_SIZE + padding);
  int WindowHeight = offset_filter + marging;
  InitWindow(WindowWidth, WindowHeight, "Convolution Neural Network");
  SetWindowPosition(0, 0);

  Font font = LoadFontEx("fonts/MonacoNerdFont-Regular.ttf", 128, NULL, 95);
  Pixel_Alpha pixels[IMG_SIZE*IMG_SIZE] = {0};
  Pixel pixels_filter[KERN_SIZE*KERN_SIZE] = {0};

  Image image = {
    .data = pixels,
    .width = IMG_SIZE,
    .height = IMG_SIZE,
    .format = PIXELFORMAT_UNCOMPRESSED_GRAY_ALPHA,
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
  NMatrix train_data = read_idx("train-images-idx3-ubyte", .normalize = true);
  NMatrix label_data = read_idx("train-labels-idx1-ubyte", .normalize = false);

  assert(train_data.rows == label_data.rows);

  int train_data_size = 500; //train_data.rows;

  // mat_scale(input, input, 1.0/255.0);
  // output = (input - mean) / std.
  // Normalize((0.1307,), (0.3081,)),

  static ConvLayer_W conv_weight = {0};
  static ConvLayer_W grad_conv_weight = {0};
  static ConvLayer_W g_grad_conv_weight = {0};

  static DenseLayer_W dense_weight = {0};
  static DenseLayer_W grad_dense_weight = {0};
  static DenseLayer_W g_grad_dense_weight = {0};

  // srand(time(0));
  srand(0);

  float std_conv = sqrtf(2.0f / (FEATURES*KERN_SIZE*KERN_SIZE));
  for (int feature = 0; feature < FEATURES; feature++) {
    for (int i = 0; i < KERN_SIZE*KERN_SIZE; i++) {
      int r = i / KERN_SIZE;
      int c = i % KERN_SIZE;
      conv_weight[feature][r][c] = random_normal(0, std_conv);
    }
  }

  float std_dense = sqrtf(2.0f / (DENSE_UNITS*DENSE_IN));
  for (int d = 0; d < DENSE_UNITS; d++) {
    for (int i = 0; i < DENSE_IN; i++)
      dense_weight[d][i] = random_normal(0, std_dense);
  }

  float lr = 0.001;

  int index_to_draw = 0;

  bool learning = true;
  int iteration = 0;

  while (!WindowShouldClose()) {

    if (IsKeyPressed(KEY_SPACE)) 
      learning = !learning;

    if (IsKeyPressed(KEY_UP) && index_to_draw < train_data_size)
      index_to_draw += 1;

    if (IsKeyPressed(KEY_DOWN) && index_to_draw > 0)
      index_to_draw -= 1;

    memset(g_grad_conv_weight, 0, sizeof(g_grad_conv_weight));
    memset(g_grad_dense_weight, 0, sizeof(g_grad_dense_weight));

    for (int train_index = 0; train_index < train_data_size; train_index++) {
      NMatrix input = mat_row(train_data, train_index);
      NMatrix label = mat_row(label_data, train_index);

      float target[DENSE_UNITS] = {0};
      target[(int)MAT_AT(label, 0, 0)] = 1;

      input.cols = IMG_SIZE;
      input.rows = IMG_SIZE;

      float input_image[IMG_SIZE][IMG_SIZE];
      for (int i = 0; i < IMG_SIZE; i++) {
          for (int j = 0; j < IMG_SIZE; j++) {
            input_image[i][j] = MAT_AT(input, i, j);
          }
      }

      // --- FORWARD PASS ---

      CHECK_MATRIX(dense_weight, DENSE_UNITS, DENSE_IN);
      for (int f = 0; f < FEATURES; f++)
        CHECK_MATRIX(conv_weight[f], KERN_SIZE, KERN_SIZE);

      // 1. Conv Layer
      static ConvLayer_Output conv_output;
      conv_forward(input_image, conv_weight, conv_output.conv_out);

      for (int f = 0; f < FEATURES; f++)
        CHECK_MATRIX(conv_output.conv_out[f], CONV_OUT, CONV_OUT);

      // 2. ReLU Activation
      static ReLULayer_Output relu_output;
      relu_forward(conv_output.conv_out, relu_output.relu_out);

      for (int f = 0; f < FEATURES; f++)
        CHECK_MATRIX(relu_output.relu_out[f], CONV_OUT, CONV_OUT);

      // 3. Flatten ReLU
      static Flatten_Output flatten_output;
      flatten_forwared(relu_output.relu_out, flatten_output.flatten_out);

      CHECK_ARRAY(flatten_output.flatten_out, DENSE_IN);

      // 4. Dense Layer
      static DenseLayer_Output dense_output;
      dense_forward(flatten_output.flatten_out, dense_weight, dense_output.final_out);

      CHECK_ARRAY(dense_output.final_out, DENSE_UNITS);

      // 5. Softmax Layer
      static Softmax_Output softmax_output;
      softmax_forward(dense_output.final_out, softmax_output.softmax_out);

      CHECK_ARRAY(softmax_output.softmax_out, DENSE_UNITS);

      // 1. Loss Gradient (dL/dfinal_out) - MSE Loss: (out - target)^2
      static float delta_loss[DENSE_UNITS];

      float L = 0;

      for (int i = 0; i < DENSE_UNITS; i++) {
        float diff = softmax_output.softmax_out[i] - target[i];
        delta_loss[i] = 2 * diff;
        L += diff * diff;
      }

      CHECK_ARRAY(delta_loss, DENSE_UNITS);

      // --- BACKWARD PASS (The "grad" Chain) ---
      if (learning) {
        // 2. Softmax
        static float delta_softmax[DENSE_UNITS];
        softmax_backward(
            delta_loss,
            softmax_output.softmax_out,
            delta_softmax
        );

        CHECK_ARRAY(delta_softmax, DENSE_UNITS);

        // 3. Dense
        static float delta_dense[DENSE_IN];
        dense_backward(
          delta_softmax,              // float delta_out[DENSE_UNITS],
          flatten_output.flatten_out, // float flat[DENSE_IN],
          dense_weight,               // float dense_w[DENSE_UNITS][DENSE_IN],
          // outputs
          grad_dense_weight,          // float grad_w[DENSE_UNITS][DENSE_IN],
                                      // float grad_b[DENSE_UNITS],
          delta_dense                 // float delta_in[DENSE_IN]
        );

        CHECK_MATRIX(grad_dense_weight, DENSE_UNITS, DENSE_IN);
        CHECK_ARRAY(delta_dense, DENSE_IN);

        // 4. Unflatten
        static float delta_dense_unflatten[FEATURES][CONV_OUT][CONV_OUT];
        flatten_backward(
          delta_dense,
          delta_dense_unflatten
        );

        for (int f = 0; f < FEATURES; f++)
          CHECK_MATRIX(delta_dense_unflatten[f], CONV_OUT, CONV_OUT);

        // 5. Relu
        static float delta_relu[FEATURES][CONV_OUT][CONV_OUT];
        relu_backward(
            delta_dense_unflatten,
            conv_output.conv_out,
            delta_relu
        );

        for (int f = 0; f < FEATURES; f++)
          CHECK_MATRIX(delta_relu[f], CONV_OUT, CONV_OUT);

        // 6. Conv
        static float delta_conv[IMG_SIZE][IMG_SIZE];
        conv_backward(
           // inputs
           delta_relu,       // float delta_out[FEATURES][CONV_OUT][CONV_OUT],
           input_image,      // float image[IMG_SIZE][IMG_SIZE],
           conv_weight,      // float conv_w[FEATURES][KERN_SIZE][KERN_SIZE],
           // outputs
           grad_conv_weight, // float grad_w[FEATURES][KERN_SIZE][KERN_SIZE],
                             // float grad_b[FEATURES],
           delta_conv        // float delta_in[IMG_SIZE][IMG_SIZE]
        );

        for (int f = 0; f < FEATURES; f++)
          CHECK_MATRIX(grad_conv_weight[f], KERN_SIZE, KERN_SIZE);

        CHECK_MATRIX(delta_conv, IMG_SIZE, IMG_SIZE);

        for (int d = 0; d < DENSE_UNITS; d++) {
          for (int i = 0; i < DENSE_IN; i++) {
            g_grad_dense_weight[d][i] += grad_dense_weight[d][i];
          }
        }

        for (int feature = 0; feature < FEATURES; feature++) {
          for (int i = 0; i < KERN_SIZE; i++) {
            for (int j = 0; j < KERN_SIZE; j++) {
              g_grad_conv_weight[feature][i][j] += grad_conv_weight[feature][i][j];
            }
          }
        }
      }

      if (train_index == index_to_draw) {
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
          filter_to_pixels(conv_weight[feature], pixels_filter);
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

        char text[1024];
        snprintf(
            text,
            sizeof(text),
            "0=%+.2f 1=%+.2f 2=%+.2f 3=%+.2f 4=%+.2f 5=%+.2f 6=%+.2f 7=%+.2f 8=%+.2f 9=%+.2f",
            softmax_output.softmax_out[0],
            softmax_output.softmax_out[1],
            softmax_output.softmax_out[2],
            softmax_output.softmax_out[3],
            softmax_output.softmax_out[4],
            softmax_output.softmax_out[5],
            softmax_output.softmax_out[6],
            softmax_output.softmax_out[7],
            softmax_output.softmax_out[8],
            softmax_output.softmax_out[9]
        );

        int index = mat_row_max(mat_init(1, DENSE_UNITS, softmax_output.softmax_out));

        Color color = index == (int)MAT_AT(label, 0, 0) ? GREEN : RED;

        DrawTextEx(font, text, (Vector2) {.x = 22, .y = 21}, 30, 0, BLACK);
        DrawTextEx(font, text, (Vector2) {.x = 20, .y = 20}, 30, 0, color);

        snprintf(text, sizeof(text), "Learning = %s", learning ? "yes" : "no");
        DrawTextEx(font, text, (Vector2) {.x = 22, .y = 51}, 30, 0, BLACK);
        DrawTextEx(font, text, (Vector2) {.x = 20, .y = 50}, 30, 0, color);

        snprintf(text, sizeof(text), "Cost = %+.8f", L);
        DrawTextEx(font, text, (Vector2) {.x = 22, .y = 81}, 30, 0, BLACK);
        DrawTextEx(font, text, (Vector2) {.x = 20, .y = 80}, 30, 0, color);

        snprintf(text, sizeof(text), "Iteration = %d", iteration);
        DrawTextEx(font, text, (Vector2) {.x = 22, .y = 111}, 30, 0, BLACK);
        DrawTextEx(font, text, (Vector2) {.x = 20, .y = 110}, 30, 0, color);

        EndDrawing();
      }
    }

    if (learning) {
      for (int d = 0; d < DENSE_UNITS; d++) {
        for (int i = 0; i < DENSE_IN; i++) {
          dense_weight[d][i] -= lr * g_grad_dense_weight[d][i];
        }
      }

      for (int feature = 0; feature < FEATURES; feature++) {
        for (int i = 0; i < KERN_SIZE; i++) {
          for (int j = 0; j < KERN_SIZE; j++) {
            conv_weight[feature][i][j] -= lr * g_grad_conv_weight[feature][i][j];
          }
        }
      }
    }

    iteration += 1;
  }

  UnloadFont(font);
  CloseWindow();

  return 0;
}

