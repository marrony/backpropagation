#include <stdio.h>
#include <string.h>
#include <math.h>

#include "nn.h"
#include "node.h"
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

// Uniform random in (0,1)
float rand_uniform(void) {
  return (rand() + 1.0f) / (RAND_MAX + 2.0f);
}

// Gaussian random using Box-Muller
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

void conv_to_pixels(NMatrix image, Pixel_Alpha* pixels) {
  float min = MAT_AT(image, 0, 0);
  float max = MAT_AT(image, 0, 0);

  for (int i = 0; i < CONV_OUT; i++) {
    for (int j = 0; j < CONV_OUT; j++) {
      float v = MAT_AT(image, i, j);
      if (v > max) max = v;
      if (v < min) min = v;
    }
  }

  float diff = max - min;

  for (int i = 0; i < CONV_OUT; i++) {
    for (int j = 0; j < CONV_OUT; j++) {
      float v = (MAT_AT(image, i, j) - min) / diff;
      int index = i*IMG_SIZE + j;
      pixels[index].intensity = v * 255;
      pixels[index].alpha = 255;
    }
  }
}

void filter_to_pixels(NMatrix image, Pixel* pixels) {
  float min = MAT_AT(image, 0, 0);
  float max = MAT_AT(image, 0, 0);

  for (int i = 0; i < KERN_SIZE; i++) {
    for (int j = 0; j < KERN_SIZE; j++) {
      float v = MAT_AT(image, i, j);
      if (v > max) max = v;
      if (v < min) min = v;
    }
  }

  float diff = max - min;

  for (int i = 0; i < KERN_SIZE; i++) {
    for (int j = 0; j < KERN_SIZE; j++) {
      float v = (MAT_AT(image, i, j) - min) / diff;
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

  int train_data_size = 1000; //train_data.rows;

  // mat_scale(input, input, 1.0/255.0);
  // output = (input - mean) / std.
  // Normalize((0.1307,), (0.3081,)),

  // srand(time(0));
  srand(0);

  float lr = 0.001;

  int index_to_draw = 0;

  Node* conv[FEATURES] = {0};
  Node* relu[FEATURES] = {0};

  Node* input_img = create_constant(1, IMG_SIZE*IMG_SIZE);

  for (int f = 0; f < FEATURES; f++) {
    conv[f] = create_conv2d(input_img, IMG_SIZE, KERN_SIZE);
    relu[f] = create_relu(conv[f]);
  }
  Node* flatten = create_flatten(relu, FEATURES);
  Node* linear = create_linear(flatten, DENSE_UNITS);
  Node* softmax = create_softmax(linear);

  float std_conv = sqrtf(2.0f / (FEATURES*KERN_SIZE*KERN_SIZE));
  float std_dense = sqrtf(2.0f / (DENSE_UNITS*DENSE_IN));

  for (int f = 0; f < FEATURES; f++) {
    for (int i = 0; i < KERN_SIZE*KERN_SIZE; i++) {
      int r = i / KERN_SIZE;
      int c = i % KERN_SIZE;
      MAT_AT(conv[f]->params[KERN_SLOT].value, r, c) = random_normal(0, std_conv);
    }
  }

  for (int d = 0; d < DENSE_UNITS; d++) {
    for (int i = 0; i < DENSE_IN; i++) {
      MAT_AT(linear->params[W_SLOT].value, d, i) = random_normal(0, std_dense);
      VEC_AT(linear->params[B_SLOT].value, d) = 0;
    }
  }

  NMatrix dL = mat_alloc(1, DENSE_UNITS);

  bool learning = true;
  int iteration = 0;

  while (!WindowShouldClose()) {

    if (IsKeyPressed(KEY_SPACE)) 
      learning = !learning;

    if (IsKeyPressed(KEY_UP) && index_to_draw < train_data_size)
      index_to_draw += 1;

    if (IsKeyPressed(KEY_DOWN) && index_to_draw > 0)
      index_to_draw -= 1;

    zero_grads(softmax);

    for (int train_index = 0; train_index < train_data_size; train_index++) {
      NMatrix input = mat_row(train_data, train_index);
      NMatrix label = mat_row(label_data, train_index);

      float target[DENSE_UNITS] = {0};
      target[(int)MAT_AT(label, 0, 0)] = 1;

      mat_copy(input_img->params[X_SLOT].value, input);
      node_forward(softmax);

      if (train_index == index_to_draw) {
        BeginDrawing();
        ClearBackground(BEIGE);

        for (int feature = 0; feature < FEATURES; feature++) {
          int feature_x = feature % FF;
          int feature_y = feature / FF;

          // draw feature maps
          memset(pixels, 0, sizeof(pixels));
          conv_to_pixels(conv[feature]->params[X_SLOT].value, pixels);
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
          filter_to_pixels(conv[feature]->params[W_SLOT].value, pixels_filter);
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
            VEC_AT(softmax->params[X_SLOT].value, 0),
            VEC_AT(softmax->params[X_SLOT].value, 1),
            VEC_AT(softmax->params[X_SLOT].value, 2),
            VEC_AT(softmax->params[X_SLOT].value, 3),
            VEC_AT(softmax->params[X_SLOT].value, 4),
            VEC_AT(softmax->params[X_SLOT].value, 5),
            VEC_AT(softmax->params[X_SLOT].value, 6),
            VEC_AT(softmax->params[X_SLOT].value, 7),
            VEC_AT(softmax->params[X_SLOT].value, 8),
            VEC_AT(softmax->params[X_SLOT].value, 9)
        );

        int index = mat_row_max(softmax->params[X_SLOT].value);

        Color color = index == (int)MAT_AT(label, 0, 0) ? GREEN : RED;

        DrawTextEx(font, text, (Vector2) {.x = 22, .y = 21}, 30, 0, BLACK);
        DrawTextEx(font, text, (Vector2) {.x = 20, .y = 20}, 30, 0, color);

        snprintf(text, sizeof(text), "Learning = %s", learning ? "yes" : "no");
        DrawTextEx(font, text, (Vector2) {.x = 22, .y = 51}, 30, 0, BLACK);
        DrawTextEx(font, text, (Vector2) {.x = 20, .y = 50}, 30, 0, color);

        snprintf(text, sizeof(text), "Iteration = %d", iteration);
        DrawTextEx(font, text, (Vector2) {.x = 22, .y = 81}, 30, 0, BLACK);
        DrawTextEx(font, text, (Vector2) {.x = 20, .y = 80}, 30, 0, color);

        EndDrawing();
      }

      if (learning) {
        for (int i = 0; i < DENSE_UNITS; i++)
          VEC_AT(dL, i) = 2 * (VEC_AT(softmax->params[X_SLOT].value, i) - target[i]);

        node_backward(softmax, dL);

        // if (iteration == 0) {
        //   printf("output  ="); mat_println(stdout, softmax->output[X_SLOT]);
        //   printf("softmax ="); mat_println(stdout, softmax->input[0]->delta[X_SLOT]);
        //   printf("linear.x="); mat_println(stdout, mat_row_slice(linear->input[0]->delta[X_SLOT], 0, DENSE_UNITS));
        //   for (int i = 0; i < DENSE_UNITS/5; i++) {
        //     printf("linear.w=");
        //     mat_println(stdout, mat_row_slice(mat_row(linear->delta[W_SLOT], i), 0, DENSE_UNITS));
        //   }
        //   for (int i = 0; i < FEATURES; i++) {
        //     printf("relu    =");
        //     mat_println(stdout, mat_row_slice(mat_row(relu[i]->input[0]->delta[X_SLOT], 0), 0, 10));
        //   }
        //   for (int i = 0; i < FEATURES; i++) {
        //     printf("conv.w  =");
        //     //mat_println(stdout, conv[i]->delta[KERN_SLOT]);
        //     mat_println(stdout, conv[i]->delta[KERN_SLOT]);
        //   }
        //   printf("=================\n");
        //   if (train_index == 0)
        //     exit(0);
        // }
        acc_grads(softmax);
      }
    }

    update_grads(softmax, lr);

    iteration += 1;
  }

  UnloadFont(font);
  CloseWindow();

  return 0;
}

// cnn-node
// output  =[+0.10614382 +0.09738087 +0.09332450 +0.11095963 +0.10072238 +0.10284732 +0.09771184 +0.10203808 +0.09754180 +0.09132969]
// softmax =[+0.02307198 +0.01946053 +0.01789279 +0.02518749 +0.02080143 -0.18401727 +0.01959135 +0.02134165 +0.01952409 +0.01714596]
// linear.x=[-0.00014116 -0.00089083 +0.00128763 +0.00117606 -0.00018488 +0.00149464 -0.00355611 +0.00109969 -0.00210105 -0.00097707]
// linear.w=[+0.00000000 +0.00000000 +0.00000000 +0.00000000 +0.00000000 +0.00000000 +0.00000000 +0.00000000 +0.00000000 +0.00000000]
// linear.w=[+0.00000000 +0.00000000 +0.00000000 +0.00000000 +0.00000000 +0.00000000 +0.00000000 +0.00000000 +0.00000000 +0.00000000]
// relu    =[-0.00000000 -0.00000000 +0.00000000 +0.00000000 -0.00000000 +0.00000000 -0.00000000 +0.00000000 -0.00000000 -0.00000000]
// relu    =[+0.00000000 -0.00000000 +0.00000000 -0.00000000 -0.00000000 -0.00000000 +0.00000000 -0.00000000 -0.00000000 -0.00000000]
// relu    =[-0.00000000 -0.00000000 -0.00000000 +0.00000000 -0.00000000 -0.00000000 +0.00000000 -0.00000000 -0.00000000 -0.00000000]
// relu    =[+0.00000000 -0.00000000 +0.00000000 +0.00000000 +0.00000000 -0.00000000 +0.00000000 -0.00000000 -0.00000000 +0.00000000]
// conv.w  =[[+0.00242896 -0.00019978 +0.00274303] [+0.00025668 -0.00011360 +0.00462382] [-0.00913077 +0.00122695 +0.00433765]]
// conv.w  =[[-0.00416979 -0.00380669 -0.00239433] [-0.00119722 +0.00099708 +0.00152477] [+0.00090541 +0.00150625 +0.00025433]]
// conv.w  =[[-0.00897218 -0.00540281 -0.00353776] [-0.00903542 -0.00142053 -0.00347301] [-0.00364891 +0.00242225 +0.00609436]]
// conv.w  =[[+0.00397105 +0.00026401 -0.00066163] [+0.00081342 -0.00007055 +0.00000000] [-0.00198205 -0.00000931 +0.00000000]]

// cnn
// output  =[+0.10614382 +0.09738087 +0.09332450 +0.11095963 +0.10072238 +0.10284732 +0.09771184 +0.10203808 +0.09754180 +0.09132969]
// softmax =[+0.02307198 +0.01946053 +0.01789279 +0.02518749 +0.02080143 -0.18401727 +0.01959135 +0.02134165 +0.01952409 +0.01714596]
// linear.x=[-0.00014116 -0.00089083 +0.00128763 +0.00117606 -0.00018488 +0.00149464 -0.00355611 +0.00109969 -0.00210105 -0.00097707]
// linear.w=[+0.00000000 +0.00000000 +0.00000000 +0.00000000 +0.00000000 +0.00000000 +0.00000000 +0.00000000 +0.00000000 +0.00000000]
// linear.w=[+0.00000000 +0.00000000 +0.00000000 +0.00000000 +0.00000000 +0.00000000 +0.00000000 +0.00000000 +0.00000000 +0.00000000]
// relu    =[-0.00000000 -0.00000000 +0.00000000 +0.00000000 -0.00000000 +0.00000000 -0.00000000 +0.00000000 -0.00000000 -0.00000000]
// relu    =[+0.00000000 -0.00000000 +0.00000000 -0.00000000 -0.00000000 -0.00000000 +0.00000000 -0.00000000 -0.00000000 -0.00000000]
// relu    =[-0.00000000 -0.00000000 -0.00000000 +0.00000000 -0.00000000 -0.00000000 +0.00000000 -0.00000000 -0.00000000 -0.00000000]
// relu    =[+0.00000000 -0.00000000 +0.00000000 +0.00000000 +0.00000000 -0.00000000 +0.00000000 -0.00000000 -0.00000000 +0.00000000]
// conv.w  =[[+0.00242896 -0.00019978 +0.00274303] [+0.00025668 -0.00011360 +0.00462382] [-0.00913077 +0.00122695 +0.00433765]]
// conv.w  =[[-0.00416979 -0.00380669 -0.00239433] [-0.00119722 +0.00099708 +0.00152477] [+0.00090541 +0.00150625 +0.00025433]]
// conv.w  =[[-0.00897218 -0.00540281 -0.00353776] [-0.00903542 -0.00142053 -0.00347301] [-0.00364891 +0.00242225 +0.00609436]]
// conv.w  =[[+0.00397105 +0.00026401 -0.00066163] [+0.00081342 -0.00007055 +0.00000000] [-0.00198205 -0.00000931 +0.00000000]]

