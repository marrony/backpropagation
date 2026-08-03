#include "nn.h"

#if 1

float accuracy(Neuron_Network nn, NMatrix* outputs, NMatrix test_data, NMatrix test_labels) {
  int corrects = 0;
  for (int i = 0; i < test_data.rows; i++) {
    NMatrix input = mat_row(test_data, i);
    int index = mat_row_argmax(forward(&nn, outputs, input));

    corrects += index == (int)MAT_AT(test_labels, i, 0);
  }
  return (float)corrects / (float)test_data.rows;
}

int main(void) {
  srand(getpid());

  NMatrix train_data = read_idx("train-images-idx3-ubyte", .normalize = true);
  NMatrix train_labels = read_idx("train-labels-idx1-ubyte", .normalize = false);

  NMatrix test_data = read_idx("t10k-images-idx3-ubyte", .normalize = true);
  NMatrix test_labels = read_idx("t10k-labels-idx1-ubyte", .normalize = false);

  train_data.rows = train_labels.rows = 10000;
  test_data.rows = test_labels.rows = 1000;
  // Note: Data is already normalized in read_idx calls (.normalize = true)

  Neuron_Layer layers[] = {
    create_layer(.inputs = 28*28, .outputs = 20, .forward = relu_old, .backward = drelu_old),
    create_layer(.inputs = 20, .outputs = 10, .forward = relu_old, .backward = drelu_old),
    create_layer(.inputs = 10, .outputs = 10, .forward = softmax, .backward = dsoftmax)
  };

  Neuron_Network nn = neuron_create(layers, ARRAY_LEN(layers));
  Neuron_Network grad = neuron_clone(nn);
  Neuron_Network delta_grad = neuron_clone(nn);
  NMatrix* outputs = create_outputs(nn);
  NMatrix* errors = create_outputs(nn);
  NMatrix* deltas = create_outputs(nn);
  NMatrix target = mat_alloc(1, 10);

  size_t graph_size = 10000;
  size_t graph_count = 0;
  float* accuracy_test_data = malloc(sizeof(float)*graph_size);
  float* accuracy_train_data = malloc(sizeof(float)*graph_size);
  float* cost_data = malloc(sizeof(float)*graph_size);

  char buf[256];

  int WindowWidth = 1000;
  int WindowHeight = 1400;
  InitWindow(WindowWidth, WindowHeight, "Backpropagation");
  SetWindowPosition(0, 0);

  Font font = LoadFontEx("fonts/MonacoNerdFont-Regular.ttf", 128, NULL, 95);
  unsigned char pixels[28*28] = {0};

  Texture2D eval_texture;
  int eval_index = 0;

  Image image = {
    .data = pixels,
    .width = 28,
    .height = 28,
    .format = PIXELFORMAT_UNCOMPRESSED_GRAYSCALE,
    .mipmaps = 1,
  };

  eval_texture = LoadTextureFromImage(image);

#if 0
  Texture2D* texture[ARRAY_LEN(layers)];
  for (size_t i = 0; i < ARRAY_LEN(layers); i++) {
    texture[i] = malloc(sizeof(Texture2D) * nn.w[i].cols);

    for (int j = 0; j < nn.w[i].cols; j++) {
      texture[i][j] = LoadTextureFromImage(image);
    }
  }
#endif

  // SetWindowPosition(0, 0);
  // SetTargetFPS(60);

  int iterations = 0;

  while (!WindowShouldClose()) {
    neuron_zero(&grad);

    if (IsKeyPressed(KEY_UP)) {
      eval_index++;
    }

    if (IsKeyPressed(KEY_DOWN)) {
      eval_index--;
    }

    float c = 0;

    int64_t train_start = get_system_micros();
    if (graph_count < graph_size) {
      for (int i = 0; i < train_data.rows; i++) {
        NMatrix input = mat_row(train_data, i);
        float label = MAT_AT(train_labels, i, 0);

        // Zero delta_grad before each training example to prevent gradient accumulation
        neuron_zero(&delta_grad);

        mat_fill(target, 0);
        VEC_AT(target, (int)label) = 1;

        forward(&nn, outputs, input);
        c += backward(
            &nn,
            outputs,
            &delta_grad,
            errors, // dL/dh
            deltas, // dL/dz
            input,
            target
        );
        // Note: backward() already includes the 2x factor from MSE derivative.
        // The learning rate in neuron_weighted_add should compensate for this.

        // accumulate weights and biases gradients
        neuron_add(&grad, &delta_grad);
      }

      // update weights and biases gradients
      neuron_weighted_add(&nn, &grad, -0.1/train_data.rows);

      iterations++;
    }
    int64_t train_time = get_system_micros() - train_start;

    int64_t eval_start = get_system_micros();
    float accuracy_train = accuracy(nn, outputs, train_data, train_labels);
    float accuracy_test = accuracy(nn, outputs, test_data, test_labels);
    int64_t eval_time = get_system_micros() - eval_start;

    if (graph_count < graph_size) {
      accuracy_test_data[graph_count] = accuracy_test;
      accuracy_train_data[graph_count] = accuracy_train;
      cost_data[graph_count] = c / train_data.rows;
      graph_count++;
    }

    BeginDrawing();
    ClearBackground(BLACK);

    snprintf(buf, sizeof(buf), "Cost = %f  Iter = %d", c, iterations);
    DrawTextEx(font, buf, (Vector2) { .x = 10, .y = 10, }, 40, 1, WHITE);

    snprintf(buf, sizeof(buf), "Train Time = %lld us  Eval Time = %lld us", train_time, eval_time);
    DrawTextEx(font, buf, (Vector2) { .x = 10, .y = 45, }, 40, 1, WHITE);

    snprintf(buf, sizeof(buf), "Accuracy Train = %f  Accuracy Test = %f", accuracy_test, accuracy_train);
    DrawTextEx(font, buf, (Vector2) { .x = 10, .y = 80, }, 40, 1, WHITE);

#if 0
    for (int layer = 0; layer < nn.layers; layer++) {
      NMatrix weights = nn.w[layer];

      for (int neuron = 0; neuron < weights.cols; neuron++) {
        memset(pixels, 255, sizeof(pixels));
        weights_to_pixels(weights, neuron, pixels);
        UpdateTexture(texture[layer][neuron], pixels);

        float scale = 2;

        Vector2 pos = {
          .x = layer*(28*scale + 10) + 30,
          .y = neuron*(28*scale + 10) + 150,
        };

        DrawTextureEx(texture[layer][neuron], pos, 0, scale, WHITE);
      }
    }
#endif

    for (size_t i = 0; i < graph_count-1; i++) {
      float s = 1.0f / ((graph_count + WindowWidth - 1) / WindowWidth);

      float x0 = i * s;
      float x1 = (i+1) * s;

      float graph_height = 500;

      for (size_t j = 1; j <= 10; j++) {
        float d = j/10.0f;
        float y = WindowHeight - d * graph_height;

        Vector2 start = { 0, y };
        Vector2 end = { WindowWidth, y };
        DrawLineV(start, end, DARKGRAY);

        snprintf(buf, sizeof(buf), "%.2f", d);
        DrawTextEx(font, buf, (Vector2) { .x = WindowWidth - 45, .y = y, }, 20, 1, DARKGRAY);
      }


      Vector2 start_test = { x0, WindowHeight - accuracy_test_data[i] * graph_height };
      Vector2 end_test = { x1, WindowHeight - accuracy_test_data[i+1] * graph_height };

      Vector2 start_train = { x0, WindowHeight - accuracy_train_data[i] * graph_height };
      Vector2 end_train = { x1, WindowHeight - accuracy_train_data[i+1] * graph_height };

      Vector2 start_cost = { x0, WindowHeight - cost_data[i] * graph_height };
      Vector2 end_cost = { x1, WindowHeight - cost_data[i+1] * graph_height };

      DrawLineV(start_test, end_test, RED);
      DrawLineV(start_train, end_train, GREEN);
      DrawLineV(start_cost, end_cost, BLUE);
    }

    if (iterations % 20 == 0) {
      eval_index = rand() % test_data.rows;
    }

    {
      NMatrix input = mat_row(test_data, eval_index);
      float scale = 5;

      float y_offset = 200;
      float x_offset = 100;

      image_to_pixels(input, pixels);
      UpdateTexture(eval_texture, pixels);
      DrawTextureEx(eval_texture, (Vector2) {.x = x_offset + 275, .y = y_offset}, 0, scale, WHITE);

      int index = mat_row_argmax(forward(&nn, outputs, input));
      int label = (int)MAT_AT(test_labels, eval_index, 0);

      NMatrix output = outputs[nn.layers-1];
      for (int i = 0; i < output.cols; i++) {
        snprintf(buf, sizeof(buf), "%2d = %.5f", i, VEC_AT(output, i));
        Color color = DARKGRAY;

        if (i == label) {
          color = (index != label) ? GREEN : RED;
        }

        if (i == index) {
          color = (index == label) ? GREEN : RED;
        }

        DrawTextEx(font, buf, (Vector2) { .x = x_offset, .y = y_offset+35*i, }, 40, 1, color);
      }

      snprintf(buf, sizeof(buf), "Label      = %d", label);
      DrawTextEx(font, buf, (Vector2) { .x = x_offset + 328 + (scale * 28), .y = y_offset+35*0, }, 40, 1, WHITE);

      snprintf(buf, sizeof(buf), "Inferred   = %d", index);
      DrawTextEx(font, buf, (Vector2) { .x = x_offset + 328 + (scale * 28), .y = y_offset+35*1, }, 40, 1, index == label ? GREEN : RED);

      snprintf(buf, sizeof(buf), "Confidence = %.2f%%", VEC_AT(output, index)*100);
      DrawTextEx(font, buf, (Vector2) { .x = x_offset + 328 + (scale * 28), .y = y_offset+35*2, }, 40, 1, WHITE);
    }
    EndDrawing();
  }  

  UnloadFont(font);

  CloseWindow();

  return 0;
}

#else

int main(void) {
  return 0;
}

#endif
