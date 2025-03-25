#include "nn.h"

#include <float.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <assert.h>
#include <unistd.h>

#include <pthread.h>
#include <sys/semaphore.h>

#include "raylib-5/raylib.h"

#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define MAX(x, y) ((x) > (y) ? (x) : (y))

typedef struct {
  int x;
  int y;
} Point2D;

typedef enum {
  ACTION_STRAIGHT,
  ACTION_RIGHT,
  ACTION_LEFT,
  ACTION_COUNT,
} Action;

bool point_equals(Point2D a, Point2D b) {
  return a.x == b.x && a.y == b.y;
}

typedef enum {
  DIR_RIGHT,
  DIR_DOWN,
  DIR_LEFT,
  DIR_UP,
  DIR_MAX,
} Snake_Direction;

typedef struct {
  bool danger_straight;
  bool danger_right;
  bool danger_left;
  bool dir_l;
  bool dir_r;
  bool dir_u;
  bool dir_d;
  bool food_l;
  bool food_r;
  bool food_u;
  bool food_d;
} State;

typedef struct {
  float reward;
  bool done;
} Game_Step;

Neuron_Network nn;
Neuron_Network grad;
Neuron_Network delta_grad;
NMatrix* activations_old;
NMatrix* activations_new;
NMatrix* errors;
NMatrix* deltas;
NMatrix input_old;
NMatrix input_new;
NMatrix target;

#define LEARNING_RATE 0.0001f
#define REWARD 2.0f

void model_start(void) {
  Neuron_Layer layers[] = {
    create_layer(.inputs = 11, .outputs = 10, .activation = sigmoid, .derivative = dsigmoid),
    create_layer(.inputs = 10, .outputs = 10, .activation = sigmoid, .derivative = dsigmoid),
    create_layer(.inputs = 10, .outputs = 3, .activation = softmax, .derivative = dsoftmax),
  };

  nn = neuron_create(layers, ARRAY_LEN(layers));
  grad = neuron_clone(nn);
  delta_grad = neuron_clone(nn);
  activations_old = create_outputs(nn);
  activations_new = create_outputs(nn);
  errors = create_outputs(nn);
  deltas = create_outputs(nn);
  input_old = mat_alloc(1, 11);
  input_new = mat_alloc(1, 11);
  target = mat_alloc(1, 3);
}

NMatrix model_predict(State state, NMatrix* activations, NMatrix input) {
  VEC_AT(input, 0) = state.danger_straight;
  VEC_AT(input, 1) = state.danger_right;
  VEC_AT(input, 2) = state.danger_left;
  VEC_AT(input, 3) = state.dir_l;
  VEC_AT(input, 4) = state.dir_r;
  VEC_AT(input, 5) = state.dir_u;
  VEC_AT(input, 6) = state.dir_d;
  VEC_AT(input, 7) = state.food_l;
  VEC_AT(input, 8) = state.food_r;
  VEC_AT(input, 9) = state.food_u;
  VEC_AT(input, 10) = state.food_d;

  return forward(&nn, activations, input);
}

void model_train_step(State old_state, Action action, Game_Step step, State new_state, Neuron_Network* delta_grad) {
  mat_copy(target, model_predict(old_state, activations_old, input_old));

  float Q_new = step.reward;

  if (!step.done) {
    // Bellman’s equation
    NMatrix new_pred = model_predict(new_state, activations_new, input_new);
    int index = mat_row_max(new_pred);
    Q_new += 0.9 * VEC_AT(new_pred, index);
  }

  VEC_AT(target, (int)action) = Q_new;

  backward(&nn, activations_old, delta_grad, errors, deltas, input_old, target);
}

#define WindowWidth 25
#define WindowHeight 25

#define BLOCK_SIZE 32
#define SNAKE_MAX_SIZE 1024

int snake_score = 0;
size_t snake_moves = 0;
size_t snake_size = 0;
Point2D snake_body[SNAKE_MAX_SIZE];
Snake_Direction snake_body_dir[SNAKE_MAX_SIZE];
Snake_Direction snake_direction = DIR_RIGHT;
Point2D food;

Point2D game_snake_head(void) {
  return snake_body[snake_size-1];
}

void game_append_body(Point2D pt, Snake_Direction dir) {
  if (snake_size < SNAKE_MAX_SIZE) {
    snake_body_dir[snake_size] = dir;
    snake_body[snake_size] = pt;
    snake_size += 1;
  } else {
    assert("snake max size reached");
  }
}

void game_pop_body(void) {
  if (snake_size > 0) {
    for (size_t i = 0; i < snake_size - 1; i++) {
      snake_body[i] = snake_body[i + 1];
      snake_body_dir[i] = snake_body_dir[i + 1];
    }

    snake_size -= 1;
  }
}

void game_draw_dir(Point2D head, Snake_Direction dir) {
  Vector2 startPos = { .x = head.x*BLOCK_SIZE + BLOCK_SIZE/2, .y = head.y*BLOCK_SIZE + BLOCK_SIZE/2 };
  Vector2 endPos = startPos;

  switch (dir) {
    case DIR_RIGHT: endPos.x += 14; break;
    case DIR_LEFT:  endPos.x -= 14; break;
    case DIR_UP:    endPos.y -= 14; break;
    case DIR_DOWN:  endPos.y += 14; break;
    default: break;
  }

  DrawLineV(startPos, endPos, RED);
  DrawCircleV(endPos, 2, RED);
}

NMatrix model_predict(State state, NMatrix* activations, NMatrix input);

void game_update_ui(Font font, int games_count, int record, State state) {
  char buf[128];

  int padding = 4;

  for (size_t i = 0; i < snake_size; i++) {
    Point2D pt = snake_body[i];
    Snake_Direction dir = snake_body_dir[i];

    DrawRectangle(pt.x*BLOCK_SIZE, pt.y*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, (Color){.r = 0, .g = 0, .b = 255, .a = 255});
    DrawRectangle(pt.x*BLOCK_SIZE+padding, pt.y*BLOCK_SIZE+padding, BLOCK_SIZE-2*padding, BLOCK_SIZE-2*padding, (Color){.r = 0, .g = 100, .b = 255, .a = 255});

    game_draw_dir(pt, dir);

    // snprintf(buf, sizeof(buf), "%dx%d", pt.x, pt.y);
    // DrawTextEx(font, buf, (Vector2) { .x = pt.x*BLOCK_SIZE, .y = pt.y*BLOCK_SIZE }, 12, 1, WHITE);
  }

  DrawRectangle(food.x*BLOCK_SIZE, food.y*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, (Color){.r = 200, .g = 0, .b = 0, .a = 255});
  DrawRectangle(food.x*BLOCK_SIZE+padding, food.y*BLOCK_SIZE+padding, BLOCK_SIZE-2*padding, BLOCK_SIZE-2*padding, (Color){.r = 200, .g = 100, .b = 0, .a = 255});

  snprintf(buf, sizeof(buf), "Score = %d | Games = %d | Record = %d", snake_score, games_count, record);
  DrawTextEx(font, buf, (Vector2) { .x = BLOCK_SIZE, .y = BLOCK_SIZE }, 28, 1, WHITE);

  snprintf(buf, sizeof(buf),
      "danger=(%d, %d, %d), dir=(%d, %d, %d, %d), food=(%d, %d, %d, %d)",
      state.danger_straight,
      state.danger_right,
      state.danger_left,
      state.dir_l,
      state.dir_r,
      state.dir_u,
      state.dir_d,
      state.food_l,
      state.food_r,
      state.food_u,
      state.food_d
  );
  DrawTextEx(font, buf, (Vector2) { .x = BLOCK_SIZE, .y = BLOCK_SIZE*2 }, 28, 1, WHITE);
  extern NMatrix* activations_new;
  extern NMatrix input_new;

  NMatrix pred = model_predict(state, activations_new, input_new);
  snprintf(buf, sizeof(buf), "prediction=(%+.8f %+.8f %+.8f)", VEC_AT(pred, 0), VEC_AT(pred, 1), VEC_AT(pred, 2));
  DrawTextEx(font, buf, (Vector2) { .x = BLOCK_SIZE, .y = BLOCK_SIZE*3 }, 28, 1, WHITE);
}

void game_place_food(void) {
  int x = rand() % WindowWidth;
  int y = rand() % WindowHeight;

  food.x = x;
  food.y = y;

  for (size_t i = 0; i < snake_size; i++) {
    Point2D body = snake_body[i];

    if (point_equals(body, food)) {
      game_place_food();
      break;
    }
  }
}

Point2D game_move(Action action) {
  Snake_Direction new_dir;

  if (action == ACTION_STRAIGHT) {
    new_dir = snake_direction;
  }

  if (action == ACTION_RIGHT) {
    new_dir = (snake_direction + 1) % DIR_MAX;
  }

  if (action == ACTION_LEFT) {
    new_dir = (snake_direction + DIR_MAX - 1) % DIR_MAX;
  }

  snake_direction = new_dir;

  Point2D head = game_snake_head();

  switch (snake_direction) {
    case DIR_RIGHT:
      head.x += 1;
      break;
    case DIR_LEFT:
      head.x -= 1;
      break;
    case DIR_UP:
      head.y -= 1;
      break;
    case DIR_DOWN:
      head.y += 1;
      break;
    case DIR_MAX:
      assert("fail");
  }

  return head;
}

bool game_check_collision(Point2D pt) {
  if (pt.x < 0 || pt.x >= WindowWidth) return true;
  if (pt.y < 0 || pt.y >= WindowHeight) return true;

  for (size_t i = 0; i < snake_size-1; i++) {
    if (point_equals(pt, snake_body[i]))
      return true;
  }

  return false;
}

Game_Step game_step(Action action) {
  snake_moves += 1;

  Point2D head = game_move(action);

  game_append_body(head, snake_direction);

  float reward = 0;

  if (snake_moves > 100*snake_size || game_check_collision(head)) {
    reward = -REWARD;
    return (Game_Step) {
        .reward = reward,
        .done = true,
    };
  }

  if (point_equals(head, food)) {
    snake_score += 1;
    reward = REWARD;
    game_place_food();
  } else {
    game_pop_body();
  }

  return (Game_Step) {
      .reward = reward,
      .done = false,
  };
}

void game_reset(void) {
  snake_score = 0;
  snake_moves = 0;

  Point2D head = {
    .x = WindowWidth/2,
    .y = WindowHeight/2,
  };

  snake_direction = DIR_RIGHT;

  snake_size = 0;
  for (int i = 5; i >= 0; i--)
    game_append_body((Point2D) { .x = head.x - i, .y = head.y }, DIR_RIGHT);

  food.x = -1;
  food.y = -1;
  game_place_food();
}

Action game_get_action_from_key(Snake_Direction new_dir) {
  //DIR_RIGHT, DIR_DOWN, DIR_LEFT, DIR_UP,

  if (snake_direction == DIR_RIGHT && new_dir == DIR_UP)    return ACTION_RIGHT;
  if (snake_direction == DIR_DOWN  && new_dir == DIR_LEFT)  return ACTION_RIGHT;
  if (snake_direction == DIR_LEFT  && new_dir == DIR_DOWN)  return ACTION_RIGHT;
  if (snake_direction == DIR_UP    && new_dir == DIR_RIGHT) return ACTION_RIGHT;

  if (snake_direction == DIR_RIGHT && new_dir == DIR_DOWN)  return ACTION_LEFT;
  if (snake_direction == DIR_DOWN  && new_dir == DIR_RIGHT) return ACTION_LEFT;
  if (snake_direction == DIR_LEFT  && new_dir == DIR_UP)    return ACTION_LEFT;
  if (snake_direction == DIR_UP    && new_dir == DIR_LEFT)  return ACTION_LEFT;

  return ACTION_STRAIGHT;
}

State agent_get_state(void) {
  Point2D head = game_snake_head();

  Point2D pt_l = { .x = head.x - 1, head.y };
  Point2D pt_r = { .x = head.x + 1, head.y };
  Point2D pt_u = { .x = head.x, head.y - 1 };
  Point2D pt_d = { .x = head.x, head.y + 1 };

  bool dir_l = snake_direction == DIR_LEFT;
  bool dir_r = snake_direction == DIR_RIGHT;
  bool dir_u = snake_direction == DIR_UP;
  bool dir_d = snake_direction == DIR_DOWN;

  bool danger_straight = (dir_r && game_check_collision(pt_r)) || (dir_l && game_check_collision(pt_l)) || (dir_u && game_check_collision(pt_u)) || (dir_d && game_check_collision(pt_d));
  bool danger_right    = (dir_u && game_check_collision(pt_r)) || (dir_d && game_check_collision(pt_l)) || (dir_l && game_check_collision(pt_u)) || (dir_r && game_check_collision(pt_d));
  bool danger_left     = (dir_d && game_check_collision(pt_r)) || (dir_u && game_check_collision(pt_l)) || (dir_r && game_check_collision(pt_u)) || (dir_l && game_check_collision(pt_d));

  return (State) {
    .danger_straight = danger_straight,
    .danger_right    = danger_right,
    .danger_left     = danger_left,
    .dir_l = dir_l,
    .dir_r = dir_r,
    .dir_u = dir_u,
    .dir_d = dir_d,
    .food_l = food.x < head.x,
    .food_r = food.x > head.x,
    .food_u = food.y < head.y,
    .food_d = food.y > head.y,
  };
}

int agent_games_count = 0;

Action agent_get_action(State state) {
  int epsilon = 80 - agent_games_count;

  if (rand() % 200 < epsilon) {
    // return (Action)rand() % ACTION_COUNT;

    int action = rand() % 100;
    if (action < 45) return ACTION_LEFT;
    if (action > 60) return ACTION_RIGHT;
    return ACTION_STRAIGHT;
  }

  return (Action)mat_row_max(model_predict(state, activations_old, input_old));
}

#define MEMORY_SIZE (100*1000)
#define BATCH_SIZE 1000

struct {
  State old_state;
  Action action;
  Game_Step step;
  State new_state;
} agent_memory[MEMORY_SIZE];

size_t agent_memory_size = 0;

void agent_train_short_memory(State old_state, Action action, Game_Step step, State new_state) {
  model_train_step(old_state, action, step, new_state, &delta_grad);
  neuron_weighted_add(&nn, &delta_grad, -LEARNING_RATE);
}

void agent_train_long_memory(void) {
  if (agent_memory_size > 0) {
    size_t train_size = MIN(agent_memory_size, BATCH_SIZE);

    neuron_zero(&grad);
    for (size_t i = 0; i < train_size; i++) {
      int index = rand() % agent_memory_size;

      model_train_step(
          agent_memory[index].old_state,
          agent_memory[index].action,
          agent_memory[index].step,
          agent_memory[index].new_state,
          &delta_grad
      );
      neuron_add(&grad, &delta_grad);
    }

    neuron_weighted_add(&nn, &grad, -LEARNING_RATE/(float)train_size);
  }
}

void agent_remember(State old_state, Action action, Game_Step step, State new_state) {
  if (agent_memory_size >= MEMORY_SIZE) {
    for (size_t i = 0; i < agent_memory_size-1; i++) {
      memcpy(agent_memory+i, agent_memory+i+1, sizeof(*agent_memory));
    }

    agent_memory_size -= 1;
  }

  agent_memory[agent_memory_size].old_state = old_state;
  agent_memory[agent_memory_size].action = action;
  agent_memory[agent_memory_size].step = step;
  agent_memory[agent_memory_size].new_state = new_state;
  agent_memory_size += 1;
}

int main(void) {
  srand(getpid());

  InitWindow(WindowWidth*BLOCK_SIZE, WindowHeight*BLOCK_SIZE, "Snake game");
  SetWindowPosition(0, 0);
  // SetTargetFPS(60);

  Font font = LoadFontEx("fonts/MonacoNerdFont-Regular.ttf", 128, NULL, 95);

  game_reset();
  model_start();

  int64_t start = get_system_micros();
  (void)start;

  bool is_done = false;
  bool is_paused = false;

  int record = 0;

  while (!WindowShouldClose()) {
    if (IsKeyPressed(KEY_SPACE)) is_paused = !is_paused;

    if (is_done) {
      BeginDrawing();
      ClearBackground(BLACK);

      game_update_ui(font, agent_games_count, record, (State){0});
      DrawText("End", (WindowWidth*BLOCK_SIZE)/2, (WindowHeight*BLOCK_SIZE)/2, 28, WHITE);

      EndDrawing();
    } else {
#if 0
      Action action = ACTION_STRAIGHT;

      if (IsKeyDown(KEY_UP))    action = game_get_action_from_key(DIR_DOWN);
      if (IsKeyDown(KEY_DOWN))  action = game_get_action_from_key(DIR_UP);
      if (IsKeyDown(KEY_LEFT))  action = game_get_action_from_key(DIR_LEFT);
      if (IsKeyDown(KEY_RIGHT)) action = game_get_action_from_key(DIR_RIGHT);

      int64_t now = get_system_micros();

      if (now - start >= 100000) {
        if (!is_paused) {
          agent_get_action(agent_get_state());

          Game_Step step = game_step(action);
          if (step.done) {
            is_done = true;
          }
        }
        start = now;
      }

      BeginDrawing();
      ClearBackground(BLACK);

      game_update_ui(font, agent_games_count, record);

      EndDrawing();
#else
      State old_state = agent_get_state();

      if (!is_paused) {
        Action action = agent_get_action(old_state);
        // const char* action_names[] = {"straight", "right", "left"};
        // printf("Action = %s\n", action_names[action]);
        Game_Step step = game_step(action);
        State new_state = agent_get_state();

        agent_train_short_memory(old_state, action, step, new_state);

        agent_remember(old_state, action, step, new_state);

        if (step.done) {
          if (snake_score > record)
            record = snake_score;

          game_reset();
          agent_train_long_memory();

          agent_games_count += 1;
        }
      }

      BeginDrawing();
      ClearBackground(BLACK);

      game_update_ui(font, agent_games_count, record, old_state);

      EndDrawing();
#endif
    }
  }

  UnloadFont(font);

  CloseWindow();

  return 0;
}
