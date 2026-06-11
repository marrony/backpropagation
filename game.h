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
  ACTION_LEFT,
  ACTION_RIGHT,
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

#define INPUT_PARAMETERS 10
#define OUTPUT_PARAMETERS 3
// 9 parameters: 3 danger + 4 dir + 2 distance
typedef struct {
  uint32_t danger_straight : 1; // is dangerous go straight?
  uint32_t danger_right : 1;    // is dangerous turn right?
  uint32_t danger_left : 1;     // is dangerous turn left?
  uint32_t dir_l : 1;   // is snake going to left
  uint32_t dir_r : 1;   // is snake going to right
  uint32_t dir_u : 1;   // is snake going up
  uint32_t dir_d : 1;   // is snake going down
  uint32_t food_ahead : 1;  // food is ahead?
  uint32_t food_left : 1;  // food is on the left?
  uint32_t food_right : 1;  // food is on the right?
} State;

typedef struct {
  float reward;
  bool done;
} Game_Step;

#define WindowWidth 25
#define WindowHeight 25

Neuron_Network nn_online;
Neuron_Network nn_target;
Neuron_Network grad;
Neuron_Network delta_grad;
// activations_* are used on forward() to hold activations
// do not confuse with neural network weights
NMatrix* activations_model_step;
NMatrix* activations_freeze;
NMatrix* errors;
NMatrix* deltas;
NMatrix input_old;
NMatrix input_new;
NMatrix target;

#define LEARNING_RATE 0.05f
#define REWARD 10.0f
#define Q_VALUE_OFFSET 0.5f  // Small offset to encourage positive initial Q-values

void model_start(void) {
  Neuron_Layer layers[] = {
    create_layer(.inputs = INPUT_PARAMETERS, .outputs = 64, .forward = sigmoid, .backward = dsigmoid),
    create_layer(.inputs = 64, .outputs = 64, .forward = sigmoid, .backward = dsigmoid),
    create_layer(.inputs = 64, .outputs = OUTPUT_PARAMETERS, .forward = linear, .backward = dlinear),  // Linear for Q-values
  };

  nn_online = neuron_create(layers, ARRAY_LEN(layers));
  nn_target = neuron_create(layers, ARRAY_LEN(layers));
  grad = neuron_clone(nn_online);
  delta_grad = neuron_clone(nn_online);
  activations_model_step = create_outputs(nn_online);
  activations_freeze = create_outputs(nn_online);
  errors = create_outputs(nn_online);
  deltas = create_outputs(nn_online);
  input_old = mat_alloc(1, INPUT_PARAMETERS);
  input_new = mat_alloc(1, INPUT_PARAMETERS);
  target = mat_alloc(1, OUTPUT_PARAMETERS);
}

void state_to_matrix(NMatrix dst, State state) {
  VEC_AT(dst, 0) = state.danger_straight;
  VEC_AT(dst, 1) = state.danger_right;
  VEC_AT(dst, 2) = state.danger_left;
  VEC_AT(dst, 3) = state.dir_l;
  VEC_AT(dst, 4) = state.dir_r;
  VEC_AT(dst, 5) = state.dir_u;
  VEC_AT(dst, 6) = state.dir_d;
  VEC_AT(dst, 7) = state.food_ahead;
  VEC_AT(dst, 8) = state.food_left;
  VEC_AT(dst, 9) = state.food_left;
}

void model_train_step(Neuron_Network* nn, State old_state, Action action, Game_Step step, State new_state, Neuron_Network* delta_grad) {
  // Get current Q-values for old state
  state_to_matrix(input_old, old_state);
  NMatrix old_pred = forward(nn, activations_model_step, input_old);
  mat_copy(target, old_pred);

  // Compute Bellman target for the taken action only (standard DQN)
  float Q_new = step.reward + Q_VALUE_OFFSET;

  if (!step.done) {
    // Bellman's equation: add max Q-value of next state
    state_to_matrix(input_new, new_state);
    NMatrix new_pred = forward(&nn_target, activations_freeze, input_new);
    int index = mat_row_max(new_pred);
    float max_q_next = VEC_AT(new_pred, index);
    Q_new += 0.9 * max_q_next;
  }

  // Update ONLY the taken action's Q-value, leave others unchanged
  // This preserves relative Q-value differences which is crucial for learning
  VEC_AT(target, action) = Q_new;

  // Train on this single sample
  backward(nn, activations_model_step, delta_grad, errors, deltas, input_old, target);
}

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

// NMatrix forward(Neuron_Network* nn, State state, NMatrix* activations, NMatrix input);

void game_update_ui(Font font, int games_count, int record, State state, NMatrix input) {
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
      "danger=(%d, %d, %d), dir=(%d, %d, %d, %d), food=(%d, %d, %d)",
      state.danger_straight,
      state.danger_right,
      state.danger_left,
      state.dir_l,
      state.dir_r,
      state.dir_u,
      state.dir_d,
      state.food_ahead,
      state.food_left,
      state.food_right
  );
  DrawTextEx(font, buf, (Vector2) { .x = BLOCK_SIZE, .y = BLOCK_SIZE*2 }, 28, 1, WHITE);

  state_to_matrix(input, state);
  NMatrix pred = forward(&nn_target, activations_freeze, input);
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

  Point2D old_head = game_snake_head();
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

  // Reward shaping: +0.1 for moving closer to food, -0.1 for moving farther
  if (!point_equals(head, food)) {
    float old_dist = sqrtf((float)(old_head.x - food.x)*(old_head.x - food.x) + 
                           (float)(old_head.y - food.y)*(old_head.y - food.y));
    float new_dist = sqrtf((float)(head.x - food.x)*(head.x - food.x) + 
                           (float)(head.y - food.y)*(head.y - food.y));
    
    if (new_dist < old_dist) {
      reward += 0.1;  // Closer to food
    } else {
      reward -= 0.1;  // Farther from food
    }
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

// Coordinate frame depends on snake direction:
// When snake moves DIR_RIGHT (+x): forward=+x, lateral=+y
// When snake moves DIR_DOWN (+y):  forward=+y, lateral=-x
// When snake moves DIR_LEFT (-x):  forward=-x, lateral=+y
// When snake moves DIR_UP (-y):    forward=-y, lateral=-x
//
// Summary table (food position relative to snake head):
//
// Direction | Movement | food_forward = | food_lateral = |
// ----------|----------|----------------|----------------|
// DIR_RIGHT | +x       | food.x-head.x  | food.y-head.y  |
// DIR_DOWN  | +y       | food.y-head.y  | head.x-food.x  |
// DIR_LEFT  | -x       | head.x-food.x  | head.y-food.y  |
// DIR_UP    | -y       | head.y-food.y  | food.x-head.x  |
State agent_get_state(Point2D head, Point2D food) {
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
  bool danger_left     = (dir_d && game_check_collision(pt_r)) || (dir_u && game_check_collision(pt_l)) || (dir_l && game_check_collision(pt_u)) || (dir_r && game_check_collision(pt_d));

  // Transform food position to snake's local frame
  // food_forward: positive if ahead of snake, negative if behind
  // food_lateral: positive if to the right, negative if to the left
  int food_forward = 0;
  int food_lateral = 0;

  if (dir_r) {
    // Facing right: forward = +x, lateral = +y (down)
    food_forward = food.x - head.x;
    food_lateral = food.y - head.y;
  } else if (dir_d) {
    // Facing down: forward = +y, lateral = -x (left)
    food_forward = food.y - head.y;
    food_lateral = head.x - food.x;
  } else if (dir_l) {
    // Facing left: forward = -x, lateral = +y (down)
    food_forward = head.x - food.x;
    food_lateral = head.y - food.y;
  } else if (dir_u) {
    // Facing up: forward = -y, lateral = -x (left)
    food_forward = head.y - food.y;
    food_lateral = food.x - head.x;
  }

  return (State) {
    .danger_straight = danger_straight,
    .danger_right    = danger_right,
    .danger_left     = danger_left,
    .dir_l = dir_l,
    .dir_r = dir_r,
    .dir_u = dir_u,
    .dir_d = dir_d,
    .food_ahead = food_forward > 0,
    .food_left = food_lateral < 0,
    .food_right = food_lateral > 0,
  };
}

int agent_games_count = 0;

Action agent_get_action(State state, NMatrix input) {
  (void)state;

  // Decay epsilon: 100% at start, 10% minimum after 500 games
  int epsilon = MAX(10, 100 * (1.0f - (float)agent_games_count / 1000.0f));

  if (rand() % 100 < epsilon) {
    //printf("Exploration\n");
    // Uniform random exploration for fair exploration of all actions
    int choice = rand() % 3;
    if (choice == 0) {
      return ACTION_STRAIGHT;
    } else if (choice == 1) {
      return ACTION_LEFT;
    } else {
      return ACTION_RIGHT;
    }
  }

  state_to_matrix(input, state);
  int action = mat_row_max(forward(&nn_target, activations_freeze, input));
  return (Action)action;
}

#define MEMORY_SIZE (100*1000)
#define BATCH_SIZE 1000

typedef struct {
  State old_state;
  Action action;
  Game_Step step;
  State new_state;
} MemoryEntry;

MemoryEntry agent_memory[MEMORY_SIZE];
size_t agent_memory_head = 0;
size_t agent_memory_size = 0;

void agent_train_short_memory(State old_state, Action action, Game_Step step, State new_state) {
  // if (snake_moves % 20 == 0) {
  //   size_t sample_idx = (agent_memory_head - (rand() % agent_memory_size)) % MEMORY_SIZE;
  //   if (sample_idx < 0) sample_idx += MEMORY_SIZE;
  //   printf("Memory size=%zu head=%zu sample=%zu\n", agent_memory_size, agent_memory_head, sample_idx);
  // }
  // if (step.reward > 0) {
  //   printf("FOOD! score=%d moves=%zu\n", snake_score, snake_moves);
  // }    
  // if (snake_moves % 50 == 0) {
  //   NMatrix pred = forward(&nn_target, old_state, activations_freeze, input_old);
  //   float q0 = VEC_AT(pred, ACTION_STRAIGHT);  // Straight
  //   float q1 = VEC_AT(pred, ACTION_LEFT);  // Left
  //   float q2 = VEC_AT(pred, ACTION_RIGHT);  // Right
  //   printf("Q=(%.2f, %.2f, %.2f) action=%d\n", q0, q1, q2, action);
  // }
  // (void)action;
  // (void)step;
  // (void)old_state;
  // (void)new_state;
  neuron_zero(&delta_grad);
  model_train_step(&nn_online, old_state, action, step, new_state, &delta_grad);
  neuron_weighted_add(&nn_online, &delta_grad, -LEARNING_RATE);
}

void agent_train_long_memory(void) {
  if (agent_memory_size == 0) return;

  // Periodic sync: train target network every 10 games
  static int games_since_sync = 0;
  games_since_sync++;
  if (games_since_sync < 10) return;
  games_since_sync = 0;

  size_t train_size = MIN(agent_memory_size, BATCH_SIZE);

  neuron_zero(&grad);
  for (size_t i = 0; i < train_size; i++) {
    int index = ((int)agent_memory_head - (int)(rand() % agent_memory_size)) % MEMORY_SIZE;
    if (index < 0) index += MEMORY_SIZE;

    neuron_zero(&delta_grad);
    model_train_step(
        &nn_target,  // Train nn_target from replay buffer (stable targets)
        agent_memory[index].old_state,
        agent_memory[index].action,
        agent_memory[index].step,
        agent_memory[index].new_state,
        &delta_grad
    );
    neuron_add(&grad, &delta_grad);
  }

  neuron_weighted_add(&nn_target, &grad, -LEARNING_RATE / (float)train_size);
}

void agent_remember(State old_state, Action action, Game_Step step, State new_state) {
  agent_memory[agent_memory_head].old_state = old_state;
  agent_memory[agent_memory_head].action = action;
  agent_memory[agent_memory_head].step = step;
  agent_memory[agent_memory_head].new_state = new_state;

  agent_memory_head = (agent_memory_head + 1) % MEMORY_SIZE;
  agent_memory_size = MIN(agent_memory_size + 1, MEMORY_SIZE);
}

