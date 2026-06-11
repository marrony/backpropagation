#include "game.h"

int main(void) {
  setvbuf(stdout, NULL, _IONBF, 0);
  srand(getpid());

  SetTraceLogLevel(LOG_NONE);

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

  while (!WindowShouldClose() && agent_games_count < 5000) {
    if (IsKeyPressed(KEY_SPACE)) is_paused = !is_paused;

    if (is_done) {
      BeginDrawing();
      ClearBackground(BLACK);

      game_update_ui(font, agent_games_count, record, (State){0}, input_old);
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
      Point2D head = game_snake_head();

      State old_state = agent_get_state(head, food);

      if (!is_paused) {
        Action action = agent_get_action(old_state, input_old);
        Game_Step step = game_step(action);
        State new_state = agent_get_state(head, food);

        agent_train_short_memory(old_state, action, step, new_state);
        agent_remember(old_state, action, step, new_state);

        // Debug: Print Q-values after training
        if (agent_games_count % 50 == 0 && step.done) {
          const char* action_names[] = {"straight", "left", "right"};
          state_to_matrix(input_old, old_state);
          NMatrix pred = forward(&nn_target, activations_freeze, input_old);
          float q0 = VEC_AT(pred, ACTION_STRAIGHT);
          float q1 = VEC_AT(pred, ACTION_LEFT);
          float q2 = VEC_AT(pred, ACTION_RIGHT);
          printf("Games=%d Q=(%.2f, %.2f, %.2f) action=%s record=%d food=(%d,%d,%d) snake_dir=(%d,%d,%d,%d) danger=(%d,%d,%d)\n",
              agent_games_count,
              q0, q1, q2, action_names[action], record,
              old_state.food_ahead, old_state.food_left, old_state.food_right,
              old_state.dir_l, old_state.dir_r, old_state.dir_u, old_state.dir_d,
              old_state.danger_straight, old_state.danger_right, old_state.danger_left
              );
        }

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

      game_update_ui(font, agent_games_count, record, old_state, input_old);

      EndDrawing();
#endif
    }
  }

  UnloadFont(font);

  CloseWindow();

  return 0;
}
