#include "nn.h"
#include <raylib.h>
#include <stdio.h>

#define WindowWidth 1000
#define WindowHeight 1000

#define WorldWidth  50.0f
#define WorldHeight 2.0f

Vector2 world_to_screen(Vector2 pt) {
  float x = (pt.x/2 + WorldWidth/2)/WorldWidth * WindowWidth;
  float y = (pt.y/2 + WorldHeight/2)/WorldHeight * WindowHeight;

  return (Vector2) {
      .x = x,
      .y = WindowHeight - y,
  };
}

Vector2 screen_to_world(Vector2 pt) {
  float y = WindowHeight - pt.y;

  return (Vector2) {
      .x = (pt.x/WindowWidth * WorldWidth - WorldWidth/2) * 2,
      .y = (y/WindowHeight * WorldHeight - WorldHeight/2) * 2,
  };
}

int main(void) {
  InitWindow(WindowWidth, WindowHeight, "Backpropagation");
  SetWindowPosition(0, 0);

  Font font = LoadFontEx("fonts/MonacoNerdFont-Regular.ttf", 128, NULL, 95);

  while (!WindowShouldClose()) {
    BeginDrawing();
    ClearBackground(BLACK);

    DrawTextEx(font, "graph", (Vector2) { .x = 10, .y = 10, }, 40, 1, WHITE);


    // sigmoid(x) = 1.0f / (1 + expf(-x))
    // dsigmoid(x) = s(x) * (1 - s(x))

    char buf[256];

    for (int i = -10; i <= 10; i++) {
      float y = WorldHeight/10.0f * i;

      Vector2 start = { -WorldWidth, y };
      Vector2 end = { +WorldWidth, y };
      DrawLineV(world_to_screen(start), world_to_screen(end), DARKGRAY);

      snprintf(buf, sizeof(buf), "%+.2f", y);
      DrawTextEx(font, buf, (Vector2) { .x = WindowWidth - 60, .y = world_to_screen(start).y, }, 20, 1, DARKGRAY);
    }

    for (int x0 = -100; x0 < 100; x0++) {
      #define F(x) sigmoidf(x)
      #define dF(x) dsigmoidf(x)

      float x1 = x0 + 1;

      Vector2 startPos = { .x = x0, .y = F(x0) };
      Vector2 endPos = { .x = x1, .y = F(x1) };

      DrawLineV(world_to_screen(startPos), world_to_screen(endPos), WHITE);

      Vector2 startPosDeriv = { .x = x0, .y = dF(startPos.y) };
      Vector2 endPosDeriv = { .x = x1, .y = dF(endPos.y) };
      DrawLineV(world_to_screen(startPosDeriv), world_to_screen(endPosDeriv), RED);

      Vector2 mouse = screen_to_world(GetMousePosition());

      Vector2 dot = { .x = mouse.x, .y = F(mouse.x) };
      float deriv = dF(dot.y);

      float dx = 1.0f;
      float dy = deriv;
      float len = sqrtf(dx*dx + dy*dy);
      Vector2 dot2 = {
          .x = dot.x + dx/len,
          .y = dot.y + dy/len
      };

      snprintf(buf, sizeof(buf), "pt = (%+.5f %+.5f)   deriv = %+.5f", dot.x, dot.y, deriv);
      DrawTextEx(font, buf, (Vector2) { .x = 10, .y = 60, }, 40, 1, WHITE);

      DrawLineV(world_to_screen(dot), world_to_screen(dot2), GREEN);
      DrawCircleV(world_to_screen(dot), 2, GREEN);
      DrawCircleV(world_to_screen(dot2), 2, GREEN);

      DrawCircleV(world_to_screen((Vector2) { .x = dot.x, .y = deriv }), 2, RED);
    }

    EndDrawing();
  }

  UnloadFont(font);

  CloseWindow();

  return 0;
}
