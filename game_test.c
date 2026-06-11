#include "game.h"
#include <stdio.h>
#include <string.h>
#include <assert.h>

void test_directions(void) {
    printf("=== Test 1: Direction flags (all 4 directions) ===\n");
    for (int dir = 0; dir < 4; dir++) {
        snake_direction = (Snake_Direction)dir;
        snake_size = 5;
        snake_body[0] = snake_body[1] = snake_body[2] = snake_body[3] = snake_body[4] = (Point2D){10, 10};
        food.x = 0; food.y = 0;
        State s = agent_get_state((Point2D){10, 10}, (Point2D){0, 0});
        uint32_t sum = s.dir_l + s.dir_r + s.dir_u + s.dir_d;
        printf("  DIR_%d: sum=%u (expect 1)\n", dir, sum);
        assert(sum == 1);
    }
    printf("  PASSED\n\n");
}

void test_food_right(void) {
    printf("=== Test 2: Food position - DIR_RIGHT ===\n");
    snake_direction = DIR_RIGHT;
    snake_size = 5;
    snake_body[0] = (Point2D){10, 10}; snake_body[1] = (Point2D){9, 10};
    snake_body[2] = (Point2D){8, 10}; snake_body[3] = (Point2D){7, 10};
    snake_body[4] = (Point2D){6, 10};
    
    food.x = 15; food.y = 10;
    State s1 = agent_get_state((Point2D){10, 10}, (Point2D){15, 10});
    assert(s1.food_ahead == 1 && s1.food_left == 0 && s1.food_right == 0);
    printf("  Ahead (15,10): PASSED\n");
    
    food.x = 5; food.y = 10;
    State s2 = agent_get_state((Point2D){10, 10}, (Point2D){5, 10});
    assert(s2.food_ahead == 0 && s2.food_left == 0 && s2.food_right == 0);
    printf("  Behind (5,10): PASSED\n");
    
    food.x = 12; food.y = 5;
    State s3 = agent_get_state((Point2D){10, 10}, (Point2D){12, 5});
    assert(s3.food_ahead == 1 && s3.food_left == 1 && s3.food_right == 0);
    printf("  Ahead/Left (12,5): PASSED\n");
    
    food.x = 12; food.y = 15;
    State s4 = agent_get_state((Point2D){10, 10}, (Point2D){12, 15});
    assert(s4.food_ahead == 1 && s4.food_left == 0 && s4.food_right == 1);
    printf("  Ahead/Right (12,15): PASSED\n");
    printf("  PASSED\n\n");
}

void test_food_up(void) {
    printf("=== Test 3: Food position - DIR_UP ===\n");
    snake_direction = DIR_UP;
    snake_size = 5;
    snake_body[0] = (Point2D){10, 10}; snake_body[1] = (Point2D){10, 9};
    snake_body[2] = (Point2D){10, 8}; snake_body[3] = (Point2D){10, 7};
    snake_body[4] = (Point2D){10, 6};
    
    food.x = 10; food.y = 5;
    State s1 = agent_get_state((Point2D){10, 10}, (Point2D){10, 5});
    assert(s1.food_ahead == 1 && s1.food_left == 0 && s1.food_right == 0);
    printf("  Ahead (10,5): PASSED\n");
    
    food.x = 10; food.y = 15;
    State s2 = agent_get_state((Point2D){10, 10}, (Point2D){10, 15});
    assert(s2.food_ahead == 0 && s2.food_left == 0 && s2.food_right == 0);
    printf("  Behind (10,15): PASSED\n");
    
    food.x = 15; food.y = 7;
    State s3 = agent_get_state((Point2D){10, 10}, (Point2D){15, 7});
    assert(s3.food_ahead == 1 && s3.food_left == 0 && s3.food_right == 1);
    printf("  Ahead/Right (15,7): PASSED\n");
    
    food.x = 5; food.y = 7;
    State s4 = agent_get_state((Point2D){10, 10}, (Point2D){5, 7});
    assert(s4.food_ahead == 1 && s4.food_left == 1 && s4.food_right == 0);
    printf("  Ahead/Left (5,7): PASSED\n");
    printf("  PASSED\n\n");
}

void test_food_down(void) {
    printf("=== Test 4: Food position - DIR_DOWN ===\n");
    snake_direction = DIR_DOWN;
    snake_size = 5;
    snake_body[0] = (Point2D){10, 10}; snake_body[1] = (Point2D){10, 11};
    snake_body[2] = (Point2D){10, 12}; snake_body[3] = (Point2D){10, 13};
    snake_body[4] = (Point2D){10, 14};
    
    food.x = 10; food.y = 15;
    State s1 = agent_get_state((Point2D){10, 10}, (Point2D){10, 15});
    assert(s1.food_ahead == 1 && s1.food_left == 0 && s1.food_right == 0);
    printf("  Ahead (10,15): PASSED\n");
    
    food.x = 10; food.y = 5;
    State s2 = agent_get_state((Point2D){10, 10}, (Point2D){10, 5});
    assert(s2.food_ahead == 0 && s2.food_left == 0 && s2.food_right == 0);
    printf("  Behind (10,5): PASSED\n");
    
    food.x = 5; food.y = 12;
    State s3 = agent_get_state((Point2D){10, 10}, (Point2D){5, 12});
    assert(s3.food_ahead == 1 && s3.food_left == 0 && s3.food_right == 1);
    printf("  Ahead/Right (5,12): PASSED\n");
    
    food.x = 15; food.y = 12;
    State s4 = agent_get_state((Point2D){10, 10}, (Point2D){15, 12});
    assert(s4.food_ahead == 1 && s4.food_left == 1 && s4.food_right == 0);
    printf("  Ahead/Left (15,12): PASSED\n");
    printf("  PASSED\n\n");
}

void test_food_left(void) {
    printf("=== Test 5: Food position - DIR_LEFT ===\n");
    snake_direction = DIR_LEFT;
    snake_size = 5;
    snake_body[0] = (Point2D){10, 10}; snake_body[1] = (Point2D){11, 10};
    snake_body[2] = (Point2D){12, 10}; snake_body[3] = (Point2D){13, 10};
    snake_body[4] = (Point2D){14, 10};
    
    food.x = 5; food.y = 10;
    State s1 = agent_get_state((Point2D){10, 10}, (Point2D){5, 10});
    assert(s1.food_ahead == 1 && s1.food_left == 0 && s1.food_right == 0);
    printf("  Ahead (5,10): PASSED\n");
    
    food.x = 15; food.y = 10;
    State s2 = agent_get_state((Point2D){10, 10}, (Point2D){15, 10});
    assert(s2.food_ahead == 0 && s2.food_left == 0 && s2.food_right == 0);
    printf("  Behind (15,10): PASSED\n");
    
    food.x = 7; food.y = 5;
    State s3 = agent_get_state((Point2D){10, 10}, (Point2D){7, 5});
    assert(s3.food_ahead == 1 && s3.food_left == 0 && s3.food_right == 1);
    printf("  Ahead/Right (7,5): PASSED\n");
    
    food.x = 7; food.y = 15;
    State s4 = agent_get_state((Point2D){10, 10}, (Point2D){7, 15});
    assert(s4.food_ahead == 1 && s4.food_left == 1 && s4.food_right == 0);
    printf("  Ahead/Left (7,15): PASSED\n");
    printf("  PASSED\n\n");
}

int main(void) {
    srand(42);
    test_directions();
    test_food_right();
    test_food_up();
    test_food_down();
    test_food_left();
    printf("=== All tests passed! ===\n");
    return 0;
}
