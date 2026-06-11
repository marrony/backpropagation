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

void test_danger_right(void) {
    printf("=== Test 6: Danger flags - DIR_RIGHT ===\n");
    snake_direction = DIR_RIGHT;

    // Wall collision on straight (right) - head at (24, 10), body at (19, 10)
    snake_size = 5;
    snake_body_dir[0] = DIR_RIGHT; snake_body[0] = (Point2D){19, 10};
    snake_body_dir[1] = DIR_RIGHT; snake_body[1] = (Point2D){20, 10};
    snake_body_dir[2] = DIR_RIGHT; snake_body[2] = (Point2D){21, 10};
    snake_body_dir[3] = DIR_RIGHT; snake_body[3] = (Point2D){22, 10};
    snake_body_dir[4] = DIR_RIGHT; snake_body[4] = (Point2D){23, 10};
    food.x = 0; food.y = 0;
    State s1 = agent_get_state((Point2D){24, 10}, (Point2D){0, 0});
    assert(s1.danger_straight == 1 && s1.danger_right == 0 && s1.danger_left == 0);
    printf("  Wall straight (24,10): PASSED\n");

    // Snake body collision on straight (right) - snake grew, body extends ahead
    // Snake was at {14,10}, moved RIGHT to {15,10}
    // Body shifted: old head {14,10} became body[0]
    // Body extends to {16,10} (snake grew)
    // Next position {16,10} collides with body[2]
    // Note: game_check_collision checks i < snake_size-1, so need snake_size=4
    snake_size = 4;
    snake_body_dir[0] = DIR_RIGHT; snake_body[0] = (Point2D){14, 10};
    snake_body_dir[1] = DIR_RIGHT; snake_body[1] = (Point2D){15, 10};
    snake_body_dir[2] = DIR_RIGHT; snake_body[2] = (Point2D){16, 10};
    snake_body_dir[3] = DIR_RIGHT; snake_body[3] = (Point2D){17, 10};
    snake_body_dir[4] = DIR_RIGHT; snake_body[4] = (Point2D){18, 10};
    State s2 = agent_get_state((Point2D){15, 10}, (Point2D){0, 0});
    assert(s2.danger_straight == 1 && s2.danger_right == 0 && s2.danger_left == 0);
    printf("  Body straight (15,10): PASSED\n");

    // No danger - head at (12, 10), body at (11, 10)
    snake_size = 2;
    snake_body_dir[0] = DIR_RIGHT; snake_body[0] = (Point2D){11, 10};
    snake_body_dir[1] = DIR_RIGHT; snake_body[1] = (Point2D){12, 10};
    snake_body_dir[2] = DIR_RIGHT; snake_body[2] = (Point2D){13, 10};
    snake_body_dir[3] = DIR_RIGHT; snake_body[3] = (Point2D){14, 10};
    snake_body_dir[4] = DIR_RIGHT; snake_body[4] = (Point2D){15, 10};
    State s3 = agent_get_state((Point2D){12, 10}, (Point2D){0, 0});
    assert(s3.danger_straight == 0 && s3.danger_right == 0 && s3.danger_left == 0);
    printf("  No danger (12,10): PASSED\n");
    printf("  PASSED\n\n");
}

void test_danger_up(void) {
    printf("=== Test 7: Danger flags - DIR_UP ===\n");
    snake_direction = DIR_UP;

    // Wall collision on straight (up) - head at (10, 0), body at (10, 5)
    snake_size = 5;
    snake_body_dir[0] = DIR_UP; snake_body[0] = (Point2D){10, 5};
    snake_body_dir[1] = DIR_UP; snake_body[1] = (Point2D){10, 4};
    snake_body_dir[2] = DIR_UP; snake_body[2] = (Point2D){10, 3};
    snake_body_dir[3] = DIR_UP; snake_body[3] = (Point2D){10, 2};
    snake_body_dir[4] = DIR_UP; snake_body[4] = (Point2D){10, 1};
    food.x = 0; food.y = 0;
    State s1 = agent_get_state((Point2D){10, 0}, (Point2D){0, 0});
    assert(s1.danger_straight == 1 && s1.danger_right == 0 && s1.danger_left == 0);
    printf("  Wall straight (10,0): PASSED\n");

    // Snake body collision on straight (up) - snake grew, body extends ahead
    // Snake was at {10,11}, moved UP to {10,12}
    // Body shifted: old head {10,11} became body[0]
    // Body extends to {10,10} (snake grew)
    // Next position {10,13} collides with body[2]
    snake_size = 4;
    snake_body_dir[0] = DIR_UP; snake_body[0] = (Point2D){10, 11};
    snake_body_dir[1] = DIR_UP; snake_body[1] = (Point2D){10, 12};
    snake_body_dir[2] = DIR_UP; snake_body[2] = (Point2D){10, 10};
    snake_body_dir[3] = DIR_UP; snake_body[3] = (Point2D){10, 13};
    snake_body_dir[4] = DIR_UP; snake_body[4] = (Point2D){10, 14};
    State s2 = agent_get_state((Point2D){10, 12}, (Point2D){0, 0});
    assert(s2.danger_straight == 1 && s2.danger_right == 0 && s2.danger_left == 0);
    printf("  Body straight (10,12): PASSED\n");

    // No danger - head at (10, 12), body behind only
    // Snake was at {10,13}, moved UP to {10,12}
    // Body shifted: old head {10,13} became body[0]
    snake_size = 2;
    snake_body_dir[0] = DIR_UP; snake_body[0] = (Point2D){10, 13};
    snake_body_dir[1] = DIR_UP; snake_body[1] = (Point2D){10, 12};
    snake_body_dir[2] = DIR_UP; snake_body[2] = (Point2D){10, 14};
    snake_body_dir[3] = DIR_UP; snake_body[3] = (Point2D){10, 15};
    snake_body_dir[4] = DIR_UP; snake_body[4] = (Point2D){10, 16};
    State s3 = agent_get_state((Point2D){10, 12}, (Point2D){0, 0});
    assert(s3.danger_straight == 0 && s3.danger_right == 0 && s3.danger_left == 0);
    printf("  No danger (10,12): PASSED\n");
    printf("  PASSED\n\n");
}

void test_danger_left(void) {
    printf("=== Test 8: Danger flags - DIR_LEFT ===\n");
    snake_direction = DIR_LEFT;

    // Wall collision on straight (left) - head at (0, 10), body at (5, 10)
    snake_size = 5;
    snake_body_dir[0] = DIR_LEFT; snake_body[0] = (Point2D){5, 10};
    snake_body_dir[1] = DIR_LEFT; snake_body[1] = (Point2D){4, 10};
    snake_body_dir[2] = DIR_LEFT; snake_body[2] = (Point2D){3, 10};
    snake_body_dir[3] = DIR_LEFT; snake_body[3] = (Point2D){2, 10};
    snake_body_dir[4] = DIR_LEFT; snake_body[4] = (Point2D){1, 10};
    food.x = 0; food.y = 0;
    State s1 = agent_get_state((Point2D){0, 10}, (Point2D){0, 0});
    assert(s1.danger_straight == 1 && s1.danger_right == 0 && s1.danger_left == 0);
    printf("  Wall straight (0,10): PASSED\n");

 // Snake body collision on straight (left) - snake grew, body extends ahead
    // Snake was at {12,10}, moved LEFT to {13,10}
    // Body shifted: old head {12,10} became body[0]
    // Body extends to {11,10} (snake grew)
    // Next position {14,10} collides with body[2]
    snake_size = 4;
    snake_body_dir[0] = DIR_LEFT; snake_body[0] = (Point2D){12, 10};
    snake_body_dir[1] = DIR_LEFT; snake_body[1] = (Point2D){13, 10};
    snake_body_dir[2] = DIR_LEFT; snake_body[2] = (Point2D){11, 10};
    snake_body_dir[3] = DIR_LEFT; snake_body[3] = (Point2D){14, 10};
    snake_body_dir[4] = DIR_LEFT; snake_body[4] = (Point2D){15, 10};
    State s2 = agent_get_state((Point2D){13, 10}, (Point2D){0, 0});
    assert(s2.danger_straight == 1 && s2.danger_right == 0 && s2.danger_left == 0);
    printf("  Body straight (13,10): PASSED\n");

    // No danger - head at (8, 10), body behind only
    // Snake was at {9,10}, moved LEFT to {8,10}
    // Body shifted: old head {9,10} became body[0]
    snake_size = 2;
    snake_body_dir[0] = DIR_LEFT; snake_body[0] = (Point2D){9, 10};
    snake_body_dir[1] = DIR_LEFT; snake_body[1] = (Point2D){8, 10};
    snake_body_dir[2] = DIR_LEFT; snake_body[2] = (Point2D){9, 9};
    snake_body_dir[3] = DIR_LEFT; snake_body[3] = (Point2D){9, 8};
    snake_body_dir[4] = DIR_LEFT; snake_body[4] = (Point2D){9, 7};
    State s3 = agent_get_state((Point2D){8, 10}, (Point2D){0, 0});
    assert(s3.danger_straight == 0 && s3.danger_right == 0 && s3.danger_left == 0);
    printf("  No danger (8,10): PASSED\n");
    printf("  PASSED\n\n");
}

void test_danger_down(void) {
    printf("=== Test 9: Danger flags - DIR_DOWN ===\n");
    snake_direction = DIR_DOWN;

    // Wall collision on straight (down) - head at (10, 24), body at (10, 19)
    snake_size = 5;
    snake_body_dir[0] = DIR_DOWN; snake_body[0] = (Point2D){10, 19};
    snake_body_dir[1] = DIR_DOWN; snake_body[1] = (Point2D){10, 20};
    snake_body_dir[2] = DIR_DOWN; snake_body[2] = (Point2D){10, 21};
    snake_body_dir[3] = DIR_DOWN; snake_body[3] = (Point2D){10, 22};
    snake_body_dir[4] = DIR_DOWN; snake_body[4] = (Point2D){10, 23};
    food.x = 0; food.y = 0;
    State s1 = agent_get_state((Point2D){10, 24}, (Point2D){0, 0});
    assert(s1.danger_straight == 1 && s1.danger_right == 0 && s1.danger_left == 0);
    printf("  Wall straight (10,24): PASSED\n");

// Snake body collision on straight (down) - body extends ahead
    // Snake was at {10,13}, moved DOWN to {10,14}
    // Body shifted: old head {10,13} became body[0]
    // Body extends to {10,13} ahead (snake grew somehow)
    // Next position {10,15} collides with body[2]
    snake_size = 4;
    snake_body_dir[0] = DIR_DOWN; snake_body[0] = (Point2D){10, 13};
    snake_body_dir[1] = DIR_DOWN; snake_body[1] = (Point2D){10, 14};
    snake_body_dir[2] = DIR_DOWN; snake_body[2] = (Point2D){10, 15};
    snake_body_dir[3] = DIR_DOWN; snake_body[3] = (Point2D){10, 11};
    snake_body_dir[4] = DIR_DOWN; snake_body[4] = (Point2D){10, 10};
    State s2 = agent_get_state((Point2D){10, 14}, (Point2D){0, 0});
    assert(s2.danger_straight == 1 && s2.danger_right == 0 && s2.danger_left == 0);
    printf("  Body straight (10,14): PASSED\n");

    // No danger - head at (10, 8), body only behind
    // Snake was at {10,9}, moved DOWN to {10,8}
    // Body shifted: old head {10,9} is ABOVE (behind) the head
    // Moving DOWN to {10,9} would collide! But we need NO danger...
    // Actually, for NO danger, body must be BELOW head
    // Snake was at {10,7}, moved DOWN to {10,8}
    // Body shifted: old head {10,7} became body[0]
    snake_size = 2;
    snake_body_dir[0] = DIR_DOWN; snake_body[0] = (Point2D){10, 7};
    snake_body_dir[1] = DIR_DOWN; snake_body[1] = (Point2D){10, 8};
    snake_body_dir[2] = DIR_DOWN; snake_body[2] = (Point2D){10, 6};
    snake_body_dir[3] = DIR_DOWN; snake_body[3] = (Point2D){10, 5};
    snake_body_dir[4] = DIR_DOWN; snake_body[4] = (Point2D){10, 4};
    State s3 = agent_get_state((Point2D){10, 8}, (Point2D){0, 0});
    assert(s3.danger_straight == 0 && s3.danger_right == 0 && s3.danger_left == 0);
    printf("  No danger (10,8): PASSED\n");
    printf("  PASSED\n\n");
}

void test_danger_turn_right(void) {
    printf("=== Test 10: Danger turn right ===\n");
    snake_direction = DIR_UP;

    // Turn right (toward +x) into snake body - head at (10, 10), body at (11, 10)
    // Right of head is (11, 10) which is occupied by body
    // snake_body[0] = body, snake_body[1] = head
    snake_size = 2;
    snake_body_dir[0] = DIR_UP; snake_body[0] = (Point2D){11, 10};
    snake_body_dir[1] = DIR_UP; snake_body[1] = (Point2D){10, 10};
    snake_body_dir[2] = DIR_UP; snake_body[2] = (Point2D){10, 11};
    snake_body_dir[3] = DIR_UP; snake_body[3] = (Point2D){10, 12};
    snake_body_dir[4] = DIR_UP; snake_body[4] = (Point2D){10, 13};
    food.x = 0; food.y = 0;
    State s1 = agent_get_state((Point2D){10, 10}, (Point2D){0, 0});
    assert(s1.danger_straight == 0 && s1.danger_right == 1 && s1.danger_left == 0);

    // No danger turning right - head at (10, 10)
    snake_size = 1;
    snake_body_dir[0] = DIR_UP; snake_body[0] = (Point2D){10, 9};
    snake_body_dir[1] = DIR_UP; snake_body[1] = (Point2D){10, 8};
    snake_body_dir[2] = DIR_UP; snake_body[2] = (Point2D){10, 7};
    State s3 = agent_get_state((Point2D){10, 10}, (Point2D){0, 0});
    assert(s3.danger_straight == 0 && s3.danger_right == 0 && s3.danger_left == 0);
    printf("  PASSED\n\n");
}

void test_danger_turn_left(void) {
    printf("=== Test 11: Danger turn left ===\n");
    snake_direction = DIR_UP;

    // Turn left (toward -x) into snake body - head at (10, 10), body at (9, 10)
    // Left of head is (9, 10) which is occupied by body
    // snake_body[0] = body, snake_body[1] = head
    snake_size = 2;
    snake_body_dir[0] = DIR_UP; snake_body[0] = (Point2D){9, 10};
    snake_body_dir[1] = DIR_UP; snake_body[1] = (Point2D){10, 10};
    snake_body_dir[2] = DIR_UP; snake_body[2] = (Point2D){10, 11};
    snake_body_dir[3] = DIR_UP; snake_body[3] = (Point2D){10, 12};
    snake_body_dir[4] = DIR_UP; snake_body[4] = (Point2D){10, 13};
    food.x = 0; food.y = 0;
    State s1 = agent_get_state((Point2D){10, 10}, (Point2D){0, 0});
    assert(s1.danger_straight == 0 && s1.danger_right == 0 && s1.danger_left == 1);

    // No danger turning left - head at (10, 10)
    snake_size = 1;
    snake_body_dir[0] = DIR_UP; snake_body[0] = (Point2D){10, 9};
    snake_body_dir[1] = DIR_UP; snake_body[1] = (Point2D){10, 8};
    snake_body_dir[2] = DIR_UP; snake_body[2] = (Point2D){10, 7};
    State s3 = agent_get_state((Point2D){10, 10}, (Point2D){0, 0});
    assert(s3.danger_straight == 0 && s3.danger_right == 0 && s3.danger_left == 0);
    printf("  PASSED\n\n");
}

int main(void) {
    srand(42);
    test_directions();
    test_food_right();
    test_food_up();
    test_food_down();
    test_food_left();
    test_danger_right();
    test_danger_up();
    test_danger_left();
    test_danger_down();
    test_danger_turn_right();
    test_danger_turn_left();
    printf("=== All tests passed! ===\n");
    return 0;
}
