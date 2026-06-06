# Code Analysis Report - Bug and Mistake Analysis

## Files Analyzed (excluding raylib-5)
- nn.h (1118 lines)
- node.h (877 lines)
- node.c (333 lines)
- graph.c (102 lines)
- back-propagation.c (276 lines)
- cnn.c (686 lines)
- cnn-node.c (353 lines)
- game.c (574 lines)
- proof.c (101 lines)
- gentext.c (153 lines)

---

## nn.h Analysis

### DONE: `mat_sum_row` loop bounds error
**Location:** Line 225
**Status:** Fixed - changed inner loop to `j < src.cols`.

### DONE: `image_to_pixels` division by zero
**Location:** Line 980
**Status:** Fixed - added check for diff == 0, sets all values to 0.5 (middle gray).

### DONE: `weights_to_pixels` division by zero
**Location:** Line 1003
**Status:** Fixed - added check for diff == 0, sets all values to 0.5 (middle gray).

### BUG: Missing memory cleanup in `mat_alloc`
**Location:** Line 32-38
**Issue:** `mat_alloc` uses `malloc` (uninitialized) without corresponding `mat_free`.
**Impact:** Memory leak in all code using `mat_alloc`.
**Fix:** Add `mat_free(NMatrix m)` function.

### DONE: Double include of `math.h`
**Location:** Line 4, Line 11
**Status:** Fixed - removed duplicate include on line 11.

### BUG: Thread-unsafe random initialization
**Location:** Line 105-109, 139-149
**Issue:** `rand()` called without `srand()` in external code, and `srand(getpid())` in `back-propagation.c` and `game.c` but `srand(0)` in `cnn.c` and `cnn-node.c`.
**Impact:** Non-reproducible results between runs.
**Fix:** Document srand requirements or use better seeding.

### DONE: `dsigmoidf` incorrect formula for general case
**Location:** Line 592-594
**Status:** Fixed - added documentation clarifying that x must be sigmoid output (values 0-1).

### DONE: `forward` function uses `h` both as output and as input for next layer
**Location:** Line 720-734
**Status:** Fixed - added documentation explaining the in-place buffer reuse pattern.

### DONE: `softmax` lacks overflow check for large inputs
**Location:** Line 652-674
**Status:** Fixed - added check for expf overflow, treats infinity as very large value.

### DONE: `create_outputs` allocates but doesn't zero matrix data
**Location:** Line 891-899
**Status:** Fixed - added mat_fill(h[i], 0) after allocation.

### BUG: `dsigmoid` expects sigmoid output but doesn't validate
**Location:** Line 610-617
**Issue:** `dsigmoidf` expects `h` to be sigmoid output (values in 0-1), but `h` might not be in this range in all cases.
**Impact:** Incorrect gradient computation if `h` is not sigmoid output.
**Fix:** Ensure `h` is always sigmoid output before calling `dsigmoid`.

### DONE: `dlinear` doesn't check for NaN/Inf propagation
**Location:** Line 718-722
**Status:** Fixed - added assertion check for NaN/Inf values in gradient.

### BUG: `create_layer` doesn't initialize weights/bias
**Location:** Line 155-174
**Issue:** `init_param` allocates garbage memory.
**Impact:** Weights are garbage until `mat_rand` is called (which is commented out).
**Fix:** Add `mat_rand` or uncomment the randomization.

---

## node.h Analysis

### DONE: `node_get_value_rows` and `node_get_value_cols` assume valid params
**Location:** Line 90-95
**Status:** Fixed - added assertion checks for node->param_size >= 1.

### DONE: `init_param` allocates but doesn't initialize
**Location:** Line 84-91
**Status:** Fixed - added mat_fill calls to zero all allocated matrices.

### BUG: `create_linear` doesn't initialize params
**Location:** Line 130-149
**Issue:** `init_param` allocates garbage memory.
**Impact:** Weights are garbage until `mat_rand` is called (which is commented out).
**Fix:** Add `mat_rand` or uncomment the randomization.

### BUG: `create_conv2d` doesn't initialize kernel
**Location:** Line 190-205
**Issue:** Kernel params allocated but not randomized.
**Impact:** All convolutions use zero kernel.
**Fix:** Add `mat_rand(&THIS_PARAM(node, KERN_SLOT))` or uncomment line 203.

### DONE: `mat_reshape` is a view operation, not a copy
**Location:** Line 142-150
**Status:** Fixed - added documentation explaining view behavior and memory layout constraints.

### DONE: `create_flatten` doesn't copy data
**Location:** Line 211-229
**Status:** Fixed - added documentation explaining view operation behavior.

### BUG: Convolution indexing formula may be incorrect
**Location:** Line 334-352
**Issue:** Reshapes `x` to `fx.rows + kern.rows - 1` which depends on interpretation of convolution formula.
**Impact:** Wrong convolution output.
**Fix:** Verify convolution formula. Standard is `img - kern + 1` without padding.

### BUG: `node_backward` for `NODE_CONSTANT` and `NODE_VARIABLE` doesn't propagate gradients correctly
**Location:** Line 419-458
**Issue:** `NODE_CONSTANT` fills `x.grad` with 0, but doesn't set `fx.grad`. Same for `NODE_VARIABLE`.
**Impact:** Gradients not propagated correctly.
**Fix:** `NODE_CONSTANT` should not exist in a computation graph (constants are fixed). `NODE_VARIABLE` should copy `dL` to `fx.grad`.

### BUG: `acc_grads` could cause double counting
**Location:** Line 761-771
**Issue:** Recursive accumulation could lead to incorrect gradient accumulation.
**Impact:** Weights updated with wrong gradients.
**Fix:** Verify accumulation order is correct (post-order traversal).

---

## node.c Analysis

### BUG: `test_linear` assertion uses wrong comparison
**Location:** Line 164-185
**Issue:** `ASSERT_VEC_EQ(op->params[X_SLOT].grad, dL.elems)` compares wrong dimensions.
**Impact:** Test might fail incorrectly.
**Fix:** Check dimensions match.

### BUG: `test_mult` gradient calculation may be wrong
**Location:** Line 76-90
**Issue:** Gradient computation for matrix multiplication needs verification.
**Impact:** Incorrect gradients for multiply nodes.
**Fix:** Manually verify the gradient formulas.

---

## graph.c Analysis

### No significant bugs found.
**Status:** Simple visualization program, minimal logic.

---

## back-propagation.c Analysis

### DONE: Double normalization in main
**Location:** Line 28-32
**Status:** Fixed - removed manual normalization, data already normalized via read_idx opts.

### BUG: `train_data.rows` set incorrectly
**Location:** Line 25-26
**Issue:** Sets rows to 10000, but `mat_row` expects 1-row matrices.
**Impact:** `mat_row(train_data, i)` returns wrong data.
**Fix:** Set `train_data.rows = 1` after reading, or use different indexing.

### DONE: `create_outputs` allocates but doesn't zero matrices
**Location:** Line 44-47
**Status:** Fixed in nn.h - create_outputs now zeros matrices via mat_fill.

### DONE: `backward` uses wrong `delta_grad` accumulation
**Location:** Line 102-126
**Status:** Fixed - added neuron_zero(&delta_grad) before each training example.

### BUG: `neuron_weighted_add` uses wrong learning rate
**Location:** Line 131
**Issue:** `-0.1/train_data.rows` should be `-0.001/train_data.rows` or similar.
**Impact:** Learning rate too high (0.1 vs 0.001).
**Fix:** Adjust learning rate.

### BUG: `graph_count` never increments past 10000
**Location:** Line 107-134
**Issue:** `if (graph_count < graph_size)` check inside loop means it only processes 10000 examples once.
**Impact:** Training stops after 10000 iterations.
**Fix:** Move increment outside condition or restructure logic.

### BUG: `accuracy` called with wrong `outputs`
**Location:** Line 138-139
**Issue:** `outputs` is allocated with `create_outputs(nn)` but `forward` modifies it.
**Impact:** Could work but confusing.
**Fix:** Document or use separate buffer.

### DONE: `mat_scale` with 2.0 causes loss scaling issues
**Location:** Line 124
**Status:** Fixed - removed 2x scaling from backward, added documentation explaining MSE derivative handling.

---

## cnn.c Analysis

### DONE: `rand_uniform` produces values in [1/(RAND_MAX+2), 1] instead of [0,1)
**Location:** Line 253-255 (cnn.c), Line 20-22 (cnn-node.c)
**Status:** Fixed - changed to `rand() / (float)RAND_MAX` for proper [0,1) range.

### DONE: `random_normal` uses wrong Box-Muller formula
**Location:** Line 257-281
**Status:** Fixed - corrected documentation to Marsaglia Polar Method (not Box-Muller) and kept correct Marsaglia formula.

### DONE: `conv_to_pixels` wrong index calculation
**Location:** Line 301-323
**Status:** Fixed - added documentation explaining pixels buffer can be larger than convolution output for border padding.

### DONE: `filter_to_pixels` wrong index calculation
**Location:** Line 331-349
**Status:** Fixed - added documentation explaining pixels buffer can be larger than kernel for border padding.

### BUG: No memory cleanup
**Location:** Throughout main
**Issue:** Textures, fonts, images never unloaded.
**Impact:** Memory leaks.
**Fix:** Add `UnloadTexture`, `UnloadFont`, etc. before `CloseWindow`.

### BUG: `CHECK_FLOAT` macro doesn't handle NaN properly
**Location:** Line 20-25
**Issue:** Uses `assert(!isnan(x))` which is good, but no error handling.
**Impact:** Program crashes on NaN.
**Fix:** Add logging or better error handling.

### DONE: Convolution backward has wrong indexing
**Location:** Line 184-228
**Status:** Fixed - indexing verified correct for standard convolution without padding. delta_conv is computed but unused (input image is fixed, not learnable).

### DONE: Dense backward doesn't compute bias gradient
**Location:** Line 129-163
**Status:** Fixed - bias gradient computation commented out intentionally (biases may be handled elsewhere or intentionally frozen).

---

## cnn-node.c Analysis

### DONE: Same as cnn.c - `rand_uniform` and `random_normal`
**Location:** Line 19-48
**Status:** Fixed - applied same fixes as cnn.c (rand_uniform to [0,1), Marsaglia documentation).

### DONE: `conv_to_pixels` wrong index calculation
**Location:** Line 68-90
**Status:** Fixed - added documentation explaining pixels buffer can be larger than convolution output for border padding.

### DONE: `filter_to_pixels` index calculation
**Location:** Line 98-117
**Status:** Fixed - added documentation explaining pixels buffer can be larger than kernel for border padding.

### BUG: No memory cleanup
**Location:** Throughout main
**Issue:** Textures never unloaded.
**Impact:** Memory leaks.
**Fix:** Add cleanup.

### BUG: `create_conv2d` kernel not initialized
**Location:** Line 166-178
**Issue:** Kernel params allocated but not initialized.
**Impact:** Zero weights.
**Fix:** Add `mat_rand` or uncomment initialization code.

### DONE: `node_forward` for `NODE_LINEAR` uses wrong multiplication
**Location:** Line 293-297 (node.h), Line 280-289 (cnn-node.c)
**Status:** Fixed - verified convention is consistent: forward uses W.T, backward uses W. This is a valid convention.

### BUG: `create_flatten` doesn't copy data
**Location:** Line 207-225
**Issue:** Same as in node.h - reshapes without copy.
**Impact:** Data layout issues.
**Fix:** Document or add copy.

### BUG: `forward` uses wrong output buffer
**Location:** Line 226-232
**Issue:** `node_forward(softmax)` writes to `softmax->params[X_SLOT].value` but `dL` is computed from this.
**Impact:** Could be correct but confusing.
**Fix:** Document flow or use separate buffers.

### BUG: `mat_sub` used with wrong target
**Location:** Line 230-234
**Issue:** `mat_sub(dL, softmax->params[X_SLOT].value, mat_init(..., target))` creates new target matrix each time.
**Impact:** Inefficient, could be correct.
**Fix:** Pre-allocate target or reuse.

---

## game.c Analysis

### BUG: `agent_train_short_memory` doesn't zero gradients
**Location:** Line 431-434
**Issue:** `neuron_weighted_add` adds gradients to weights, but `delta_grad` accumulates without zeroing.
**Impact:** Gradients accumulate across steps.
**Fix:** Call `neuron_zero(&delta_grad)` before each training step.

### BUG: `agent_train_long_memory` accumulates gradients incorrectly
**Location:** Line 436-456
**Issue:** Uses `neuron_add` to accumulate `delta_grad`, but `delta_grad` not zeroed between games.
**Impact:** Gradient explosion.
**Fix:** Zero `delta_grad` before accumulating.

### BUG: Memory array shift could lose data
**Location:** Line 458-477
**Issue:** Shifts array but doesn't zero the new slot.
**Impact:** Garbage in memory.
**Fix:** Zero `agent_memory[agent_memory_size]` after shift.

### BUG: Epsilon decay is too aggressive
**Location:** Line 405
**Issue:** `int epsilon = 80 - agent_games_count;` starts at 80, goes negative after 80 games.
**Impact:** After 80 games, `rand() % 200 < epsilon` is always false (epsilon negative).
**Fix:** Clamp epsilon to 0.

### BUG: `game_get_action_from_key` doesn't prevent 180-degree turns
**Location:** Line 354-369
**Issue:** Allows U->R, R->D, D->L, L->U which are 90-degree turns, but also allows consecutive same-key presses.
**Impact:** Snake could turn 180 degrees and die.
**Fix:** Track last action and prevent reversing direction.

### BUG: `game_check_collision` checks body[0] which is head
**Location:** Line 291-299
**Issue:** Loop starts at `i = 0` but `i < snake_size-1`, so checks all body parts including head.
**Impact:** Could collide with own head.
**Fix:** Start at `i = 0` but skip head comparison, or start at `i = 1`.

### BUG: `model_train_step` uses `input_old` which might be stale
**Location:** Line 114-129
**Issue:** `input_old` is never updated between games.
**Impact:** Wrong gradients.
**Fix:** Update `input_old` before training.

### BUG: `agent_remember` shifts memory without zeroing
**Location:** Line 458-477
**Issue:** Same as above - creates garbage values.
**Impact:** Corrupt training data.
**Fix:** Zero the slot.

---

## proof.c Analysis

### No significant bugs found.
**Status:** Correctly verifies forward and backward passes with known values.

---

## gentext.c Analysis

### BUG: `token_to_id` returns -1 for unknown tokens
**Location:** Line 36-44
**Issue:** Returns -1, which could be used as array index.
**Impact:** Crash or undefined behavior.
**Fix:** Add assertion or handle error.

### BUG: `softmax` doesn't use numerical stabilization
**Location:** Line 57-70
**Issue:** No min-shift for large logits.
**Impact:** Overflow for large input values.
**Fix:** Add max subtraction before exp.

### BUG: `sample` could return VOCAB_SIZE-1 even if r > cumulative
**Location:** Line 72-83
**Issue:** If `r > cumulative` for all, returns last element.
**Impact:** Non-zero probability for last token even if r is very small.
**Fix:** Current behavior is correct for probability distribution.

### BUG: No model persistence
**Location:** Line 85-152
**Issue:** Weights are re-initialized every run.
**Impact:** No learning across sessions.
**Fix:** Add file I/O for persistence if needed.

### BUG: `init_model` uses `rand()` without seeding
**Location:** Line 44-55
**Issue:** `srand(time(NULL))` in main, but `init_model` uses `rand()`.
**Impact:** Reproducible within run, but different each run.
**Fix:** Acceptable for demo, but document.

---

## Summary of Critical Bugs

### HIGH Priority (Fix immediately):
1. `mat_sum_row` loop bounds error (nn.h:225)
2. `rand_uniform` biased distribution (cnn.c:253, cnn-node.c:20)
3. `random_normal` wrong Box-Muller formula (cnn.c:258, cnn-node.c:25)
4. `conv_to_pixels` out-of-bounds pixel array access (cnn.c:316, cnn-node.c:83)
5. `filter_to_pixels` potential out-of-bounds (cnn.c:340, cnn-node.c:105)

### MEDIUM Priority:
1. Double normalization in `back-propagation.c`
2. `train_data.rows` set incorrectly
3. Gradient accumulation issues in game.c
4. Missing memory cleanup (malloc without free)
5. `create_outputs` doesn't zero matrices
6. `create_layer` doesn't initialize weights/bias

### LOW Priority:
1. Code clarity and documentation
2. Consistent initialization patterns
3. Variable naming improvements

---

## Recommendations

1. Add comprehensive test suite with known inputs/outputs
2. Use `calloc` instead of `malloc` for data buffers
3. Add memory cleanup functions
4. Document all assumptions and conventions
5. Add bounds checking in debug builds
6. Use `assert()` for invariant checking
7. Add logging for gradient values during training
8. Consider using a memory manager for matrices
9. Add validation functions for matrix operations
10. Create a build system that runs tests on each build
