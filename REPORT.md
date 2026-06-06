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

### BUG: `mat_sum_row` loop bounds error
**Location:** Line 225
```c
for (int j = 0; j < src.rows; j++) {
  MAT_AT(dst, i, 0) += MAT_AT(src, i, j);
}
```
**Issue:** Inner loop iterates `j < src.rows` but should be `j < src.cols`.
**Impact:** Incorrect row sums when `src.rows != src.cols`.
**Fix:** Change to `j < src.cols`.

### BUG: `image_to_pixels` division by zero
**Location:** Line 980
```c
float diff = max - min;
for (int i = 0; i < image.cols; i++) {
  float v = (MAT_AT(image, 0, i) - min) / diff;
```
**Issue:** If `max == min` (all pixels same value), `diff == 0` causes division by zero.
**Impact:** NaN values in output.
**Fix:** Add check for `diff == 0` and set all values to 128 (middle gray).

### BUG: `weights_to_pixels` division by zero
**Location:** Line 1003
```c
float diff = max - min;
for (int i = 0; i < weights.rows; i++) {
  float v = (MAT_AT(weights, i, neuron) - min) / diff;
```
**Issue:** If `max == min` (all weights same value), `diff == 0` causes division by zero.
**Impact:** NaN values in output.
**Fix:** Add check for `diff == 0` and set all values to 128.

### BUG: Missing memory cleanup in `mat_alloc`
**Location:** Line 32-38
**Issue:** `mat_alloc` uses `malloc` (uninitialized) without corresponding `mat_free`.
**Impact:** Memory leak in all code using `mat_alloc`.
**Fix:** Add `mat_free(NMatrix m)` function.

### BUG: Double include of `math.h`
**Location:** Line 4, Line 11
**Issue:** `math.h` included twice.
**Fix:** Remove duplicate include.

### BUG: Thread-unsafe random initialization
**Location:** Line 105-109, 139-149
**Issue:** `rand()` called without `srand()` in external code, and `srand(getpid())` in `back-propagation.c` and `game.c` but `srand(0)` in `cnn.c` and `cnn-node.c`.
**Impact:** Non-reproducible results between runs.
**Fix:** Document srand requirements or use better seeding.

### BUG: `dsigmoidf` incorrect formula for general case
**Location:** Line 592-594
```c
static inline float dsigmoidf(float x) {
  return x * (1.0f - x);
}
```
**Issue:** The derivative of sigmoid is `sigmoid(x) * (1 - sigmoid(x))`, but this function expects `x` to already be the sigmoid output. The formula is only correct if `x` is in range (0,1) and represents sigmoid output.
**Impact:** Potential confusion but actually correct usage in code when called with sigmoid outputs.
**Fix:** Rename to `dsigmoid_of_sigmoid` or document that `x` should be sigmoid output.

### BUG: `forward` function uses `h` both as output and as input for next layer
**Location:** Line 718-789
**Issue:** `h` array is reused for each layer, but `h[i-1]` is read before it's fully computed in the current iteration. This works but is confusing.
**Fix:** Use separate input/output buffers or document the pattern clearly.

### BUG: `softmax` lacks overflow check for large inputs
**Location:** Line 652-674
**Issue:** Uses min-shift stabilization (subtracts max value) which is good, but doesn't check for `expf` overflow on very large inputs.
**Impact:** Very large inputs (>> 887) could still cause overflow.
**Fix:** Add check after subtracting max.

### BUG: `create_outputs` allocates but doesn't zero matrix data
**Location:** Line 882-893
**Issue:** `h[i] = mat_alloc(1, nn.w[i].rows)` allocates unzeroed memory.
**Impact:** Output matrices contain garbage values until forward pass.
**Fix:** Use `calloc` or fill with zeros after allocation.

### BUG: `dsigmoid` expects sigmoid output but doesn't validate
**Location:** Line 610-617
**Issue:** `dsigmoidf` expects `h` to be sigmoid output (values in 0-1), but `h` might not be in this range in all cases.
**Impact:** Incorrect gradient computation if `h` is not sigmoid output.
**Fix:** Ensure `h` is always sigmoid output before calling `dsigmoid`.

### BUG: `dlinear` doesn't check for NaN/Inf propagation
**Location:** Line 712-716
**Issue:** `dL_dh` could contain NaN/Inf from previous operations.
**Impact:** NaN propagation through network.
**Fix:** Add assertions or checks.

### BUG: `create_layer` doesn't initialize weights/bias
**Location:** Line 155-174
**Issue:** `init_param` allocates garbage memory.
**Impact:** Weights are garbage until `mat_rand` is called (which is commented out).
**Fix:** Add `mat_rand` or uncomment the randomization.

---

## node.h Analysis

### BUG: `node_get_value_rows` and `node_get_value_cols` assume valid params
**Location:** Line 91-97
**Issue:** Accesses `THIS_PARAM(node, X_SLOT)` without checking if node has params.
**Impact:** Crashes on invalid node types.
**Fix:** Add assertion or check `node->param_size >= 1`.

### BUG: `init_param` allocates but doesn't initialize
**Location:** Line 85-88
**Issue:** Uses `mat_alloc` instead of `calloc` or `memset`.
**Impact:** Uninitialized memory (garbage values).
**Fix:** Change to `calloc` or add `memset(param->value, 0, ...)` after allocation.

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

### BUG: `mat_reshape` is a view operation, not a copy
**Location:** Line 143-151
**Issue:** Returns pointer to original data, changing shape doesn't change data layout.
**Impact:** Incorrect data access after reshape if data layout changes.
**Fix:** Document this behavior clearly.

### BUG: `create_flatten` doesn't copy data
**Location:** Line 207-225
**Issue:** Reshapes without copying data.
**Impact:** Data layout mismatch if original buffer layout differs.
**Fix:** This might be intentional as a view. Document or add copy.

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

### BUG: Double normalization in main
**Location:** Line 28-32
**Issue:** Data normalized in `read_idx_opt` AND manually normalized again (lines 28-32).
**Impact:** Values divided by 255 twice (0-0.0039 instead of 0-1).
**Fix:** Remove manual normalization or fix `read_idx_opt` to not normalize.

### BUG: `train_data.rows` set incorrectly
**Location:** Line 25-26
**Issue:** Sets rows to 10000, but `mat_row` expects 1-row matrices.
**Impact:** `mat_row(train_data, i)` returns wrong data.
**Fix:** Set `train_data.rows = 1` after reading, or use different indexing.

### BUG: `create_outputs` allocates but doesn't zero matrices
**Location:** Line 44-47
**Issue:** Allocates `h` array but matrices inside are unzeroed.
**Impact:** Output matrices contain garbage values.
**Fix:** Zero each matrix after allocation.

### BUG: `backward` uses wrong `delta_grad` accumulation
**Location:** Line 107-124
**Issue:** `delta_grad` is cloned once and used for all training examples without reset between examples.
**Impact:** Gradients accumulate incorrectly across batches.
**Fix:** Call `neuron_zero(&delta_grad)` before each example.

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

### BUG: `mat_scale` with 2.0 causes loss scaling issues
**Location:** Line 124
**Issue:** MSE loss is multiplied by 2, making gradients 2x larger.
**Impact:** Learning rate needs adjustment.
**Fix:** Remove `mat_scale(dL, dL, 2.0)` or adjust learning rate.

---

## cnn.c Analysis

### BUG: `rand_uniform` produces values in [1/(RAND_MAX+2), 1] instead of [0,1)
**Location:** Line 253-255
```c
return (rand() + 1.0f) / (RAND_MAX + 2.0f);
```
**Issue:** Formula produces (1 to RAND_MAX+1) / (RAND_MAX+2), which is biased toward 1.
**Impact:** Biased initialization.
**Fix:** Use `rand() / (float)RAND_MAX` for [0,1).

### BUG: `random_normal` uses wrong Box-Muller formula
**Location:** Line 258-285
**Issue:** Box-Muller should use `sqrt(-2*log(s))` not `sqrt(-2*log(s)/s)`.
**Impact:** Incorrect normal distribution.
**Fix:** Change to `sqrtf(-2.0f * logf(s))` and remove `/s`.

### BUG: `conv_to_pixels` wrong index calculation
**Location:** Line 298-320
**Issue:** `pixels[index].intensity = v * 255;` where `index = i*IMG_SIZE + j`.
**Impact:** Out of bounds array access when `CONV_OUT < IMG_SIZE`.
**Fix:** Change index to `i*CONV_OUT + j` if pixels array is sized correctly.

### BUG: `filter_to_pixels` wrong index calculation
**Location:** Line 322-345
**Issue:** Uses `i*KERN_SIZE + j` but should verify array size is `KERN_SIZE*KERN_SIZE`.
**Impact:** Out of bounds if array is sized incorrectly.
**Fix:** Use `i*KERN_SIZE + j` but verify `pixels` array size.

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

### BUG: Convolution backward has wrong indexing
**Location:** Line 184-228
**Issue:** Similar to forward - indexing might be off by one.
**Impact:** Incorrect gradients.
**Fix:** Verify convolution formula.

### BUG: Dense backward doesn't compute bias gradient
**Location:** Line 129-163
**Issue:** `grad_b` is commented out and never computed.
**Impact:** Biases never updated.
**Fix:** Uncomment and add bias gradient computation.

---

## cnn-node.c Analysis

### BUG: Same as cnn.c - `rand_uniform` and `random_normal`
**Location:** Line 20-48
**Issue:** Same bugs as in cnn.c.
**Impact:** Biased initialization, wrong distribution.
**Fix:** Apply same fixes as cnn.c.

### BUG: `conv_to_pixels` wrong index calculation
**Location:** Line 65-87
**Issue:** `index = i*IMG_SIZE + j` but should use `i*CONV_OUT + j`.
**Impact:** Out of bounds array access.
**Fix:** Change to `i*CONV_OUT + j`.

### BUG: `filter_to_pixels` index calculation
**Location:** Line 89-109
**Issue:** Uses `i*KERN_SIZE + j` which is correct if array size is `KERN_SIZE*KERN_SIZE`.
**Impact:** Verify array size.
**Fix:** Ensure `pixels` array is sized correctly.

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

### BUG: `node_forward` for `NODE_LINEAR` uses wrong multiplication
**Location:** Line 280-289
**Issue:** `mat_mult_transpose_add` computes `x*W.T + b`, but standard is `x*W + b`.
**Impact:** Weight interpretation swapped.
**Fix:** Decide on convention and use consistently.

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
