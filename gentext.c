#include <_stdlib.h>
#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/signal.h>
#include <time.h>
#include <signal.h>
#include <unistd.h>
#include "nn.h"
#include "node.h"
#include "tokenizer.h"

#include "array.h"

#define BYTEBUFFER_IMPLEMENTATION
#define ALLOCATOR_IMPLEMENATION
#define HASHMAP_IMPLMENTATION

#include "bytebuffer.h"
#include "allocator.h"
#include "hashmap.h"

#include "generated/alice.h"
#include "generated/vocab.h"

const char *training_text[] = {
  (const char*)alice_txt,
  "the desert was silent except for the low, rhythmic hum of the wind sweeping across the dunes. for miles in every direction, nothing broke the horizon line but shifting sand and the occasional skeletal remains of ancient shrubs. evelyn checked her gps device, frowning at the uncoordinated coordinates flashing across the screen. the signal was dead. she had exactly two liters of water left, a compass that could not find true north, and six hours of daylight remaining before the temperature dropped below freezing.",
  "the desert is a landscape of surprising contrasts and quiet resilience. during the day, the sun beats down relentlessly, turning the sand into a glowing sea of gold. cacti and deep-rooted shrubs stand as silent sentinels, conserving every drop of precious moisture in their thick stems. yet, as twilight approaches, the extreme heat yields to a crisp, cooling breeze. the sky shifts into a canvas of violet and deep indigo. nocturnal creatures, such as the kit fox and the rattlesnake, emerge from their underground burrows to hunt and forage, breathing vibrant life into the quiet night.",
  "artificial intelligence has rapidly transformed from a theoretical concept into an everyday reality. machine learning algorithms now power everything from basic email filters to complex autonomous vehicles. by processing massive amounts of historical data, these systems can identify hidden patterns, make accurate predictions, and automate tedious tasks. however, this technological leap brings significant ethical challenges, including data privacy concerns and algorithmic bias. as these neural networks become increasingly sophisticated, developers face the crucial responsibility of ensuring transparency and fairness, so that these powerful digital tools ultimately benefit society as a whole.",
  "the roman empire was one of the most powerful and enduring civilizations in human history, fundamentally shaping the trajectory of the western world. beginning as a modest republic on the italian peninsula, it expanded rapidly through strategic military conquests and advanced engineering. roman legions established dominance across the mediterranean, bringing law, architecture, and commerce to diverse cultures. at its peak, the empire spanned from the rainy hills of britannia to the arid deserts of egypt. even after its eventual collapse, the architectural marvels, legal systems, and cultural innovations of rome continued to influence modern societies for centuries.",
  "baking the perfect loaf of artisan bread requires patience, precision, and a deep understanding of basic ingredients. the process begins with just four simple components: flour, water, salt, and yeast. when combined, these elements undergo a magical transformation. the yeast feeds on the natural sugars in the flour, releasing carbon dioxide that causes the dough to rise and develop a complex network of air pockets. kneading and resting the dough properly are essential steps that build gluten structure. finally, baking the dough in a scorching hot oven creates a beautiful, crispy crust while keeping the interior soft.",
  "earth is a dynamic, ever-changing planet covered mostly by vast, interconnected oceans. beneath the water lies a complex topography of deep trenches, underwater mountain ranges, and expansive plains. these marine ecosystems are home to an astonishing variety of life, ranging from microscopic phytoplankton to massive whales. the oceans also play a critical role in regulating the global climate by absorbing massive amounts of carbon dioxide and distributing heat across the globe. despite their importance, these fragile aquatic environments are currently facing severe threats from pollution, overfishing, and rising water temperatures caused by climate change.",
  "the powerful king ruled. this man wore gold. the wise queen ruled. this woman wore gold. the brave king led men. that man commanded troops. the brave queen led men. that woman commanded troops. the noble king signed laws. a man signed laws. the noble queen signed laws. a woman signed laws.",
  "the young prince smiled. a happy boy smiled. the young princess smiled. a happy girl smiled. the small prince played. that young boy played. the small princess played. that young girl played. the royal prince learned. every smart boy learned. the royal princess learned. every smart girl learned.",
  "the great lord feasted. his proud husband feasted. the great lady feasted. her proud wife feasted. the rich lord rested. this loyal husband rested. the rich lady rested. this loyal wife rested.",
  "the loving father built homes. the young son built homes. the loving mother built homes. the young daughter built homes. the proud father worked hard. that brave son worked hard. the proud mother worked hard. that brave daughter worked hard. a kind father teaches youth. the elder son teaches youth. a kind mother teaches youth. the elder daughter teaches youth.",
  "the strict chairman signed deals. that male executive signed deals. the strict chairwoman signed deals. that female executive signed deals. the smart chairman led teams. a top director led teams. the smart chairwoman led teams. a top director led teams.",
  "the ancient god created life. this divine wizard created life. the ancient goddess created life. this divine witch created life. the powerful god cast spells. that cruel wizard cast spells. the powerful goddess cast spells. that cruel witch cast spells.",
  "the heavy bull ate grass. that male rooster ate grass. the heavy cow ate grass. that female hen ate grass. the loud bull woke farmers. a fierce rooster woke farmers. the loud cow woke farmers. a fierce hen woke farmers.",
};

#define CONTEXT 16
#define EMBED_DIM 16
#define INPUT_DIM (CONTEXT * EMBED_DIM)
#define MAX_GENERATE 100

#define TO_STR_HELPER(x) #x
#define TO_STR(x) TO_STR_HELPER(x)
#define SCAN_TOKEN_FMT "%" TO_STR(TOK_SIZE) "s"

// ------------------------------------------

typedef struct {
  int32_t context[CONTEXT];
  int32_t target;
} Sample;

DEFINE_ARRAY_SLICE(Sample);

Malloc_Allocator mallocator = MALLOC_CREATE();

Node* embedding_node = NULL;
Node* linear_node = NULL;
Node* relu_node = NULL;
Node* hidden1_node = NULL;
Node* softmax_cross_entropy_node = NULL;
Node* softmax_node = NULL;
NMatrix target_logit = NULL_MATRIX;
Optimizer optimizer = {0};
int32_t epochs = 0;

void mat_write(NMatrix mat, FILE* fp) {
  fwrite(&mat.rows, sizeof(int32_t), 1, fp);
  fwrite(&mat.cols, sizeof(int32_t), 1, fp);
  fwrite(mat.elems, sizeof(float), mat.rows*mat.cols, fp);
}

void mat_read(NMatrix mat, FILE* fp) {
  int32_t rows = 0;
  int32_t cols = 0;
  fread(&rows, sizeof(int32_t), 1, fp);
  fread(&cols, sizeof(int32_t), 1, fp);
  assert(rows == mat.rows);
  assert(cols == mat.cols);
  fread(mat.elems, sizeof(float), mat.rows*mat.cols, fp);
}

void save_model(void) {
  FILE* fp = fopen("models/gentext.bin", "wb");
  mat_write(embedding_node->embeddings.value, fp);
  mat_write(linear_node->weight.value, fp);
  mat_write(linear_node->bias.value, fp);
  mat_write(hidden1_node->weight.value, fp);
  mat_write(hidden1_node->bias.value, fp);
  fclose(fp);
}

void save_optimizer(Optimizer* optimizer, int32_t epochs) {
  FILE* fp = fopen("models/gentext.opt", "wb");
  fwrite(&optimizer->updates, sizeof(size_t), 1, fp);
  fwrite(&optimizer->learning_rate, sizeof(float), 1, fp);
  fwrite(&epochs, sizeof(int32_t), 1, fp);
  fwrite(&optimizer->tensors.count, sizeof(size_t), 1, fp);
  for (size_t i = 0; i < optimizer->tensors.count; i++) {
    mat_write(optimizer->history.elems[i], fp); // 1st moment
    mat_write(optimizer->second.elems[i], fp); // 2nd moment
  }
  fclose(fp);
}

void load_model(void) {
  FILE* fp = fopen("models/gentext.bin", "rb");
  if (fp != NULL) {
    printf("model exists, continue training\n");
    mat_read(embedding_node->embeddings.value, fp);
    mat_read(linear_node->weight.value, fp);
    mat_read(linear_node->bias.value, fp);
    mat_read(hidden1_node->weight.value, fp);
    mat_read(hidden1_node->bias.value, fp);
    fclose(fp);
  }
}

void load_optimizer(Optimizer* optimizer, int32_t* epochs) {
  FILE* fp = fopen("models/gentext.opt", "rb");
  if (fp != NULL) {
    printf("optimizer exists, reading\n");
    fread(&optimizer->updates, sizeof(size_t), 1, fp);
    fread(&optimizer->learning_rate, sizeof(float), 1, fp);
    fread(epochs, sizeof(int32_t), 1, fp);
    size_t tensors_count = 0;
    fread(&tensors_count, sizeof(size_t), 1, fp);
    assert(tensors_count == optimizer->tensors.count);

    for (size_t i = 0; i < optimizer->tensors.count; i++) {
      mat_read(optimizer->history.elems[i], fp); // 1st moment
      mat_read(optimizer->second.elems[i], fp); // 2nd moment
    }
    fclose(fp);
  }
}

void init_model(void) {
  optimizer = (Optimizer) {
    .tensors = ARRAY_CREATE(&mallocator.alloc),
    .history = ARRAY_CREATE(&mallocator.alloc),
    .second = ARRAY_CREATE(&mallocator.alloc),
    .learning_rate = 0.01f,
    .updates = 0,
  };

  // context = 8
  // embed_dim = 8
  // vocab_size = 512
  // embeddings (64) -> linear (64x128) -> relu -> linear (128x512) -> softmax
  int hidden_dim = 24*CONTEXT;
  embedding_node = create_embeddings(MAX_VOCAB, EMBED_DIM, CONTEXT);
  linear_node = create_linear(embedding_node, INPUT_DIM, hidden_dim);
  relu_node = create_relu(linear_node);
  hidden1_node = create_linear(relu_node, hidden_dim, MAX_VOCAB);
  target_logit = mat_alloc(1, MAX_VOCAB);
  softmax_cross_entropy_node = create_softmax_cross_entropy(hidden1_node, target_logit);
  softmax_cross_entropy_node->temperature = 1.0;

  softmax_node = create_softmax(hidden1_node);
  softmax_node->temperature = 1.0;

  // Xavier (Glorot): sqrt(1.0 / x)
  // Used For: Linear layers, Tanh layers, or Softmax inputs.
  //
  // He (Kaiming): sqrt(2.0 / x)
  // Used For: Layers followed by ReLU or LeakyReLU.

  float std_embeddings = sqrtf(6.0f / (MAX_VOCAB+EMBED_DIM)) * 1;
  float std_linear = sqrtf(6.0f / (EMBED_DIM+hidden_dim)) * 1;
  float std_hidden = sqrtf(6.0f / (hidden_dim+MAX_VOCAB)) * 1;

  printf("inits = %f %f %f\n", std_embeddings, std_linear, std_hidden);

  // Initialize embeddings and weights with small random values
  mat_rand_uniform(embedding_node->embeddings.value, -std_embeddings, +std_embeddings);
  mat_rand_uniform(linear_node->weight.value, -std_linear, +std_linear);
  mat_zero(linear_node->bias.value);
  mat_rand_uniform(hidden1_node->weight.value, -std_hidden, +std_hidden);
  mat_zero(hidden1_node->bias.value);

  mat_zero(mat_row(embedding_node->embeddings.value, PAD_TOKEN));

  // todo: move these calls to create_* functions
  register_tensor(&optimizer, &embedding_node->embeddings);
  register_tensor(&optimizer, &linear_node->weight);
  register_tensor(&optimizer, &linear_node->bias);
  register_tensor(&optimizer, &hidden1_node->weight);
  register_tensor(&optimizer, &hidden1_node->bias);
  // register_tensor(&optimizer, target_node->value);

  load_model();
  load_optimizer(&optimizer, &epochs);
}

volatile sig_atomic_t keep_running = 1;

void handle_sigint(int sig) {
  (void)sig;

  keep_running = 0;
}

void mat_one_minus_memberwise_div(NMatrix dst, NMatrix a, NMatrix b, float eps) {
  assert(dst.cols == a.cols);
  assert(dst.rows == a.rows);
  assert(dst.cols == b.cols);
  assert(dst.rows == b.rows);

  for (int i = 0; i < dst.rows; i++) {
    for (int j = 0; j < dst.cols; j++) {
      MAT_AT(dst, i, j) = 1.0f - (MAT_AT(a, i, j) / (MAT_AT(b, i, j) + eps));
    }
  }
}

float mat_min(NMatrix m) {
  float min = MAT_AT(m, 0, 0);

  for (int i = 0; i < m.rows; i++) {
    for (int j = 0; j < m.cols; j++) {
      if (MAT_AT(m, i, j) < min) min = MAT_AT(m, i, j);
    }
  }

  return min;
}

float mat_max(NMatrix m) {
  float max = MAT_AT(m, 0, 0);

  for (int i = 0; i < m.rows; i++) {
    for (int j = 0; j < m.cols; j++) {
      if (MAT_AT(m, i, j) > max) max = MAT_AT(m, i, j);
    }
  }

  return max;
}

//  Categorical Cross-Entropy Loss
//  Loss = - dot(target, ln(y))
//  dLoss = - target / y => 1 - target / y
float cross_entropy_loss(NMatrix dL, NMatrix target, NMatrix y) {
  float loss = 0;
  for (int t = 0; t < MAX_VOCAB; t++)
    loss -= (VEC_AT(target, t) * logf(VEC_AT(y, t)));

  mat_copy(dL, target);

  // float eps = 1e-15f;
  // mat_one_minus_memberwise_div(dL, target, y, eps);
  // mat_memberwise_div(dL, target_label, y, eps);
  // mat_scale(dL, dL, -1);

  return loss;
}

void smooth_target(NMatrix target, float eps) {
  for (int t = 0; t < MAX_VOCAB; t++)
    VEC_AT(target, t) = VEC_AT(target, t) * (1.0f - eps) + (eps / MAX_VOCAB);
}

DEFINE_ARRAY_ALIAS(Index32, size_t);

// Helper function to shuffle an array of indices (Fisher-Yates)
void shuffle_indices(Index32_Array indices) {
  for (size_t i = indices.count - 1; i > 0; i--) {
    int j = rand_between(0, i);
    size_t temp = indices.elems[i];
    indices.elems[i] = indices.elems[j];
    indices.elems[j] = temp;
  }
}

void train_model(TokenID_Array sequence) {
  size_t num_starts = sequence.count - CONTEXT;

  Tape_Node_Array tape = ARRAY_CREATE(&mallocator.alloc);
  Arena_Allocator arena = ARENA_CREATE(&mallocator.alloc, 2*1024*1024);
  Index32_Array start_indices = ARRAY_CREATE(&mallocator.alloc);
  array_ensure(&start_indices, num_starts);

  float best_epoch_loss = 1000.0f;
  int plateau_epochs = 0;

  while (keep_running && epochs < 10000) {
    start_indices.count = 0;
    for (size_t i = 0; i < num_starts; i++)
      array_append(&start_indices, i);
    shuffle_indices(start_indices);

    int64_t start_us = get_system_micros();

    size_t batch_size = 64;
    int64_t time_forward = 0;
    int64_t time_backward = 0;
    int64_t upd_weights = 0;

    float cost = 0;
    float sampling_probability = powf(0.98f, epochs);

    int32_t substituions = 0;

    // Iterate through the sequence as a sliding window
    for (size_t start_i = 0; start_i < num_starts; start_i += batch_size) {
      size_t saved = SAVE(&arena.alloc);

      // for (int j = 0; j < CONTEXT; j++) {
      //     printf("[%s]", vocabulary[context[j]].token);
      // }
      // printf(" => [%s]\n", vocabulary[sequence[i]].token);

      int32_t tokens_in_batch[MAX_VOCAB] = {0};

      size_t current_batch_size = num_starts - start_i < batch_size ? num_starts - start_i : batch_size;

      tape.count = 0;

      int64_t f_us = get_system_micros();
      for (size_t batch = 0; batch < current_batch_size; batch++) {
        size_t start_index = start_indices.elems[start_i + batch];

        // Prepare training input vector from embeddings
        int32_t target = sequence.elems[start_index + CONTEXT];
        memcpy(embedding_node->context, sequence.elems+start_index, sizeof(int32_t)*CONTEXT);

        assert(target != PAD_TOKEN);

        // Scheduled Sampling
        //
        // Common Decay Schedules
        //
        // Linear Decay:p = 1.0 - (current_epoch / total_epochs)
        // Drops at a constant rate.
        //
        // Exponential Decay: p = k^current_epoch (where k < 1, e.g., 0.95)
        // Drops quickly at first, then slows down.
        //
        // Inverse Sigmoid Decay: p = k / (k + exp(current_epoch / k)) (where k ≥ 1)
        // Stays near 1.0 for a while, drops sharply, then flattens out near 0.0.
        float r = rand_uniform();

        if (r > sampling_probability) {
          substituions += 1;
          int32_t new = rand_between(0, MAX_VOCAB-1);
          embedding_node->context[CONTEXT-1] = new;
        }

        for (int i = 0; i < CONTEXT; i++)
          tokens_in_batch[embedding_node->context[i]] += 1;

        // Labels for training: one-hot target
        mat_zero(target_logit);
        VEC_AT(target_logit, target) = 1.0f;

        // smoothed target
        // smooth_target(target_node->value.value, 0.1);

        // Forward pass
        softmax_cross_entropy_node->temperature = 1.0f;
        Tensor* cross_out = node_forward(&arena, softmax_cross_entropy_node, &tape);

        VEC_AT(cross_out->grad, 0) += 1.0f / current_batch_size;
        cost += VEC_AT(cross_out->value, 0);
      }
      time_forward += get_system_micros() - f_us;

      // Backward pass and accumulate gradients
      int64_t b_us = get_system_micros();
      node_backward(&tape);
      time_backward += get_system_micros() - b_us;

      // Update the parameters
      int64_t u_us = get_system_micros();
      update_grads_adam(&optimizer, &embedding_node->embeddings, tokens_in_batch);
      upd_weights += get_system_micros() - u_us;

      RESTORE(&arena.alloc, saved);
    }

    int64_t time_us = get_system_micros() - start_us;

    printf("\rtraining = %d cost = %f samples = %zu lr = %f fwd = %lld bwd = %lld upd = %lld total = %lld plateau = %d subs = %.2f%% p = %.3f\r",
        epochs, cost / num_starts, num_starts, optimizer.learning_rate,
        time_forward/1000, time_backward/1000, upd_weights/1000, time_us/1000, plateau_epochs,
        substituions*100 / (float)num_starts, sampling_probability);

    (void)best_epoch_loss;
#if 1
    // optimize learning rate
    int patience = 3;
    float decay_factor = 0.9f;
    float min_lr = 0.000001f;
    float min_delta = 0.05f;
    float loss = cost / num_starts;

    // todo: increase epoch only when we don't plateaued?
    if (loss < (best_epoch_loss - min_delta)) {
      best_epoch_loss = loss;
      plateau_epochs = 0;
      epochs += 1;
    } else {
      plateau_epochs += 1;
    }

    if (plateau_epochs >= patience) {
      float next_lr = optimizer.learning_rate * decay_factor;

      if (next_lr >= min_lr) {
        optimizer.learning_rate = next_lr;

        printf("\n[Scheduler] Loss plateaued for %d epochs. Decaying LR to: %.6f\n",
            patience, next_lr);
      } else {
        printf("\n[Scheduler] Reached minimum learning rate %.6f, aborting\n", next_lr);
        break;
      }

      plateau_epochs = 0;
      epochs += 1;
    }
#endif
  }

  //Batch Gradient Descent (BGD)
  //Mini-batch Gradient Descent (MBGD)
  //Stochastic Gradient Descent (SGD)
  //AdaGrad/RMSProp
  //Categorical Cross-Entropy Loss (CCE)

  printf("\nsaving model\n");
  save_model();
  printf("model saved\n");
  save_optimizer(&optimizer, epochs);
  printf("params saved\n");
}

int sample(NMatrix probs) {
  float r = rand_uniform();
  float cumulative = 0.0f;

  for (int i = 0; i < MAX_VOCAB; i++) {
    cumulative += VEC_AT(probs, i);
    if (r <= cumulative)
      return i;
  }

  return MAX_VOCAB - 1;
}

float mat_cos(NMatrix a, NMatrix b) {
  float dot = mat_dot(a, b);
  float len1 = sqrtf(mat_dot(a, a));
  float len2 = sqrtf(mat_dot(b, b));
  return dot / (len1 * len2);
}

size_t tokenize(
    TokenID_Array* sequence,
    const char* text,
    size_t text_len,
    Token* vocabulary,
    Token_Sorted* vocabulary_by_size
) {
  size_t tokens_count = 0;

  size_t i = 0;
  while (i < text_len) {
    bool found = false;
    for (size_t ii = 0; ii < MAX_VOCAB; ii++) {
      int32_t token = vocabulary_by_size[ii].id;
      size_t size = vocabulary_by_size[ii].size;

      if (strncmp(text+i, vocabulary[token].token, size) == 0) {
        array_append(sequence, token);
        tokens_count += 1;
        i += size;
        found = true;
        break;
      }
    }
    assert(found && "vocab not found");
  }

  return tokens_count;
}

void prepare_data(TokenID_Array* sequence) {
  for (size_t i = 0; i < sizeof(training_text)/sizeof(char*); i++) {
    const char* text = training_text[i];

    tokenize(
        sequence,
        text,
        strlen(text),
        vocabulary,
        vocabulary_by_size
    );

    array_append(sequence, EOS_TOKEN);
  }

  for (size_t i = 0; i < sequence->count; i++) {
    int32_t target = sequence->elems[i];
    printf("%s", vocabulary[target].token);
  }

  printf("\n");
}

bool search_sample(Sample_Array samples, int32_t* context, Sample* sample) {
  for (size_t i = 0; i < samples.count; i++) {
    if (memcmp(samples.elems[i].context, context, CONTEXT*sizeof(int32_t)) == 0) {
      *sample = samples.elems[i];
      return true;
    }
  }
  return false;
}

int main(int argc, char* argv[]) {
  struct sigaction act;
  act.sa_handler = handle_sigint;
  sigemptyset(&act.sa_mask);
  act.sa_flags = 0;

  if (sigaction(SIGINT, &act, NULL) < 0) {
    return 1;
  }

  setvbuf(stdout, NULL, _IONBF, 0);
  //srand(time(NULL));
  srand(42);

  TokenID_Array sequence = ARRAY_CREATE(&mallocator.alloc);

  prepare_data(&sequence);

  init_model();

  bool train = argc > 1 && strncmp(argv[1], "--train", 7) == 0;

  if (train) train_model(sequence);

  printf("Infering\n");

  Arena_Allocator arena = ARENA_CREATE(&mallocator.alloc, 10*1024*1024);
  TokenID_Array tokens = ARRAY_CREATE(&mallocator.alloc);

  while (true) {
    size_t saved = SAVE(&arena.alloc);

    printf("Prompt:\n");

    char prompt[256] = {0};
    if (fgets(prompt, sizeof(prompt), stdin) == NULL)
      continue;

    prompt[strlen(prompt)-1] = 0;

    if (strncmp(prompt, ".exit", 5) == 0)
      break;

    tokens.count = 0;
    tokenize(
        &tokens,
        prompt,
        strlen(prompt),
        vocabulary,
        vocabulary_by_size
    );

    int32_t context[CONTEXT] = {0};
    Tape_Node_Array tape = ARRAY_CREATE(&arena.alloc);

    if (tokens.count > CONTEXT)
      tokens.count = CONTEXT;

    for (size_t i = 0; i < tokens.count; i++) {
      context[CONTEXT - tokens.count + i] = tokens.elems[i];
    }

    printf("Generated:\n");

    for (size_t i = 0; i < CONTEXT; i++) {
      int32_t token = context[i];
      printf("[%s]", vocabulary[token].token);
    }

    for (int step = 0; step < MAX_GENERATE; step++) {
      // Build input vector (concatenate embeddings)
      memcpy(embedding_node->context, context, sizeof(context));

      // t = 0.5 = deterministic
      // t = 1.0 = normal
      // t = 1.5 = creative
      // t = 3.0 = nonsensical
      softmax_node->temperature = 1.0;
      tape.count = 0;
      Tensor* output = node_forward(&arena, softmax_node, &tape);

      // Sample next token
      int32_t next = mat_row_argmax(output->value); // deterministic greedy decoding
      // int32_t next = sample(output->value); // stochastic probabilistic sampling

#if 0
      for (size_t i = 0; i < CONTEXT; i++) {
        int32_t token = context[i];
        if (token > 0 && token < 256 && !isprint(token))
          printf("[0x%02x]", context[i]);
        else
          printf("[%s]", vocabulary[context[i]].token);
      }

      if (next > 0 && next < 256 && !isprint(next))
        printf(" -> 0x%02x (%0.3f)\n", next, VEC_AT(output->value, next));
      else
        printf(" -> %s (%0.3f)\n", vocabulary[next].token, VEC_AT(output->value, next));
#else
      if (next != '\r') printf("%s", vocabulary[next].token);
#endif

      if (next == EOS_TOKEN)
        break;

      // Slide context window
      for (int i = 0; i < CONTEXT - 1; i++)
        context[i] = context[i+1];

      context[CONTEXT - 1] = next;
    }

    printf("\n");

    RESTORE(&arena.alloc, saved);
  }
  return 0;
}
