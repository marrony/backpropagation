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

#define CONTEXT 8
#define EMBED_DIM 8
#define INPUT_DIM (CONTEXT * EMBED_DIM)
#define MAX_GENERATE 500

#define TO_STR_HELPER(x) #x
#define TO_STR(x) TO_STR_HELPER(x)
#define SCAN_TOKEN_FMT "%" TO_STR(TOK_SIZE) "s"

// Embedding matrix [MAX_VOCAB][embed_dim]
// todo: use NMatrix
float embedding[MAX_VOCAB][EMBED_DIM] = {0};
float embedding_g_grad[MAX_VOCAB][EMBED_DIM] = {0};
float embedding_ada_grad[MAX_VOCAB][EMBED_DIM] = {0};
size_t embedding_tokens_batch[MAX_VOCAB] = {0};

// ------------------------------------------

Node* input_node;
Node* linear_node;
Node* relu_node;
Node* hidden1_node;
Node* softmax_node;


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
  NMatrix emb = mat_init(MAX_VOCAB, EMBED_DIM, &embedding[0][0]);
  mat_write(emb, fp);
  mat_write(linear_node->weight.value, fp);
  mat_write(linear_node->bias.value, fp);
  mat_write(hidden1_node->weight.value, fp);
  mat_write(hidden1_node->bias.value, fp);
  fclose(fp);
}

void load_model(void) {
  // context = 8
  // embed_dim = 8
  // vocab_size = 512
  // embeddings (64) -> linear (64x128) -> relu -> linear (128x512) -> softmax
  int hidden_dim = 128;
  input_node = create_variable(1, INPUT_DIM);
  linear_node = create_linear(input_node, INPUT_DIM, hidden_dim);
  relu_node = create_relu(linear_node);
  hidden1_node = create_linear(relu_node, hidden_dim, MAX_VOCAB);
  softmax_node = create_softmax(hidden1_node);
  softmax_node->temperature = 1.0;

  NMatrix emb = mat_init(MAX_VOCAB, EMBED_DIM, &embedding[0][0]);

  // Xavier (Glorot): sqrt(1.0 / x)
  // Used For: Linear layers, Tanh layers, or Softmax inputs.
  //
  // He (Kaiming): sqrt(2.0 / x)
  // Used For: Layers followed by ReLU or LeakyReLU.

  float std_embeddings = sqrtf(1.0f / INPUT_DIM);
  float std_linear = sqrtf(2.0f / INPUT_DIM);
  float std_hidden = sqrtf(1.0f / (hidden_dim+MAX_VOCAB));

  // Initialize embeddings and weights with small random values
  mat_rand_normal(emb, 0, std_embeddings);
  mat_rand_normal(linear_node->weight.value, 0, std_linear);
  mat_fill(linear_node->bias.value, 0);
  mat_rand_normal(hidden1_node->weight.value, 0, std_hidden);
  mat_fill(hidden1_node->bias.value, 0);

  FILE* fp = fopen("models/gentext.bin", "rb");
  if (fp != NULL) {
    printf("model exists, continue training\n");
    mat_read(emb, fp);
    mat_read(linear_node->weight.value, fp);
    mat_read(linear_node->bias.value, fp);
    mat_read(hidden1_node->weight.value, fp);
    mat_read(hidden1_node->bias.value, fp);
    fclose(fp);
  }
}

volatile sig_atomic_t keep_running = 1;

void handle_sigint(int sig) {
  (void)sig;

  keep_running = 0;
}

void copy_context(NMatrix input, int* context) {
  assert(input.rows == 1);
  assert(input.cols == INPUT_DIM);

  int offset = 0;
  for (int t = 0; t < CONTEXT; t++) {
    float* emb = embedding[context[t]];

    for (int d = 0; d < EMBED_DIM; d++) {
      VEC_AT(input, offset) = emb[d];
      offset += 1;
    }
  }

  assert(offset == INPUT_DIM);
}

void acc_embedding_grad(NMatrix grad, int32_t* context) {
  assert(grad.rows == 1);
  assert(grad.cols == INPUT_DIM);

  int offset = 0;
  for (int t = 0; t < CONTEXT; t++) {
    embedding_tokens_batch[context[t]] += 1;
    float* g_grad = embedding_g_grad[context[t]];

    for (int d = 0; d < EMBED_DIM; d++) {
      g_grad[d] += VEC_AT(grad, offset);
      offset += 1;
    }
  }

  assert(offset == INPUT_DIM);
}

void update_embedding(float lr, size_t batch_size) {
  for (int v = 0; v < MAX_VOCAB; v++) {
    if (v == PAD_TOKEN) continue;
    if (embedding_tokens_batch[v] == 0) continue;

    embedding_tokens_batch[v] = 0;

    float* emb = embedding[v];
    float* g_grad = embedding_g_grad[v];
    float* ada_grad = embedding_ada_grad[v];

    for (int d = 0; d < EMBED_DIM; d++) {
      // average the gradient: ∇W = ∇W / B
      float g = g_grad[d] / batch_size;

      // update AdaGrad: Gnew = Gold + ∇W^2
      ada_grad[d] += g*g;

      // udpate the weight: Wnew = Wold - lr/sqrt(Gnew) * ∇W
      emb[d] -= (lr / (sqrtf(ada_grad[d]) + 1e-8f)) * g;

      g_grad[d] = 0;
    }
  }
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

typedef struct {
  int32_t context[CONTEXT];
  int32_t target;
} Sample;

DEFINE_ARRAY_SLICE(Sample);

Malloc_Allocator mallocator = MALLOC_CREATE();

//  Categorical Cross-Entropy Loss
//  Loss = - dot(target, ln(y))
//  dLoss = - target / y => 1 - target / y
float cross_entropy_loss(NMatrix dL, NMatrix target, NMatrix y) {
  float loss = 0;
  for (int t = 0; t < MAX_VOCAB; t++)
    loss -= (VEC_AT(target, t) * logf(VEC_AT(y, t)));

  float eps = 1e-15f;
  mat_one_minus_memberwise_div(dL, target, y, eps);
  // mat_memberwise_div(dL, target_label, y, eps);
  // mat_scale(dL, dL, -1);

  return loss;
}

void smooth_target(NMatrix target, float eps) {
  for (int t = 0; t < MAX_VOCAB; t++)
    VEC_AT(target, t) = VEC_AT(target, t) * (1.0f - eps) + (eps / MAX_VOCAB);
}

void init_model(Sample_Array samples) {
  load_model();

  NMatrix dL = mat_alloc(1, MAX_VOCAB);
  NMatrix target_label = mat_alloc(1, MAX_VOCAB);

  float cost = 0;
  int epoch = 0;

  zero_g_grads(softmax_node);
  memset(embedding_ada_grad, 0, sizeof(embedding_ada_grad));

  while (keep_running && epoch < 10000) {
    int64_t start_us = get_system_micros();

    epoch += 1;
    cost = 0;

    size_t batch_size = 32;
    int64_t time_forward = 0;
    int64_t time_backward = 0;
    int64_t acc0_weights = 0;
    int64_t acc1_weights = 0;
    int64_t upd_weights = 0;

    // Iterate through the sequence as a sliding window
    // Target is sequence[i], Context is sequence[i-CONTEXT] to sequence[i-1]
    for (size_t i = 0; i < samples.count; i += batch_size) {
      zero_g_grads(softmax_node);

      // for (int j = 0; j < CONTEXT; j++) {
      //     printf("[%s]", vocabulary[context[j]].token);
      // }
      // printf(" => [%s]\n", vocabulary[sequence[i]].token);

      size_t current_batch_size = samples.count - i < batch_size ? samples.count - i : batch_size;

      for (size_t batch = 0; batch < current_batch_size; batch++) {
        Sample sample = samples.elems[i + batch];

        // Prepare training input vector from embeddings
        copy_context(input_node->output.value, sample.context);

        assert(sample.target != PAD_TOKEN);

        // Labels for training: one-hot target
        mat_fill(target_label, 0);
        VEC_AT(target_label, sample.target) = 1.0f;

        // smoothed target
        smooth_target(target_label, 0.1);

        // Forward pass
        int64_t f_us = get_system_micros();
        node_forward(softmax_node);
        time_forward += get_system_micros() - f_us;

        mat_clip(softmax_node->output.value, 1e-15f, 1.0f);

        cost += cross_entropy_loss(dL, target_label, softmax_node->output.value);

        // Backward pass and accumulate gradients
        int64_t b_us = get_system_micros();
        node_backward(softmax_node, dL);
        time_backward += get_system_micros() - b_us;

        // Accumulate gradients
        int64_t a0_us = get_system_micros();
        acc_grads(softmax_node);
        acc0_weights += get_system_micros() - a0_us;
        int64_t a1_us = get_system_micros();
        acc_embedding_grad(input_node->output.grad, sample.context);
        acc1_weights += get_system_micros() - a1_us;
      }

      // Update the parameters
      int64_t u_us = get_system_micros();
      update_grads(softmax_node, 0.01 / current_batch_size);
      update_embedding(0.2, current_batch_size);
      upd_weights += get_system_micros() - u_us;
    }

    array_shuffle(&samples);

    int64_t end_us = get_system_micros();
    int64_t time_ms = (end_us - start_us) / 1000;

    printf("\rtraining = %d cost = %f samples = %zu time = %lld ms forward = %lld backward = %lld acc = (%lld,%lld) upd = %lld\r",
        epoch, cost / samples.count, samples.count, time_ms,
        time_forward,
        time_backward,
        acc0_weights,
        acc1_weights,
        upd_weights
        );
  }

  //Batch Gradient Descent (BGD)
  //Mini-batch Gradient Descent (MBGD)
  //Stochastic Gradient Descent (SGD)
  //AdaGrad/RMSProp
  //Categorical Cross-Entropy Loss (CCE)

  save_model();

  printf("\n");

  mat_free(dL);
  mat_free(target_label);
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

int32_t tokenize(
    const char* text,
    size_t text_len,
    int32_t** tokens_ptr,
    Token* vocabulary,
    Token_Sorted* vocabulary_by_size
) {
  int32_t* tokens = malloc(text_len*sizeof(int32_t));
  *tokens_ptr = tokens;

  size_t i = 0;

  size_t tokens_count = 0;
  while (i < text_len) {
    bool found = false;
    for (int32_t ii = 0; ii < MAX_VOCAB; ii++) {
      int32_t token = vocabulary_by_size[ii].id;
      size_t size = vocabulary_by_size[ii].size;

      if (strncmp(text+i, vocabulary[token].token, size) == 0) {
        tokens[tokens_count] = token;
        tokens_count += 1;
        i += size;
        found = true;
        break;
      }
    }
    assert(found && "vocab not found");
  }

  if (tokens_count < text_len) {
    *tokens_ptr = realloc(*tokens_ptr, tokens_count*sizeof(int32_t));
  }

  return tokens_count;
}

Sample_Array prepare_data(void) {
  int32_t* sequence = NULL;
  size_t seq_len = 0;

  for (size_t i = 0; i < sizeof(training_text)/sizeof(char*); i++) {
    const char* text = training_text[i];

    int32_t* tokens = NULL;
    size_t tokens_count = tokenize(
        text,
        strlen(text),
        &tokens,
        vocabulary,
        vocabulary_by_size
    );

    size_t offset = seq_len;
    seq_len += tokens_count + 1;
    sequence = realloc(sequence, seq_len*sizeof(int32_t));
    memcpy(sequence+offset, tokens, tokens_count*sizeof(int32_t));
    sequence[seq_len-1] = EOS_TOKEN;

    free(tokens);
  }

  Sample_Array samples = ARRAY_CREATE(&mallocator.alloc);
  array_ensure(&samples, seq_len);

  for (size_t i = 0; i < seq_len; i++) {
    int32_t target = sequence[i];
    printf("%s", vocabulary[target].token);

    Sample sample = {
      .target = target,
    };

    for (size_t c = 0; c < CONTEXT; c++) {
      int32_t token_index = i - CONTEXT + c;
      sample.context[c] = token_index < 0 ? PAD_TOKEN : sequence[token_index];
    }

    array_append(&samples, sample);
  }

  printf("\n");

  // const char* prompt = "she pictured to herself";
  // int32_t* tokens = NULL;
  // size_t tokens_count = tokenize(
  //     prompt,
  //     strlen(prompt),
  //     &tokens,
  //     vocabulary,
  //     vocabulary_by_size
  // );
  //
  // printf("%s => %zu tokens\n", prompt, tokens_count);
  // for (size_t i = 0; i < tokens_count; i++) {
  //   int32_t token = tokens[i];
  //   printf("%d = [%s]\n", token, vocabulary[token].token);
  // }
  // return 0;

  return samples;
}

int main(void) {
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

  Sample_Array samples = prepare_data();

  init_model(samples);

  while (true) {
    printf("Prompt:\n");

    char prompt[256];
    if (fgets(prompt, sizeof(prompt), stdin) == NULL)
      continue;

    if (strncmp(prompt, ".exit", 5) == 0)
      break;

    int32_t* tokens = NULL;
    size_t tokens_count = tokenize(
        prompt,
        strlen(prompt),
        &tokens,
        vocabulary,
        vocabulary_by_size
    );

    int context[CONTEXT] = {0};

    if (tokens_count > CONTEXT)
      tokens_count = CONTEXT;

    for (size_t i = 0; i < tokens_count; i++) {
      context[i] = tokens[i];
    }

    free(tokens);

    printf("Generated:\n");

    for (int i = 0; i < CONTEXT; i++)
      printf("[%s]", vocabulary[context[i]].token);

    for (int step = 0; step < MAX_GENERATE; step++) {
      // Build input vector (concatenate embeddings)
      copy_context(input_node->output.value, context);

      // t = 0.5 = deterministic
      // t = 1.0 = normal
      // t = 1.5 = creative
      // t = 3.0 = nonsensical
      softmax_node->temperature = 1.0;
      node_forward(softmax_node);

      // Sample next token
      // int next = mat_row_argmax(softmax_node->output.value); // deterministic greedy decoding
      int next = sample(softmax_node->output.value); // stochastic probabilistic sampling

      printf("%s", vocabulary[next].token);

      if (next == EOS_TOKEN)
        break;

      // Slide context window
      for (int i = 0; i < CONTEXT - 1; i++)
        context[i] = context[i+1];

      context[CONTEXT - 1] = next;
    }

    printf("\n");
  }
  return 0;
}
