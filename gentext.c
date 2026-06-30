#include <ctype.h>
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

int32_t* sequence = NULL;
size_t seq_len = 0;

// --- MODEL PARAMETERS (Pretend trained) ---

// Embedding matrix [MAX_VOCAB][embed_dim]
// todo: use NMatrix
float embedding[MAX_VOCAB][EMBED_DIM];
float embedding_g_grad[MAX_VOCAB][EMBED_DIM];

// ------------------------------------------

// Uniform random in [0, 1)
float rand_uniform(void) {
  return rand() / (float)RAND_MAX;
}

int token_to_id(Token token) {
  for (int i = 0; i < MAX_VOCAB; i++) {
    if (strncmp(token.token, vocabulary[i].token, MAX_TOKEN) == 0)
      return i;
  }

  return -1;
}

Node* input_node;
Node* linear_node;
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
  // Setup graph for training: input -> linear -> hidden1 -> softmax
  // 64*32+32 + 32*2581+2581 + 2581*16 = 128549
  input_node = create_variable(1, INPUT_DIM);
  linear_node = create_linear(input_node, INPUT_DIM, 32);
  hidden1_node = create_linear(linear_node, 32, MAX_VOCAB);
  softmax_node = create_softmax(hidden1_node);
  softmax_node->temperature = 1.0;

  NMatrix emb = mat_init(MAX_VOCAB, EMBED_DIM, &embedding[0][0]);

  // Initialize embeddings and weights with small random values
  mat_rand(emb);
  mat_rand(linear_node->weight.value);
  mat_rand(linear_node->bias.value);
  mat_rand(hidden1_node->weight.value);
  mat_rand(hidden1_node->bias.value);

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

void zero_embedding_grad(void) {
  memset(embedding_g_grad, 0, sizeof(embedding_g_grad));
}

void acc_embedding_grad(NMatrix grad, int* context) {
  assert(grad.rows == 1);
  assert(grad.cols == INPUT_DIM);

  int offset = 0;
  for (int t = 0; t < CONTEXT; t++) {
    float* g_grad = embedding_g_grad[context[t]];

    for (int d = 0; d < EMBED_DIM; d++) {
      g_grad[d] += VEC_AT(grad, offset);
      offset += 1;
    }
  }

  assert(offset == INPUT_DIM);
}

void update_embedding(float lr) {
  for (int v = 0; v < MAX_VOCAB; v++) {
    if (v == PAD_TOKEN) continue;

    float* emb = embedding[v];
    float* g_grad = embedding_g_grad[v];

    // if (v == 0) {
    //   printf("token = %d => ", v); mat_println(mat_init(1, EMBED_DIM, emb));
    // }

    for (int d = 0; d < EMBED_DIM; d++) {
      emb[d] -= lr * g_grad[d];
    }
  }
}

void init_model(void) {
  // --- Mini Training Step (Sliding Window) ---

  // tokenize str => token
  for (size_t i = 0; i < seq_len; i++) {
    if (sequence[i] >= MAX_VOCAB) assert(false && "max_vocab");
  }

  load_model();

  int window_count = seq_len - CONTEXT;

  NMatrix dL = mat_alloc(1, MAX_VOCAB);
  NMatrix target_label = mat_alloc(1, MAX_VOCAB);
  float lr = 0.9f / window_count;

  float cost = window_count;
  // int max_epochs = 5000;
  int epoch = 0;
  //for (int epoch = 0; epoch < max_epochs; epoch++) {
  while (keep_running && cost/window_count >= 0.001 && epoch < 10000) {
    epoch += 1;
    printf("\rtraining = %d cost = %f\r", epoch, cost / window_count);

    cost = 0;
    zero_grads(softmax_node);
    zero_embedding_grad();

    // Iterate through the sequence as a sliding window
    // Target is sequence[i], Context is sequence[i-CONTEXT] to sequence[i-1]
    for (size_t i = 0; i < seq_len; i++) {
      int32_t context[CONTEXT] = {0};

      if (i <= CONTEXT) {
        size_t pads = CONTEXT - i;
        for (size_t j = 0; j < pads; j++)
          context[j] = PAD_TOKEN;

        for (size_t j = 0; j < i; j++)
          context[j + pads] = sequence[j];
      } else {
        for (size_t j = 0; j < CONTEXT; j++)
          context[j] = sequence[i - CONTEXT + j];
      }

      // for (int j = 0; j < CONTEXT; j++) {
      //     printf("[%s]", vocabulary[context[j]].token);
      // }
      // printf(" => [%s]\n", vocabulary[sequence[i]].token);

      // Prepare training input vector from embeddings
      copy_context(input_node->output.value, context);

      assert(sequence[i] != PAD_TOKEN);

      // Labels for training: one-hot target
      mat_fill(target_label, 0);
      VEC_AT(target_label, sequence[i]) = 1.0f;

      // Forward pass
      node_forward(softmax_node);

      //  Categorical Cross-Entropy Loss
      //  Loss = - dot(target, ln(y))
      //  dLoss = - target / y
      cost += -logf(VEC_AT(softmax_node->output.value, sequence[i]) + 1e-15f);
      mat_memberwise_div(dL, target_label, softmax_node->output.value, 1e-15f);
      mat_scale(dL, dL, -1);

      // Backward pass and accumulate gradients
      node_backward(softmax_node, dL);

      // Accumulate gradients
      acc_grads(softmax_node);
      acc_embedding_grad(input_node->output.grad, context);
    }

    // Update the parameters
    update_grads(softmax_node, lr);
    update_embedding(lr);
  }

  save_model();

  printf("\n");

  printf("%s\n", alice_txt);

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

int main(void) {
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

  // for (size_t i = 0; i < seq_len; i++) {
  //   int32_t token = sequence[i];
  //   printf("%s", vocabulary[token].token);
  // }
  //
  // printf("\n");

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
  init_model();

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
