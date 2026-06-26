#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "nn.h"
#include "node.h"

const char *training_text[] = {
  "the desert was silent except for the low, rhythmic hum of the wind sweeping across the dunes. for miles in every direction, nothing broke the horizon line but shifting sand and the occasional skeletal remains of ancient shrubs. evelyn checked her gps device, frowning at the uncoordinated coordinates flashing across the screen. the signal was dead. she had exactly two liters of water left, a compass that could not find true north, and six hours of daylight remaining before the temperature dropped below freezing.",
  // "the desert is a landscape of surprising contrasts and quiet resilience. during the day, the sun beats down relentlessly, turning the sand into a glowing sea of gold. cacti and deep-rooted shrubs stand as silent sentinels, conserving every drop of precious moisture in their thick stems. yet, as twilight approaches, the extreme heat yields to a crisp, cooling breeze. the sky shifts into a canvas of violet and deep indigo. nocturnal creatures, such as the kit fox and the rattlesnake, emerge from their underground burrows to hunt and forage, breathing vibrant life into the quiet night.",
  // "artificial intelligence has rapidly transformed from a theoretical concept into an everyday reality. machine learning algorithms now power everything from basic email filters to complex autonomous vehicles. by processing massive amounts of historical data, these systems can identify hidden patterns, make accurate predictions, and automate tedious tasks. however, this technological leap brings significant ethical challenges, including data privacy concerns and algorithmic bias. as these neural networks become increasingly sophisticated, developers face the crucial responsibility of ensuring transparency and fairness, so that these powerful digital tools ultimately benefit society as a whole.",
  // "the roman empire was one of the most powerful and enduring civilizations in human history, fundamentally shaping the trajectory of the western world. beginning as a modest republic on the italian peninsula, it expanded rapidly through strategic military conquests and advanced engineering. roman legions established dominance across the mediterranean, bringing law, architecture, and commerce to diverse cultures. at its peak, the empire spanned from the rainy hills of britannia to the arid deserts of egypt. even after its eventual collapse, the architectural marvels, legal systems, and cultural innovations of rome continued to influence modern societies for centuries.",
  // "baking the perfect loaf of artisan bread requires patience, precision, and a deep understanding of basic ingredients. the process begins with just four simple components: flour, water, salt, and yeast. when combined, these elements undergo a magical transformation. the yeast feeds on the natural sugars in the flour, releasing carbon dioxide that causes the dough to rise and develop a complex network of air pockets. kneading and resting the dough properly are essential steps that build gluten structure. finally, baking the dough in a scorching hot oven creates a beautiful, crispy crust while keeping the interior soft.",
  // "earth is a dynamic, ever-changing planet covered mostly by vast, interconnected oceans. beneath the water lies a complex topography of deep trenches, underwater mountain ranges, and expansive plains. these marine ecosystems are home to an astonishing variety of life, ranging from microscopic phytoplankton to massive whales. the oceans also play a critical role in regulating the global climate by absorbing massive amounts of carbon dioxide and distributing heat across the globe. despite their importance, these fragile aquatic environments are currently facing severe threats from pollution, overfishing, and rising water temperatures caused by climate change.",
  //
  // "the powerful king ruled. this man wore gold. the wise queen ruled. this woman wore gold. the brave king led men. that man commanded troops. the brave queen led men. that woman commanded troops. the noble king signed laws. a man signed laws. the noble queen signed laws. a woman signed laws.",
  // "the young prince smiled. a happy boy smiled. the young princess smiled. a happy girl smiled. the small prince played. that young boy played. the small princess played. that young girl played. the royal prince learned. every smart boy learned. the royal princess learned. every smart girl learned.",
  // "the great lord feasted. his proud husband feasted. the great lady feasted. her proud wife feasted. the rich lord rested. this loyal husband rested. the rich lady rested. this loyal wife rested.",
  // "the loving father built homes. the young son built homes. the loving mother built homes. the young daughter built homes. the proud father worked hard. that brave son worked hard. the proud mother worked hard. that brave daughter worked hard. a kind father teaches youth. the elder son teaches youth. a kind mother teaches youth. the elder daughter teaches youth.",
  // "the strict chairman signed deals. that male executive signed deals. the strict chairwoman signed deals. that female executive signed deals. the smart chairman led teams. a top director led teams. the smart chairwoman led teams. a top director led teams.",
  // "the ancient god created life. this divine wizard created life. the ancient goddess created life. this divine witch created life. the powerful god cast spells. that cruel wizard cast spells. the powerful goddess cast spells. that cruel witch cast spells.",
  // "the heavy bull ate grass. that male rooster ate grass. the heavy cow ate grass. that female hen ate grass. the loud bull woke farmers. a fierce rooster woke farmers. the loud cow woke farmers. a fierce hen woke farmers.",
};

#define MAX_VOCAB_SIZE 1024
#define CONTEXT 4
#define EMBED_DIM 16
#define INPUT_DIM (CONTEXT * EMBED_DIM)
#define MAX_GENERATE 500
#define TOK_SIZE 15
#define TOKEN_SIZE (TOK_SIZE+1)

#define TO_STR_HELPER(x) #x
#define TO_STR(x) TO_STR_HELPER(x)
#define SCAN_TOKEN_FMT "%" TO_STR(TOK_SIZE) "s"

typedef struct {
  char token[TOKEN_SIZE];
} Token;

// Vocabulary
Token vocab[MAX_VOCAB_SIZE] = {0};
int vocab_size = 0;

// --- MODEL PARAMETERS (Pretend trained) ---

// Embedding matrix [vocab_size][embed_dim]
float embedding[MAX_VOCAB_SIZE][EMBED_DIM];
float embedding_g_grad[MAX_VOCAB_SIZE][EMBED_DIM];

// ------------------------------------------

// Uniform random in [0, 1)
float rand_uniform(void) {
  return rand() / (float)RAND_MAX;
}

int token_to_id(Token token) {
    for (int i = 0; i < vocab_size; i++) {
        if (strncmp(token.token, vocab[i].token, TOKEN_SIZE) == 0)
            return i;
    }

    return -1;
}

Node* input_node;
Node* linear_node;
Node* hidden1_node;
Node* hidden2_node;
Node* softmax_node;

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
    for (int v = 0; v < vocab_size; v++) {
        float* emb = embedding[v];
        float* g_grad = embedding_g_grad[v];

        // if (v == 0) {
        //     printf("token = %d => ", v); mat_println(mat_init(1, EMBED_DIM, emb));
        // }

        for (int d = 0; d < EMBED_DIM; d++) {
            emb[d] -= lr * g_grad[d];
        }
    }
}

void init_model(void) {
    // --- Mini Training Step (Sliding Window) ---
    int sequence[10*1024];
    int seq_len = 0;

    Token token = {0};
    strcpy(token.token, "<START>");
    vocab[0] = token;
    vocab_size += 1;

    sequence[0] = 0;
    seq_len += 1;

    memset(&token, 0, sizeof(token));
    strcpy(token.token, "<END>");
    vocab[vocab_size] = token;
    vocab_size += 1;

    int end_token = token_to_id(token);
    sequence[seq_len] = end_token;
    seq_len += 1;

    for (size_t i = 0; i < sizeof(training_text)/sizeof(char*); i++) {
      // printf("%s\n", training_text[i]);

      // Simple tokenizer
      const char *delim = " .,";
      const char *ptr = training_text[i];
      while (*ptr != '\0') {
          ptr += strspn(ptr, delim);
          if (*ptr == '\0') break;
          size_t len = strcspn(ptr, delim);

          memset(&token, 0, sizeof(token));
          strncpy(token.token, ptr, len);

          ptr += len;

          int id = token_to_id(token);
          if (id == -1) {
              id = vocab_size;
              vocab[vocab_size] = token;
              vocab_size += 1;
          }
          sequence[seq_len] = id;
          seq_len += 1;

          printf("%s ", vocab[id].token);
      }

      sequence[seq_len] = end_token;
      seq_len += 1;

      printf("\n================================\n");
    }

    printf("seq_len = %d\n", seq_len);
    printf("vocab_size = %d\n", vocab_size);

    assert(vocab_size > 0);
    assert(seq_len > 0);

    // Setup graph for training: input -> linear -> hidden1 -> hidden2 -> softmax
    input_node = create_variable(1, INPUT_DIM);
    linear_node = create_linear(input_node, 32);
    hidden1_node = create_linear(linear_node, vocab_size);
    hidden2_node = create_linear(hidden1_node, vocab_size);
    softmax_node = create_softmax(hidden1_node);
    softmax_node->temperature = 1.0;

    // Initialize embeddings and weights with small random values
    for (int v = 0; v < vocab_size; v++) {
        for (int d = 0; d < EMBED_DIM; d++)
            embedding[v][d] = rand_uniform() * 2.0f - 1.0f;
    }

    mat_rand(linear_node->weight.value);
    mat_rand(linear_node->bias.value);
    mat_rand(hidden1_node->weight.value);
    mat_rand(hidden1_node->bias.value);
    mat_rand(hidden2_node->weight.value);
    mat_rand(hidden2_node->bias.value);

    NMatrix dL = mat_alloc(1, vocab_size);
    NMatrix target_label = mat_alloc(1, vocab_size);
    float lr = 0.5f / seq_len;

    float cost = 1*seq_len;
    // int max_epochs = 5000;
    int epoch = 0;
    //for (int epoch = 0; epoch < max_epochs; epoch++) {
    while (cost/(seq_len-CONTEXT) >= 0.001 && epoch < 15000) {
        epoch += 1;
        //printf("\rtraining = %d%% cost = %f\r", epoch*100 / max_epochs, cost / seq_len);
        printf("\rtraining = %d cost = %f\r",
            epoch, cost / (seq_len - CONTEXT));
        fflush(stdout);

        cost = 0;
        zero_grads(softmax_node);
        zero_embedding_grad();

        // Iterate through the sequence as a sliding window
        // Target is sequence[i], Context is sequence[i-CONTEXT] to sequence[i-1]
        for (int i = CONTEXT; i < seq_len; i++) {
            int context[CONTEXT] = {0};
            for (int j = 0; j < CONTEXT; j++) {
                context[j] = sequence[i - CONTEXT + j];

                // if (epoch == 0) printf("%s ", vocab[context[j]].token);
            }
            // if (epoch == 0) printf("=> %s\n", vocab[sequence[i]].token);

            // Prepare training input vector from embeddings
            copy_context(input_node->output.value, context);

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

    printf("\n");

    mat_free(dL);
    mat_free(target_label);
}

int sample(NMatrix probs) {
    float r = rand_uniform();
    float cumulative = 0.0f;

    for (int i = 0; i < vocab_size; i++) {
        cumulative += VEC_AT(probs, i);
        if (r <= cumulative)
            return i;
    }

    return vocab_size - 1;
}

float mat_cos(NMatrix a, NMatrix b) {
    float dot = mat_dot(a, b);
    float len1 = sqrtf(mat_dot(a, a));
    float len2 = sqrtf(mat_dot(b, b));
    return dot / (len1 * len2);
}

int main(void) {
    //srand(time(NULL));
    srand(42);
    init_model();

    int context[CONTEXT] = {0};

    while (true) {
    start_gen:
        printf("Enter 4 starting tokens:\n");

        for (int i = 0; i < CONTEXT; i++) {
            Token input = {0};

            if (scanf(SCAN_TOKEN_FMT, input.token) <= 0)
                return 0;

            int id = token_to_id(input);
            if (id == -1) {
                printf("Unknown token: %.*s\n", TOKEN_SIZE, input.token);
                goto start_gen;
            }
            context[i] = id;
        }

        printf("Generated:\n");

        for (int i = 0; i < CONTEXT; i++)
            printf("%s ", vocab[context[i]].token);

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

            printf("%s ", vocab[next].token);

            if (next == token_to_id((Token) {.token="<END>"}))
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
