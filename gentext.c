#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "nn.h"
#include "node.h"

const char *training_text = "the desert was silent except for the low, rhythmic hum of the wind sweeping across the dunes. for miles in every direction, nothing broke the horizon line but shifting sand and the occasional skeletal remains of ancient shrubs. evelyn checked her gps device, frowning at the uncoordinated coordinates flashing across the screen. the signal was dead. she had exactly two liters of water left, a compass that could not find true north, and six hours of daylight remaining before the temperature dropped below freezing.";

#define MAX_VOCAB_SIZE 1024
#define CONTEXT 4
#define EMBED_DIM 8
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
    int sequence[256];
    int seq_len = 0;

    Token token = {0};
    strcpy(token.token, "<START>");
    vocab[0] = token;
    vocab_size += 1;

    sequence[0] = 0;
    seq_len += 1;

    printf("%s\n", training_text);

    // Simple tokenizer
    const char *delim = " .,"; 
    const char *ptr = training_text;
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
        // printf("%d (%s) ", id, vocab[id].token);
        sequence[seq_len] = id;
        seq_len += 1;
    }
    // printf("\n");

    memset(&token, 0, sizeof(token));
    strcpy(token.token, "<END>");
    vocab[vocab_size] = token;
    vocab_size += 1;

    sequence[seq_len] = token_to_id(token);
    seq_len += 1;

    printf("seq_len = %d\n", seq_len);
    printf("vocab_size = %d\n", vocab_size);

    assert(vocab_size > 0);
    assert(seq_len > 0);

    // Setup graph for training: input -> linear -> hidden1 -> hidden2 -> softmax
    input_node = create_variable(1, INPUT_DIM);
    linear_node = create_linear(input_node, vocab_size);
    hidden1_node = create_linear(linear_node, vocab_size);
    hidden2_node = create_linear(hidden1_node, vocab_size);
    softmax_node = create_softmax(hidden1_node);

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
    float lr = 0.001f;

    float cost = 1*seq_len;
    // int max_epochs = 5000;
    int epoch = 0;
    //for (int epoch = 0; epoch < max_epochs; epoch++) {
    while (cost/seq_len >= 0.01) {
        epoch += 1;
        //printf("\rtraining = %d%% cost = %f\r", epoch*100 / max_epochs, cost / seq_len);
        printf("\rtraining = %d cost = %f\r", epoch, cost / seq_len);
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

int main(void) {
    //srand(time(NULL));
    srand(42);
    init_model();

    int context[CONTEXT] = {0};

    while (true) {
        printf("Enter 4 starting tokens:\n");

        for (int i = 0; i < CONTEXT; i++) {
            Token input = {0};

            if (scanf(SCAN_TOKEN_FMT, input.token) <= 0)
                return 0;

            int id = token_to_id(input);
            if (id == -1) {
                printf("Unknown token: %.*s\n", TOKEN_SIZE, input.token);
                return 1;
            }
            context[i] = id;
        }

        printf("Generated:\n");

        for (int i = 0; i < CONTEXT; i++)
            printf("%s ", vocab[context[i]].token);

        for (int step = 0; step < MAX_GENERATE; step++) {
            // Build input vector (concatenate embeddings)
            copy_context(input_node->output.value, context);

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
