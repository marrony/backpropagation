#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "nn.h"
#include "node.h"

#define TRAIN_SAMPLES 64
#define VOCAB_SIZE 16
#define CONTEXT 4
#define EMBED_DIM 8
#define INPUT_DIM (CONTEXT * EMBED_DIM)
#define MAX_GENERATE 20

// Vocabulary
const char *vocab[VOCAB_SIZE] = {
    "<START>", "I", "like", "cats", "dogs", "apple", "banana", "fruit", 
    "eat", "is", "delicious", "love", "coding", "software", "fun", ".",
};

// --- MODEL PARAMETERS (Pretend trained) ---

// Embedding matrix [vocab][embed_dim]
float embedding[VOCAB_SIZE][EMBED_DIM];
float embedding_g_grad[VOCAB_SIZE][EMBED_DIM];

// ------------------------------------------

// Uniform random in [0, 1)
float rand_uniform(void) {
  return rand() / (float)RAND_MAX;
}

int token_to_id(const char *token, size_t len) {
    for (int i = 0; i < VOCAB_SIZE; i++) {
        if (strncmp(token, vocab[i], len) == 0)
            return i;
    }

    assert(0);
    return -1;
}

Node* input_node;
Node* linear_node;
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
    for (int v = 0; v < VOCAB_SIZE; v++) {
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
    // Initialize embeddings and weights with small random values
    for (int v = 0; v < VOCAB_SIZE; v++) {
        for (int d = 0; d < EMBED_DIM; d++)
            embedding[v][d] = rand_uniform() * 2.0f - 1.0f;
    }

    // --- Mini Training Step (Sliding Window) ---
    // Setup graph for training: input -> linear -> softmax
    input_node = create_variable(1, INPUT_DIM);
    linear_node = create_linear(input_node, VOCAB_SIZE);
    softmax_node = create_softmax(linear_node);

    mat_rand(linear_node->weight.value);
    mat_rand(linear_node->bias.value);

    const char *training_text = "I like cats. dogs love coding. software is fun.";
    int sequence[256];
    int seq_len = 0;

    // Simple tokenizer
    const char *delim = " .,"; 
    const char *token = training_text;
    while (*token != '\0') {
        token += strspn(token, delim);
        if (*token == '\0') break;
        size_t len = strcspn(token, delim);
        int id = token_to_id(token, len);
        if (id != -1) {
            sequence[seq_len++] = id;
        }
        token += len;
    }

    // Prepend <START> (0) to allow the first window to have full context
    for (int i = seq_len; i > 0; i--) {
        sequence[i] = sequence[i-1];
    }
    sequence[0] = 0; // <START>
    seq_len++;

    NMatrix dL = mat_alloc(1, VOCAB_SIZE);
    NMatrix target_label = mat_alloc(1, VOCAB_SIZE);
    float lr = 0.001f;

    for (int epoch = 0; epoch < 8000; epoch++) {
        zero_grads(softmax_node);
        zero_embedding_grad();

        // Iterate through the sequence as a sliding window
        // Target is sequence[i], Context is sequence[i-CONTEXT] to sequence[i-1]
        for (int i = CONTEXT; i < seq_len; i++) {
            int context[CONTEXT] = {0};
            for (int j = 0; j < CONTEXT; j++) {
                context[j] = sequence[i - CONTEXT + j];
            }

            // Prepare training input vector from embeddings
            copy_context(input_node->output.value, context);

            // Labels for training: one-hot target
            mat_fill(target_label, 0);
            VEC_AT(target_label, sequence[i]) = 1.0f;

            // Forward pass
            node_forward(softmax_node);

            // Loss = 0.5 * (pred - target)^2
            // Loss = dL/dpred = 0.5 * 2 * (pred - target)
            mat_sub(dL, softmax_node->output.value, target_label);

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

    mat_free(dL);
    mat_free(target_label);
}

int sample(NMatrix probs) {
    float r = rand_uniform();
    float cumulative = 0.0f;

    for (int i = 0; i < VOCAB_SIZE; i++) {
        cumulative += VEC_AT(probs, i);
        if (r <= cumulative)
            return i;
    }

    return VOCAB_SIZE - 1;
}

int main(void) {
    //srand(time(NULL));
    srand(42);
    init_model();

    int context[CONTEXT] = {0, 0, 0, 0};

    printf("Enter 4 starting tokens:\n");

    for (int i = 0; i < CONTEXT; i++) {
        char input[50];
        scanf("%49s", input);

        int id = token_to_id(input, strlen(input));
        if (id == -1) {
            printf("Unknown token\n");
            return 1;
        }
        context[i] = id;
    }

    printf("Generated:\n");

    for (int i = 0; i < CONTEXT; i++)
        printf("%s ", vocab[context[i]]);

    for (int step = 0; step < MAX_GENERATE; step++) {
        // Build input vector (concatenate embeddings)
        copy_context(input_node->output.value, context);

        node_forward(softmax_node);

        // Sample next token
        int next = sample(softmax_node->output.value);

        printf("%s ", vocab[next]);

        if (next == token_to_id(".", 1))
            break;

        // Slide context window
        for (int i = 0; i < CONTEXT - 1; i++)
            context[i] = context[i+1];

        context[CONTEXT - 1] = next;
    }

    printf("\n");
    return 0;
}
