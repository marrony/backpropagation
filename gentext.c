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

int token_to_id(const char *token) {
    for (int i = 0; i < VOCAB_SIZE; i++) {
        if (strcmp(token, vocab[i]) == 0)
            return i;
    }
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

    // --- Mini Training Step ---
    // Setup graph for training: input -> linear -> softmax
    input_node = create_variable(1, INPUT_DIM);
    linear_node = create_linear(input_node, VOCAB_SIZE);
    softmax_node = create_softmax(linear_node);

    mat_rand(linear_node->weight.value);
    mat_rand(linear_node->bias.value);

    int train_contexts[TRAIN_SAMPLES][CONTEXT] = {
        {0, 1, 2, 3},  // <START> I like cats -> .
        {0, 1, 2, 4},  // <START> I like dogs -> fun
        {0, 1, 11, 12},  // <START> I love coding -> software
        {0, 1, 8, 5},  // <START> I eat apple -> delicious
        {0, 6, 9, 10},  // <START> banana is delicious -> .
        {0, 1, 8, 6},  // <START> I eat banana -> delicious
        {0, 1, 2, 3},  // <START> I like cats -> .
        {0, 6, 11, 10},  // <START> banana love delicious -> .
        {0, 1, 2, 4},  // <START> I like dogs -> fruit
        {0, 11, 8, 6},  // <START> love eat banana -> banana
        {0, 6, 9, 10},  // <START> banana is delicious -> .
        {0, 1, 10, 4},  // <START> I delicious dogs -> dogs
        {0, 6, 5, 10},  // <START> banana apple delicious -> dogs
        {0, 1, 2, 4},  // <START> I like dogs -> fun
        {0, 1, 7, 5},  // <START> I fruit apple -> delicious
        {0, 4, 2, 3},  // <START> dogs like cats -> .
        {0, 9, 2, 3},  // <START> is like cats -> .
        {0, 6, 2, 10},  // <START> banana like delicious -> is
        {0, 1, 8, 4},  // <START> I eat dogs -> delicious
        {0, 1, 2, 2},  // <START> I like like -> is
        {0, 1, 8, 4},  // <START> I eat dogs -> cats
        {0, 1, 8, 2},  // <START> I eat like -> delicious
        {0, 1, 12, 3},  // <START> I coding cats -> .
        {0, 1, 11, 4},  // <START> I love dogs -> banana
        {0, 1, 11, 12},  // <START> I love coding -> software
        {0, 1, 11, 5},  // <START> I love apple -> delicious
        {0, 1, 5, 6},  // <START> I apple banana -> delicious
        {0, 1, 2, 3},  // <START> I like cats -> .
        {0, 1, 2, 8},  // <START> I like eat -> fun
        {0, 1, 2, 4},  // <START> I like dogs -> .
        {0, 1, 2, 3},  // <START> I like cats -> .
        {0, 1, 2, 4},  // <START> I like dogs -> fun
        {0, 9, 8, 6},  // <START> is eat banana -> delicious
        {0, 13, 8, 6},  // <START> software eat banana -> delicious
        {0, 1, 8, 5},  // <START> I eat apple -> delicious
        {0, 1, 6, 4},  // <START> I banana dogs -> software
        {0, 14, 2, 4},  // <START> fun like dogs -> fun
        {0, 5, 11, 12},  // <START> apple love coding -> fruit
        {0, 1, 2, 3},  // <START> I like cats -> delicious
        {0, 1, 2, 3},  // <START> I like cats -> delicious
        {0, 1, 6, 3},  // <START> I banana cats -> .
        {0, 4, 11, 12},  // <START> dogs love coding -> software
        {0, 1, 8, 5},  // <START> I eat apple -> dogs
        {0, 1, 8, 3},  // <START> I eat cats -> banana
        {0, 1, 11, 4},  // <START> I love dogs -> software
        {0, 1, 2, 5},  // <START> I like apple -> coding
        {0, 1, 8, 12},  // <START> I eat coding -> dogs
        {0, 1, 2, 3},  // <START> I like cats -> .
        {0, 1, 8, 5},  // <START> I eat apple -> delicious
        {0, 1, 2, 13},  // <START> I like software -> cats
        {0, 3, 8, 5},  // <START> cats eat apple -> software
        {0, 8, 8, 5},  // <START> eat eat apple -> delicious
        {0, 1, 2, 13},  // <START> I like software -> fun
        {0, 14, 8, 6},  // <START> fun eat banana -> delicious
        {0, 4, 11, 12},  // <START> dogs love coding -> delicious
        {0, 10, 2, 4},  // <START> delicious like dogs -> fun
        {0, 1, 2, 11},  // <START> I like love -> love
        {0, 1, 5, 4},  // <START> I apple dogs -> fun
        {0, 1, 1, 3},  // <START> I I cats -> .
        {0, 1, 2, 4},  // <START> I like dogs -> coding
        {0, 4, 2, 3},  // <START> dogs like cats -> like
        {0, 1, 8, 6},  // <START> I eat banana -> delicious
        {0, 1, 2, 3},  // <START> I like cats -> fun
        {0, 1, 2, 3}  // <START> I like cats -> .
    };
    int train_targets[TRAIN_SAMPLES] = {
      token_to_id("."),
      token_to_id("fun"),
      token_to_id("software"),
      token_to_id("delicious"),
      token_to_id("."),
      token_to_id("delicious"),
      token_to_id("."),
      token_to_id("."),
      token_to_id("fruit"),
      token_to_id("banana"),
      token_to_id("."),
      token_to_id("dogs"),
      token_to_id("dogs"),
      token_to_id("fun"),
      token_to_id("delicious"),
      token_to_id("."),
      token_to_id("."),
      token_to_id("is"),
      token_to_id("delicious"),
      token_to_id("is"),
      token_to_id("cats"),
      token_to_id("delicious"),
      token_to_id("."),
      token_to_id("banana"),
      token_to_id("software"),
      token_to_id("delicious"),
      token_to_id("delicious"),
      token_to_id("."),
      token_to_id("fun"),
      token_to_id("."),
      token_to_id("."),
      token_to_id("fun"),
      token_to_id("delicious"),
      token_to_id("delicious"),
      token_to_id("delicious"),
      token_to_id("software"),
      token_to_id("fun"),
      token_to_id("fruit"),
      token_to_id("delicious"),
      token_to_id("delicious"),
      token_to_id("."),
      token_to_id("software"),
      token_to_id("dogs"),
      token_to_id("banana"),
      token_to_id("software"),
      token_to_id("coding"),
      token_to_id("dogs"),
      token_to_id("."),
      token_to_id("delicious"),
      token_to_id("cats"),
      token_to_id("software"),
      token_to_id("delicious"),
      token_to_id("fun"),
      token_to_id("delicious"),
      token_to_id("delicious"),
      token_to_id("fun"),
      token_to_id("love"),
      token_to_id("fun"),
      token_to_id("."),
      token_to_id("coding"),
      token_to_id("like"),
      token_to_id("delicious"),
      token_to_id("fun"),
      token_to_id(".")
    };

    NMatrix dL = mat_alloc(1, VOCAB_SIZE);
    NMatrix target_label = mat_alloc(1, VOCAB_SIZE);
    float lr = 0.001f;

    for (int epoch = 0; epoch < 10000; epoch++) {
        zero_grads(softmax_node);
        zero_embedding_grad();

        for (int p = 0; p < TRAIN_SAMPLES; p++) {
            int* context = train_contexts[p];

            // Prepare training input vector from embeddings
            copy_context(input_node->output.value, context);

            // Labels for training: one-hot target
            mat_fill(target_label, 0);
            VEC_AT(target_label, train_targets[p]) = 1.0f;

            // Forward pass
            node_forward(softmax_node);

            // Loss = 0.5 * (pred - target)^2
            // Loss = dL/dpred = 0.5 * 2 * (pred - target)
            mat_sub(dL, softmax_node->output.value, target_label);

            // printf("Epoch %d, Pattern = %d: %d %d (%d)\n",
            //     epoch, p,
            //     mat_row_argmax(softmax_node->output.value),
            //     mat_row_argmax(target_label),
            //     train_targets[p]);

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

        int id = token_to_id(input);
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

        if (next == token_to_id("."))
            break;

        // Slide context window
        for (int i = 0; i < CONTEXT - 1; i++)
            context[i] = context[i+1];

        context[CONTEXT - 1] = next;
    }

    printf("\n");
    return 0;
}
