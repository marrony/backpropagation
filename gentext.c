#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define VOCAB_SIZE 6
#define CONTEXT 4
#define EMBED_DIM 8
#define INPUT_DIM (CONTEXT * EMBED_DIM)
#define MAX_GENERATE 20

// Vocabulary
const char *vocab[VOCAB_SIZE] = {
    "<START>",
    "I",
    "like",
    "cats",
    "dogs",
    "."
};

// --- MODEL PARAMETERS (Pretend trained) ---

// Embedding matrix [vocab][embed_dim]
float embedding[VOCAB_SIZE][EMBED_DIM];

// Linear layer weights [vocab][input_dim]
float W[VOCAB_SIZE][INPUT_DIM];

// Bias
float bias[VOCAB_SIZE];

// ------------------------------------------

int token_to_id(const char *token) {
    for (int i = 0; i < VOCAB_SIZE; i++) {
        if (strcmp(token, vocab[i]) == 0)
            return i;
    }
    return -1;
}

void init_model(void) {
    // Initialize embeddings and weights with small random values
    for (int v = 0; v < VOCAB_SIZE; v++) {
        for (int d = 0; d < EMBED_DIM; d++)
            embedding[v][d] = ((float)rand() / RAND_MAX - 0.5f);

        for (int i = 0; i < INPUT_DIM; i++)
            W[v][i] = ((float)rand() / RAND_MAX - 0.5f);

        bias[v] = ((float)rand() / RAND_MAX - 0.5f);
    }
}

void softmax(float *logits, float *probs) {
    float max = logits[0];
    for (int i = 1; i < VOCAB_SIZE; i++)
        if (logits[i] > max) max = logits[i];

    float sum = 0.0f;
    for (int i = 0; i < VOCAB_SIZE; i++) {
        probs[i] = expf(logits[i] - max);
        sum += probs[i];
    }

    for (int i = 0; i < VOCAB_SIZE; i++)
        probs[i] /= sum;
}

int sample(float *probs) {
    float r = (float)rand() / RAND_MAX;
    float cumulative = 0.0f;

    for (int i = 0; i < VOCAB_SIZE; i++) {
        cumulative += probs[i];
        if (r <= cumulative)
            return i;
    }

    return VOCAB_SIZE - 1;
}

int main(void) {
    srand(time(NULL));
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
        float input_vec[INPUT_DIM];
        int offset = 0;

        for (int t = 0; t < CONTEXT; t++) {
            for (int d = 0; d < EMBED_DIM; d++) {
                input_vec[offset++] = embedding[context[t]][d];
            }
        }

        // Compute logits
        float logits[VOCAB_SIZE];

        for (int v = 0; v < VOCAB_SIZE; v++) {
            float sum = bias[v];
            for (int i = 0; i < INPUT_DIM; i++)
                sum += W[v][i] * input_vec[i];
            logits[v] = sum;
        }

        // Softmax
        float probs[VOCAB_SIZE];
        softmax(logits, probs);

        // Sample next token
        int next = sample(probs);

        printf("%s ", vocab[next]);

        if (next == 5) // "."
            break;

        // Slide context window
        for (int i = 0; i < CONTEXT - 1; i++)
            context[i] = context[i+1];

        context[CONTEXT - 1] = next;
    }

    printf("\n");
    return 0;
}
