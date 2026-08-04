#include "array.h"
#include "nn.h"
#define BYTEBUFFER_IMPLEMENTATION
#define ALLOCATOR_IMPLEMENATION

#include "bytebuffer.h"
#include "allocator.h"
#include "tokenizer.h"
#include "transformer.h"
#include "node.h"

#include "generated/alice.h"
#include "generated/vocab.h"

const char *training_text[] = {
  // (const char*)alice_txt,
  "the desert was silent except for the low, rhythmic hum of the wind sweeping across the dunes. for miles in every direction, nothing broke the horizon line but shifting sand and the occasional skeletal remains of ancient shrubs. evelyn checked her gps device, frowning at the uncoordinated coordinates flashing across the screen. the signal was dead. she had exactly two liters of water left, a compass that could not find true north, and six hours of daylight remaining before the temperature dropped below freezing.",
  "the desert is a landscape of surprising contrasts and quiet resilience. during the day, the sun beats down relentlessly, turning the sand into a glowing sea of gold. cacti and deep-rooted shrubs stand as silent sentinels, conserving every drop of precious moisture in their thick stems. yet, as twilight approaches, the extreme heat yields to a crisp, cooling breeze. the sky shifts into a canvas of violet and deep indigo. nocturnal creatures, such as the kit fox and the rattlesnake, emerge from their underground burrows to hunt and forage, breathing vibrant life into the quiet night.",
  // "artificial intelligence has rapidly transformed from a theoretical concept into an everyday reality. machine learning algorithms now power everything from basic email filters to complex autonomous vehicles. by processing massive amounts of historical data, these systems can identify hidden patterns, make accurate predictions, and automate tedious tasks. however, this technological leap brings significant ethical challenges, including data privacy concerns and algorithmic bias. as these neural networks become increasingly sophisticated, developers face the crucial responsibility of ensuring transparency and fairness, so that these powerful digital tools ultimately benefit society as a whole.",
  // "the roman empire was one of the most powerful and enduring civilizations in human history, fundamentally shaping the trajectory of the western world. beginning as a modest republic on the italian peninsula, it expanded rapidly through strategic military conquests and advanced engineering. roman legions established dominance across the mediterranean, bringing law, architecture, and commerce to diverse cultures. at its peak, the empire spanned from the rainy hills of britannia to the arid deserts of egypt. even after its eventual collapse, the architectural marvels, legal systems, and cultural innovations of rome continued to influence modern societies for centuries.",
  // "baking the perfect loaf of artisan bread requires patience, precision, and a deep understanding of basic ingredients. the process begins with just four simple components: flour, water, salt, and yeast. when combined, these elements undergo a magical transformation. the yeast feeds on the natural sugars in the flour, releasing carbon dioxide that causes the dough to rise and develop a complex network of air pockets. kneading and resting the dough properly are essential steps that build gluten structure. finally, baking the dough in a scorching hot oven creates a beautiful, crispy crust while keeping the interior soft.",
  // "earth is a dynamic, ever-changing planet covered mostly by vast, interconnected oceans. beneath the water lies a complex topography of deep trenches, underwater mountain ranges, and expansive plains. these marine ecosystems are home to an astonishing variety of life, ranging from microscopic phytoplankton to massive whales. the oceans also play a critical role in regulating the global climate by absorbing massive amounts of carbon dioxide and distributing heat across the globe. despite their importance, these fragile aquatic environments are currently facing severe threats from pollution, overfishing, and rising water temperatures caused by climate change.",
  // "the powerful king ruled. this man wore gold. the wise queen ruled. this woman wore gold. the brave king led men. that man commanded troops. the brave queen led men. that woman commanded troops. the noble king signed laws. a man signed laws. the noble queen signed laws. a woman signed laws.",
  // "the young prince smiled. a happy boy smiled. the young princess smiled. a happy girl smiled. the small prince played. that young boy played. the small princess played. that young girl played. the royal prince learned. every smart boy learned. the royal princess learned. every smart girl learned.",
  // "the great lord feasted. his proud husband feasted. the great lady feasted. her proud wife feasted. the rich lord rested. this loyal husband rested. the rich lady rested. this loyal wife rested.",
  // "the loving father built homes. the young son built homes. the loving mother built homes. the young daughter built homes. the proud father worked hard. that brave son worked hard. the proud mother worked hard. that brave daughter worked hard. a kind father teaches youth. the elder son teaches youth. a kind mother teaches youth. the elder daughter teaches youth.",
  // "the strict chairman signed deals. that male executive signed deals. the strict chairwoman signed deals. that female executive signed deals. the smart chairman led teams. a top director led teams. the smart chairwoman led teams. a top director led teams.",
  // "the ancient god created life. this divine wizard created life. the ancient goddess created life. this divine witch created life. the powerful god cast spells. that cruel wizard cast spells. the powerful goddess cast spells. that cruel witch cast spells.",
  // "the heavy bull ate grass. that male rooster ate grass. the heavy cow ate grass. that female hen ate grass. the loud bull woke farmers. a fierce rooster woke farmers. the loud cow woke farmers. a fierce hen woke farmers.",
};

Malloc_Allocator mallocator = MALLOC_CREATE();

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
}

size_t count_parameters(Optimizer* optmizer) {
  size_t count = 0;

  for (size_t i = 0; i < optmizer->tensors.count; i++) {
    NMatrix value = optmizer->tensors.elems[i].value;
    count += value.rows*value.cols;
  }

  return count;
}

int main(void) {
  Optimizer optimizer = (Optimizer) {
    .tensors = ARRAY_CREATE(&mallocator.alloc),
    .history = ARRAY_CREATE(&mallocator.alloc),
    .second = ARRAY_CREATE(&mallocator.alloc),
    .learning_rate = 0.075f,
    .updates = 0,
  };
  Arena_Allocator arena = ARENA_CREATE(&mallocator.alloc, 2*1024*1024);
  TokenID_Array sequence = ARRAY_CREATE(&mallocator.alloc);
  TokenID_Array tokens = ARRAY_CREATE(&mallocator.alloc);
  TokenID_Array targets = ARRAY_CREATE(&mallocator.alloc);

  prepare_data(&sequence);

  size_t C = 16;
  size_t D = 8;
  size_t H = 4;
  size_t F = 8;
  size_t V = MAX_VOCAB;

  // optimize learning rate
  int patience = 3;
  float decay_factor = 0.9f;
  float min_lr = 0.00001f;
  float min_delta = 0.0005f;

  array_ensure(&tokens, C);
  array_ensure(&targets, C);

  Transformer trans_in = {0};

  init_transformer(
      .alloc = &arena.alloc,
      .trans = &trans_in,
      .context_size = C,
      .vocab_size = V,
      .emb_size = D,
      .ff_size = F,
  );

  register_tensor(&optimizer, trans_in.tok_emb);
  register_tensor(&optimizer, trans_in.pos_emb);
  register_tensor(&optimizer, trans_in.ln.gamma);
  register_tensor(&optimizer, trans_in.ln.beta);
  register_tensor(&optimizer, trans_in.H.weight);
  register_tensor(&optimizer, trans_in.H.bias);

  for (size_t i = 0; i < NUM_BLOCKS; i++) {
    register_tensor(&optimizer, trans_in.blocks[i].ln1.gamma);
    register_tensor(&optimizer, trans_in.blocks[i].ln1.beta);
    register_tensor(&optimizer, trans_in.blocks[i].ln2.gamma);
    register_tensor(&optimizer, trans_in.blocks[i].ln2.beta);

    register_tensor(&optimizer, trans_in.blocks[i].ff1.weight);
    register_tensor(&optimizer, trans_in.blocks[i].ff1.bias);
    register_tensor(&optimizer, trans_in.blocks[i].ff2.weight);
    register_tensor(&optimizer, trans_in.blocks[i].ff2.bias);

    register_tensor(&optimizer, trans_in.blocks[i].attn.K.weight);
    register_tensor(&optimizer, trans_in.blocks[i].attn.K.bias);
    register_tensor(&optimizer, trans_in.blocks[i].attn.Q.weight);
    register_tensor(&optimizer, trans_in.blocks[i].attn.Q.bias);
    register_tensor(&optimizer, trans_in.blocks[i].attn.V.weight);
    register_tensor(&optimizer, trans_in.blocks[i].attn.V.bias);
    register_tensor(&optimizer, trans_in.blocks[i].attn.O.weight);
    register_tensor(&optimizer, trans_in.blocks[i].attn.O.bias);
  }

  printf("dataset size = %zu\n", sequence.count);
  printf("total parameters = %zu\n", count_parameters(&optimizer));

  float best_epoch_loss = 1000.0f;
  int plateau_epochs = 0;

  for (size_t epoch = 0; epoch < 2000; epoch++) {
    float loss = 0;
    size_t count = 0;

    for (size_t i = 0; i < sequence.count - C; i++) {
      size_t saved = SAVE(&arena.alloc);

      tokens.count = 0;
      targets.count = 0;

      for (size_t c = 0; c < C; c++) {
        array_append(&tokens, sequence.elems[i+c]);
        array_append(&targets, sequence.elems[i+c+1]);
      }

      size_t N = tokens.count;

      Transformer_Output trans_out = {0};
      init_transformer_output(
          .arena = &arena,
          .trans_out = &trans_out,
          .vocab_size = trans_in.vocab_size,
          .emb_size = trans_in.emb_size,
          .ff_size = trans_in.ff_size,
          .sequence_size = N,
          .heads_count = H,
      );

      transformer_forward(tokens, &trans_out, &trans_in, 1.0f);

      loss += cross_entropy(trans_out.probs, targets);
      count += 1;

      transformer_backward(tokens, targets, &trans_out, &trans_in);

      RESTORE(&arena.alloc, saved);
    }

    loss /= count;

    // todo: increase epoch only when we don't plateaued?
    if (loss < (best_epoch_loss - min_delta)) {
      best_epoch_loss = loss;
      plateau_epochs = 0;
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
    }

    printf("\rloss(%zu) = %f %zu %f", epoch, loss, count, optimizer.learning_rate);
    fflush(stdout);
    update_grads_adam(&optimizer);
  }

  printf("dataset size = %zu\n", sequence.count);
  printf("total parameters = %zu\n", count_parameters(&optimizer));

  const char* prompt = "the desert";

  tokens.count = 0;
  tokenize(
      &tokens,
      prompt,
      strlen(prompt),
      vocabulary,
      vocabulary_by_size
  );

  if (tokens.count > C) tokens.count = C;

  for (size_t i = 0; i < tokens.count; i++)
      printf("%s", vocabulary[tokens.elems[i]].token);

  for (size_t i = 0; i < 500; i++) {
    size_t saved = SAVE(&arena.alloc);

    size_t N = tokens.count;
    Transformer_Output trans_out = {0};
    init_transformer_output(
        .arena = &arena,
        .trans_out = &trans_out,
        .vocab_size = trans_in.vocab_size,
        .emb_size = trans_in.emb_size,
        .ff_size = trans_in.ff_size,
        .sequence_size = N,
        .heads_count = H,
    );

    transformer_forward(tokens, &trans_out, &trans_in, 1.0f);

    int32_t next = mat_row_argmax(mat_row(trans_out.probs, N - 1));

    if (next == '\r') next = '\n';
    printf("%s", vocabulary[next].token);

    if (tokens.count >= C) {
      for (size_t c = 0; c < C - 1; c++)
        tokens.elems[c] = tokens.elems[c+1];
      tokens.count -= 1;
    }

    array_append(&tokens, next);

    RESTORE(&arena.alloc, saved);

    if (next == EOS_TOKEN)
      break;
  }

  return 0;
}

