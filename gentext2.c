#include "array.h"
#include "nn.h"
#include <_time.h>
#include <math.h>
#include <stdlib.h>
#define BYTEBUFFER_IMPLEMENTATION
#define ALLOCATOR_IMPLEMENATION

#include "bytebuffer.h"
#include "allocator.h"
#include "tokenizer.h"
#include "transformer.h"
#include "node.h"

#include "generated/alice.h"
#include "generated/vocab.h"

// ratio = tokens / parameters
//
// 20:1 to 100:1

const char *training_text[] = {
  (const char*)alice_txt,
  "the desert was silent except for the low, rhythmic hum of the wind sweeping across the dunes. for miles in every direction, nothing broke the horizon line but shifting sand and the occasional skeletal remains of ancient shrubs. evelyn checked her gps device, frowning at the uncoordinated coordinates flashing across the screen. the signal was dead. she had exactly two liters of water left, a compass that could not find true north, and six hours of daylight remaining before the temperature dropped below freezing.",
  "the desert is a landscape of surprising contrasts and quiet resilience. during the day, the sun beats down relentlessly, turning the sand into a glowing sea of gold. cacti and deep-rooted shrubs stand as silent sentinels, conserving every drop of precious moisture in their thick stems. yet, as twilight approaches, the extreme heat yields to a crisp, cooling breeze. the sky shifts into a canvas of violet and deep indigo. nocturnal creatures, such as the kit fox and the rattlesnake, emerge from their underground burrows to hunt and forage, breathing vibrant life into the quiet night.",
  "the desert was an oven of white heat that felt entirely unescapable. jackson tapped the cracked glass of his analog barometer, watching the needle fluctuate wildly against the glass. his satellite uplink to the desert research base had been dark for forty-eight hours, leaving him completely isolated. he rationed his final half-liter of water, scanning the shimmering horizon for any sign of shelter before the sub-zero desert night winds set in.",
  "the roman empire was one of the most powerful and enduring civilizations in human history, fundamentally shaping the trajectory of the western world. beginning as a modest republic on the italian peninsula, it expanded rapidly through strategic military conquests and advanced engineering. roman legions established dominance across the mediterranean, bringing law, architecture, and commerce to diverse cultures. at its peak, the empire spanned from the rainy hills of britannia to the arid deserts of egypt. even after its eventual collapse, the architectural marvels, legal systems, and cultural innovations of rome continued to influence modern societies for centuries.",
  "a sudden dust devil whirled across the gravel flats, peppering maya's goggles with sharp desert grit. she pulled up her scarf, squinting at the digital compass on her wrist watch, which kept looping through calibration cycles. the drone tracking her movements across this barren desert had lost its rotors to a sudden thermal draft miles ago. she had four hours of functional twilight left to navigate the maze of canyon vents before visibility dropped to zero.",
  "the steering column of the overland truck was burning to the touch in the midday sun. tom climbed down into the blinding white glare of the playa, his boots sinking into the fine, powdery desert silt. the main truck battery was dead, fried by the extreme desert temperature peak. armed with only a basic multi-tool, a single thermal blanket, and three miles of open sand ahead of him, he began his long trek.",
  "the sandstone arroyo offered a tiny sliver of shade, but the shifting sun was shrinking it by the minute. rachel checked her emergency radio, met only by the steady hiss of static from the surrounding desert mountain iron deposits. her digital mapping tablets had overheated and shut down an hour ago. she calculated her remaining energy against a steep, five-hundred-foot climb to get a line of sight across the desert before the air froze over.",
  "nothing but sun-bleached shale stretched between ben and the horizon line. he shook his heavy aluminum flask, disturbed by the light, hollow thud of the last few drops of water inside. the automated waypoint markers he followed had been completely washed away by a previous desert flash flood. with the sun dipping low and turning the desert hills a dangerous orange, he needed to find high ground to avoid the nocturnal predators.",
  "the abandoned mining outpost was a skeletal ruin collapsing into the sand. mark kicked the rusted iron door open, desperate to escape the relentless glare of the afternoon sun. he was entirely offline, his satellite phone battery ruined by the intense desert heat wave. he checked his supplies, noting he had exactly one liter of water left to survive his long trek across the open desert landscape.",
  "the desert plateau exists in a perpetual state of extreme thermal swings. by afternoon, the black basalt rocks absorb enough heat to scorch anything that touches them, forcing even the hardiest desert insects deep into volcanic fissures. but as the sun slips below the jagged peaks, the rock faces rapidly radiate their warmth away. the sudden chill triggers the emergence of nocturnal creatures, transforming the barren stone into a bustling hunting ground.",
  "adaptability is the ultimate currency for survival in the parched desert badlands. desert ironwood trees grow incredibly slowly, producing dense, heavy wood that resists both intense heat and parasitic rot. their deep taproots lock onto moisture hidden deep within underground aquifers. when nightfall brings a cool mist, the entire desert plant community shifts, opening microscopic pores to breathe in the damp air.",
  "the dry clay pans appear dead and heavily fractured under the blinding summer sky. this brittle crust, however, protects millions of dormant desert organisms and microscopic algae spores waiting for a rare rain event. when twilight cools the baked earth to a pleasant crispness, the landscape undergoes a subtle awakening as desert owls sweep low over the cracked ground, searching for movement.",
  "the oasis acts as a vibrant focal point within a vast, silent desert wilderness. tall fan palms cluster tightly around a hidden artesian spring, creating a microclimate where delicate ferns can survive the surrounding glare. at dusk, the division between the harsh desert and this green sanctuary blurs as kit foxes, bats, and migrating birds descend upon the water, filling the night with a chorus of calls.",
  "artificial intelligence has rapidly transformed from a theoretical concept into an everyday reality. machine learning algorithms now power everything from basic email filters to complex autonomous vehicles. by processing massive amounts of historical data, these systems can identify hidden patterns, make accurate predictions, and automate tedious tasks. however, this technological leap brings significant ethical challenges, including data privacy concerns and algorithmic bias. as these neural networks become increasingly sophisticated, developers face the crucial responsibility of ensuring transparency and fairness, so that these powerful digital tools ultimately benefit society as a whole.",
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

int32_t sample(NMatrix probs) {
  float r = rand_uniform();
  float cumulative = 0.0f;

  for (int i = 0; i < MAX_VOCAB; i++) {
    cumulative += VEC_AT(probs, i);
    if (r <= cumulative)
      return i;
  }

  return MAX_VOCAB - 1;
}

typedef struct {
  float prob;
  size_t index;
} Entry;

DEFINE_ARRAY(Entry);

int compare_entry(const void* a, const void* b) {
  const Entry* entry_a = a;
  const Entry* entry_b = b;

  if (entry_a->prob < entry_b->prob) return +1;
  if (entry_a->prob > entry_b->prob) return -1;
  return 0;
}

int32_t sample_topp(Arena_Allocator* arena, NMatrix probs, float topp) {
  Entry_Array entries = ARRAY_CREATE(&arena->alloc);
  array_ensure(&entries, MAX_VOCAB);

  for (size_t i = 0; i < MAX_VOCAB; i++) {
    Entry entry = {
      .prob = VEC_AT(probs, i),
      .index = i,
    };

    array_append(&entries, entry);
  }

  qsort(entries.elems, entries.count, sizeof(Entry), compare_entry);

  Entry_Array nucleus = ARRAY_CREATE(&arena->alloc);
  array_ensure(&nucleus, MAX_VOCAB);

  float cummulative = 0;
  for (size_t i = 0; i < MAX_VOCAB; i++) {
    Entry entry = entries.elems[i];
    array_append(&nucleus, entry);
    cummulative += entry.prob;
    if (cummulative >= topp)
      break;
  }

  float r = rand_uniform() * cummulative;
  int32_t sampled = nucleus.elems[0].index;

  cummulative = 0;
  for (size_t i = 0; i < nucleus.count; i++) {
    Entry entry = nucleus.elems[i];
    cummulative += entry.prob;
    if (r <= cummulative) {
      return entry.index;
    }
  }

  return sampled;
}

void save_model(Transformer* trans_in) {
  FILE* fp = fopen("models/gentext2.bin", "wb");
  mat_write(trans_in->tok_emb.value, fp);
  mat_write(trans_in->ln.gamma.value, fp);
  mat_write(trans_in->ln.beta.value, fp);
  mat_write(trans_in->H.weight.value, fp);
  mat_write(trans_in->H.bias.value, fp);

  fwrite(&trans_in->num_blocks, sizeof(size_t), 1, fp);

  for (size_t i = 0; i < trans_in->num_blocks; i++) {
    mat_write(trans_in->blocks[i].ln1.gamma.value, fp);
    mat_write(trans_in->blocks[i].ln1.beta.value, fp);
    mat_write(trans_in->blocks[i].ln2.gamma.value, fp);
    mat_write(trans_in->blocks[i].ln2.beta.value, fp);
    mat_write(trans_in->blocks[i].ff1.weight.value, fp);
    mat_write(trans_in->blocks[i].ff1.bias.value, fp);
    mat_write(trans_in->blocks[i].ff2.weight.value, fp);
    mat_write(trans_in->blocks[i].ff2.bias.value, fp);
    mat_write(trans_in->blocks[i].attn.K.weight.value, fp);
    mat_write(trans_in->blocks[i].attn.K.bias.value, fp);
    mat_write(trans_in->blocks[i].attn.Q.weight.value, fp);
    mat_write(trans_in->blocks[i].attn.Q.bias.value, fp);
    mat_write(trans_in->blocks[i].attn.V.weight.value, fp);
    mat_write(trans_in->blocks[i].attn.V.bias.value, fp);
    mat_write(trans_in->blocks[i].attn.O.weight.value, fp);
    mat_write(trans_in->blocks[i].attn.O.bias.value, fp);
  }

  fclose(fp);
}

bool load_model(Transformer* trans_in) {
  FILE* fp = fopen("models/gentext2.bin", "rb");
  if (fp == NULL) return false;

  mat_read(trans_in->tok_emb.value, fp);
  mat_read(trans_in->ln.gamma.value, fp);
  mat_read(trans_in->ln.beta.value, fp);
  mat_read(trans_in->H.weight.value, fp);
  mat_read(trans_in->H.bias.value, fp);

  fread(&trans_in->num_blocks, sizeof(size_t), 1, fp);

  for (size_t i = 0; i < trans_in->num_blocks; i++) {
    mat_read(trans_in->blocks[i].ln1.gamma.value, fp);
    mat_read(trans_in->blocks[i].ln1.beta.value, fp);
    mat_read(trans_in->blocks[i].ln2.gamma.value, fp);
    mat_read(trans_in->blocks[i].ln2.beta.value, fp);
    mat_read(trans_in->blocks[i].ff1.weight.value, fp);
    mat_read(trans_in->blocks[i].ff1.bias.value, fp);
    mat_read(trans_in->blocks[i].ff2.weight.value, fp);
    mat_read(trans_in->blocks[i].ff2.bias.value, fp);
    mat_read(trans_in->blocks[i].attn.K.weight.value, fp);
    mat_read(trans_in->blocks[i].attn.K.bias.value, fp);
    mat_read(trans_in->blocks[i].attn.Q.weight.value, fp);
    mat_read(trans_in->blocks[i].attn.Q.bias.value, fp);
    mat_read(trans_in->blocks[i].attn.V.weight.value, fp);
    mat_read(trans_in->blocks[i].attn.V.bias.value, fp);
    mat_read(trans_in->blocks[i].attn.O.weight.value, fp);
    mat_read(trans_in->blocks[i].attn.O.bias.value, fp);
  }

  fclose(fp);

  return true;
}

void set_dropout(Transformer* trans_in, float p) {
  trans_in->x0_p = p;
  for (size_t i = 0; i < trans_in->num_blocks; i++) {
    trans_in->blocks[i].attn.p = p;
  }
}

void generate_text(
  Arena_Allocator* arena,
  Transformer* trans_in,
  const char* prompt,
  TokenID_Array* tokens
) {
  size_t C = trans_in->context_size;

  tokens->count = 0;
  tokenize(
      tokens,
      prompt,
      strlen(prompt),
      vocabulary,
      vocabulary_by_size
  );

  size_t lines_count = 0;

  if (tokens->count > C) tokens->count = C;

  const char* separator = "\u22c5";

  for (size_t i = 0; i < tokens->count; i++)
      printf("%s%s", vocabulary[tokens->elems[i]].token, separator);

  for (size_t i = 0; i < 500; i++) {
    size_t saved = SAVE(&arena->alloc);

    size_t N = tokens->count;
    Transformer_Output trans_out = {0};
    init_transformer_output(
        .arena = arena,
        .trans_out = &trans_out,
        .num_blocks = trans_in->num_blocks,
        .vocab_size = trans_in->vocab_size,
        .heads_count = trans_in->heads_count,
        .emb_size = trans_in->emb_size,
        .ff_size = trans_in->ff_size,
        .sequence_size = N,
    );

    // 0.5 = deterministic
    // 1.0 = normal
    // 1.5 = creative
    // 3.0 = nonsensical
    transformer_forward(*tokens, &trans_out, trans_in, 1.5f, 0);

    // int32_t next = mat_row_argmax(mat_row(trans_out.probs, N - 1));
    // int32_t next = sample(mat_row(trans_out.probs, N - 1));
    int32_t next = sample_topp(arena, mat_row(trans_out.probs, N - 1), 0.9f);

    if (next == '\r') next = '\n';

    printf("%s%s", vocabulary[next].token, separator);

    if (tokens->count >= C) {
      for (size_t c = 0; c < C - 1; c++)
        tokens->elems[c] = tokens->elems[c+1];
      tokens->count -= 1;
    }

    array_append(tokens, next);

    RESTORE(&arena->alloc, saved);

    if (next == EOS_TOKEN)
      break;

    if (next == '\n')
      lines_count += 1;

    if (lines_count >= 5)
      break;
  }

  printf("\n");
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

void print_timestamp(void) {
  time_t raw_time;
  struct tm tm;
  char buffer[128];

  time(&raw_time);
  localtime_r(&raw_time, &tm);
  strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", &tm);

  printf("%s", buffer);
}

int find_token(const char* token) {
  for (int i = 0; i < MAX_VOCAB; i++) {
    if (strcmp(token, vocabulary[i].token) == 0)
      return i;
  }
  return -1;
}

float cosine_learning_rate(size_t t, size_t t_warmup, size_t t_total, float lr_max) {
  if (t <= t_warmup) {
    return lr_max * (float)t / (float)t_warmup;
  }

  float lr_min = 0.01f * lr_max;

  float alpha = (float)(t  - t_warmup) / (float)(t_total - t_warmup);
  return lr_min + 0.5f*(lr_max - lr_min) * (1.0f + cosf(alpha * M_PI));
}

void print_tokens(Transformer* trans_in) {
  (void)trans_in;
  // printf("\033[38;5;130m");
  // printf("desert  = %.5f\n",
  //   mat_row_similarity(
  //     mat_row(trans_in->tok_emb.value, find_token("desert")),
  //     mat_row(trans_in->tok_emb.value, find_token(" desert"))
  //   )
  // );
  // printf("\033[0m");
}

void hsl_to_rgb(float h, float s, float l, int *r, int *g, int *b) {
  // Calculate Chroma
  float c = (1.0f - fabsf(2.0f * l - 1.0f)) * s;

  // Calculate intermediate value X
  // fmodf(h / 60.0f, 2.0f) handles the modulo behavior for floating points
  float x = c * (1.0f - fabsf(fmodf(h / 60.0f, 2.0f) - 1.0f));

  // Match value m
  float m = l - c / 2.0f;

  float rp = 0, gp = 0, bp = 0;

  if (h >= 0.0f && h < 60.0f) {
    rp = c;
    gp = x;
  } else if (h >= 60.0f && h < 120.0f) {
    rp = x;
    gp = c;
  } else if (h >= 120.0f && h < 180.0f) {
    gp = c;
    bp = x;
  } else if (h >= 180.0f && h < 240.0f) {
    gp = x;
    bp = c;
  } else if (h >= 240.0f && h < 300.0f) {
    rp = x;
    bp = c;
  } else if (h >= 300.0f && h <= 360.0f) {
    rp = c;
    bp = x;
  }

  // Scale values to standard 8-bit integers [0, 255]
  *r = (int)roundf((rp + m) * 255.0f);
  *g = (int)roundf((gp + m) * 255.0f);
  *b = (int)roundf((bp + m) * 255.0f);
}

int main(int argc, char* argv[]) {
  srand(getpid());

  Optimizer optimizer = (Optimizer) {
    .tensors = ARRAY_CREATE(&mallocator.alloc),
    .history = ARRAY_CREATE(&mallocator.alloc),
    .second = ARRAY_CREATE(&mallocator.alloc),
    .learning_rate = 0.05f,
    .decay_factor = 0.00001f,
    .updates = 0,
  };
  size_t arena_size = MAX_VOCAB*MAX_VOCAB*sizeof(float);
  Arena_Allocator arena = ARENA_CREATE(&mallocator.alloc, arena_size + 1024*1024*1024);
  TokenID_Array sequence = ARRAY_CREATE(&mallocator.alloc);
  TokenID_Array tokens = ARRAY_CREATE(&mallocator.alloc);
  TokenID_Array targets = ARRAY_CREATE(&mallocator.alloc);
  Index32_Array start_indices = ARRAY_CREATE(&mallocator.alloc);

  prepare_data(&sequence);

  size_t C = 16;
  size_t D = 16;
  size_t H = 4; //headDim = D / H;
  size_t F = D*4;
  size_t V = MAX_VOCAB;

  float peak_lr = 0.005f;
  size_t total_epochs = 200;
  size_t warmup_epochs = 10;
  size_t batch_size = 128;
  float dropout_pct = 0.25;
  size_t num_blocks = 20;

  // optimize learning rate
  // int patience = 100;
  // float decay_factor = 0.9f;
  // float min_lr = 0.0001f;
  // float min_delta = 0.0005f;
  // float best_epoch_loss = INFINITY;
  // int plateau_epochs = 0;

  array_ensure(&tokens, C);
  array_ensure(&targets, C);

  Transformer trans_in = {0};

  init_transformer(
      .alloc = &arena.alloc,
      .trans = &trans_in,
      .num_blocks = num_blocks,
      .context_size = C,
      .heads_count = H,
      .vocab_size = V,
      .emb_size = D,
      .ff_size = F,
  );

  register_tensor(&optimizer, trans_in.tok_emb);
  register_tensor(&optimizer, trans_in.ln.gamma);
  register_tensor(&optimizer, trans_in.ln.beta);
  register_tensor(&optimizer, trans_in.H.weight);
  register_tensor(&optimizer, trans_in.H.bias);

  for (size_t i = 0; i < num_blocks; i++) {
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

  size_t params = count_parameters(&optimizer);

  size_t epoch = 0;
  bool loaded = load_model(&trans_in);
  load_optimizer(&optimizer, "models/gentext2.adam", &epoch);

  bool train = false;

  for (int i = 1; i < argc; i++) {
    if (strncmp(argv[i], "--train", 7) == 0)
      train = true;

    if (strncmp(argv[i], "--epoch", 7) == 0) {
      sscanf(argv[i], "--epoch=%zu", &epoch);
    }
  }

  if (loaded && !train) goto generate_text;

  printf("dataset size = %zu\n", sequence.count);
  printf("total parameters = %zu\n", params);
  printf("ratio = %f\n", sequence.count / (float)params);

  init_opencl();
  ensure_buffer_size(sizeof(float)*MAX_VOCAB*MAX_VOCAB);

  mat_zero(mat_row(trans_in.tok_emb.value, 0));
  print_tokens(&trans_in);

  printf("\033[38;5;28m");
  print_timestamp();
  printf(" Start training\n");
  printf("\033[0m");

  size_t batches_per_epoch = sequence.count / batch_size;

  size_t warmup_steps = warmup_epochs*batches_per_epoch;
  size_t total_steps = total_epochs*batches_per_epoch;

  size_t global_step = epoch * batches_per_epoch;
  optimizer.learning_rate = cosine_learning_rate(global_step, warmup_steps, total_steps, peak_lr);

  while (epoch < total_epochs) {
    float loss = 0;
    size_t count = 0;

    size_t num_starts = sequence.count - C - 1;

    start_indices.count = 0;
    for (size_t i = 0; i < num_starts; i++)
      array_append(&start_indices, i);
    shuffle_indices(start_indices);

    set_dropout(&trans_in, dropout_pct);
    mat_zero(mat_row(trans_in.tok_emb.value, 0));

    for (size_t start_i = 0; start_i < num_starts; start_i += batch_size) {
      size_t current_batch_size = num_starts - start_i < batch_size ? num_starts - start_i : batch_size;

      float batch_loss = 0;
      size_t batch_count = 0;

      // start batch
      for (size_t sample = 0; sample < current_batch_size; sample++) {
        size_t saved = SAVE(&arena.alloc);

        size_t start_index = start_indices.elems[start_i + sample];

        tokens.count = 0;
        targets.count = 0;

        for (size_t c = 0; c < C; c++) {
          array_append(&tokens, sequence.elems[start_index+c]);
          array_append(&targets, sequence.elems[start_index+c+1]);
        }

        size_t N = tokens.count;

        Transformer_Output trans_out = {0};
        init_transformer_output(
            .arena = &arena,
            .trans_out = &trans_out,
            .num_blocks = trans_in.num_blocks,
            .vocab_size = trans_in.vocab_size,
            .heads_count = trans_in.heads_count,
            .emb_size = trans_in.emb_size,
            .ff_size = trans_in.ff_size,
            .sequence_size = N,
        );

        transformer_forward(tokens, &trans_out, &trans_in, 1.0f, 0);

        float c_loss = cross_entropy(trans_out.probs, targets);
        batch_loss += c_loss;
        batch_count += 1;

        transformer_backward(tokens, targets, &trans_out, &trans_in, 0);

        RESTORE(&arena.alloc, saved);

        float max_loss = logf(MAX_VOCAB);
        float alpha = c_loss / max_loss;
        float red = 0.0f;
        float green = 120.0f;

        if (alpha < 0) alpha = 0;
        if (alpha > 1) alpha = 1;

        int r, g, b;
        hsl_to_rgb(red*alpha + (1.0f - alpha)*green, 1.0f, 0.5f, &r, &g, &b);

        const char* block[] = {
          "\u25AE", // vertical rectangle
          "\u2588", // full block
          "\u258C", // left half block
        };
        printf("\033[38;2;%d;%d;%dm%s\033[38;0m", r, g, b, block[1]);

        // int index = (int)(ptc * 10);
        // int colors[] = {196, 202, 208, 214, 220, 226, 190, 154, 118, 82, 46};
        // printf("\033[38;5;%dm%s\033[38;0m", colors[index], block[1]);
      }

      batch_loss = batch_loss / batch_count;

      if (current_batch_size < batch_size) {
        for (size_t i = current_batch_size; i < batch_size; i++)
          printf(" ");
      }

      printf(" = %.6f %.2f\r", optimizer.learning_rate, expf(batch_loss));
      // end batch

      mat_zero(mat_row(trans_in.tok_emb.grad, 0));
      update_grads_adam(&optimizer);

      // cosine annealing
      optimizer.learning_rate = cosine_learning_rate(global_step, warmup_steps, total_steps, peak_lr);

      global_step += 1;

      loss += batch_loss;
      count += 1;
    }
    printf("\n");

    {
      print_tokens(&trans_in);

      size_t saved = SAVE(&arena.alloc);
      set_dropout(&trans_in, 0.0f);
      generate_text(&arena, &trans_in, "the desert", &tokens);
      RESTORE(&arena.alloc, saved);
    }

    loss /= count;

    printf("\033[38;5;28m");
    print_timestamp();
    printf(" loss(%zu,%zu) = %f learning_rate = %f perplexity = %f\n",
        epoch, global_step, loss, optimizer.learning_rate, expf(loss));
    printf("\033[0m");

    save_model(&trans_in);
    save_optimizer(&optimizer, "models/gentext2.adam", epoch+1);

    epoch += 1;
  }

  save_model(&trans_in);
  save_optimizer(&optimizer, "models/gentext2.adam", epoch);

generate_text:
  printf("dataset size = %zu\n", sequence.count);
  printf("total parameters = %zu\n", params);
  printf("ratio = %f\n", sequence.count / (float)params);

  generate_text(&arena, &trans_in, "the desert", &tokens);

  return 0;
}

