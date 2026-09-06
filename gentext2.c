#include "array.h"
#include "nn.h"
#include <_time.h>
#include <math.h>
#include <stdio.h>
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
  // (const char*)alice_txt,
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

void set_color(int r, int g, int b) {
  printf("\033[38;2;%d;%d;%dm", r, g, b);
}

void rst_color(void) {
  printf("\033[38;0m");
}

// #define MIN(a, b) ((a) < (b) ? (a) : (b))
// #define MAX(a, b) ((a) > (b) ? (a) : (b))

Malloc_Allocator mallocator = MALLOC_CREATE();

void prepare_data(TokenID_Array* sequence) {
//Non-overlapping chunks, shuffled. That's what real pretraining does, and for 1M tokens it's the right call.
//
// The recipe
//
// Read C+1 tokens per chunk, advance by C:
//
// chunk k covers  ids[k*C  ..  k*C + C]      (C+1 tokens)
//   input        ids[k*C   .. k*C + C - 1]   (C tokens)
//   targets      ids[k*C+1 .. k*C + C]       (C tokens)
//
// The +1 overlap is what makes coverage exact: the last target of chunk k is ids[(k+1)*C], which is the first input of chunk k+1. Every token is a target exactly once and an input exactly once. No gaps, no duplication.
//
// With C = 512:
//
// chunks       = (1,000,000 - 1) / 512  =  1953
// tokens/epoch = 1953 × 512             =  999,936    (64 left over, drop them)
//
// One epoch = 1953 forward/backward passes, or 244 optimizer steps at batch 8.
//
// Shuffle the chunk order
//
// size_t n_chunks = (n_tokens - 1) / C;
// size_t *order = malloc(n_chunks * sizeof(size_t));
// for (size_t i = 0; i < n_chunks; i++) { order[i] = i; }
//
// for (int epoch = 0; epoch < epochs; epoch++) {
//     shuffle(order, n_chunks);
//     for (size_t i = 0; i < n_chunks; i++) {
//         const int32_t *chunk = &ids[order[i] * C];   /* reads C+1 */
//         /* forward on chunk[0..C-1], targets chunk[1..C] */
//     }
// }
//
// This matters. Consecutive chunks come from the same chapter — same topic, same vocabulary — so feeding them in reading order gives you long runs of correlated gradients. Shuffling decorrelates the batch. Same order-array pattern you already have.
//
// The one real downside, and a free fix
//
// Position 0 of each chunk has zero left context, position 1 has one token, and so on. Fixed boundaries mean the same tokens are handicapped every epoch.
//
// Fix: randomize the starting offset each epoch.
//
// size_t off = rand_below(C);              /* different every epoch */
// const int32_t *chunk = &ids[off + order[i] * C];
//
// Now epoch 1 cuts at 0, 512, 1024…; epoch 2 cuts at 137, 649, 1161…. Every token sees a different amount of context across epochs, and it costs one addition. You get most of the benefit overlapping strides were supposed to buy, for free.
//
// Why not stride 1
//
// From the last message: stride 1 is 512× the compute for the same 1M unique tokens. The only thing it buys is that each token is predicted at every possible context length instead of one — and the offset trick above approximates that across epochs at 1/512th the cost.
//
// If you want a middle ground, stride = C/2 doubles compute and guarantees every token at least C/2 context. That's the most anyone reasonably does.

  for (size_t i = 0; i < sizeof(training_text)/sizeof(char*); i++) {
    const char* text = training_text[i];

    tokenize(
        sequence,
        text,
        strlen(text),
        vocabulary,
        vocabulary_by_size,
        MAX_VOCAB
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
  // size_t C = trans_in->context_size;

  tokens->count = 0;
  tokenize(
      tokens,
      prompt,
      strlen(prompt),
      vocabulary,
      vocabulary_by_size,
      MAX_VOCAB
  );

  size_t lines_count = 0;

  //const char* separator = "\u22c5";
  const char* separator = "";

  for (size_t i = 0; i < tokens->count; i++)
      printf("%s%s", vocabulary[tokens->elems[i]].token, separator);

  for (size_t i = 0; i < 256; i++) {
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
    transformer_forward(*tokens, &trans_out, trans_in, 1.0f, 0);

    NMatrix last_token = mat_row(trans_out.probs, N - 1);

    int32_t next = mat_row_argmax(last_token);
    // int32_t next = sample(last_token);
    // int32_t next = sample_topp(arena, last_token, 0.9f);

    if (next == '\r') next = '\n';

    float red = 0.0f;
    float green = 120.0f;
    float alpha = VEC_AT(last_token, next);

    int r, g, b;
    hsl_to_rgb(green*alpha + (1.0f - alpha)*red, 1.0f, 0.5f, &r, &g, &b);
    set_color(r, g, b);
    printf("%s%s", vocabulary[next].token, separator);
    fflush(stdout);

    // if (tokens->count >= C) {
    //   for (size_t c = 0; c < C - 1; c++)
    //     tokens->elems[c] = tokens->elems[c+1];
    //   tokens->count -= 1;
    // }

    array_append(tokens, next);

    RESTORE(&arena->alloc, saved);

    if (next == EOS_TOKEN)
      break;

    if (next == '\n')
      lines_count += 1;

    if (lines_count >= 5)
      break;
  }

  rst_color();
  printf("\n");
}

DEFINE_ARRAY_ALIAS(Index32, size_t);

// Helper function to shuffle an array of indices (Fisher-Yates)
void shuffle_indices(Index32_Array indices) {
  for (size_t i = indices.count - 1; i > 0; i--) {
    size_t j = rand_between(0, i);
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

  if (t >= t_total) {
    return lr_min;
  }

  float alpha = (float)(t - t_warmup) / (float)(t_total - t_warmup);
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

typedef struct {
  int32_t offset;
  int32_t prompt_len;
  int32_t total_len;
} Rec;
_Static_assert(sizeof(Rec) == 12, "Rec must be 3 packed int32");

typedef struct {
  int32_t *ids;
  size_t n_ids;
  Rec *recs;
  size_t n_recs;
} Corpus;

void *slurp(const char *path, size_t *nbytes) {
  FILE *f = fopen(path, "rb");
  if (!f) { perror(path); exit(1); }
  fseek(f, 0, SEEK_END);
  long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  void *p = malloc((size_t)n);
  if (fread(p, 1, (size_t)n, f) != (size_t)n) { perror(path); exit(1); }
  fclose(f);
  *nbytes = (size_t)n;
  return p;
}

Corpus corpus_load(const char *ids_path, const char *index_path) {
  Corpus c;
  size_t nb;
  c.ids   = slurp(ids_path, &nb);
  c.n_ids = nb / sizeof(int32_t);
  c.recs   = slurp(index_path, &nb);
  c.n_recs = nb / sizeof(Rec);
  return c;
}

const int32_t *example(const Corpus *c, size_t i, int *prompt_len, int *total_len) {
  Rec r = c->recs[i];
  *prompt_len = r.prompt_len;
  *total_len  = r.total_len;
  return c->ids + r.offset;
}

int main(int argc, char* argv[]) {
  srand(getpid());

  Optimizer optimizer = (Optimizer) {
    .tensors = ARRAY_CREATE(&mallocator.alloc),
    .history = ARRAY_CREATE(&mallocator.alloc),
    .second = ARRAY_CREATE(&mallocator.alloc),
    .decay = ARRAY_CREATE(&mallocator.alloc),
    .learning_rate = 0.05f,
    .decay_factor = 0.00001f,
    .updates = 0,
  };
  size_t arena_size = MAX_VOCAB*MAX_VOCAB*sizeof(float);
  Arena_Allocator arena = ARENA_CREATE(&mallocator.alloc, arena_size + 1024*1024*1024);
  // TokenID_Array sequence = ARRAY_CREATE(&mallocator.alloc);
  TokenID_Array tokens = ARRAY_CREATE(&mallocator.alloc);
  TokenID_Array targets = ARRAY_CREATE(&mallocator.alloc);
  Index32_Array start_indices = ARRAY_CREATE(&mallocator.alloc);

  // prepare_data(&sequence);

  Corpus corpus = corpus_load("distill/cdata/train.ids.bin", "distill/cdata/train.index.bin");

  int32_t sum_tokens = 0;
  int32_t max_len = 0;
  for (size_t i = 0; i < corpus.n_recs; ++i) {
      max_len = MAX(max_len, corpus.recs[i].total_len);
      sum_tokens += corpus.recs[i].total_len;
  }

  // for (size_t i = 0; i < corpus.n_recs; i++) {
  //   int prompt_len = 0;
  //   int total_len = 0;
  //   const int32_t *ids = example(&corpus, i, &prompt_len, &total_len);
  //
  //   printf("prompt_len = %d | total_len = %d\n", prompt_len, total_len);
  //
  //   //const char* sep = "\u22c5";
  //   const char* sep = "";
  //
  //   for (int j = 0; j < total_len; j++) {
  //     if (j == prompt_len) printf(" -> ");
  //     printf("%s%s", vocabulary[ids[j]].token, sep);
  //   }
  //
  //   printf("\n");
  // }

  size_t D = 64;
  size_t H = 4; //headDim = D / H;
  size_t F = D*4;
  size_t V = MAX_VOCAB;
  size_t num_blocks = 4;

  float peak_lr = 0.02f;
  size_t batch_size = 8;
  size_t steps_per_epoch = corpus.n_recs / batch_size;
  size_t total_epochs = 4;
  size_t warmup_epochs = 1;
  float dropout_pct = 0.0;

  // optimize learning rate
  // int patience = 100;
  // float decay_factor = 0.9f;
  // float min_lr = 0.0001f;
  // float min_delta = 0.0005f;
  // float best_epoch_loss = INFINITY;
  // int plateau_epochs = 0;

  // array_ensure(&tokens, C);
  // array_ensure(&targets, C);

  Transformer trans_in = {0};

  init_transformer(
      .alloc = &arena.alloc,
      .trans = &trans_in,
      .num_blocks = num_blocks,
      .heads_count = H,
      .vocab_size = V,
      .emb_size = D,
      .ff_size = F,
  );

  set_color(255, 165, 0);
  {
    size_t params = count_parameters(&optimizer);
    register_tensor(&optimizer, trans_in.tok_emb, true);
    printf("token emdeddings = %zu\n", count_parameters(&optimizer) - params);
  }
  {
    size_t params1 = count_parameters(&optimizer);
    register_tensor(&optimizer, trans_in.H.weight, true);
    printf("output head = %zu\n", count_parameters(&optimizer) - params1);
    size_t params2 = count_parameters(&optimizer);
    register_tensor(&optimizer, trans_in.H.bias, false);
    printf("output bias = %zu\n", count_parameters(&optimizer) - params2);
  }

  for (size_t i = 0; i < num_blocks; i++) {
    size_t params = count_parameters(&optimizer);
    register_tensor(&optimizer, trans_in.blocks[i].ln1.gamma, false);
    register_tensor(&optimizer, trans_in.blocks[i].ln1.beta, false);
    register_tensor(&optimizer, trans_in.blocks[i].ln2.gamma, false);
    register_tensor(&optimizer, trans_in.blocks[i].ln2.beta, false);

    register_tensor(&optimizer, trans_in.blocks[i].ff1.weight, true);
    register_tensor(&optimizer, trans_in.blocks[i].ff1.bias, false);
    register_tensor(&optimizer, trans_in.blocks[i].ff2.weight, true);
    register_tensor(&optimizer, trans_in.blocks[i].ff2.bias, false);

    register_tensor(&optimizer, trans_in.blocks[i].attn.K.weight, true);
    register_tensor(&optimizer, trans_in.blocks[i].attn.K.bias, false);
    register_tensor(&optimizer, trans_in.blocks[i].attn.Q.weight, true);
    register_tensor(&optimizer, trans_in.blocks[i].attn.Q.bias, false);
    register_tensor(&optimizer, trans_in.blocks[i].attn.V.weight, true);
    register_tensor(&optimizer, trans_in.blocks[i].attn.V.bias, false);
    register_tensor(&optimizer, trans_in.blocks[i].attn.O.weight, true);
    register_tensor(&optimizer, trans_in.blocks[i].attn.O.bias, false);
    printf("transformer block = %zu\n", count_parameters(&optimizer) - params);
  }

  {
    size_t params0 = count_parameters(&optimizer);
    register_tensor(&optimizer, trans_in.ln.gamma, false);
    register_tensor(&optimizer, trans_in.ln.beta, false);
    printf("final layerNorm = %zu\n", count_parameters(&optimizer) - params0);
  }

  size_t params = count_parameters(&optimizer);
  printf("corpus size = %zu\n", corpus.n_recs);
  printf("tokens count = %d\n", sum_tokens);
  printf("total parameters = %zu\n", params);
  printf("chinchilla target = %zu tokens\n", 20*params);
  printf("epochs to get there = %zu\n", (20*params + sum_tokens - 1) / sum_tokens);
  printf("steps per epoch = %zu\n", steps_per_epoch);
  printf("batch size = %zu\n", batch_size);
  printf("ratio = %f tokens per parameter\n", sum_tokens / (float)params);

  rst_color();

  // Muennighoff et al. (2023), Scaling Data-Constrained Language Models, is the empirical answer:
  //
  //  - Up to ~4 epochs: repeated tokens are worth nearly as much as fresh ones
  //  - 4 to ~16 epochs: returns decay steadily
  //  - Past ~16: essentially zero value
  size_t epoch = 0;
  size_t optimizer_steps = 0;

  bool loaded = load_model(&trans_in);
  load_optimizer(&optimizer, "models/gentext2.adam", &epoch, &optimizer_steps);

  char* prompt = "User: Write a horror story.";
  bool train = false;

  for (int i = 1; i < argc; i++) {
    if (strncmp(argv[i], "--train", 7) == 0)
      train = true;

    if (strncmp(argv[i], "--epoch", 7) == 0) {
      sscanf(argv[i], "--epoch=%zu", &epoch);
    }

    if (strncmp(argv[i], "--prompt", 8) == 0) {
      prompt = argv[i] + 9;
      goto generate_text;
    }
  }

  if (loaded && !train) goto generate_text;

  init_opencl();
  ensure_buffer_size(sizeof(float)*MAX_VOCAB*MAX_VOCAB);

  mat_zero(mat_row(trans_in.tok_emb.value, 0));
  print_tokens(&trans_in);

  set_color(0, 255, 0);
  print_timestamp();
  printf(" Start training\n");
  rst_color();

  optimizer.learning_rate = cosine_learning_rate(
      epoch,
      warmup_epochs*steps_per_epoch,
      total_epochs*steps_per_epoch,
      peak_lr
  );

  for (size_t i = 0; i < corpus.n_recs; i++)
    array_append(&start_indices, i);
  shuffle_indices(start_indices);

  size_t cursor = 0;

  {
    size_t saved = SAVE(&arena.alloc);
    set_dropout(&trans_in, 0.0f);
    generate_text(&arena, &trans_in, prompt, &tokens);
    RESTORE(&arena.alloc, saved);
  }

  while (epoch < total_epochs) {
    float loss_sum = 0;
    size_t token_count = 0;

    set_dropout(&trans_in, dropout_pct);
    mat_zero(mat_row(trans_in.tok_emb.value, 0));

    for (size_t sample = 0; sample < batch_size; sample++) {
      size_t saved = SAVE(&arena.alloc);

      if (cursor == start_indices.count) {
        // todo: increment epoch here
        shuffle_indices(start_indices);
        cursor = 0;

        epoch += 1;

        save_model(&trans_in);
        save_optimizer(&optimizer, "models/gentext2.adam", epoch, optimizer_steps);

        size_t saved = SAVE(&arena.alloc);
        set_dropout(&trans_in, 0.0f);
        generate_text(&arena, &trans_in, prompt, &tokens);
        RESTORE(&arena.alloc, saved);
      }

      size_t start_index = start_indices.elems[cursor];
      cursor += 1;

      int prompt_len = 0;
      int total_len = 0;
      const int32_t *ids = example(&corpus, start_index, &prompt_len, &total_len);

      const int first = prompt_len - 1;             /* first loss position */
      const int npos  = total_len - prompt_len;     /* how many */

      if (npos <= 0)
        continue;

      tokens.count = 0;
      targets.count = 0;

      for (int c = 0; c < total_len - 1; c++) {
        array_append(&tokens, ids[c]);
        array_append(&targets, ids[c+1]);
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

      // dLoss/dlogits = softmax(logits) - onehot(target).
      NMatrix dlogits = trans_out.logits.grad;
      mat_copy(dlogits, trans_out.probs);

      for (size_t i = 0; i < N; i++) {
        int32_t target = targets.elems[i];

        if (i >= (size_t)first) {
          MAT_AT(dlogits, i, target) -= 1.0f;
          loss_sum -= logf(MAT_AT(trans_out.probs, i, target) + 1e-10f);
          token_count += 1;
        } else {
          mat_zero(mat_row(dlogits, i));
        }
      }

      transformer_backward(tokens, &trans_out, &trans_in);

      RESTORE(&arena.alloc, saved);
    }

    mat_zero(mat_row(trans_in.tok_emb.grad, 0));
    update_grads_adam(&optimizer, 1.0f / (float)token_count);

    // cosine annealing
    optimizer.learning_rate = cosine_learning_rate(
        optimizer_steps,
        warmup_epochs*steps_per_epoch,
        total_epochs*steps_per_epoch,
        peak_lr
    );

    float loss = loss_sum / token_count;

    set_color(0, 255, 0);
    print_timestamp();
    printf(" epoch=%zu/%zu opt=%zu/%zu lr=%f loss=%f perp=%f\n",
        epoch+1, total_epochs, optimizer_steps, total_epochs*steps_per_epoch,
        optimizer.learning_rate, loss, expf(loss));
    rst_color();

    optimizer_steps += 1;

    if (optimizer_steps % 10 == 0) {
      save_model(&trans_in);
      save_optimizer(&optimizer, "models/gentext2.adam", epoch, optimizer_steps);

      size_t saved = SAVE(&arena.alloc);
      set_dropout(&trans_in, 0.0f);
      generate_text(&arena, &trans_in, prompt, &tokens);
      RESTORE(&arena.alloc, saved);
    }
  }

  save_model(&trans_in);
  save_optimizer(&optimizer, "models/gentext2.adam", epoch, optimizer_steps);

generate_text:
  generate_text(&arena, &trans_in, prompt, &tokens);

  return 0;
}

