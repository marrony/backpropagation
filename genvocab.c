#include "tokenizer.h"

#define BYTEBUFFER_IMPLEMENTATION
#define ALLOCATOR_IMPLEMENATION
#define HASHMAP_IMPLMENTATION

#include "bytebuffer.h"
#include "allocator.h"
#include "hashmap.h"

#include "generated/alice.h"

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

int comp_token(const void* a, const void* b) {
  const Token* token_a = a;
  const Token* token_b = b;

  size_t len_a = strlen(token_a->token);
  size_t len_b = strlen(token_b->token);

  if (len_a > len_b) return -1;
  if (len_a < len_b) return +1;
  return 0;
}

Malloc_Allocator mallocator = MALLOC_CREATE();

#define MAX_VOCAB 512

int main(void) {
  int32_t* tokens = NULL;
  Token vocabulary[MAX_VOCAB] = {0};

  size_t token_count = gen_vocabulary(
      &mallocator.alloc,
      training_text,
      sizeof(training_text)/sizeof(char*),
      &tokens,
      vocabulary,
      MAX_VOCAB
  );

  printf("#define MAX_VOCAB %d\n", MAX_VOCAB);
  printf("Token vocabulary[MAX_VOCAB] = {\n");
  for (int i = 0; i < MAX_VOCAB; i++) {
    printf("  { .id = %d, .token = {", vocabulary[i].id);
    for (int j = 0; j < MAX_TOKEN; j++) {
      if (vocabulary[i].token[j] == '\'')
        printf("'\\'', ");
      else if (vocabulary[i].token[j] == '\\')
        printf("'\\\\', ");
      else if (isprint(vocabulary[i].token[j]))
        printf("'%c', ", vocabulary[i].token[j]);
      else
        printf("0x%02x, ", 0xff & vocabulary[i].token[j]);
    }
    printf("} },\n");
  }
  printf("};\n");

  (void)token_count;
  for (size_t i = 0; i < token_count; i++) {
    int32_t token = tokens[i];
    fprintf(stderr, "%s", vocabulary[token].token);
  }

  qsort(vocabulary, MAX_VOCAB, sizeof(Token), comp_token);

  printf("Token_Sorted vocabulary_by_size[MAX_VOCAB] = {\n");
  for (int i = 0; i < MAX_VOCAB; i++) {
    printf("  { .id = %d, .size = %zu },\n", vocabulary[i].id, strlen(vocabulary[i].token));
  }
  printf("};\n");

  return 0;
}

