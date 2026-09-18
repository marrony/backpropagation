#ifndef ALLOCATOR_H
#define ALLOCATOR_H

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <strings.h>

#include "bytebuffer.h"

#define OOM_RETURN_CODE (137)

struct Allocator;

typedef Byte_Buffer (*Alloc_Fn)(struct Allocator*, size_t, const char*, size_t);
typedef void (*Free_Fn)(struct Allocator*, Byte_Buffer, const char*, size_t);
typedef size_t (*Save_Fn)(struct Allocator*, const char*, size_t);
typedef void (*Restore_Fn)(struct Allocator*, size_t, const char*, size_t);

typedef struct Allocator {
  Alloc_Fn alloc;
  Free_Fn free;
  Save_Fn save;
  Restore_Fn restore;
} Allocator;

#define ALLOC(a, n) (a)->alloc((a), (n), __FILE__, __LINE__)

#define FREE_NON_NULL(a, b)                    \
  do {                                         \
    if (!byte_buffer_null((b)))                \
      (a)->free((a), (b), __FILE__, __LINE__); \
  } while (0)

#define FREE(a, b)           \
  do {                       \
    FREE_NON_NULL((a), (b)); \
    (b) = NULL_BYTE_BUFFER;  \
  } while (0)

#define SAVE(a) (a)->save((a), __FILE__, __LINE__)
#define RESTORE(a, b) (a)->restore((a), (b), __FILE__, __LINE__)

#define ALLOC_TYPE(a, t) (t*)ALLOC((a), sizeof(t)).ptr
#define FREE_TYPE(a, p) FREE_NON_NULL((a), byte_buffer_from_parts((p), sizeof(*(p))))

typedef struct Malloc_Entry {
  struct Malloc_Entry* previous;
  struct Malloc_Entry* next;

  size_t nbytes;
  const char* file;
  size_t line;
} Malloc_Entry;

typedef struct Malloc_Allocator {
  Allocator alloc;
  size_t allocated;
  Malloc_Entry* entries;
} Malloc_Allocator;

#define MALLOC_CREATE() (Malloc_Allocator) { \
  .alloc = {                                 \
    .alloc = (Alloc_Fn)malloc_alloc,         \
    .free = (Free_Fn)malloc_free,            \
    .save = (Save_Fn)malloc_save,            \
    .restore = (Restore_Fn)malloc_restore,   \
  },                                         \
  .allocated = 0,                            \
  .entries = NULL,                           \
}

Byte_Buffer malloc_alloc(Malloc_Allocator* alloc, size_t nbytes, const char* file, size_t line);
void malloc_free(Malloc_Allocator* alloc, Byte_Buffer buffer, const char* file, size_t line);
size_t malloc_save(Malloc_Allocator* alloc, const char* file, size_t line);
void malloc_restore(Malloc_Allocator* alloc, size_t allocated, const char* file, size_t line);

typedef struct Arena_Allocator {
  Allocator alloc;
  Byte_Buffer buffer;
  size_t allocated;
} Arena_Allocator;

#define ARENA_CREATE(a, s) (Arena_Allocator) { \
  .alloc = {                                   \
    .alloc = (Alloc_Fn)arena_alloc,            \
    .free = (Free_Fn)arena_free,               \
    .save = (Save_Fn)arena_save,               \
    .restore = (Restore_Fn)arena_restore,      \
  },                                           \
  .buffer = ALLOC((a), (s)),                   \
  .allocated = 0,                              \
}

#define ARENA_DESTROY(alloc, arena)     \
do {                                    \
  FREE((alloc), (arena)->buffer);       \
  (arena)->allocated = 0;               \
} while (0)

Byte_Buffer arena_alloc(Arena_Allocator* alloc, size_t nbytes, const char* file, size_t line);
void arena_free(Arena_Allocator* alloc, Byte_Buffer buffer, const char* file, size_t line);
size_t arena_save(Arena_Allocator* alloc, const char* file, size_t line);
void arena_restore(Arena_Allocator* alloc, size_t allocated, const char* file, size_t line);

#endif // ALLOCATOR_H

#ifdef ALLOCATOR_IMPLEMENATION

#ifndef ALLOCATOR_IMPLEMENATION_C
#define ALLOCATOR_IMPLEMENATION_C

#include "bytebuffer.h"

Byte_Buffer malloc_alloc(Malloc_Allocator* alloc, size_t nbytes, const char* file, size_t line) {
  uint8_t* ptr = malloc(nbytes + sizeof(Malloc_Entry));
  if (ptr == NULL) {
    fprintf(stderr, "malloc error: tried to allocate %zu bytes\n", nbytes + sizeof(Malloc_Entry));
    exit(OOM_RETURN_CODE);
  }

  memset(ptr, 0, nbytes + sizeof(Malloc_Entry));

  Malloc_Entry* entry = (Malloc_Entry*)ptr;

  entry->nbytes = nbytes;
  entry->file = file;
  entry->line = line;
  entry->next = NULL;
  entry->previous = NULL;

  alloc->allocated += nbytes;

  if (alloc->entries != NULL) {
    alloc->entries->previous = entry;
  }

  entry->next = alloc->entries;
  alloc->entries = entry;

  return byte_buffer_from_parts(ptr + sizeof(Malloc_Entry), nbytes);
}

void malloc_free(Malloc_Allocator* alloc, Byte_Buffer buffer, const char* file, size_t line) {
  (void)file;
  (void)line;

  void* ptr = buffer.uptr - sizeof(Malloc_Entry);
  Malloc_Entry* entry = (Malloc_Entry*)ptr;

  size_t nbytes = entry->nbytes;

  if (nbytes != buffer.len) {
    if (buffer.len < nbytes)
      fprintf(stderr, "%s:%zu warning freeing less than allocated: %zu < %zu\n", entry->file, entry->line, buffer.len, nbytes);
    else
      fprintf(stderr, "%s:%zu warning freeing more than allocated: %zu != %zu\n", entry->file, entry->line, buffer.len, nbytes);
  }

  alloc->allocated -= nbytes;

  Malloc_Entry* previous = entry->previous;
  Malloc_Entry* next = entry->next;

  if (previous != NULL) {
    previous->next = next;
  }

  if (next != NULL) {
    next->previous = previous;
  }

  if (alloc->entries == entry) {
    alloc->entries = next;
  }

  free(ptr);
}

size_t malloc_save(Malloc_Allocator* alloc, const char* file, size_t line) {
  (void)alloc;
  (void)file;
  (void)line;
  return 0;
}

void malloc_restore(Malloc_Allocator* alloc, size_t allocated, const char* file, size_t line) {
  (void)alloc;
  (void)allocated;
  (void)file;
  (void)line;
}

Byte_Buffer arena_alloc(Arena_Allocator* alloc, size_t nbytes, const char* file, size_t line) {
  (void)file;
  (void)line;

  size_t remaining = alloc->buffer.len - alloc->allocated;

  if (nbytes > remaining) {
    fprintf(stderr, "arena error: tried to allocate %zu bytes, remaining %zu\n", nbytes, remaining);
    fprintf(stderr, "arena error: total=%zu allocated=%zu\n", alloc->buffer.len, alloc->allocated+nbytes);
    exit(OOM_RETURN_CODE);
  }

  void* ptr = alloc->buffer.uptr + alloc->allocated;

  memset(ptr, 0, nbytes);

  alloc->allocated += nbytes;

  return byte_buffer_from_parts(ptr, nbytes);
}

void arena_free(Arena_Allocator* alloc, Byte_Buffer buffer, const char* file, size_t line) {
  (void)alloc;
  (void)buffer;
  (void)file;
  (void)line;
}

size_t arena_save(Arena_Allocator* alloc, const char* file, size_t line) {
  (void)file;
  (void)line;
  return alloc->allocated;
}

void arena_restore(Arena_Allocator* alloc, size_t allocated, const char* file, size_t line) {
  (void)file;
  (void)line;
  alloc->allocated = allocated;
}

#endif // ALLOCATOR_IMPLEMENATION_C

#endif // ALLOCATOR_IMPLEMENATION
