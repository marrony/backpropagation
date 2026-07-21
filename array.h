#ifndef ARRAY_H
#define ARRAY_H

#include "bytebuffer.h"

#define SLICE_BODY(type) \
  type* elems;           \
  size_t count

#define ARRAY_BODY(type) \
  SLICE_BODY(type);      \
  size_t capacity;       \
  Allocator* allocator

#define DEFINE_ARRAY_ALIAS(name, type) \
typedef struct {                       \
  ARRAY_BODY(type);                    \
} name##_Array

#define DEFINE_ARRAY(type) \
  DEFINE_ARRAY_ALIAS(type, type)

#define DEFINE_SLICE(type)          \
typedef struct type##_Array_Slice { \
  SLICE_BODY(type);                 \
} type##_Array_Slice

#define DEFINE_ARRAY_SLICE(type) \
  DEFINE_ARRAY(type);            \
  DEFINE_SLICE(type)

typedef void* Generic_Ptr;

DEFINE_ARRAY_SLICE(Generic_Ptr);

void _array_ensure(Generic_Ptr_Array* array, size_t elem_size, size_t init_capacity) {
  if (array->capacity == 0) {
    Byte_Buffer new_ptr = ALLOC(array->allocator, init_capacity * elem_size);

    array->elems = (void*)new_ptr.ptr;
    array->capacity = init_capacity;
  } else if (array->count >= array->capacity) {
    size_t new_capacity = array->capacity * 2;

    Byte_Buffer new_ptr = ALLOC(array->allocator, new_capacity * elem_size);

    Byte_Buffer old_ptr = byte_buffer_from_parts(array->elems, array->capacity * elem_size);

    byte_buffer_copy(new_ptr, old_ptr);

    FREE(array->allocator, old_ptr);

    array->elems = (void*)new_ptr.ptr; 
    array->capacity = new_capacity;
  }
}

void _array_destroy(Generic_Ptr_Array* array, size_t elem_size) {
  if (array->allocator != NULL && array->elems != NULL) {
    Byte_Buffer old_ptr = byte_buffer_from_parts(array->elems, array->capacity * elem_size);

    FREE(array->allocator, old_ptr);
  }

  array->elems = NULL;
  array->count = 0;
  array->capacity = 0;
  //array->allocator = NULL;
}

#define array_destroy(array) \
  _array_destroy((Generic_Ptr_Array*)(array), sizeof((array)->elems[0]))

#define array_ensure(array, size) \
  do { \
    _array_ensure((Generic_Ptr_Array*)(array), sizeof((array)->elems[0]), size); \
  } while (0)

#define array_append(array, e) \
  do { \
    _array_ensure((Generic_Ptr_Array*)(array), sizeof((array)->elems[0]), 32); \
    (array)->elems[(array)->count++] = (e); \
  } while (0)

#define array_empty(array) \
  ((array)->count == 0)

#define array_at(array, ith) \
  ((array)->elems[(ith)])

#define array_first(array) \
  array_at(array, 0)

#define array_last(array) \
  array_at(array, (array)->count - 1)

#define array_pop_last(array) \
  ((array)->elems[--(array)->count])

#define array_swap(array, x, y)                                     \
  do {                                                              \
    if ((x) != (y)) {                                               \
      char temp[sizeof((array)->elems[0])];                         \
      memcpy(temp, (array)->elems+(x), sizeof(temp));               \
      memcpy((array)->elems+(x), (array)->elems+(y), sizeof(temp)); \
      memcpy((array)->elems+(y), temp, sizeof(temp));               \
    }                                                               \
  } while (0)

#define array_shuffle(array)                             \
  do {                                                   \
    if ((array)->count > 1) {                            \
      for (size_t i = (array)->count - 1; i > 0; i--) {  \
        size_t j = (size_t)rand() % (i + 1);             \
        array_swap((array), i, j);                       \
      }                                                  \
    }                                                    \
  } while (0)

#define ARRAY_CREATE(alloc) { .allocator = (alloc) }

#endif // ARRAY_H

