#include <asm-generic/errno-base.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdlib.h>
#include <errno.h>
#include <stdio.h>
#include <string.h>

#define VEC_GROWTH 2
#define VEC_GET(type, vec, i) ((type*)vec_get(vec, i))

typedef struct Vector { // d_array
  void* data; // int, uint, float, double
  size_t capacity;
  size_t length;
  size_t element_size;
}Vector;


Vector* vec_new(size_t initial_cap, size_t element_size) {
    if (element_size == 0) {
        errno = EINVAL;
        return NULL;
    }

    if (initial_cap == 0) {
        initial_cap = 32; // default cap
    }

    Vector* vec = malloc(sizeof(Vector));
    if (vec == NULL) {
        perror("Failed to allocate memory for the vector\n");
        return NULL;
    }

    vec->data = malloc(initial_cap * element_size);
    if (vec->data == NULL) {
        perror("Failed to allocate memory for the vector data\n");
        free(vec);
        return NULL;
    }

    vec->capacity = initial_cap;
    vec->length = 0;
    vec->element_size = element_size;
    return vec;
}


void vec_free(Vector* vec) {
  if (vec == NULL) {
    return;
  }

  free(vec->data);
  free(vec);
}

int vec_resize(Vector* vec, size_t new_capacity) {
  if (vec == NULL) {
    return -1;
  }

  void* new_data = realloc(vec->data, new_capacity * vec->element_size);
  if (new_data == NULL) {
    return -1;
  }

  vec->data = new_data;
  vec->capacity = new_capacity;
  return 0;
}

int vec_append(Vector* vec, const void* element) {
  if (vec == NULL || element == NULL) {
    errno = EINVAL;
    return -1;
  }

  if (vec->length == vec->capacity) {
    size_t new_cap = vec->capacity * VEC_GROWTH;
    if (vec_resize(vec, new_cap) != 0) {
      return -1;
    }
  }

  void* dest = (char*)vec->data + vec->length * vec->element_size; // getting the mem addr
  memcpy(dest, element, vec->element_size);

  vec->length++;
  return 0;
}

size_t vec_len(Vector* vec) {
  return vec->length;
}

void* vec_get(Vector* vec, size_t idx) {
  if (vec == NULL || idx >= vec->length) {
    return NULL;
  }

  return (char*)vec->data + idx * vec->element_size;
}


