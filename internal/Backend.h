#ifndef BACKEND_H
#define BACKEND_H

#include <stdbool.h>

typedef enum{
    CPU,
    SIMD,
    GPU,
    BACKEND_COUNT
}Backend;


typedef struct array array;

bool to_backend(array *a, Backend backend);

#endif // BACKEND_H