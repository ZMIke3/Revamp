#include "tensor.h"

int main () {

    Tensor *a = tensor_fill(3, (uint64_t[]){3, 2, 2}, DTYPE_F64, CPU, 2, false);
    Tensor *b = tensor_fill(3, (uint64_t[]){3, 2, 5}, DTYPE_F64, CPU, 3, false);

    Tensor *c = tensor_matmul(a, b);

    tensor_print(c);

    error_stack_print();

}