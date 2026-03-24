# Revamp

## Overview

Revamp is a low performance, 😅, tensor, and graph computation library written in pure C.  Right now it's not developed enough to be viable in an actual project but the goal is to update the library in incremental steps till we can support most modern neural network workloads.

## 🚀 Usage

Here are some operations you can try out for yourself
- Tensor Matmul
```C
#include "tensor.h"

int main () {

    Tensor *a = tensor_fill(3, (uint64_t[]){3, 2, 2}, DTYPE_F64, CPU, 2, false);
    Tensor *b = tensor_fill(3, (uint64_t[]){3, 2, 5}, DTYPE_F64, CPU, 3, false);

    Tensor *c = tensor_matmul(a, b);

    tensor_print(c);

    error_stack_print();

}
```
```
[[[12.000000, 12.000000, 12.000000, 12.000000, 12.000000], 
  [12.000000, 12.000000, 12.000000, 12.000000, 12.000000]], 
 [[12.000000, 12.000000, 12.000000, 12.000000, 12.000000], 
  [12.000000, 12.000000, 12.000000, 12.000000, 12.000000]], 
 [[12.000000, 12.000000, 12.000000, 12.000000, 12.000000], 
  [12.000000, 12.000000, 12.000000, 12.000000, 12.000000]]]
```

- Adding two tensors
```C
int64_t data[6] = {1, 2, 3, 4, 5, 6};

Tensor *t1 = tensor_create(3, (uint64_t[]){1, 2, 3}, CPU, DTYPE_I64, data);

Tensor *t2 = tensor_create(3, (uint64_t[]){1, 2, 3}, CPU, DTYPE_I64, data);

Tensor *t3 = tensor_add(t1, t2);

tensor_print(t3);
```
```
[[[2, 4, 6], 
  [8, 10, 12]]]
```

- Broadcasting

The library also supports implicit broadcasting

```C
int64_t data[6] = {1, 2, 3, 4, 5, 6};

int64_t data1[2] = {2, 4};

Tensor *t1 = tensor_create(3, (uint64_t[]){1, 2, 3}, CPU, DTYPE_I64, data);

Tensor *t2 = tensor_create(2, (uint64_t[]){2, 1}, CPU, DTYPE_I64, data1);

Tensor *t3 = tensor_add(t1, t2);

tensor_print(t3);
```
```
[[[3, 4, 5],
  [8, 9, 10]]]
```

- Taking the gradient
We can also take the gradient of operations
```C
double w_data[6] = {1, 2, 3, 4, 5, 6};

double a_data[1] = {2};

double b_data[1] = {5};

Tensor *w = tensor_create(3, (uint64_t[]){1, 2, 3}, CPU, DTYPE_F64, w_data);

Tensor *a = tensor_create(1, (uint64_t[]){1}, CPU, DTYPE_F64, a_data);

Tensor *b = tensor_create(1, (uint64_t[]){1}, CPU, DTYPE_F64, b_data);

tensor_retain_grad(w);

tensor_retain_grad(a);

tensor_retain_grad(b);

Tensor *out = tensor_add(tensor_mul(w, a), b);

grad(out);

tensor_print_grad(w);

tensor_print_grad(a);

tensor_print_grad(b);
```
*For w, a, and b respectively*
```
[[[2.000000, 2.000000, 2.000000], 
  [2.000000, 2.000000, 2.000000]]]

[21.000000]

[6.000000]
```

## ⬇️ Installation

```bash
  git clone https://github.com/ZMIke3/Revamp.git
  mingw32-make install
```
1. Update makefile prefix with installation path
2. Add bin to path
3. Add include folder path to VScode configurations Ctrl+Shift+P -> C/C++:Edit Configurations (UI)
4. Make sure Revamp.dll is in project folder
5. Compile with gcc main.c -o main -IC:/Programming/RevampTestInstall/include -LC:/Programming/RevampTestInstall/lib -lrevamp
6. Matmul requires OpenBLAS to run so make sure you have it installed and update the makefile with its path

## 🤝 Contributions
Any contributions are welcome! Have a look at the Contributions.md...to be written

## 🔧 Quick Notes
- Right now matmul is a bit buggy, doesn't produce full output for 4d @ 1d calls or vice versa

