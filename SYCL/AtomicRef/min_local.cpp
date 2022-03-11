// RUN: %clangxx -fsycl -fsycl-unnamed-lambda -fsycl-targets=%sycl_triple %s -o %t.out -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_70
// RUN: %HOST_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

// Barrier is not supported on host. HIP and ACC do not support floating
// point atomics.
// XFAIL: host, hip, acc

#include "min.h"

int main() { min_test_all<access::address_space::local_space>(); }
