// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %HOST_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

#include "store.h"
#include <iostream>
using namespace sycl;

int main() {
  queue q;

  constexpr int N = 32;
  store_test<int>(q, N);
  store_test<unsigned int>(q, N);
  store_test<float>(q, N);

  // Include long tests if they are 32 bits wide
  if constexpr (sizeof(long) == 4) {
    store_test<long>(q, N);
    store_test<unsigned long>(q, N);
  }

  // Include pointer tests if they are 32 bits wide
  if constexpr (sizeof(char *) == 4) {
    store_test<char *>(q, N);
  }

  std::cout << "Test passed." << std::endl;
}
