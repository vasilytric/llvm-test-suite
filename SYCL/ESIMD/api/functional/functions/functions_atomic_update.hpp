//===-- functions_atomic_update.hpp - Generic code for tests on simd
//      atomic_update -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file provides common code for tests on simd atomic_update function
///
//===----------------------------------------------------------------------===//

#pragma once

#define ESIMD_TESTS_DISABLE_DEPRECATED_TEST_DESCRIPTION_FOR_LOGS

#include "../common.hpp"

#include <vector>

namespace esimd = sycl::ext::intel::esimd;

namespace esimd_test::api::functional::functions {

// Structure that provides a call operator that lets filter values by different
// algorithm.
namespace masks {
struct ChangeByStep {
  bool operator()(size_t val) SYCL_ESIMD_KERNEL { return val % 2 == 0; }
};

struct ChangeAll {
  bool operator()(size_t val) SYCL_ESIMD_KERNEL { return true; }
};

struct ChangeNothing {
  bool operator()(size_t val) SYCL_ESIMD_KERNEL { return false; }
};
} // namespace masks

enum class offset_generation { all, ordered_step, non_ordered_step };

// Provides std::vector with offset values.
template <int N, offset_generation Algorithm, typename VectorT>
void fill_offsets(VectorT &vector) {
  size_t step = 1;

  if constexpr (Algorithm == offset_generation::ordered_step) {
    step = 2;
  }

  if constexpr (Algorithm == offset_generation::non_ordered_step) {
    size_t max_value = 2 * N;
    for (size_t i = 0; i < N; ++i) {
      vector.push_back(max_value - 2 * i);
    }
  } else {
    for (size_t i = 0; i < N; ++i) {
      vector.push_back(step * i);
    }
  }
}

} // namespace esimd_test::api::functional::functions
