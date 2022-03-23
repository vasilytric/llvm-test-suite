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
namespace filters {
struct ChangeByStep {
  bool operator()(size_t val) SYCL_ESIMD_KERNEL { return val % 2 == 0; }
};

struct ChangeAll {
  bool operator()(size_t val) SYCL_ESIMD_KERNEL { return true; }
};

struct ChangeNothing {
  bool operator()(size_t val) SYCL_ESIMD_KERNEL { return false; }
};
} // namespace filters

enum class algorithm_to_change { all, ordered_step, non_ordered_step };

template <int N, algorithm_to_change Algorithm>
std::vector<size_t> get_indexess_to_change() {
  std::vector<size_t> data;
  size_t step;

  if constexpr (Algorithm == algorithm_to_change::all) {
    step = 1;
  } else if constexpr (Algorithm == algorithm_to_change::ordered_step) {
    step = 2;
  }

  if constexpr (Algorithm == algorithm_to_change::non_ordered_step) {
    data = std::vector<size_t>{1, 3, 7, 5};
    for (size_t i = data.size(); i < N; ++i) {
      data.push_back(2 * i + 1);
    }
  } else {
    for (size_t i = 0; i < N; ++i) {
      data.push_back(step * i);
    }
  }
  return data;
}

template <int N, esimd::atomic_op Operator, typename T>
std::vector<T> get_init_values(T initial_base_value) {
  std::vector<T> data;
  T base_value;
  T step;

  if constexpr (Operator == esimd::atomic_op::dec) {
    base_value = initial_base_value;
    step = 0;
  } else if constexpr (Operator == esimd::atomic_op::inc) {
    base_value = initial_base_value;
    step = 0;
  }

  for (size_t i = 0; i < N; ++i) {
    data.push_back(base_value + step * i);
  }
  return data;
}

template <esimd::atomic_op Operator, typename T>
T get_expected_value(T base_value, int number_teractions) {
  T expected;

  if constexpr (Operator == esimd::atomic_op::dec) {
    expected = base_value - number_teractions;
  } else if constexpr (Operator == esimd::atomic_op::inc) {
    expected = base_value + number_teractions;
  }
  return expected;
}

} // namespace esimd_test::api::functional::functions
