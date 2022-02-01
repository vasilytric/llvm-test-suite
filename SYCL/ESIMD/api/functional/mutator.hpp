//===-- mutator.hpp - This file provides common function and classes to mutate
//      reference data. ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file provides common function and classes to mutate reference data.
///
//===----------------------------------------------------------------------===//

#include "value.hpp"
#include <sycl/sycl.hpp>

#include <algorithm>
#include <vector>

#pragma once

namespace esimd_test::api::functional {

// In some test cases it’s possible to pass a reference data as the input data
// directly without any modification. For example, if we check some copy
// constructor or memory move operation it’s OK to have any data values as the
// input ones. But in most cases we need to consider the possibility of UB for
// C++ operations due to modification of input values.
//
// Mutation mechanism covers such requirement.
// The mutator namespace is intended to store such mutators alongside with
// the generic ones provided below.
namespace mutator {

// Replace specific reference values to the bigger ones, so we can safely
// substract `val` later.
template <typename T> class For_subtraction {
  T m_value;

public:
  For_subtraction(T val)
      : m_value((assert(val > 0 && "Invalid value."), val)) {}

  void operator()(T &val) {
    if constexpr (std::is_signed_v<T>) {
      if (val < value<T>::lowest() + m_value) {
        val = value<T>::lowest() + m_value;
      }
    }
  }
};

// Replace specific reference values to the smaller ones, so we can safely add
// `val` later.
template <typename T> class For_addition {
  T m_value;

public:
  For_addition(T val) : m_value((assert(val > 0 && "Invalid value."), val)) {}

  void operator()(T &val) {
    if constexpr (std::is_signed_v<T> || std::is_same_v<T, sycl::half>) {
      if (val > value<T>::max() - m_value) {
        val = value<T>::max() - m_value;
      }
    }
  }
};

// Replace specific reference values to the divided ones, so we can safely
// multiply to `val` later.
template <typename T> class For_division {
  T m_value;

public:
  For_division(T val) : m_value((assert(val > 0 && "Invalid value."), val)) {}

  void operator()(T &val) {
    if (val != value<T>::denorm_min()) {
      val /= m_value;
    }
  }
};

} // namespace mutator

// Applies provided mutator to each value for provided container.
template <typename T, typename MutatorT>
void mutate(std::vector<T> &input_vector, MutatorT &&mutator) {
  std::for_each(input_vector.begin(), input_vector.end(), mutator);
}

} // namespace esimd_test::api::functional
