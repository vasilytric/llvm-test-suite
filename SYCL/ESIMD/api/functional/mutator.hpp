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

// Provides base functions to mutate value.
namespace mutator {

template <typename T> class Add {
  T m_value;

public:
  Add(T value_for_mutation) : m_value(value_for_mutation) {}

  void operator()(T &value_for_mutation) {
    if constexpr (std::is_signed_v<T>) {
      if (value<T>::max() - m_value <= value_for_mutation) {
        value_for_mutation = value<T>::max();
      } else {
        value_for_mutation += m_value;
      }
    } else {
      value_for_mutation += m_value;
    }
  }
};

template <typename T> class Sub {
  T m_value;

public:
  Sub(T value_for_mutation) : m_value(value_for_mutation) {}

  void operator()(T &value_for_mutation) {
    if constexpr (std::is_signed_v<T> || std::is_same_v<T, sycl::half>) {
      if (value<T>::lowest() + m_value >= value_for_mutation) {
        value_for_mutation = value<T>::lowest();
      } else {
        value_for_mutation -= m_value;
      }
    } else {
      value_for_mutation -= m_value;
    }
  }
};

template <typename T> class Divide {
  T m_value;

public:
  Divide(T value_for_mutation) : m_value(value_for_mutation) {}

  void operator()(T &value_for_mutation) {
    if (value_for_mutation != value<T>::denorm_min()) {
      value_for_mutation /= m_value;
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
