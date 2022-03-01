//===-- common.hpp - Define common code for simd functions tests ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file provides common things for simd functions tests.
///
//===----------------------------------------------------------------------===//

#pragma once

#include "../common.hpp"

namespace esimd_test::api::functional::functions {

template <typename DataT, int NumElems>
class TestDescription : public ITestDescription {
public:
  TestDescription(size_t index, DataT retrieved_val, DataT expected_val,
                  int number_selected_values, int stride,
                  const std::string &data_type)
      : m_data_type(data_type), m_retrieved_val(retrieved_val),
        m_expected_val(expected_val), m_index(index),
        m_number_selected_values(number_selected_values), m_stride(stride) {}

  std::string to_string() const override {
    // TODO: Make strings for fp values more short during failure output, may be
    // by using hex representation
    std::string log_msg("Failed for simd<");

    log_msg += m_data_type + ", " + std::to_string(NumElems) + ">";
    log_msg += ", retrieved: " + std::to_string(m_retrieved_val);
    log_msg += ", expected: " + std::to_string(m_expected_val);
    log_msg += ", number selected elements: " +
               std::to_string(m_number_selected_values);
    log_msg += ", with stride: " + std::to_string(m_stride);
    log_msg += ", at index: " + std::to_string(m_index);

    return log_msg;
  }

private:
  const std::string m_data_type;
  const DataT m_retrieved_val;
  const DataT m_expected_val;
  const int m_number_selected_values;
  const int m_stride;
  const size_t m_index;
};

} // namespace esimd_test::api::functional::functions
