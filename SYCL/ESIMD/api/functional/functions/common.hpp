//===-- common.hpp - Define common code for simd functions tests ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file provides common code for simd functions tests.
///
//===----------------------------------------------------------------------===//

#pragma once

#include "../common.hpp"

namespace esimd_test::api::functional::functions {

namespace esimd = sycl::ext::intel::esimd;

template <int NumElems, typename TestCaseT>
class TestDescription : public ITestDescription {
public:
  TestDescription(const std::string &data_type) : m_data_type(data_type) {}

  std::string to_string() const override {
    std::string test_description = TestCaseT::get_description();
    test_description += " with simd<" + m_data_type;
    test_description += ", " + std::to_string(NumElems) + ">";
    return test_description;
  }

private:
  const std::string m_data_type;
};

} // namespace esimd_test::api::functional::functions
