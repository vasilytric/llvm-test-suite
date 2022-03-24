//===-- sycl_nd_range.hpp - Provides support for sycl::nd_range usage in
//      tests -------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Provides all neccessary helpers and wrappers to use with tests with
/// sycl::nd_range
///
//===----------------------------------------------------------------------===//

#pragma once

#include "logger.hpp"

namespace esimd_test::api::functional {

template <int Dims> sycl::nd_range<Dims> get_sycl_nd_range(size_t count);

template <> sycl::nd_range<1> get_sycl_nd_range<1>(size_t count) {
  return sycl::nd_range<1>(count, 1);
}
template <> sycl::nd_range<2> get_sycl_nd_range<2>(size_t count) {
  if (count % 2 != 0) {
    sycl::range<2> global_range(count, 1);
    sycl::range<2> local_range(1, 1);
    return sycl::nd_range<2>(global_range, local_range);
  }
  sycl::range<2> global_range(count / 2, 1);
  sycl::range<2> local_range(2, 1);
  return sycl::nd_range<2>(global_range, local_range);
}
template <> sycl::nd_range<3> get_sycl_nd_range<3>(size_t count) {
  if (count % 2 != 0) {
    sycl::range<3> global_range(count, 1, 1);
    sycl::range<3> local_range(1, 1, 1);
    return sycl::nd_range<3>(global_range, local_range);
  }
  sycl::range<3> global_range(count / 2, 1, 1);
  sycl::range<3> local_range(1, 1, 1);
  return sycl::nd_range<3>(global_range, local_range);
}

namespace log {
// Specialization of StringMaker for sycl::range logging purposes
template <int Dims> struct StringMaker<sycl::range<Dims>> {
  static std::string stringify(const sycl::nd_range<Dims> &nd_range) {
    std::ostringstream stream;
    stream << "sycl::nd_range<" << std::to_string(Dims);
    stream << ">{" << nd_range[0];
    if constexpr (Dims >= 2) {
      stream << ", " << nd_range[1];
    }
    if constexpr (Dims >= 3) {
      stream << ", " << nd_range[2];
    }
    stream << "}";
    return stream.str();
  }
};
} // namespace log

} // namespace esimd_test::api::functional
