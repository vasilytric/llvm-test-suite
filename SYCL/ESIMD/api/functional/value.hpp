//===-- value.hpp - This file provides common functions generate values for
//      testing. ----------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file provides functions that let obtain data for test according to
/// current underlying type.
///
//===----------------------------------------------------------------------===//

#pragma once
#include "type_traits.hpp"
#include <CL/sycl.hpp>

#include <climits>
#include <limits>
#include <type_traits>

namespace esimd_test::api::functional {

namespace details {

// Initializes the sycl::half value by using two bytes given:
//  - the higher byte, including the sign bit
//  - the lower byte, including the part of mantissa
//
// This implementation doesn't depend on:
//  - the byte order of both the unsigned types and the floating type itself;
//  - the existence of the optional std::uint16_t type
//  - compiler optimisations related to the strict aliasing rules
sycl::half half_from_bytes(unsigned char hi, unsigned char lo) {
  const size_t size = sizeof(sycl::half);
  static_assert(CHAR_BIT == 8, "Unexpected byte size, input values may broke");
  static_assert(size == 2, "Invalid number of bytes for half type");

  const unsigned char in[size] = {lo, hi};
  unsigned char index[size];
  unsigned char out[size];

  // We are using specific half value to initialize the bits required to differ
  // the lowest and the highest byte
  sycl::half indexHint = 2;
  const unsigned char indexCoeff = 64;

  memcpy(index, &indexHint, size);
  index[0] /= indexCoeff;
  index[1] /= indexCoeff;

  // Ensure there is no overflow possible
  assert(index[0] + index[1] == 1);

  out[0] = in[index[0]];
  out[1] = in[index[1]];
  return esimd_test::bit_cast<sycl::half>(out);
}

// Constructs std::vector with from provided vectors.
template <typename DataT, int NumElems>
std::vector<DataT> construct_ref_data(double step,
                                      std::vector<DataT> first_vector,
                                      std::vector<DataT> second_vector = {}) {
  std::vector<DataT> ref_data{};

  ref_data.reserve((NumElems > 1) ? NumElems : first_vector.size());

  ref_data.insert(ref_data.end(), first_vector.begin(), first_vector.end());
  if constexpr (NumElems != 1) {
    ref_data.insert(ref_data.end(), second_vector.begin(), second_vector.end());

    for (size_t i = ref_data.size(); i < NumElems; ++i) {
      ref_data.push_back(i + step);
    }
  }

  return ref_data;
}

} // namespace details

// Utility class to retrieve specific values for tests depending on the data
// type May be used to retrieve reference data or for generation of golden
// values
template <typename DataT> struct value {
  static DataT inf() {
    static_assert(
        type_traits::is_sycl_floating_point_v<DataT>,
        "Infinity is required only for the floating point data types.");

    if constexpr (std::is_same_v<DataT, sycl::half>) {
      return details::half_from_bytes(0b01111100u, 0b00000000u);
    } else {
      return std::numeric_limits<DataT>::infinity();
    }
  }

  static DataT lowest() {
    if constexpr (std::is_same_v<DataT, sycl::half>) {
      return -max();
    } else {
      return std::numeric_limits<DataT>::lowest();
    }
  }

  static DataT denorm_min() {
    if constexpr (std::is_same_v<DataT, sycl::half>) {
      return details::half_from_bytes(0b00000000u, 0b00000001u);
    } else {
      return std::numeric_limits<DataT>::denorm_min();
    }
  }

  static DataT nan(unsigned char opcode = 42u) {
    static_assert(type_traits::is_sycl_floating_point_v<DataT>,
                  "NaN has meaning only for floating point data types.");
    if constexpr (std::is_same_v<DataT, double>) {
      return sycl::nan(static_cast<unsigned long>(opcode));
    } else if constexpr (std::is_same_v<DataT, float>) {
      return sycl::nan(static_cast<unsigned int>(opcode));
    } else if constexpr (std::is_same_v<DataT, sycl::half>) {
      return details::half_from_bytes(0b11111110u, 0b00000000u + opcode);
    }
  }

  static DataT max() {
    if constexpr (std::is_same_v<DataT, sycl::half>) {
      return details::half_from_bytes(0b01111011u, 0b11111111u);
    } else {
      return std::numeric_limits<DataT>::max();
    }
  }
};

// Provides std::vector with the reference data according to the currently
// tested data type and number of elements.
template <typename DataT, int NumElems> std::vector<DataT> generate_ref_data() {
  static_assert(
      std::is_integral_v<DataT> || type_traits::is_sycl_floating_point_v<DataT>,
      "Invalid data type provided to the generate_ref_data function.");

  // Create values with the strict type guarantee
  static const DataT min = value<DataT>::lowest();
  static const DataT min_half = min / 2;
  static const DataT max = value<DataT>::max();
  static const DataT max_half = max / 2;
  static const DataT min_plus_one = min + 1;
  static const DataT max_minus_one = max - 1;

  std::vector<DataT> ref_data{};

  if constexpr (type_traits::is_sycl_floating_point_v<DataT>) {
    static const DataT nan = value<DataT>::nan();
    static const DataT inf = value<DataT>::inf();

    // We are using the `double` literals to avoid precision loss for case of
    // the `double` DataT on unexact values like 0.1
    ref_data = details::construct_ref_data<DataT, NumElems>(
        0.25, {-inf, nan, min, max, -0.0, 0.1}, {-0.1, +0.0});
  } else if constexpr (std::is_signed_v<DataT>) {
    ref_data = details::construct_ref_data<DataT, NumElems>(
        0, {min, min_half, max, max_half, 0},
        {min_plus_one, max_minus_one, -1});
  } else {
    ref_data = details::construct_ref_data<DataT, NumElems>(
        0, {max, max_half, 0}, {max_minus_one});
  }
  return ref_data;
}

// Provides std::vector with the reference data according to the obtained data
// types and number of elements.
template <typename SrcT, typename DstT, int NumElems>
std::vector<SrcT> generate_converted_ref_data() {
  static_assert(std::is_integral_v<SrcT> ||
                    type_traits::is_sycl_floating_point_v<SrcT>,
                "Invalid the first data type provided to the "
                "generate_converted_ref_data function as source type.");
  static_assert(std::is_integral_v<DstT> ||
                    type_traits::is_sycl_floating_point_v<DstT>,
                "Invalid the second data type provided to the "
                "generate_converted_ref_data function as destination type.");

  static const SrcT min =
      std::max(value<SrcT>::lowest(), static_cast<SrcT>(value<DstT>::lowest()));
  static const SrcT min_half = min / 2;
  static const SrcT max =
      std::min(value<SrcT>::max(), static_cast<SrcT>(value<DstT>::max()));
  static const SrcT max_half = max / 2;
  static const SrcT nan = value<SrcT>::nan();
  static const SrcT inf = value<SrcT>::inf();
  static const SrcT denorm = std::max(
      value<SrcT>::denorm_min(), static_cast<SrcT>(value<DstT>::denorm_min()));
  static const SrcT positive = static_cast<SrcT>(126.75);

  std::vector<SrcT> ref_data{};

  if constexpr (type_traits::is_sycl_floating_point_v<SrcT> &&
                type_traits::is_sycl_floating_point_v<DstT>) {
    ref_data = details::construct_ref_data<SrcT, NumElems>(
        0.25, {min, max, -0.0, +0.0, 0.1, denorm, nan, -inf});
  } else if constexpr (type_traits::is_sycl_floating_point_v<SrcT> &&
                       std::is_unsigned_v<DstT>) {
    ref_data = details::construct_ref_data<SrcT, NumElems>(
        0.75, {-0.0, max, max_half, -max_half});
  } else if constexpr (type_traits::is_sycl_floating_point_v<SrcT> &&
                       std::is_signed_v<DstT>) {
    ref_data = details::construct_ref_data<SrcT, NumElems>(
        0.75, {-0.0, max, max_half, min, min_half});
  } else if constexpr (std::is_signed_v<SrcT> && std::is_signed_v<DstT>) {
    ref_data = details::construct_ref_data<SrcT, NumElems>(
        0, {min, min_half, 0, max_half, max});
  } else if constexpr (std::is_signed_v<SrcT> && std::is_unsigned_v<DstT>) {
    static const SrcT src_min = value<SrcT>::lowest();
    static const SrcT src_min_half = src_min / 2;

    ref_data = details::construct_ref_data<SrcT, NumElems>(
        0, {src_min, src_min_half, 0, max_half, max});
  } else if constexpr (std::is_unsigned_v<SrcT>) {
    ref_data =
        details::construct_ref_data<SrcT, NumElems>(0, {0, max_half, max});
  } else {
    static_assert(!std::is_same_v<SrcT, SrcT>, "Unexpected types combination");
  }

  return ref_data;
}

} // namespace esimd_test::api::functional
