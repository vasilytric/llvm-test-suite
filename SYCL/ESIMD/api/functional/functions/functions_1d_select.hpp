//===-- functions_1d_select.hpp - Functions for tests on simd rvalue select
//      function. ---------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file provides functions for tests on simd 1d select function.
///
//===----------------------------------------------------------------------===//

#pragma once

#include "../shared_element.hpp"
#include "common.hpp"

// for std::ceil
#include <cmath>
// for std::numeric_limits
#include <limits>

namespace esimd = sycl::ext::intel::experimental::esimd;

namespace esimd_test::api::functional::functions {

namespace details {

constexpr int ceil(float arg) {
  return (arg > static_cast<int>(arg)) ? arg + 1 : arg;
}

} // namespace details

// Aliases to provide size or stride values to test.
// This is the syntax sugar and they have the same type.
template <int N> using stride_type = std::integral_constant<int, N>;
template <int N> using size_type = std::integral_constant<int, N>;
template <int N> using offset_type = std::integral_constant<int, N>;

using use_offset = std::true_type;
using do_not_use_offset = std::true_type;

// Descriptor class for the case of calling simd<T,N>::select function.
struct select_rval {
  static std::string get_description() { return "select rvalue"; }

  template <typename DataT, int NumElems, int NumSelectedElems, int Stride>
  static bool call_operator(const DataT *const ref_1, const DataT *const ref_2,
                            DataT *const out, size_t offset) {
    esimd::simd<DataT, NumElems> simd_obj;
    simd_obj.copy_from(ref_1);
    auto select_result =
        simd_obj.template select<NumSelectedElems, Stride>(offset);

    for (size_t i = 0; i < NumSelectedElems; ++i) {
      select_result[i] = ref_2[i];
    }
    simd_obj.copy_to(out);

    return std::is_same_v<
        decltype(select_result),
        simd_view<esimd::simd<DataT, NumElems>,
                  region1d_t<DataT, NumSelectedElems, Stride>>>;
  }
};

// Descriptor class for the case of calling simd<T,N>::select function.
struct select_lval {
  static std::string get_description() { return "select lvalue"; }

  template <typename DataT, int NumElems, int NumSelectedElems, int Stride>
  static bool call_operator(const DataT *const ref_1, const DataT *const ref_2,
                            DataT *const out, size_t offset) {
    simd<DataT, NumElems> src_simd_obj;
    src_simd_obj.copy_from(ref_1);

    simd<DataT, NumSelectedElems> dst_simd_obj;
    dst_simd_obj.copy_from(ref_2);

    src_simd_obj.template select<NumSelectedElems, Stride>(offset) =
        dst_simd_obj;
    src_simd_obj.copy_to(out);

    return true;
  }
};

// Descriptor class for the case of calling simd<T,N>::select function.
struct select_simd_view {
  static std::string get_description() { return "select simd view"; }

  template <typename DataT, int NumElems, int NumSelectedElems, int Stride>
  static bool call_operator(const DataT *const ref_1, const DataT *const ref_2,
                            DataT *const out, size_t offset) {
    simd<DataT, NumElems> src_simd_obj;
    src_simd_obj.copy_from(ref_1);

    auto simd_view_instance = src_simd_obj.template bit_cast_view<DataT>();

    auto selected_elems =
        simd_view_instance.template select<NumSelectedElems, Stride>(offset);

    for (size_t i = 0; i < NumSelectedElems; ++i) {
      selected_elems[i] = ref_2[i];
    }

    src_simd_obj.copy_to(out);

    return true;
  }
};

// The main test routine.
// Using functor class to be able to iterate over the pre-defined data types.
template <typename TestCaseT, typename NumSelectedElemsT, typename StrideT,
          typename OffsetT, typename DataT, typename DimT>
class run_test {
  static constexpr int NumElems = DimT::value;
  static constexpr int NumSelectedElems = NumSelectedElemsT::value;
  static constexpr int Stride = StrideT::value;
  static constexpr int Offset = OffsetT::value;

public:
  bool operator()(sycl::queue &queue, const std::string &data_type) {
    static_assert(NumElems >= NumSelectedElems * Stride + Offset &&
                  "Offset should be less than number selected elements.");
    bool passed = true;
    size_t alignment_value = alignof(DataT);

    constexpr size_t value_for_increase_ref_data_for_fill = 50;
    static_assert(std::numeric_limits<signed char>::max() >
                      value_for_increase_ref_data_for_fill + NumElems,
                  "Value that used for increase ref data for fill plus simd "
                  "size should be less than char max value.");

    shared_allocator<DataT> allocator(queue);
    shared_vector<DataT> result(NumElems, allocator);
    shared_vector<DataT> initial_ref_data(NumElems, allocator);
    shared_vector<DataT> ref_data_for_fill(NumElems, allocator);

    shared_element<bool> is_correct_type(queue, true);

    for (size_t i = 0; i < NumElems; ++i) {
      initial_ref_data[i] = i + 1;
    }
    // We should have different values in the first reference data and in the
    // second reference data.
    for (size_t i = 0; i < NumElems; ++i) {
      ref_data_for_fill[i] =
          initial_ref_data[i] + value_for_increase_ref_data_for_fill;
    }

    queue.submit([&](sycl::handler &cgh) {
      DataT *init_ref_ptr = initial_ref_data.data();
      DataT *ref_data_for_fill_ptr = ref_data_for_fill.data();
      DataT *const out_ptr = result.data();
      auto is_correct_type_ptr = is_correct_type.data();

      cgh.single_task<
          Kernel<DataT, NumElems, TestCaseT, NumSelectedElemsT, StrideT>>(
          [=]() SYCL_ESIMD_KERNEL {
            *is_correct_type_ptr =
                TestCaseT::template call_operator<DataT, NumElems,
                                                  NumSelectedElems, Stride>(
                    init_ref_ptr, ref_data_for_fill_ptr, out_ptr, Offset);
          });
    });
    queue.wait_and_throw();

    std::vector<size_t> selected_indexsess;
    // Collect the indexess that has been selected.
    for (size_t i = Offset; i < Stride * NumSelectedElems + Offset;
         i += Stride) {
      selected_indexsess.push_back(i);
    }

    // Push the largest value to avoid the following error: can't dereference
    // out of range vector iterator.
    selected_indexsess.push_back(std::numeric_limits<size_t>::max());
    auto selected_indexsess_ptr = selected_indexsess.begin();

    // Verify that values, that do not was selected has initial values.
    for (size_t i = 0; i < NumElems; ++i) {
      // If current index is less than selected index verify that this element
      // hasn't been selected and changed.
      if (i < *selected_indexsess_ptr) {
        DataT expected = initial_ref_data[i];
        DataT retrieved = result[i];
        if (expected != retrieved) {
          passed = fail_test(i, expected, retrieved, data_type);
        }
      } else {
        DataT expected = ref_data_for_fill[(i - Offset) / Stride];
        DataT retrieved = result[i];
        if (expected != retrieved) {
          passed = fail_test(i, expected, retrieved, data_type);
        }
        selected_indexsess_ptr++;
      }
    }

    if (!is_correct_type.value()) {
      passed = false;
      log::print_line("Test failed due to type of the object that returns " +
                      TestCaseT::get_description() +
                      " operator is not equal to the expected one for simd<" +
                      data_type + ", " + std::to_string(NumElems) + ">.");
    }

    return passed;
  }

private:
  bool fail_test(size_t i, DataT expected, DataT retrieved,
                 const std::string &data_type) {
    log::fail(TestDescription<NumElems, TestCaseT>(data_type),
              "Unexpected value at index ", i, ", retrieved: ", retrieved,
              ", expected: ", expected, ", with size: ", NumSelectedElems,
              ", stride: ", Stride, ", offset: ", Offset);

    return false;
  }
};

template <tested_types TestedTypes, typename SelectT>
bool run_test_for_types(sycl::queue &queue) {
  bool passed = true;
  constexpr int desired_simd_small_size = 1;
  constexpr int desired_simd_large_size = 16;
  constexpr int coefficient_of_division = 3;
  constexpr int zero_offset_value = 0;
  constexpr int small_offset_value = 1;
  constexpr int large_offset_value =
      desired_simd_large_size - details::ceil(2 * desired_simd_large_size / 3);

  const auto small_size = get_dimensions<desired_simd_small_size>();
  const auto great_size = get_dimensions<desired_simd_large_size>();
  const auto types = get_tested_types<TestedTypes>();

  // Checks are run for specific combinations of types, sizes, strides and
  // offsets.
  passed &=
      for_all_combinations<run_test, SelectT, size_type<1>, stride_type<1>,
                           offset_type<zero_offset_value>>(types, small_size,
                                                           queue);
  passed &=
      for_all_combinations<run_test, SelectT, size_type<1>, stride_type<1>,
                           offset_type<zero_offset_value>>(types, small_size,
                                                           queue);
  passed &= for_all_combinations<
      run_test, SelectT,
      size_type<desired_simd_large_size / coefficient_of_division>,
      stride_type<coefficient_of_division>, offset_type<zero_offset_value>>(
      types, great_size, queue);
  passed &= for_all_combinations<
      run_test, SelectT,
      size_type<desired_simd_large_size / coefficient_of_division>,
      stride_type<3>, offset_type<zero_offset_value>>(types, great_size, queue);
  passed &= for_all_combinations<
      run_test, SelectT, size_type<3>,
      stride_type<desired_simd_large_size / coefficient_of_division>,
      offset_type<zero_offset_value>>(types, great_size, queue);

  passed &= for_all_combinations<
      run_test, SelectT,
      size_type<desired_simd_large_size - small_offset_value>,
      stride_type<desired_simd_small_size>, offset_type<small_offset_value>>(
      types, great_size, queue);
  passed &=
      for_all_combinations<run_test, SelectT,
                           size_type<desired_simd_large_size / 3>,
                           stride_type<2>, offset_type<large_offset_value>>(
          types, great_size, queue);

  return passed;
}

} // namespace esimd_test::api::functional::functions
