//===-- functions_select_rvalue.hpp - Functions for tests on simd rvalue select
//      function. ---------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file provides functions for tests on simd rvalue select function.
///
//===----------------------------------------------------------------------===//

#pragma once

#include "../shared_element.hpp"
#include "common.hpp"

// for std::ceil
#include <cmath>

namespace esimd = sycl::ext::intel::experimental::esimd;

namespace esimd_test::api::functional::functions {

// Aliases to provide size or stride values to test.
template <int N> using stride = std::integral_constant<int, N>;
template <int N> using size = std::integral_constant<int, N>;

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

// The main test routine.
// Using functor class to be able to iterate over the pre-defined data types.
template <typename TestCaseT, typename NumSelectedElemsT, typename StrideT,
          typename UseOffsetT, typename DataT, typename DimT>
class run_test {
  static constexpr int NumElems = DimT::value;
  static constexpr int NumSelectedElems = NumSelectedElemsT::value;
  static constexpr int Stride = StrideT::value;
  static constexpr bool UseOffset = UseOffsetT::value;

public:
  bool operator()(sycl::queue &queue, size_t offset,
                  const std::string &data_type) {
    assert(NumElems >= NumSelectedElems * Stride + offset &&
           "Offset should be less than number selected elements.");

    bool passed = true;
    size_t alignment_value = alignof(DataT);

    constexpr size_t value_for_increase_ref_data_for_fill = 50;
    static_assert(std::numeric_limits<char>::max() >
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
                    init_ref_ptr, ref_data_for_fill_ptr, out_ptr, offset);
          });
    });
    queue.wait_and_throw();

    // Verify that selected values has been changed to values from referece data
    // for fill.
    for (size_t i = offset; i < Stride * NumSelectedElems + offset;
         i += Stride) {
      DataT expected = ref_data_for_fill[(i - offset) / Stride];
      DataT retrieved = result[i];
      if (expected != retrieved) {
        passed = fail_test(i, expected, retrieved, data_type);
      }
    }
    // Verify that values, that do not was selected has initial values.
    for (size_t i = 0; i < NumElems; ++i) {
      if (i % Stride != 0) {
        DataT expected = initial_ref_data[i];
        DataT retrieved = result[i];
        if (expected != retrieved) {
          passed = fail_test(i, expected, retrieved, data_type);
        }
      }
    }

    if (!is_correct_type.value()) {
      passed = false;
      log::note("Test failed due to type of the object that returns " +
                TestCaseT::get_description() +
                " operator is not equal to the expected one for simd<" +
                data_type + ", " + std::to_string(NumElems) + ">.");
    }

    return passed;
  }

private:
  bool fail_test(size_t i, DataT expected, DataT retrieved,
                 const std::string &data_type) {
    const auto description = TestDescription<DataT, NumElems>(
        i, retrieved, expected, NumSelectedElems, Stride, data_type);
    log::fail(description);

    return false;
  }
};

template <tested_types TestedTypesT>
bool run_test_with_chosen_data_types(sycl::queue &queue) {
  bool passed = true;
  constexpr int desired_simd_small_size = 1;
  constexpr int desired_simd_large_size = 16;
  constexpr int coefficient_of_division = 3;
  constexpr int small_offset_value = 1;
  const int large_offset_value =
      desired_simd_large_size - std::ceil(2 * desired_simd_large_size / 3);

  const auto small_size = get_dimensions<desired_simd_small_size>();
  const auto great_size = get_dimensions<desired_simd_large_size>();
  const auto all_types = get_tested_types<TestedTypesT>();

  passed &=
      for_all_combinations<run_test, select_rval, size<1>, stride<1>,
                           do_not_use_offset>(all_types, small_size, queue, 0);
  passed &=
      for_all_combinations<run_test, select_rval, size<1>, stride<1>,
                           do_not_use_offset>(all_types, small_size, queue, 0);
  passed &= for_all_combinations<
      run_test, select_rval,
      size<desired_simd_large_size / coefficient_of_division>,
      stride<coefficient_of_division>, do_not_use_offset>(all_types, great_size,
                                                          queue, 0);
  passed &= for_all_combinations<
      run_test, select_rval,
      size<desired_simd_large_size / coefficient_of_division>, stride<3>,
      do_not_use_offset>(all_types, great_size, queue, 0);
  passed &= for_all_combinations<
      run_test, select_rval, size<3>,
      stride<desired_simd_large_size / coefficient_of_division>,
      do_not_use_offset>(all_types, great_size, queue, 0);
  passed &=
      for_all_combinations<run_test, select_rval,
                           size<desired_simd_large_size - small_offset_value>,
                           stride<desired_simd_small_size>, use_offset>(
          all_types, great_size, queue, small_offset_value);
  passed &= for_all_combinations<run_test, select_rval,
                                 size<desired_simd_large_size / 3>, stride<2>,
                                 use_offset>(all_types, great_size, queue,
                                             large_offset_value);

  return passed;
}

} // namespace esimd_test::api::functional::functions
