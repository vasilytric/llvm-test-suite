//===-- functions_select_2d.hpp - Functions for tests on simd rvalue select
//      function. ---------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file provides common code for tests on simd 2d select function.
///
//===----------------------------------------------------------------------===//

#pragma once

#include "../shared_element.hpp"
#include "common.hpp"

namespace esimd = sycl::ext::intel::esimd;

namespace esimd_test::api::functional::functions {

// Descriptor class for the case of calling simd<T,N>::select function.
struct select_2d {
  static std::string get_description() { return "2d select"; }

  template <typename DataT, int NumElems, int SizeY, int StrideY, int SizeX,
            int StrideX, int Height, int Width>
  static void call_esimd_function(const DataT *const ref_1,
                                  const DataT *const ref_2, DataT *const out,
                                  size_t offset_x, size_t offset_y) {
    esimd::simd<DataT, NumElems> src_simd_obj;
    src_simd_obj.copy_from(ref_1);

    auto simd_view_instance =
        src_simd_obj.template bit_cast_view<DataT, Height, Width>();
    auto selected_elems =
        simd_view_instance.template select<SizeY, StrideY, SizeX, StrideX>(
            offset_y, offset_x);

    size_t ref_2_index = 0;
    for (int i = 0; i < SizeY; ++i) {
      for (int j = 0; j < SizeX; ++j) {
        auto a = selected_elems.row(i);
        a[j] = ref_2[ref_2_index];
        ++ref_2_index;
      }
    }

    src_simd_obj.copy_to(out);
  }
};

// The main test routine.
// Using functor class to be able to iterate over the pre-defined data types.
template <typename TestCaseT, typename SizeXT, typename SizeYT,
          typename StrideXT, typename StrideYT, typename OffsetXT,
          typename OffsetYT, typename SimdSplitCoeffT, typename DataT,
          typename DimT>
class run_test {
  static constexpr int NumElems = DimT::value;
  static constexpr int SizeY = SizeYT::value;
  static constexpr int StrideY = StrideYT::value;
  static constexpr int SizeX = SizeXT::value;
  static constexpr int StrideX = StrideXT::value;
  static constexpr int OffsetX = OffsetXT::value;
  static constexpr int OffsetY = OffsetYT::value;
  static constexpr int Height = NumElems / SimdSplitCoeffT::value;
  static constexpr int Width = NumElems / Height;

public:
  bool operator()(sycl::queue &queue, const std::string &data_type) {
    bool passed = true;

    static_assert(NumElems == Height * Width,
                  "Invalid SimdSplitCoeff, height and width multiplacation "
                  "should be equal to simd size.");
    static_assert(Width > SizeX, "Number selected elements can't be greater "
                                 "than number elements in one line");
    static_assert(
        Height > SizeY,
        "Number selected elements can't be greater than lines number");
    assert(Height > OffsetY &&
           "Offset of height should be less than simd_view height size");
    assert(Width > OffsetX &&
           "Offset of with should be less than simd_view with size");

    constexpr size_t value_for_increase_ref_data_for_fill = 50;
    static_assert(std::numeric_limits<char>::max() >
                      value_for_increase_ref_data_for_fill + NumElems,
                  "Value that used for increase ref data for fill plus simd "
                  "size should be less than char max value.");

    shared_allocator<DataT> allocator(queue);
    shared_vector<DataT> result(NumElems, allocator);
    shared_vector<DataT> initial_ref_data(NumElems, allocator);
    shared_vector<DataT> ref_data_for_fill(SizeY * SizeX, allocator);

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
      cgh.single_task<
          Kernel<DataT, NumElems, TestCaseT, SizeYT, StrideYT, SizeXT, StrideXT,
                 SimdSplitCoeffT, OffsetXT, OffsetYT>>([=]() SYCL_ESIMD_KERNEL {
        TestCaseT::template call_esimd_function<DataT, NumElems, SizeY, StrideY,
                                                SizeX, StrideX, Height, Width>(
            init_ref_ptr, ref_data_for_fill_ptr, out_ptr, OffsetX, OffsetY);
      });
    });
    queue.wait_and_throw();

    std::vector<size_t> selected_indexsess;
    // Collect the indexess that has been selected.
    for (int i = 0; i < SizeY; ++i) {
      for (int j = 0; j < SizeX; ++j) {
        selected_indexsess.push_back(OffsetY * Width + i * Width * StrideY +
                                     j * StrideX + OffsetX);
      }
    }

    // Push the largest value to avoid the following error: can't dereference
    // out of range vector iterator.
    selected_indexsess.push_back(std::numeric_limits<size_t>::max());
    auto selected_indexsess_ptr = selected_indexsess.begin();

    for (int i = 0; i < NumElems; ++i) {
      // If current index is less than selected index verify that this element
      // hasn't been selected and changed.
      if (i < *selected_indexsess_ptr) {
        DataT retrieved = result[i];
        DataT expected = initial_ref_data[i];
        if (expected != retrieved) {
          passed = fail_test(i, expected, retrieved, data_type);
        }
      } else {
        DataT retrieved = result[i];
        DataT expected = ref_data_for_fill.front();
        ref_data_for_fill.erase(ref_data_for_fill.begin());

        if (expected != retrieved) {
          passed = fail_test(i, expected, retrieved, data_type);
        }
        selected_indexsess_ptr++;
      }
    }

    return passed;
  }

private:
  bool fail_test(size_t i, DataT expected, DataT retrieved,
                 const std::string &data_type) {
    log::fail(TestDescription<NumElems, TestCaseT>(data_type),
              "Unexpected value at index ", i, ", retrieved: ", retrieved,
              ", expected: ", expected, ", with size x: ", SizeX,
              ", stride x: ", StrideX, ", size y: ", SizeY,
              ", stride y: ", StrideY, ", height: ", Height,
              ", offset x: ", OffsetX, ", offset y: ", OffsetY,
              ", width: ", Width);

    return false;
  }
};

// Aliases to provide size or stride values to test.
// This is the syntax sugar just for code readability.
template <int N> using stride_type = std::integral_constant<int, N>;
template <int N> using size_type = std::integral_constant<int, N>;
template <int N> using offset_type = std::integral_constant<int, N>;
template <int N> using simd_split_coeff = std::integral_constant<int, N>;

template <typename TestCaseT, int SizeX, int SizeY, int StrideX, int StrideY,
          int OffsetX, int OffsetY, int SimdSplitCoeff, typename... ArgsT>
bool run_with_size_stride_offset(ArgsT &&...args) {
  bool passed = for_all_combinations<
      run_test, select_2d, size_type<SizeX>, size_type<SizeY>,
      stride_type<StrideX>, stride_type<StrideY>, offset_type<OffsetX>,
      offset_type<OffsetY>, simd_split_coeff<SimdSplitCoeff>>(
      std::forward<ArgsT>(args)...);

  return passed;
}

// Helping function that lets launch test with provided tested_types.
// IMPORTANT: do not set offset value that greater than expected number of
// strings or element in one string in simd_view.
template <tested_types TestedTypesT>
bool run_test_with_chosen_data_types(sycl::queue &queue) {
  bool passed = true;
  constexpr int desired_simd_large_size = 32;
  constexpr int coefficient_of_division = 3;
  constexpr int default_simd_split_value = 4;
  constexpr int expected_height =
      desired_simd_large_size / default_simd_split_value;
  constexpr int expected_width = desired_simd_large_size / expected_height;
  const int heights_large_offset =
      expected_height - round_up_int_division(2 * expected_height, 3);
  const int width_large_offset =
      expected_width - round_up_int_division(2 * expected_width, 3);

  const auto great_size = get_dimensions<desired_simd_large_size>();
#ifdef SIMD_RUN_TEST_WITH_SYCL_HALF_TYPE
  const auto all_types = get_tested_types<TestedTypesT>();
#else
  const auto all_types = named_type_pack<double>::generate("double");
#endif

  // Verify correctness for different select sizes.
  passed &= run_with_size_stride_offset<select_2d, 1, 1, 1, 1, 1, 1,
                                        default_simd_split_value>(
      all_types, great_size, queue);
  passed &= run_with_size_stride_offset<select_2d, 1, 1, 1, 1, 0, 1,
                                        default_simd_split_value>(
      all_types, great_size, queue);
  passed &= run_with_size_stride_offset<select_2d, 1, 1, 1, 1, 1, 0,
                                        default_simd_split_value>(
      all_types, great_size, queue);
  passed &= run_with_size_stride_offset<select_2d, 1, 1, 1, 1, 0, 0,
                                        default_simd_split_value>(
      all_types, great_size, queue);

  passed &=
      run_with_size_stride_offset<select_2d, 1, 1, 1, 1, expected_width - 1, 0,
                                  default_simd_split_value>(all_types,
                                                            great_size, queue);
  passed &=
      run_with_size_stride_offset<select_2d, 1, 1, 1, 1, 0, expected_height - 1,
                                  default_simd_split_value>(all_types,
                                                            great_size, queue);
  passed &= run_with_size_stride_offset<select_2d, 1, 1, 1, 1,
                                        expected_width - 1, expected_height - 1,
                                        default_simd_split_value>(
      all_types, great_size, queue);

  passed &= run_with_size_stride_offset<
      select_2d, expected_height / coefficient_of_division, 1,
      coefficient_of_division, 1, 0, 0, default_simd_split_value>(
      all_types, great_size, queue);
  passed &= run_with_size_stride_offset<
      select_2d, 1, expected_width / coefficient_of_division, 1,
      coefficient_of_division, 0, 0, default_simd_split_value>(
      all_types, great_size, queue);
  passed &= run_with_size_stride_offset<
      select_2d, expected_height / coefficient_of_division,
      expected_width / coefficient_of_division, coefficient_of_division,
      coefficient_of_division, 0, 0, default_simd_split_value>(
      all_types, great_size, queue);

  // Verify correctness with small offset values.
  passed &= run_with_size_stride_offset<select_2d, expected_width - 1, 1, 1, 1,
                                        1, 0, default_simd_split_value>(
      all_types, great_size, queue);
  passed &= run_with_size_stride_offset<select_2d, 1, expected_height - 1, 1, 1,
                                        0, 1, default_simd_split_value>(
      all_types, great_size, queue);
  passed &= run_with_size_stride_offset<select_2d, expected_width - 1,
                                        expected_height - 1, 1, 1, 1, 1,
                                        default_simd_split_value>(
      all_types, great_size, queue);

  // Verify correctness with large offset values.
  passed &= run_with_size_stride_offset<
      select_2d, expected_width / coefficient_of_division, 1, 2, 1,
      width_large_offset, 0, default_simd_split_value>(all_types, great_size,
                                                       queue);
  passed &= run_with_size_stride_offset<
      select_2d, 1, expected_height / coefficient_of_division, 1, 1, 0,
      heights_large_offset, default_simd_split_value>(all_types, great_size,
                                                      queue);
  passed &= run_with_size_stride_offset<
      select_2d, expected_width / coefficient_of_division,
      expected_height / coefficient_of_division, 1, 1, width_large_offset,
      heights_large_offset, default_simd_split_value>(all_types, great_size,
                                                      queue);

  return passed;
}

} // namespace esimd_test::api::functional::functions
