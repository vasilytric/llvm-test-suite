//==------- operator_bitwise_not.cpp  - DPC++ ESIMD on-device test ---------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: gpu, level_zero
// XREQUIRES: gpu
// TODO gpu and level_zero in REQUIRES due to only this platforms supported yet.
// The current "REQUIRES" should be replaced with "gpu" only as mentioned in
// "XREQUIRES".
// UNSUPPORTED: cuda, hip
// RUN: %clangxx -fsycl %s -fsycl-device-code-split=per_kernel -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
//
// Test for simd bitwise not operator.
// The test creates source simd instance with reference data and invokes bitwise
// not operator.
// The test verifies that data from simd is not corrupted after calling bitwise
// not operator, that bitwise not operator return type is as expected and
// bitwise not operator result values is correct.

#include "../mutator.hpp"
#include "../shared_element.hpp"
#include "common.hpp"
// For std::abs
#include <cmath>

using namespace sycl::ext::intel::experimental::esimd;
using namespace esimd_test::api::functional;

// Descriptor class for the case of calling bitwise not operator.
struct bitwise_not_operator {
  static std::string get_description() { return "bitwise not"; }

  template <typename DataT, int NumElems>
  static bool call_operator(const DataT *const ref_data,
                            DataT *const source_simd_result,
                            DataT *const operator_result) {
    auto simd_obj = simd<DataT, NumElems>();
    simd_obj.copy_from(ref_data);
    auto bitwise_not_result = ~simd_obj;
    simd_obj.copy_to(source_simd_result);
    bitwise_not_result.copy_to(operator_result);
    return std::is_same_v<decltype(~simd_obj), simd<DataT, NumElems>>;
  }
};

// Replace specific reference values to lower once.
template <typename T> struct For_bitwise_not {
  For_bitwise_not() = default;

  void operator()(T &val) {
    static_assert(!type_traits::is_sycl_floating_point_v<T>,
                  "Invalid data type.");
    if constexpr (std::is_signed_v<T>) {
      // We could have UB for negative zero in different integral type
      // representations: two's complement, ones' complement and signed
      // magnitude.
      // See http://www.open-std.org/jtc1/sc22/wg14/www/docs/n2218.htm
      // Note that there is no check for UB with padding bits here, as it would
      // effectively disable any possible check for signed integer bitwise
      // operations.
      static const T max = value<T>::max();
      static const T lowest = value<T>::lowest();
      if (std::abs(lowest + 1) == (max - 1)) {
        // C11 standard mentions that it's possible to have a `0b100...0` value
        // as a trap value for twos' complement representation. In such case the
        // condition above would trigger for twos' complement representation
        // also.
        if (val == max) {
          // Would result in trap representation for signed magnitude
          val -= 1;
        } else if (val == 0) {
          // Would result in trap representation for ones' complement
          val = 1;
        }
      }
    } //  signed integral types
  }
};

// The main test routine.
// Using functor class to be able to iterate over the pre-defined data types.
template <typename TestCaseT, typename DataT, typename DimT> class run_test {
  static constexpr int NumElems = DimT::value;

public:
  bool operator()(sycl::queue &queue, const std::string &data_type) {
    bool passed = true;
    std::vector<DataT> ref_data = generate_ref_data<DataT, NumElems>();

    mutate(ref_data, For_bitwise_not<DataT>());

    // If current number of elements is equal to one, then run test with each
    // one value from reference data.
    // If current number of elements is greater than one, then run tests with
    // whole reference data.
    if constexpr (NumElems == 1) {
      for (size_t i = 0; i < ref_data.size(); ++i) {
        passed = run_verification(queue, {ref_data[i]}, data_type);
      }
    } else {
      passed = run_verification(queue, ref_data, data_type);
    }
    return passed;
  }

private:
  bool run_verification(sycl::queue &queue, const std::vector<DataT> &ref_data,
                        const std::string &data_type) {
    assert(ref_data.size() == NumElems &&
           "Reference data size is not equal to the simd vector length.");

    bool passed = true;

    shared_allocator<DataT> allocator(queue);
    shared_vector<DataT> shared_ref_data(ref_data.begin(), ref_data.end(),
                                         allocator);
    shared_vector<DataT> source_simd_result(NumElems, allocator);
    shared_vector<DataT> operator_result(NumElems, allocator);

    shared_element<bool> is_correct_type(queue, true);

    queue.submit([&](sycl::handler &cgh) {
      const DataT *const ref = shared_ref_data.data();
      DataT *const source_simd_result_data_ptr = source_simd_result.data();
      DataT *const operator_result_data_ptr = operator_result.data();
      auto is_correct_type_ptr = is_correct_type.data();

      cgh.single_task<Kernel<DataT, NumElems, TestCaseT>>(
          [=]() SYCL_ESIMD_KERNEL {
            *is_correct_type_ptr =
                TestCaseT::template call_operator<DataT, NumElems>(
                    ref, source_simd_result_data_ptr, operator_result_data_ptr);
          });
    });
    queue.wait_and_throw();

    for (size_t i = 0; i < NumElems; ++i) {
      if (!are_bitwise_equal(ref_data[i], source_simd_result[i])) {
        passed = false;

        const auto description = operators::TestDescription<DataT, NumElems>(
            i, source_simd_result[i], ref_data[i], data_type);
        log::fail(description);
      }

      DataT retrieved = operator_result[i];
      DataT expected = ~shared_ref_data[i];
      if (!are_bitwise_equal(expected, retrieved)) {
        passed = false;
        const auto description = operators::TestDescription<DataT, NumElems>(
            i, retrieved, expected, data_type);
        log::fail(description);
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
};

int main(int, char **) {
  sycl::queue queue(esimd_test::ESIMDSelector{},
                    esimd_test::createExceptionHandler());

  bool passed = true;

  const auto uint_types = get_tested_types<tested_types::uint>();
  const auto sint_types = get_tested_types<tested_types::sint>();
  const auto all_dims = get_all_dimensions();

  passed &= for_all_combinations<run_test, bitwise_not_operator>(
      uint_types, all_dims, queue);
  passed &= for_all_combinations<run_test, bitwise_not_operator>(
      sint_types, all_dims, queue);

  std::cout << (passed ? "=== Test passed\n" : "=== Test FAILED\n");
  return passed ? 0 : 1;
}
