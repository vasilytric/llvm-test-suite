//==------- operator_logical_not.cpp  - DPC++ ESIMD on-device test ---------==//
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
// Test for simd logical not operator.
// The test creates source simd instance with reference data and invokes logical
// not operator.
// The test verifies that data from simd is not corrupted after calling logical
// not operator, that logical not operator return type is as expected and
// logical not operator result values is correct.

#include "common.hpp"

using namespace sycl::ext::intel::experimental::esimd;
using namespace esimd_test::api::functional;

// Descriptor class for the case of calling logical not operator.
struct logical_not_operator {
  static std::string get_description() { return "logical not"; }

  template <typename DataT, int NumElems, typename FlagT>
  static bool call_simd_ctor(const DataT *const ref_data, DataT *const out,
                             FlagT *const logical_not_res_is_correct) {
    auto simd_obj = simd<DataT, NumElems>();
    simd_obj.copy_from(ref_data);
    const auto logical_not_result = !simd_obj;

    for (size_t i = 0; i < NumElems; ++i) {
      *logical_not_res_is_correct &=
          logical_not_result[i] == (ref_data[i] == 0);
    }
    simd_obj.copy_to(out);
    return std::is_same_v<decltype(!simd_obj), simd_mask<NumElems>>;
  }
};

// The main test routine.
// Using functor class to be able to iterate over the pre-defined data types.
template <typename DataT, typename DimT, typename TestCaseT> class run_test {
  static constexpr int NumElems = DimT::value;

public:
  bool operator()(sycl::queue &queue, const std::string &data_type) {
    bool passed = true;
    const std::vector<DataT> ref_data = generate_ref_data<DataT, NumElems>();

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
    shared_vector<DataT> result(NumElems, allocator);
    shared_vector<DataT> shared_ref_data(ref_data.begin(), ref_data.end(),
                                         allocator);
    // TODO replace using std::vector with using shared_element
    shared_vector<int> logical_not_res_is_correct(1,
                                                  shared_allocator<int>(queue));
    shared_vector<int> logical_not_res_type_is_correct(
        1, shared_allocator<int>(queue));

    queue.submit([&](sycl::handler &cgh) {
      const DataT *const ref = shared_ref_data.data();
      DataT *const out = result.data();
      auto logical_not_res_is_correct_storage =
          logical_not_res_is_correct.data();
      *logical_not_res_is_correct_storage = true;
      auto logical_not_res_type_is_correct_storage =
          logical_not_res_type_is_correct.data();

      cgh.single_task<ctors::Kernel<DataT, NumElems, TestCaseT>>(
          [=]() SYCL_ESIMD_KERNEL {
            *logical_not_res_type_is_correct_storage =
                TestCaseT::template call_simd_ctor<DataT, NumElems>(
                    ref, out, logical_not_res_is_correct_storage);
          });
    });
    queue.wait_and_throw();

    for (size_t i = 0; i < result.size(); ++i) {
      if (!are_bitwise_equal(ref_data[i], result[i])) {
        passed = false;

        const auto description =
            ctors::TestDescription<DataT, NumElems, TestCaseT>(
                i, result[i], ref_data[i], data_type);
        log::fail(description);
      }
    }

    if (logical_not_res_is_correct.at(0)) {
      passed = false;
      log::note("Test failed due to some elements from logical not operator is "
                "not equal to \"reference_data == 0 \" for simd<" +
                data_type + ", " + std::to_string(NumElems) + ">.");
    }
    if (logical_not_res_type_is_correct.at(0)) {
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

  const auto u_types = get_tested_types<tested_types::uint>();
  const auto s_types = get_tested_types<tested_types::sint>();
  const auto dims = get_all_dimensions();
  const auto context = unnamed_type_pack<logical_not_operator>::generate();

  passed &= for_all_combinations<run_test>(u_types, dims, context, queue);
  passed &= for_all_combinations<run_test>(s_types, dims, context, queue);

  std::cout << (passed ? "=== Test passed\n" : "=== Test FAILED\n");
  return passed ? 0 : 1;
}
