//===-- operator_assignment.hpp - Functions for tests on simd assignment
//      operators. --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file provides functios for tests on simd assignment operators.
///
//===----------------------------------------------------------------------===//

#pragma once

// The test proxy is used to verify the move constructor was called actually.
#define __ESIMD_ENABLE_TEST_PROXY

#include "common.hpp"

namespace esimd_test::api::functional::ctors {

// The main test routine.
// Using functor class to be able to iterate over the pre-defined data types.
template <typename DataT, int NumElems, typename TestCaseT,
          typename ShouldMoveCtorBeCalled>
class run_test {
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

    shared_vector<int> was_moved(1, shared_allocator<int>(queue));

    queue.submit([&](sycl::handler &cgh) {
      const DataT *const ref = shared_ref_data.data();
      DataT *const out = result.data();
      const auto was_moved_storage = was_moved.data();

      cgh.single_task<ctors::Kernel<DataT, NumElems, TestCaseT>>(
          [=]() SYCL_ESIMD_KERNEL {
            *was_moved_storage =
                TestCaseT::template call_simd_ctor<DataT, NumElems>(ref, out);
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

    if (was_moved.at(0) != ShouldMoveCtorBeCalled::value) {
      passed = false;
      log::note("Test failed due to move assignment operator hasn't called.");
    }

    return passed;
  }
};

} // namespace esimd_test::api::functional::ctors
