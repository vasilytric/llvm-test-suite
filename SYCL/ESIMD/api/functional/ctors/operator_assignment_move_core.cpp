//==------- operator_assignment_move_core.cpp  - DPC++ ESIMD on-device test
//          ----------------------------------------------------------------==//
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
// RUN: %clangxx -fsycl %s -fsycl-device-code-split=per_kernel -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// XFAIL: *
// TODO Remove XFAIL once the simd vector provides move assignment operator
//
// Test for simd move assignment operator.
// The test creates source simd instance with reference data and invokes move
// assignment operator. It is expected that simd object that called move
// assignment operator will stor a bitwise same data as in the source simd
// object.

// The test proxy is used to verify the move constructor was called actually.
#define __ESIMD_ENABLE_TEST_PROXY

#include "common.hpp"

using namespace sycl::ext::intel::experimental::esimd;
using namespace esimd_test::api::functional;

// Descriptor class for the case of calling constructor in initializer context.
struct move_assignment {
  static std::string get_description() { return "move assignment operator"; }

  template <typename DataT, int NumElems>
  static bool call_simd_ctor(const DataT *const ref_data, DataT *const out) {
    simd<DataT, NumElems> source_simd;
    source_simd.copy_from(ref_data);
    simd<DataT, NumElems> simd_obj;
    simd_obj = std::move(source_simd);
    simd_obj.copy_to(out);
    return simd_obj.get_test_proxy().was_move_destination();
  }
};

// The main test routine.
// Using functor class to be able to iterate over the pre-defined data types.
template <typename DataT, int NumElems, typename TestCaseT> class run_test {
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

    if (!was_moved.at(0)) {
      passed = false;
      log::note("Test failed due to move assignment operator hasn't called.");
    }

    return passed;
  }
};

int main(int, char **) {
  sycl::queue queue(esimd_test::ESIMDSelector{},
                    esimd_test::createExceptionHandler());

  bool passed = true;

  const auto types = get_tested_types<tested_types::all>();
  const auto dims = get_all_dimensions();

  passed &=
      for_all_types_and_dims<run_test, move_assignment>(types, dims, queue);

  std::cout << (passed ? "=== Test passed\n" : "=== Test FAILED\n");
  return passed ? 0 : 1;
}
