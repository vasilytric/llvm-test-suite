//==------- ctor_fill_accuracy.cpp  - DPC++ ESIMD on-device test
//---------------==//
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
// The test verifies that simd fill constructor has no precision differences.
// The test do the following actions:
//  - call simd with predefined base and step values
//  - bitwise comparing that output[0] value is equal to base value and
//    output[i] is equal to output[i -1]

#include "common.hpp"
// For std::find
#include <algorithm>

using namespace sycl::ext::intel::experimental::esimd;
using namespace esimd_test::api::functional::ctors;
using namespace esimd_test::api::functional;

struct context_description {
  static std::string get_description() { return "variable declaration"; }
};

enum class init_val { denorm, inexact, min, ulp, ulp_half };

template <typename T> T get_value(init_val InitVal, T base_val = T()) {
  if (InitVal == init_val::denorm) {
    return value<T>::denorm_min();
  } else if (InitVal == init_val::inexact) {
    return 0.1;
  } else if (InitVal == init_val::min) {
    return value<T>::lowest();
  } else if (InitVal == init_val::ulp || InitVal == init_val::ulp_half) {
    T next_step_val{};

    if constexpr (std::is_same_v<T, sycl::half>) {
      next_step_val = static_cast<sycl::half>(
          (sycl::nextafter(base_val, sycl::half(1.f)) - base_val) * 8192);
    } else {
      next_step_val = std::nextafter(base_val, T(1.f)) - base_val;
    }
    if (InitVal == init_val::ulp_half) {
      next_step_val = next_step_val / 2;
    }

    return next_step_val;
  } else {
    assert(InitVal != InitVal && "Unexpected init value type was received.");
  }
}

template <typename, int> class kernel_1;

// The main test routine.
// Using functor class to be able to iterate over the pre-defined data types.
template <typename DataT, int NumElems> struct run_test {
  bool operator()(sycl::queue &queue, init_val BaseVal, init_val Step,
                  const std::string &data_type) {
    bool passed = true;
    DataT base_value{get_value<DataT>(BaseVal)};
    DataT step_value{get_value<DataT>(Step, base_value)};

    shared_vector<DataT> result(NumElems, shared_allocator<DataT>(queue));

    queue.submit([&](sycl::handler &cgh) {
      DataT *const out = result.data();

      cgh.single_task<kernel_1<DataT, NumElems>>([=]() SYCL_ESIMD_KERNEL {
        simd<DataT, NumElems> simd_obj(base_value, step_value);
        simd_obj.copy_to(out);
      });
    });
    queue.wait_and_throw();

    DataT expected_value{base_value};
    for (size_t i = 0; i < result.size(); ++i) {
      if (!are_bitwise_equal(result[i], expected_value)) {
        passed = false;

        const auto description =
            ctors::TestDescription<DataT, NumElems, context_description>(
                i, result[i], expected_value, data_type);
        log::fail(description);
      }
      expected_value += step_value;
    }

    return passed;
  }
};

int main(int, char **) {
  sycl::queue queue(esimd_test::ESIMDSelector{},
                    esimd_test::createExceptionHandler());

  bool passed = true;

  auto fp_types{get_tested_types<tested_types::fp>()};
  const auto single_dim = values_pack<8>();

  const std::vector<sycl::info::fp_config> dev_info =
      queue.get_device().get_info<sycl::info::device::single_fp_config>();

  if (std::find(dev_info.begin(), dev_info.end(),
                sycl::info::fp_config::denorm) != dev_info.end()) {
    passed &= for_all_types_and_dims<run_test>(fp_types, single_dim, queue,
                                               init_val::denorm, init_val::ulp);
  } else {
    log::note("The test for denorm as base value and ulp as step value was "
              "skipped due to denorms is not supported on current device.");
  }
  passed &= for_all_types_and_dims<run_test>(fp_types, single_dim, queue,
                                             init_val::inexact, init_val::ulp);
  passed &= for_all_types_and_dims<run_test>(fp_types, single_dim, queue,
                                             init_val::min, init_val::ulp);

  passed &= for_all_types_and_dims<run_test>(
      fp_types, single_dim, queue, init_val::inexact, init_val::ulp_half);
  passed &= for_all_types_and_dims<run_test>(fp_types, single_dim, queue,
                                             init_val::min, init_val::ulp_half);

  std::cout << (passed ? "=== Test passed\n" : "=== Test FAILED\n");
  return passed ? 0 : 1;
}
