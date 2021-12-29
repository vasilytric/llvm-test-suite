//==------- ctor_fill_accuracy_fp_extra.cpp  - DPC++ ESIMD on-device test --==//
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
// XRUN: %clangxx -fsycl %s -fsycl-device-code-split=per_kernel -o %t.out
// XRUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: false
// XFAIL: *
// TODO The simd can't be constructed with sycl::half data type. The issue was
// created (https://github.com/intel/llvm/issues/5077) and the test must be
// enabled when it is resolved.
//
// The test verifies that simd fill constructor has no precision differences.
// The test do the following actions:
//  - call simd with predefined base and step values
//  - bitwise comparing that output[0] value is equal to base value and
//    output[i] is equal to output[i -1] + step_value

#include "ctor_fill.hpp"

using namespace sycl::ext::intel::experimental::esimd;
using namespace esimd_test::api::functional;

int main(int, char **) {
  sycl::queue queue(esimd_test::ESIMDSelector{},
                    esimd_test::createExceptionHandler());

  bool passed = true;

  const auto fp_types = get_tested_types<tested_types::fp_extra>();
  const auto single_dim = values_pack<8>();

  // Run for specific combinations of types, base and step values and vector
  // length.
  // The first init_val value it's a base value and the second init_val value
  // it's a step value.
  passed &= ctors::run_verification<ctors::var_dec, ctors::init_val::denorm,
                                    ctors::init_val::ulp>(queue, single_dim,
                                                          fp_types);
  passed &= ctors::run_verification<ctors::var_dec, ctors::init_val::inexact,
                                    ctors::init_val::ulp>(queue, single_dim,
                                                          fp_types);
  passed &= ctors::run_verification<ctors::var_dec, ctors::init_val::min,
                                    ctors::init_val::ulp>(queue, single_dim,
                                                          fp_types);

  passed &= ctors::run_verification<ctors::var_dec, ctors::init_val::inexact,
                                    ctors::init_val::ulp_half>(
      queue, single_dim, fp_types);
  passed &= ctors::run_verification<ctors::var_dec, ctors::init_val::min,
                                    ctors::init_val::ulp_half>(
      queue, single_dim, fp_types);

  std::cout << (passed ? "=== Test passed\n" : "=== Test FAILED\n");
  return passed ? 0 : 1;
}
