//==------- ctor_fill_accuracy.cpp  - DPC++ ESIMD on-device test -----------==//
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
// TODO simd<float, 32> fills with unexpected values while base value is denorm
// and step is ulp. The test must be enabled when this problem will be resolved.
//
// The test verifies that simd fill constructor has no precision differences.
// The test do the following actions:
//  - call simd with predefined base and step values
//  - bitwise comparing that output[0] value is equal to base value and
//    output[i] is equal to output[i -1] + step_value

#include "ctor_fill.hpp"

using namespace sycl::ext::intel::experimental::esimd;
using namespace esimd_test::api::functional::ctors;
using namespace esimd_test::api::functional;

int main(int, char **) {
  sycl::queue queue(esimd_test::ESIMDSelector{},
                    esimd_test::createExceptionHandler());

  bool passed = true;

  const auto fp_types = get_tested_types<tested_types::fp>();
  const auto single_dim = values_pack<8>();

  passed &= run_verification<var_dec, init_val::denorm, init_val::ulp>(
      queue, single_dim, fp_types);
  passed &= run_verification<var_dec, init_val::inexact, init_val::ulp>(
      queue, single_dim, fp_types);
  passed &= run_verification<var_dec, init_val::min, init_val::ulp>(
      queue, single_dim, fp_types);

  passed &= run_verification<var_dec, init_val::inexact, init_val::ulp_half>(
      queue, single_dim, fp_types);
  passed &= run_verification<var_dec, init_val::min, init_val::ulp_half>(
      queue, single_dim, fp_types);

  std::cout << (passed ? "=== Test passed\n" : "=== Test FAILED\n");
  return passed ? 0 : 1;
}