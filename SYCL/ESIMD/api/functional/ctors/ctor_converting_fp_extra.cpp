//==------- ctor_converting_fp_extra.cpp  - DPC++ ESIMD on-device test -----==//
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
// TODO Unexpected static_assert was retrieved while calling simd::copy_from()
// function. The issue was created (https://github.com/intel/llvm/issues/5112)
// and the test must be enabled when it is resolved.
//
// Test for simd converting constructor for extra fp types.
// This test uses extra fp data types with different dimensionality, base and
// step values and different simd constructor invocation contexts.
// The test do the following actions:
//  - construct simd with source data type, then construct simd with destination
//    type from the earlier constructed simd
//  - compare retrieved and expected values

#include "ctor_converting.hpp"

using namespace sycl::ext::intel::experimental::esimd;
using namespace esimd_test::api::functional;

int main(int, char **) {
  sycl::queue queue(esimd_test::ESIMDSelector{},
                    esimd_test::createExceptionHandler());

  bool passed = true;

  const auto types = get_tested_types<tested_types::fp_extra>();
  const auto single_dim = values_pack<8>();

  // Run for specific combinations of types, vector length, base and step values
  // and invocation contexts.
  ctors::run_test<float, 8, double, ctors::initializer>{}(queue, "float",
                                                          "double");

  std::cout << (passed ? "=== Test passed\n" : "=== Test FAILED\n");
  return passed ? 0 : 1;
}
