//==------- ctor_array_core.cpp  - DPC++ ESIMD on-device test --------------==//
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
// TODO Unexpected runtime error "error: unsupported type for load/store" while
// try to use simd::copy_from(), then simd::copy_to() with fixed-size array that
// was defined on device side and the test must be enabled when it is resolved.
//
// Test for simd constructor from an array.
// This test uses different data types, dimensionality and different simd
// constructor invocation contexts.
// The test does the following actions:
//  - construct fixed-size array that filled with reference values
//  - use std::move() to provide it to simd constructor
//  - bitwise compare expected and retrieved values

#include "ctor_array.hpp"

using namespace esimd_test::api::functional;

int main(int, char **) {
  sycl::queue queue(esimd_test::ESIMDSelector{},
                    esimd_test::createExceptionHandler());

  bool passed = true;

  const auto types = get_tested_types<tested_types::all>();
  const auto dims = get_all_dimensions();

  // Run for specific combinations of types, vector length, and invocation
  // contexts.
  passed &= for_all_types_and_dims<ctors::run_test, ctors::initializer>(
      types, dims, queue);
  passed &= for_all_types_and_dims<ctors::run_test, ctors::var_decl>(
      types, dims, queue);
  passed &= for_all_types_and_dims<ctors::run_test, ctors::rval_in_expr>(
      types, dims, queue);
  passed &= for_all_types_and_dims<ctors::run_test, ctors::const_ref>(
      types, dims, queue);

  std::cout << (passed ? "=== Test passed\n" : "=== Test FAILED\n");
  return passed ? 0 : 1;
}
