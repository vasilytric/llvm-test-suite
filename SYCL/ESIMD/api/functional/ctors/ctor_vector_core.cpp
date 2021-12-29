//==------- ctor_vector_core.cpp  - DPC++ ESIMD on-device test -------------==//
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
// Test for simd constructor from vector.
// This test uses different data types, dimensionality and different simd
// constructor invocation contexts.
// The test do the following actions:
//  - call init_simd.data() to retreive vector_type and then provide it to the
//    simd constructor
//  - bitwise comparing expected and retrieved values

#include "ctor_vector.hpp"

using namespace sycl::ext::intel::experimental::esimd;
using namespace esimd_test::api::functional;

int main(int, char **) {
  sycl::queue queue(esimd_test::ESIMDSelector{},
                    esimd_test::createExceptionHandler());

  bool passed = true;

  const auto types = get_tested_types<tested_types::all>();
  const auto dims = get_all_dimensions();

  // Run for specific combinations of types, vector length and invocation
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
