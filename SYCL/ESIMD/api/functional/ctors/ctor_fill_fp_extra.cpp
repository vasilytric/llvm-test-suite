//==------- ctor_fill_fp_extra.cpp  - DPC++ ESIMD on-device test -----------==//
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
// created (https://github.com/intel/llvm/issues/5077) and the this test must be
// enabled when it is resolved.
//
// Test for esimd fill constructor for extra fp types.

#include "ctor_fill.hpp"

using namespace sycl::ext::intel::experimental::esimd;
using namespace esimd_test::api::functional::ctors;
using namespace esimd_test::api::functional;

int main(int argc, char **argv) {
  sycl::queue queue(esimd_test::ESIMDSelector{},
                    esimd_test::createExceptionHandler());

  bool passed = true;

  const auto fp_types = get_tested_types<tested_types::fp_extra>();
  const auto single_dim = values_pack<8>();

  passed &= run_verification<var_dec, init_val::neg_inf, init_val::zero>(
      queue, single_dim, fp_types);
  passed &= run_verification<var_dec, init_val::max, init_val::neg_inf>(
      queue, single_dim, fp_types);
  passed &= run_verification<var_dec, init_val::nan, init_val::negative>(
      queue, single_dim, fp_types);
  passed &= run_verification<var_dec, init_val::zero, init_val::nan>(
      queue, single_dim, fp_types);

  std::cout << (passed ? "=== Test passed\n" : "=== Test FAILED\n");
  return passed ? 0 : 1;
}
