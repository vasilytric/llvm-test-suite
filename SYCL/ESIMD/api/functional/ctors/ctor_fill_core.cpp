//==------- ctor_fill_core.cpp  - DPC++ ESIMD on-device test ---------------==//
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
// Test for esimd fill constructor for core types.

#include "ctor_fill.hpp"

using namespace sycl::ext::intel::experimental::esimd;
using namespace esimd_test::api::functional::ctors;
using namespace esimd_test::api::functional;

int main(int argc, char **argv) {
  sycl::queue queue{};

  bool pass{true};
  const auto one_and_eighth_dims = values_pack<1, 8>();
  auto char_int_types{named_type_pack<char, int>({"char", "int"})};
  pass &= run_verifying<initializer, init_val::min_half, init_val::zero>(
      queue, one_and_eighth_dims, char_int_types);

  pass &= run_verifying<initializer, init_val::zero, init_val::positive>(
      queue, one_and_eighth_dims, char_int_types);
  pass &= run_verifying<initializer, init_val::min_half, init_val::positive>(
      queue, one_and_eighth_dims, char_int_types);

  pass &= run_verifying<var_dec, init_val::min_half, init_val::zero>(
      queue, one_and_eighth_dims, char_int_types);
  pass &= run_verifying<var_dec, init_val::zero, init_val::positive>(
      queue, one_and_eighth_dims, char_int_types);
  pass &= run_verifying<var_dec, init_val::min_half, init_val::positive>(
      queue, one_and_eighth_dims, char_int_types);

  pass &= run_verifying<rval_in_express, init_val::min_half, init_val::zero>(
      queue, one_and_eighth_dims, char_int_types);
  pass &= run_verifying<rval_in_express, init_val::zero, init_val::positive>(
      queue, one_and_eighth_dims, char_int_types);
  pass &=
      run_verifying<rval_in_express, init_val::min_half, init_val::positive>(
          queue, one_and_eighth_dims, char_int_types);

  pass &= run_verifying<const_ref, init_val::min_half, init_val::zero>(
      queue, one_and_eighth_dims, char_int_types);
  pass &= run_verifying<const_ref, init_val::zero, init_val::positive>(
      queue, one_and_eighth_dims, char_int_types);
  pass &= run_verifying<const_ref, init_val::min_half, init_val::positive>(
      queue, one_and_eighth_dims, char_int_types);

  const auto all_dims = values_pack<1, 8, 16, 32>();
  auto all_types{get_tested_types<tested_types::all>()};
  pass &= run_verifying<var_dec, init_val::min, init_val::zero>(queue, all_dims,
                                                                all_types);
  pass &= run_verifying<var_dec, init_val::max_half, init_val::zero>(
      queue, all_dims, all_types);
  pass &= run_verifying<var_dec, init_val::zero, init_val::positive>(
      queue, all_dims, all_types);
  pass &= run_verifying<var_dec, init_val::max_half, init_val::positive>(
      queue, all_dims, all_types);
  pass &= run_verifying<var_dec, init_val::zero, init_val::negative>(
      queue, all_dims, all_types);
  pass &= run_verifying<var_dec, init_val::max_half, init_val::negative>(
      queue, all_dims, all_types);

  const auto single_dim = values_pack<8>();
  auto uint_types{get_tested_types<tested_types::uint>()};
  pass &= run_verifying<var_dec, init_val::min, init_val::positive>(
      queue, single_dim, uint_types);
  pass &= run_verifying<var_dec, init_val::min, init_val::negative>(
      queue, single_dim, uint_types);
  pass &= run_verifying<var_dec, init_val::max, init_val::positive>(
      queue, single_dim, uint_types);
  pass &= run_verifying<var_dec, init_val::max, init_val::negative>(
      queue, single_dim, uint_types);
  auto sint_types{get_tested_types<tested_types::sint>()};
  pass &= run_verifying<var_dec, init_val::min, init_val::positive>(
      queue, single_dim, uint_types);
  pass &= run_verifying<var_dec, init_val::max, init_val::negative>(
      queue, single_dim, uint_types);

  auto fp_types{get_tested_types<tested_types::fp>()};
  pass &= run_verifying<var_dec, init_val::neg_inf, init_val::max>(
      queue, single_dim, fp_types);
  pass &= run_verifying<var_dec, init_val::max, init_val::neg_inf>(
      queue, single_dim, fp_types);
  pass &= run_verifying<var_dec, init_val::nan, init_val::negative>(
      queue, single_dim, fp_types);
  pass &= run_verifying<var_dec, init_val::negative, init_val::nan>(
      queue, single_dim, fp_types);

  std::cout << (pass ? "=== Test passed\n" : "=== Test FAILED\n");
  return pass ? 0 : 1;
}
