//===-- ctor_fill.hpp - Functions for tests on simd fill constructor
//      definition. -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file provides functions for tests on simd fill constructor.
///
//===----------------------------------------------------------------------===//

#pragma once

#include "common.hpp"
// For std::isnan
#include <cmath>

namespace esimd = sycl::ext::intel::experimental::esimd;
namespace esimd_tests = esimd_test::api::functional::ctors;
namespace esimd_functional = esimd_test::api::functional;

// Descriptor class for the case of calling constructor in initializer context.
struct initializer {
  static std::string get_description() { return "initializer"; }

  template <typename DataT, int NumElems>
  static void call_simd_ctor(DataT init_value, DataT step, DataT *out) {
    esimd::simd<DataT, NumElems> simd_by_init =
        esimd::simd<DataT, NumElems>(init_value, step);
    simd_by_init.copy_to(out);
  }
};

// Descriptor class for the case of calling constructor in variable declaration
// context.
struct var_dec {
  static std::string get_description() { return "variable declaration"; }

  template <typename DataT, int NumElems>
  static void call_simd_ctor(DataT init_value, DataT step, DataT *out) {
    esimd::simd<DataT, NumElems> simd_by_var_decl(init_value, step);
    simd_by_var_decl.copy_to(out);
  }
};

// Descriptor class for the case of calling constructor in rvalue in an
// expression context.
struct rval_in_express {
  static std::string get_description() { return "rvalue in an expression"; }

  template <typename DataT, int NumElems>
  static void call_simd_ctor(DataT init_value, DataT step, DataT *out) {
    esimd::simd<DataT, NumElems> simd_by_rval;
    simd_by_rval = esimd::simd<DataT, NumElems>(init_value, step);
    simd_by_rval.copy_to(out);
  }
};

// Descriptor class for the case of calling constructor in const reference
// context.
class const_ref {
public:
  static std::string get_description() { return "const reference"; }

  template <typename DataT, int NumElems>
  static void call_simd_ctor(DataT init_value, DataT step, DataT *out) {
    return call_simd_by_const_ref<DataT, NumElems>(
        esimd::simd<DataT, NumElems>(init_value, step), out);
  }

private:
  template <typename DataT, int NumElems>
  static void
  call_simd_by_const_ref(const esimd::simd<DataT, NumElems> &simd_by_const_ref,
                         DataT *out) {
    simd_by_const_ref.copy_to(out);
  }
};

// Enumeration of possible values for base value and step that will be provided
// into simd constructor.
enum class init_val {
  min,
  max,
  zero,
  min_half,
  max_half,
  neg_inf,
  nan,
  positive,
  negative
};

// Dummy kernel for submitting some code into device side.
template <typename DataT, int NumElems, typename T, init_val BaseVal,
          init_val StepVal>
struct kernel_for_fill;

// Constructing a value for step and base values that depends on input
// parameters.
template <typename DataT, init_val BaseVal> DataT get_base_value() {
  if constexpr (BaseVal == init_val::min) {
    return value<DataT>::lowest();
  } else if constexpr (BaseVal == init_val::max) {
    return value<DataT>::max();
  } else if constexpr (BaseVal == init_val::zero) {
    return 0;
  } else if constexpr (BaseVal == init_val::positive) {
    return static_cast<DataT>(1.25);
  } else if constexpr (BaseVal == init_val::negative) {
    return static_cast<DataT>(-10.75);
  } else if constexpr (BaseVal == init_val::min_half) {
    return value<DataT>::lowest() / 2;
  } else if constexpr (BaseVal == init_val::max_half) {
    return value<DataT>::max() / 2;
  } else if constexpr (BaseVal == init_val::neg_inf) {
    return -value<DataT>::inf();
  } else if constexpr (BaseVal == init_val::nan) {
    return value<DataT>::nan();
  } else {
    static_assert(BaseVal != BaseVal, "Unexpected base value value");
  }
}

// Struct that will be used for invocating of simd constructor.
template <init_val BaseVal, init_val Step> struct call_simd_struct {
  template <typename DataT, int NumElems, typename TestCaseT>
  esimd_tests::shared_vector<DataT>
  call_simd(sycl::queue &queue, const std::vector<DataT> &ref_data) {
    esimd_tests::shared_vector<DataT> result{NumElems,
                                             shared_allocator<DataT>{queue}};
    esimd_tests::shared_vector<DataT> shared_ref_data{
        ref_data.begin(), ref_data.end(), shared_allocator<DataT>{queue}};
    const auto ref = ref_data.data();
    DataT step_value{get_base_value<DataT, Step>()};
    DataT base_value{get_base_value<DataT, BaseVal>()};
    queue.submit([&](sycl::handler &cgh) {
      const auto ref = ref_data.data();
      auto out = result.data();
      cgh.single_task<
          kernel_for_fill<DataT, NumElems, TestCaseT, BaseVal, Step>>(
          [=]() SYCL_ESIMD_KERNEL {
            TestCaseT::template call_simd_ctor<DataT, NumElems>(
                base_value, step_value, out);
          });
    });

    return result;
  }
};

// Constructing verbose failure message.
template <int NumElems>
void on_failure(const std::string &step_value, const std::string &base_value,
                const std::string &elem_from_ref_data,
                const std::string &elem_from_result_data,
                const std::string &data_type,
                const std::string &test_case_description) {
  log::fail<NumElems>("Simd by " + test_case_description +
                          " failed, retrieved: " + elem_from_result_data +
                          ", expected: " + elem_from_ref_data +
                          ", with step value: " + step_value +
                          ", with base value: " + base_value,
                      data_type);
}

// Struct for verifying test result, failed test if a retrived value is not
// equal to an expected value.
template <init_val BaseVal, init_val Step> struct verify_obtained_result {
  template <typename DataT, int NumElems>
  bool verify_test_result(const std::vector<DataT> &ref_data,
                          const esimd_tests::shared_vector<DataT> &result_data,
                          const std::string &data_type,
                          const std::string &test_case_description) {
    bool pass{true};
    DataT step_value{get_base_value<DataT, Step>()};
    DataT base_value{get_base_value<DataT, BaseVal>()};
    size_t zero_elem{0};

    if (!are_bitwise_equal(ref_data[zero_elem], base_value)) {
      pass = false;
      on_failure<NumElems>(std::to_string(step_value),
                           std::to_string(base_value),
                           std::to_string(ref_data[zero_elem]),
                           std::to_string(result_data[zero_elem]), data_type,
                           test_case_description);
    }
    for (size_t i = 1; i < ref_data.size(); ++i) {
      if (!are_bitwise_equal(
              result_data[i],
              static_cast<DataT>(result_data[i - 1] + step_value))) {
        if constexpr (type_traits::is_sycl_floating_point_v<DataT>) {
          if (std::isnan(result_data[i]) &&
              (std::isnan(step_value) || std::isnan(base_value))) {
            continue;
          }
        }
        pass = false;
        on_failure<NumElems>(
            std::to_string(step_value), std::to_string(base_value),
            std::to_string(result_data[i - 1] + step_value),
            std::to_string(result_data[i]), data_type, test_case_description);
      }
    }

    return pass;
  }
};

// Iterating over provided types and dimensions, running test for each of them.
template <typename TestT, init_val BaseVal, init_val Step, typename... Types,
          int... Ints>
bool run_verifying(sycl::queue &queue,
                   const esimd_functional::values_pack<Ints...> &dimensions,
                   const esimd_functional::named_type_pack<Types...> &types) {
  bool pass{true};

  pass &= esimd_functional::for_all_types_and_dims<esimd_tests::test, TestT>(
      types, dimensions, queue, verify_obtained_result<BaseVal, Step>{},
      call_simd_struct<BaseVal, Step>{});

  return pass;
}