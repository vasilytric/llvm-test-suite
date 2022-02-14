//===-- operator_decrement_and_increment.hpp - Functions for tests on simd
//      increment and decrement operators definition. ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file provides functions for tests on simd decrement and increment
/// operators.
///
//===----------------------------------------------------------------------===//

#pragma once

#include "../mutator.hpp"
#include "common.hpp"

namespace esimd = sycl::ext::intel::experimental::esimd;
namespace esimd_functional = esimd_test::api::functional;

namespace esimd_test::api::functional::operators {

// Descriptor class for the case of calling constructor in initializer context.
struct pre_decrement {
  static std::string get_operator_type() { return "pre decrement"; }

  template <typename DataT, int NumElems>
  static void call_simd_ctor(const DataT *const ref_data,
                             DataT *const source_simd_out,
                             DataT *const result_simd_out) {
    auto source_simd = esimd::simd<DataT, NumElems>();
    source_simd.copy_from(ref_data);
    esimd::simd<DataT, NumElems> result_simd = --source_simd;
    source_simd.copy_to(source_simd_out);
    result_simd.copy_to(result_simd_out);
  }

  template <typename DataT> static DataT apply_operator(DataT &val) {
    return --val;
  }

  static constexpr bool is_increment() { return false; }
};

// Descriptor class for the case of calling constructor in initializer context.
struct post_decrement {
  static std::string get_operator_type() { return "post decrement"; }

  template <typename DataT, int NumElems>
  static void call_simd_ctor(const DataT *const ref_data,
                             DataT *const source_simd_out,
                             DataT *const result_simd_out) {
    auto source_simd = esimd::simd<DataT, NumElems>();
    source_simd.copy_from(ref_data);
    esimd::simd<DataT, NumElems> result_simd = source_simd--;
    source_simd.copy_to(source_simd_out);
    result_simd.copy_to(result_simd_out);
  }

  template <typename DataT> static DataT apply_operator(DataT &val) {
    return val--;
  }

  static constexpr bool is_increment() { return false; }
};

// Descriptor class for the case of calling constructor in initializer context.
struct pre_increment {
  static std::string get_operator_type() { return "pre increment"; }

  template <typename DataT, int NumElems>
  static void call_simd_ctor(const DataT *const ref_data,
                             DataT *const source_simd_out,
                             DataT *const result_simd_out) {
    auto source_simd = esimd::simd<DataT, NumElems>();
    source_simd.copy_from(ref_data);
    esimd::simd<DataT, NumElems> result_simd = ++source_simd;
    source_simd.copy_to(source_simd_out);
    result_simd.copy_to(result_simd_out);
  }

  template <typename DataT> static DataT apply_operator(DataT &val) {
    return ++val;
  }

  static constexpr bool is_increment() { return true; }
};

// Descriptor class for the case of calling constructor in initializer context.
struct post_increment {
  static std::string get_operator_type() { return "post increment"; }

  template <typename DataT, int NumElems>
  static void call_simd_ctor(const DataT *const ref_data,
                             DataT *const source_simd_out,
                             DataT *const result_simd_out) {
    auto source_simd = esimd::simd<DataT, NumElems>();
    source_simd.copy_from(ref_data);
    esimd::simd<DataT, NumElems> result_simd = source_simd++;
    source_simd.copy_to(source_simd_out);
    result_simd.copy_to(result_simd_out);
  }

  template <typename DataT> static DataT apply_operator(DataT &val) {
    return val++;
  }

  static constexpr bool is_increment() { return true; }
};

template <typename DataT, int NumElems, typename TestCaseT>
class IncrementAndDecrementTestDescription : public ITestDescription {
public:
  IncrementAndDecrementTestDescription(size_t index, DataT retrieved_val,
                                       DataT expected_val,
                                       const std::string &simd_type,
                                       const std::string &data_type)
      : m_data_type(data_type), m_retrieved_val(retrieved_val),
        m_expected_val(expected_val), m_index(index), m_simd_type(simd_type) {}

  std::string to_string() const override {
    std::string log_msg("Failed for simd<");

    log_msg += m_data_type + ", " + std::to_string(NumElems) + ">";
    log_msg += ", retrieved: " + std::to_string(m_retrieved_val);
    log_msg += ", expected: " + std::to_string(m_expected_val);
    log_msg += ", at index: " + std::to_string(m_index);
    log_msg += " for simd as a " + m_simd_type;
    log_msg += " " + TestCaseT::get_operator_type() + " operator.";

    return log_msg;
  }

private:
  const std::string m_data_type;
  const DataT m_retrieved_val;
  const DataT m_expected_val;
  const size_t m_index;
  const std::string m_simd_type;
};

// The main test routine.
// Using functor class to be able to iterate over the pre-defined data types.
template <typename DataT, typename SizeT, typename TestCaseT> class run_test {
  static constexpr int NumElems = SizeT::value;

public:
  bool operator()(sycl::queue &queue, const std::string &data_type) {
    bool passed = true;
    std::vector<DataT> ref_data = generate_ref_data<DataT, NumElems>();

    if constexpr (TestCaseT::is_increment()) {
      mutate(ref_data, mutator::For_addition<DataT>(1));
    } else if constexpr (!TestCaseT::is_increment()) {
      mutate(ref_data, mutator::For_subtraction<DataT>(1));
    }

    // If current number of elements is equal to one, then run test with each
    // one value from reference data.
    // If current number of elements is greater than one, then run tests with
    // whole reference data.
    if constexpr (NumElems == 1) {
      for (size_t i = 0; i < ref_data.size(); ++i) {
        passed &= run_verification(queue, {ref_data[i]}, data_type);
      }
    } else {
      passed &= run_verification(queue, ref_data, data_type);
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
    shared_vector<DataT> source_simd_out(NumElems, allocator);
    shared_vector<DataT> result_simd_out(NumElems, allocator);
    shared_vector<DataT> shared_ref_data(ref_data.begin(), ref_data.end(),
                                         allocator);

    queue.submit([&](sycl::handler &cgh) {
      const DataT *const ref = shared_ref_data.data();
      DataT *source_simd_data_storage = source_simd_out.data();
      DataT *result_simd_data_storage = result_simd_out.data();

      cgh.single_task<Kernel<DataT, NumElems, TestCaseT>>(
          [=]() SYCL_ESIMD_KERNEL {
            TestCaseT::template call_simd_ctor<DataT, NumElems>(
                ref, source_simd_data_storage, result_simd_data_storage);
          });
    });
    queue.wait_and_throw();

    for (size_t i = 0; i < NumElems; ++i) {
      DataT expected_source_value = shared_ref_data[i];

      const DataT expected_return_value =
          TestCaseT::apply_operator(expected_source_value);

      passed &= verify_result(i, expected_source_value, source_simd_out[i],
                              "result value after", data_type);
      passed &= verify_result(i, expected_return_value, result_simd_out[i],
                              "return value for", data_type);
    }

    return passed;
  }

  bool verify_result(size_t i, DataT expected, DataT retrieved,
                     std::string &&simd_type, const std::string &data_type) {
    bool passed = true;
    if constexpr (type_traits::is_sycl_floating_point_v<DataT>) {
      if (std::isnan(expected) && !std::isnan(retrieved)) {
        passed = false;

        // TODO: Make ITestDescription architecture more flexible.
        // We are assuming that the NaN opcode may differ
        std::string log_msg("Failed for simd<");
        log_msg += data_type + ", " + std::to_string(NumElems) + ">";
        log_msg += ". The element at index: " + std::to_string(i) +
                   ", is not nan, but it should.";

        log::note(log_msg);
      }
    }
    if (!are_bitwise_equal(expected, retrieved)) {
      passed = false;

      const auto description =
          IncrementAndDecrementTestDescription<DataT, NumElems, TestCaseT>(
              i, retrieved, expected, simd_type, data_type);
      log::fail(description);
    }
    return passed;
  }
};

} // namespace esimd_test::api::functional::operators
