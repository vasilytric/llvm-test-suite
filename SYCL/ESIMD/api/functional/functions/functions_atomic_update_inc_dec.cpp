//==------- functions_atomic_update_inc_dec.cpp  - DPC++ ESIMD on-device
//          test -----------------------------------------------------------==//
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
// Test for simd atomic_update function.
// The test uses increment and decrement atomic operators. Invokes atomic_update
// with different offset and simd_mask values.
// It is expected that values that should be changed has been changed.

#include "../sycl_nd_range.hpp"
#include "common.hpp"
#include "functions_atomic_update.hpp"

using namespace esimd_test::api::functional;
using namespace sycl::ext::intel::esimd;

// Descriptor class for the case of calling bitwise not operator.
struct atomic_update_0_operands {
  static std::string get_description() {
    return "atomic update with zero operands";
  }

  template <typename DataT, int NumElems, int NumElemsToChange,
            typename ChangeElemFilterT, atomic_op ChosenOperator>
  static void simd_function(DataT *input_data, DataT *output_data,
                            size_t *const indexes_to_change,
                            size_t work_item_index) {
    simd<DataT, NumElemsToChange> offset;
    mask_type_t<NumElemsToChange> changed_indexes;

    ChangeElemFilterT filter{};
    for (size_t i = 0; i < NumElemsToChange; ++i) {
      changed_indexes[i] = filter(i);
      offset[i] = indexes_to_change[i] * sizeof(DataT);
    }

    auto values_before_update =
        atomic_update<ChosenOperator, DataT, NumElemsToChange>(
            input_data, offset, changed_indexes);

    unsigned int values_sum = 0;
    for (size_t i = 0; i < NumElemsToChange; ++i) {
      values_sum += values_before_update[i];
    }

    output_data[work_item_index] = values_sum;
  }
};

// The main test routine.
// Using functor class to be able to iterate over the pre-defined data types.
template <typename TestCaseT, typename DataT, typename NumElemsT,
          typename NDRangeDimT, typename NumElemsToChangeT,
          typename ChangeElemFilterT, typename ChosenOperatorT,
          typename AlgorithmToChangeT>
class run_test {
  static constexpr int NumElems = NumElemsT::value;
  static constexpr int NDRangeDim = NDRangeDimT::value;
  static constexpr int NumElemsToChange = NumElemsToChangeT::value;
  static constexpr atomic_op ChosenOperator = ChosenOperatorT::value;
  static constexpr functions::algorithm_to_change AlgorithmToChange =
      AlgorithmToChangeT::value;
  using TestDescriptionT = functions::TestDescription<NumElems, TestCaseT>;

public:
  bool operator()(sycl::queue &queue, const std::string &data_type) {
    static_assert(NDRangeDim <= 3 && NDRangeDim > 0);
    bool passed = true;
    constexpr DataT base_value = 100;
    constexpr DataT result_fill_value = 5;

    auto init_values =
        functions::get_init_values<NumElems, ChosenOperator, DataT>(base_value);
    shared_allocator<DataT> allocator(queue);
    shared_vector<DataT> shared_init_values(init_values.begin(),
                                            init_values.end(), allocator);
    shared_vector<DataT> shared_result(NumElems, result_fill_value, allocator);

    auto indexes_to_change =
        functions::get_indexess_to_change<NumElems, AlgorithmToChange>();
    shared_vector<size_t> shared_indexes_to_change(
        indexes_to_change.begin(), indexes_to_change.end(),
        shared_allocator<size_t>(queue));

    constexpr int NumberIteractions = 16;
    sycl::nd_range<NDRangeDim> nd_range =
        get_sycl_nd_range<NDRangeDim>(NumberIteractions);

    queue.submit([&](sycl::handler &cgh) {
      DataT *shared_init_values_ptr = shared_init_values.data();
      DataT *result_ptr = shared_result.data();
      size_t *const indexes_to_change_ptr = shared_indexes_to_change.data();

      cgh.parallel_for<
          Kernel<DataT, NumElems, TestCaseT, ChosenOperatorT, NDRangeDimT,
                 NumElemsToChangeT, ChangeElemFilterT, AlgorithmToChangeT>>(
          nd_range, [=](sycl::nd_item<1> nd_item) SYCL_ESIMD_KERNEL {
            const size_t work_item_index = nd_item.get_global_linear_id();
            TestCaseT::template simd_function<DataT, NumElems, NumElemsToChange,
                                              ChangeElemFilterT,
                                              ChosenOperator>(
                shared_init_values_ptr, result_ptr, indexes_to_change_ptr,
                work_item_index);
          });
    });
    queue.wait_and_throw();

    std::sort(shared_result.begin(), shared_result.end());

    ChangeElemFilterT filter{};
    size_t num_changed_elems = 0;
    for (size_t i = 0; i < NumElemsToChange; ++i) {
      if (filter(i) == 1) {
        ++num_changed_elems;
      }
    }

    for (size_t i = 0; i < NumElems; ++i) {
      const DataT &retrieved = shared_result[i];
      if (i < NumberIteractions) {
        if (result_fill_value != retrieved) {
          passed = fail_test(i, result_fill_value, retrieved, data_type,
                             "atomic_update return value");
        }
      } else {
        const DataT &expected = base_value * num_changed_elems +
                                (i - NumberIteractions) * num_changed_elems;
        if (expected != retrieved) {
          passed = fail_test(i, expected, retrieved, data_type,
                             "atomic_update return value");
        }
      }
    }

    std::vector<size_t> selected_indexes;
    // Collect the indexess that has been selected.
    for (size_t i = 0; i < NumElemsToChange; ++i) {
      if (filter(i) == 1) {
        selected_indexes.push_back(indexes_to_change[i]);
      }
    }

    // Push the largest value to avoid the following error: can't dereference
    // out of range vector iterator.
    selected_indexes.push_back(std::numeric_limits<size_t>::max());
    auto next_selected_index = selected_indexes.begin();

    const DataT &expected = base_value;
    DataT expected_after_change = functions::get_expected_value<ChosenOperator>(
        base_value, NumberIteractions);
    // Verify that values, that do not was changed has initial values.
    for (size_t i = 0; i < NumElems; ++i) {
      // If current index is less than selected index verify that this element
      // hasn't been changed.
      const DataT &retrieved = shared_init_values[i];
      if (i < *next_selected_index) {
        if (expected != retrieved) {
          passed = fail_test(i, expected, retrieved, data_type,
                             "value that should be updated");
        }
      } else {
        if (expected_after_change != retrieved) {
          passed = fail_test(i, expected_after_change, retrieved, data_type,
                             "value that should be updated");
        }
        next_selected_index++;
      }
    }

    return passed;
  }

private:
  bool fail_test(size_t i, DataT expected, DataT retrieved,
                 const std::string &data_type,
                 const std::string &verification_type) {
    log::fail(TestDescriptionT(data_type), "Unexpected value at index ", i,
              ", retrieved: ", retrieved, ", expected: ", expected,
              ", for verification type: ", verification_type,
              ", with number elements to change: ", NumElemsToChange);

    return false;
  }
};

int main(int, char **) {
  sycl::queue queue(esimd_test::ESIMDSelector{},
                    esimd_test::createExceptionHandler());

  bool passed = true;

  const auto uint_types = named_type_pack<uint32_t>::generate("uint32_t");
  const auto single_size = get_sizes<16>();
  const auto nd_range_dims = integer_pack<1, 2, 3>::generate_unnamed();
  const auto num_elems_to_change = integer_pack<1, 8>::generate_unnamed();

  const auto filter_types =
      unnamed_type_pack<functions::filters::ChangeByStep,
                        functions::filters::ChangeAll,
                        functions::filters::ChangeNothing>::generate();
  const auto atomic_op_types =
      value_pack<atomic_op, atomic_op::inc, atomic_op::dec>::generate_unnamed();
  const auto algorithm_to_change_types = value_pack<
      functions::algorithm_to_change, functions::algorithm_to_change::all,
      functions::algorithm_to_change::ordered_step,
      functions::algorithm_to_change::non_ordered_step>::generate_unnamed();

  // Running test for combinations for data types, simd sizes, dn_range
  // dimensions, number element to change, filter types, atomic operator types
  // and algorithms to change.
  passed &= for_all_combinations<run_test, atomic_update_0_operands>(
      uint_types, single_size, nd_range_dims, num_elems_to_change, filter_types,
      atomic_op_types, algorithm_to_change_types, queue);

  std::cout << (passed ? "=== Test passed\n" : "=== Test FAILED\n");
  return passed ? 0 : 1;
}
