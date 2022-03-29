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
// XRUN: %GPU_RUN_PLACEHOLDER %t.out
// XFAIL: *
// TODO Remove XFAIL once unexpected error "Unknown type name 'uint8_t'" within
// integration header is gone.
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

// Descriptor class for the case of calling atomic_update overload for zero
// operands.
struct atomic_update_0_operands {
  static std::string get_description() {
    return "atomic update with zero operands";
  }

  template <typename DataT, int NumElemsToChange, typename ChangeElemFilterT,
            atomic_op ChosenOperator>
  static int call_esimd_function(DataT *input_data, DataT *output_data,
                                 size_t *const offsets) {
    simd<DataT, NumElemsToChange> offset;
    mask_type_t<NumElemsToChange> mask;

    ChangeElemFilterT filter{};
    for (size_t i = 0; i < NumElemsToChange; ++i) {
      mask[i] = filter(i);
      offset[i] = offsets[i] * sizeof(DataT);
    }

    auto values_before_update =
        atomic_update<ChosenOperator, DataT, NumElemsToChange>(input_data,
                                                               offset, mask);

    unsigned int values_sum = 0;
    for (size_t i = 0; i < NumElemsToChange; ++i) {
      values_sum += values_before_update[i];
    }

    return values_sum;
  }
};

// Provides expected value for updated memory locations.
template <esimd::atomic_op Operator, typename T>
T get_expected_value(T base_value, int number_of_iterations) {
  if constexpr (Operator == esimd::atomic_op::dec) {
    return base_value - number_of_iterations;
  } else if constexpr (Operator == esimd::atomic_op::inc) {
    return base_value + number_of_iterations;
  } else {
    static_assert(Operator != Operator, "Unexpected  operator type.");
  }
}

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
  static constexpr functions::offset_generation AlgorithmToChange =
      AlgorithmToChangeT::value;
  using TestDescriptionT = functions::TestDescription<NumElems, TestCaseT>;

public:
  bool operator()(sycl::queue &queue, const std::string &data_type,
                  const std::string &operator_type) {
    static_assert(NDRangeDim <= 3 && NDRangeDim > 0);
    static_assert(NumElemsToChange <= NumElems);
    bool passed = true;
    constexpr DataT base_value = 100;
    constexpr DataT result_fill_value = 5;
    shared_allocator<DataT> allocator(queue);
    shared_vector<DataT> shared_init_values(NumElems, allocator);
    functions::fill_init_values<NumElems, ChosenOperator, DataT>(
        base_value, shared_init_values);

    shared_vector<size_t> shared_offsets(NumElemsToChange,
                                         shared_allocator<size_t>(queue));
    functions::fill_offsets<NumElemsToChange, AlgorithmToChange>(
        shared_offsets);
    shared_vector<size_t> local_offsets = shared_offsets;
    constexpr int NumberIterations = 16;
    shared_vector<DataT> shared_result(NumberIterations, result_fill_value,
                                       allocator);
    sycl::nd_range<NDRangeDim> nd_range =
        get_sycl_nd_range<NDRangeDim>(NumberIterations);

    queue.submit([&](sycl::handler &cgh) {
      DataT *shared_init_values_ptr = shared_init_values.data();
      DataT *shared_result_ptr = shared_result.data();
      size_t *const offsets_ptr = shared_offsets.data();

      cgh.parallel_for<
          Kernel<DataT, NumElems, TestCaseT, ChosenOperatorT, NDRangeDimT,
                 NumElemsToChangeT, ChangeElemFilterT, AlgorithmToChangeT>>(
          nd_range, [=](sycl::nd_item<1> nd_item) SYCL_ESIMD_KERNEL {
            const size_t work_item_index = nd_item.get_global_linear_id();
            shared_result_ptr[work_item_index] =
                TestCaseT::template call_esimd_function<
                    DataT, NumElemsToChange, ChangeElemFilterT, ChosenOperator>(
                    shared_init_values_ptr, shared_result_ptr, offsets_ptr);
          });
    });
    queue.wait_and_throw();

    ChangeElemFilterT filter{};
    size_t num_changed_elems = 0;
    // Collect the number of elements that should be changed. This value will be
    // used to calculate expected value for all elements that will be returned
    // after atomic_update call the sum of changed elements.
    for (size_t i = 0; i < NumElemsToChange; ++i) {
      if (filter(i) == 1) {
        ++num_changed_elems;
      }
    }

    std::sort(shared_result.begin(), shared_result.end());
    for (size_t i = 0; i < NumberIterations; ++i) {
      const DataT &retrieved = shared_result[i];
      const DataT &expected = base_value * num_changed_elems +
                              (i - NumberIterations) * num_changed_elems;
      if (expected != retrieved) {
        passed = fail_test(i, expected, retrieved, data_type,
                           "atomic_update return value", operator_type);
      }
    }

    std::vector<size_t> changed_elems_indexes;
    // Collect the indexess that has been changed.
    for (size_t i = 0; i < NumElemsToChange; ++i) {
      if (filter(i) == 1) {
        changed_elems_indexes.push_back(local_offsets[i]);
      }
    }

    // Push the largest value to avoid the following error: can't dereference
    // out of range vector iterator.
    changed_elems_indexes.push_back(std::numeric_limits<size_t>::max());
    auto updated_elem_next_index = changed_elems_indexes.begin();

    const DataT &expected = base_value;
    DataT expected_after_change =
        get_expected_value<ChosenOperator>(base_value, NumberIterations);
    // Verify that values, that do not was changed has initial values.
    for (size_t i = 0; i < NumElems; ++i) {
      // If current index is less than changed element index verify that this
      // element hasn't been changed.
      const DataT &retrieved = shared_init_values[i];
      if (i < *updated_elem_next_index) {
        if (expected != retrieved) {
          passed = fail_test(i, expected, retrieved, data_type,
                             "value that should not be updated", operator_type);
        }
      } else {
        if (expected_after_change != retrieved) {
          passed = fail_test(i, expected_after_change, retrieved, data_type,
                             "value that should be updated", operator_type);
        }
        updated_elem_next_index++;
      }
    }

    return passed;
  }

private:
  bool fail_test(size_t i, DataT expected, DataT retrieved,
                 const std::string &data_type,
                 const std::string &verification_type,
                 const std::string &operator_type) {
    log::fail(TestDescriptionT(data_type), "Unexpected value at index ", i,
              ", retrieved: ", retrieved, ", expected: ", expected,
              ", for verification type: ", verification_type,
              ", with number elements to change: ", NumElemsToChange,
              ", with operator type: ", operator_type);

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
      unnamed_type_pack<functions::masks::ChangeByStep,
                        functions::masks::ChangeAll,
                        functions::masks::ChangeNothing>::generate();
  const auto atomic_op_types =
      value_pack<atomic_op, atomic_op::inc, atomic_op::dec>::generate_named(
          " atomic_op::inc", "atomic_op::dec");
  const auto algorithm_to_change_offset = value_pack<
      functions::offset_generation, functions::offset_generation::all,
      functions::offset_generation::ordered_step,
      functions::offset_generation::non_ordered_step>::generate_unnamed();

  // Running test for combinations for data types, simd sizes, dn_range
  // dimensions, number element to change, filter types, atomic operator types
  // and algorithms to change.
  passed &= for_all_combinations<run_test, atomic_update_0_operands>(
      uint_types, single_size, nd_range_dims, num_elems_to_change, filter_types,
      atomic_op_types, algorithm_to_change_offset, queue);

  std::cout << (passed ? "=== Test passed\n" : "=== Test FAILED\n");
  return passed ? 0 : 1;
}
