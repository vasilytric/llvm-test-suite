//==------- ctor_load_core.cpp  - DPC++ ESIMD on-device test ---------------==//
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
// Test for simd load constructor.
// The test uses reference data and different alignment flags. Invokes simd
// constructors in different contexts with provided reference data and alignment
// flag.
// It is expected for destination simd instance to store a bitwise same data as
// the reference one.

#include "common.hpp"

using namespace sycl::ext::intel::experimental::esimd;
using namespace esimd_test::api::functional;

// Descriptor class for the case of calling constructor in initializer context.
struct initializer {
  static std::string get_description() { return "initializer"; }

  template <typename DataT, int NumElems, typename AlignmentT>
  static void call_simd_ctor(const DataT *ref_data, DataT *const out,
                             AlignmentT alignment) {
    simd<DataT, NumElems> simd_by_init =
        simd<DataT, NumElems>(ref_data, alignment);
    simd_by_init.copy_to(out);
  }
};

// Descriptor class for the case of calling constructor in variable declaration
// context.
struct var_decl {
  static std::string get_description() { return "variable declaration"; }

  template <typename DataT, int NumElems, typename AlignmentT>
  static void call_simd_ctor(const DataT *ref_data, DataT *const out,
                             AlignmentT alignment) {
    simd<DataT, NumElems> simd_by_var_decl(ref_data, alignment);
    simd_by_var_decl.copy_to(out);
  }
};

// Descriptor class for the case of calling constructor in rvalue in an
// expression context.
struct rval_in_expr {
  static std::string get_description() { return "rvalue in an expression"; }

  template <typename DataT, int NumElems, typename AlignmentT>
  static void call_simd_ctor(const DataT *ref_data, DataT *const out,
                             AlignmentT alignment) {
    simd<DataT, NumElems> simd_by_rval;
    simd_by_rval = simd<DataT, NumElems>(ref_data, alignment);
    simd_by_rval.copy_to(out);
  }
};

// Descriptor class for the case of calling constructor in const reference
// context.
class const_ref {
public:
  static std::string get_description() { return "const reference"; }

  template <typename DataT, int NumElems, typename AlignmentT>
  static void call_simd_ctor(const DataT *ref_data, DataT *const out,
                             AlignmentT alignment) {
    call_simd_by_const_ref<DataT, NumElems>(
        simd<DataT, NumElems>(ref_data, alignment), out);
  }

private:
  template <typename DataT, int NumElems>
  static void
  call_simd_by_const_ref(const simd<DataT, NumElems> &simd_by_const_ref,
                         DataT *out) {
    simd_by_const_ref.copy_to(out);
  }
};

// Dummy kernel for submitting some code into device side.
template <typename DataT, int NumElems, typename T, typename Alignment>
struct Kernel_for_load_ctor;

namespace alignment {

struct element {
  template <typename DataT, int> static size_t get_size() {
    return alignof(DataT);
  }
  static constexpr auto get_value() { return element_aligned; }
};

struct vector {
  template <typename DataT, int NumElems> static size_t get_size() {
    // Referring to the simd class specialization on the host side is by design.
    return alignof(simd<DataT, NumElems>);
  }
  static constexpr auto get_value() { return vector_aligned; }
};

struct overal {
  // Use 16 instead of std::max_align_t because of the fact that long double is
  // not a native type in Intel GPUs. So 16 is not driven by any type, but
  // rather the "oword alignment" requirement for all block loads. In that
  // sense, std::max_align_t would give wrong idea.
  template <typename, int> static size_t get_size() { return alignof(16); }

  static constexpr auto get_value() { return overaligned<alignof(16)>; }
};

} // namespace alignment

// The main test routine.
// Using functor class to be able to iterate over the pre-defined data types.
template <typename DataT, typename SizeT, typename TestCaseT,
          typename AlignmentT>
class run_test {
  static constexpr int NumElems = SizeT::value;

public:
  bool operator()(sycl::queue &queue, const std::string &data_type) {
    bool passed = true;
    const std::vector<DataT> ref_data = generate_ref_data<DataT, NumElems>();

    // If current number of elements is equal to one, then run test with each
    // one value from reference data.
    // If current number of elements is greater than one, then run tests with
    // whole reference data.
    if constexpr (NumElems == 1) {
      for (size_t i = 0; i < ref_data.size(); ++i) {
        passed = run_verification(queue, {ref_data[i]}, data_type);
      }
    } else {
      passed = run_verification(queue, ref_data, data_type);
    }
    return passed;
  }

private:
  bool run_verification(sycl::queue &queue, const std::vector<DataT> &ref_data,
                        const std::string &data_type) {
    assert(ref_data.size() == NumElems &&
           "Reference data size is not equal to the simd vector length.");

    bool passed = true;

    const size_t alignment_value =
        AlignmentT::template get_size<DataT, NumElems>();
    const size_t container_extra_size = alignment_value / sizeof(DataT) + 1;
    const size_t offset = 1;

    shared_allocator<DataT> allocator(queue);
    shared_vector<DataT> result(NumElems, allocator);
    shared_vector<DataT> shared_ref_data(NumElems + container_extra_size +
                                             offset,
                                         shared_allocator<DataT>(queue));

    const size_t object_size = NumElems * sizeof(DataT);
    size_t buffer_size = object_size + container_extra_size * sizeof(DataT);

    // When we allocate USM there is a high probability that this memory will
    // have stronger alignment that required. We increment our pointer by fixed
    // offset value to avoid bigger alignment of USM shared.
    // The std::align can provide expected alignment on the small values of an
    // alignment.
    void *ref = shared_ref_data.data() + offset;
    if (std::align(alignment_value, object_size, ref, buffer_size) == nullptr) {
      return false;
    }
    DataT *const ref_aligned = static_cast<DataT *>(ref);

    for (size_t i = 0; i < NumElems; ++i) {
      ref_aligned[i] = ref_data[i];
    }

    queue.submit([&](sycl::handler &cgh) {
      DataT *const out = result.data();

      cgh.single_task<
          Kernel_for_load_ctor<DataT, NumElems, TestCaseT, AlignmentT>>(
          [=]() SYCL_ESIMD_KERNEL {
            const auto alignment = AlignmentT::get_value();
            TestCaseT::template call_simd_ctor<DataT, NumElems>(ref_aligned,
                                                                out, alignment);
          });
    });
    queue.wait_and_throw();

    for (size_t i = 0; i < result.size(); ++i) {
      if (!are_bitwise_equal(ref_data[i], result[i])) {
        passed = false;

        const auto description =
            ctors::TestDescription<DataT, NumElems, TestCaseT>(
                i, result[i], ref_data[i], data_type);
        log::fail(description);
      }
    }

    return passed;
  }
};

int main(int, char **) {
  sycl::queue queue(esimd_test::ESIMDSelector{},
                    esimd_test::createExceptionHandler());

  bool passed = true;
  const auto types = get_tested_types<tested_types::core>();
  const auto sizes = get_all_sizes();

  const auto contexts = unnamed_type_pack<initializer, var_decl, rval_in_expr,
                                          const_ref>::generate();
  const auto alignments =
      unnamed_type_pack<alignment::element, alignment::vector,
                        alignment::overal>::generate();

  passed &=
      for_all_combinations<run_test>(types, sizes, contexts, alignments, queue);

  std::cout << (passed ? "=== Test passed\n" : "=== Test FAILED\n");
  return passed ? 0 : 1;
}
