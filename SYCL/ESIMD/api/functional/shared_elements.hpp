//===-- shared_elements.hpp - Function that provides USM with a smart pointer.
//      -------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file provides data struct that lets interact with USM with a smart
/// pointer that lets avoid memory leaks.
///
//===----------------------------------------------------------------------===//

#pragma once

#include <functional>
#include <memory>
#include <sycl/sycl.hpp>

namespace esimd_test::api::functional {

template <typename T> class shared_elements {
  std::unique_ptr<T, std::function<void(T *)>> m_allocated_data;

public:
  shared_elements(sycl::queue &queue, size_t num_elements = 1) {
    const auto &device{queue.get_device()};
    const auto &context{queue.get_context()};

    auto deleter = [=](T *ptr) { sycl::free(ptr, context); };

    m_allocated_data = std::unique_ptr<T, decltype(deleter)>(
        sycl::malloc_shared<T>(num_elements, device, context), deleter);
  }

  T *data() { return m_allocated_data.get(); }

  T operator[](size_t i) { return m_allocated_data.get()[i]; }
};

} // namespace esimd_test::api::functional
