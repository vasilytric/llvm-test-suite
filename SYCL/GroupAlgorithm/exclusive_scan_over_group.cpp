// UNSUPPORTED: cuda || hip
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

#include <CL/sycl.hpp>
#include <algorithm>
#include <iostream>

template <typename T>
cl::sycl::event compiler_group_scan_impl(cl::sycl::queue *queue, T *in_data,
                                         T *out_data, int num_wg,
                                         int group_size) {
  cl::sycl::nd_range<1> thread_range(num_wg * group_size, group_size);
  cl::sycl::event event = queue->submit([&](cl::sycl::handler &cgh) {
    cgh.parallel_for(thread_range, [=](cl::sycl::nd_item<1> item) {
      auto id = item.get_global_linear_id();
      auto group = item.get_group();
      T data = in_data[id];

      T updated_data = cl::sycl::exclusive_scan_over_group(
          group, data, cl::sycl::multiplies<T>());
      out_data[id] = updated_data;
    });
  });
  return event;
}

template <typename T>
void test_compiler_group_scan(cl::sycl::queue *queue, T *in_data, T *out_data,
                              int num_wg, int group_size) {
  compiler_group_scan_impl(queue, in_data, out_data, num_wg, group_size);
}

int main(int argc, const char **argv) {
  int num_wg = 1;
  int group_size = 16;

  cl::sycl::queue queue{
      cl::sycl::gpu_selector{},
      cl::sycl::property_list{cl::sycl::property::queue::enable_profiling(),
                              cl::sycl::property::queue::in_order()}};

  typedef int T;
  size_t nelems = num_wg * group_size;
  T *data = cl::sycl::malloc_shared<T>(nelems, queue);
  T *result = cl::sycl::malloc_shared<T>(nelems, queue);
  queue.fill<T>(data, T(2), nelems).wait();
  queue.memset(result, 0, nelems * sizeof(T)).wait();

  test_compiler_group_scan(&queue, data, result, num_wg, group_size);
  queue.wait();
  T expected[] = {1,   2,   4,    8,    16,   32,   64,    128,
                  256, 512, 1024, 2048, 4096, 8192, 16384, 32768};
  for (int i = 0; i < sizeof(expected) / sizeof(T); ++i) {
    assert(result[i] == expected[i]);
  }
  return 0;
}
