//===-- logger.hpp - Declarate functions and containers with that will be used
//      for run tests with different types. -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of the function that let iterate over
/// provided types and provide each type to the call call-operator from the
/// provided class/struct.
///
//===----------------------------------------------------------------------===//

#pragma once

#include <string>
#include <type_traits>
#include <utility>

namespace esimd_test {
namespace api {
namespace functional {

// int data type, because of we can use only one type of the data, that can
// determinate in the for_all_types function for "action" argument. for change
// it we should change it in the for_all_types function and in the function,
// that will be chosen as action.
template <int... T> struct values_pack {
  values_pack() {}
};

// Type pack to store types and underlying data type names to use with
// type_name_string
template <typename... T> struct named_type_pack {
  const std::string names[sizeof...(T)];

  template <typename... nameListT>
  named_type_pack(nameListT &&...nameList)
      : names{std::forward<nameListT>(nameList)...} {}
};

enum class tested_types { all, fp, uint, sint };

// Constructing named_type_pack with types that depent on chosen field of
// tested_types enumeration
template <tested_types required> auto get_tested_types() {
  if constexpr (required == tested_types::all) {
    return named_type_pack<char, unsigned char, signed char, short,
                           unsigned short, int, unsigned int, long,
                           unsigned long, float, double, long long,
                           unsigned long long>(
        {"char", "unsigned char", "signed char", "short", "unsigned short",
         "int", "unsigned int", "long", "unsigned long", "float", "double",
         "long long", "unsigned long long"});
  } else if constexpr (required == tested_types::fp) {
    return named_type_pack<float, sycl::half, double>(
        {"float", "sycl::half", "double"});
  } else if constexpr (required == tested_types::uint) {
    if constexpr (!std::is_signed_v<char>) {
      return named_type_pack<unsigned char, unsigned short, unsigned int,
                             unsigned long, unsigned long long, char>(
          {"unsigned char", "unsigned short", "unsigned int", "unsigned long",
           "unsigned long long", "char"});
    } else {
      return named_type_pack<unsigned char, unsigned short, unsigned int,
                             unsigned long, unsigned long long>(
          {"unsigned char", "unsigned short", "unsigned int", "unsigned long",
           "unsigned long long"});
    }
  } else if constexpr (required == tested_types::sint) {
    if constexpr (std::is_signed_v<char>) {
      return named_type_pack<signed char, short, int, long, long long, char>(
          {"signed char", "short", "int", "long", "long long", "char"});
    } else {
      return named_type_pack<signed char, short, int, long, long long>(
          {"signed char", "short", "int", "long", "long long"});
    }
  } else {
    static_assert(required != required, "Unexpected tested type");
  }
}

// Run action for each of types given by named_type_pack instance
template <template <typename, int, typename...> class action, int N,
          typename... actionArgsT, typename... types, typename... argsT>
inline bool for_all_types(const named_type_pack<types...> &type_list,
                          argsT &&...args) {
  bool passed{true};

  // run action for each type from types... parameter pack
  size_t type_name_index = 0;

  ((passed &= action<types, N, actionArgsT...>{}(
        std::forward<argsT>(args)..., type_list.names[type_name_index]),
    ++type_name_index),
   ...);

  return passed;
}

// Calls for_all_types for each vector length by values_pack instance
template <template <typename, int, typename...> class action,
          typename... actionArgsT, typename... types, int... ints,
          typename... argsT>
inline bool for_all_types_and_dims(const named_type_pack<types...> &type_list,
                                   const values_pack<ints...> &intList,
                                   argsT &&...args) {
  bool passed{true};

  // run action for each value from values_pack... parameter pack
  int int_index = 0;

  ((passed &= for_all_types<action, ints, actionArgsT...>(type_list, args...),
    ++int_index),
   ...);

  return passed;
}

} // namespace functional
} // namespace api
} // namespace esimd_test