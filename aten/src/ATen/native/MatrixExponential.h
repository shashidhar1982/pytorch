#pragma once

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>

namespace at { namespace native {

using matrix_exp_fn = void(*)(Tensor&, const Tensor&);

DECLARE_DISPATCH(matrix_exp_fn, matrix_exp_stub);

// helper functions
static inline Tensor operator_1_norm(const Tensor& t) {
  return std::get<0>(t.abs().sum(-2).max(-1));
}

}} // namespace at::native
