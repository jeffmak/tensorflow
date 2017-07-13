/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_KERNELS_IDENTITY_OP_H_
#define TENSORFLOW_KERNELS_IDENTITY_OP_H_

#include "tensorflow/core/framework/op_kernel.h"
#include <iostream>

namespace tensorflow {

class IdentityOp : public OpKernel {
 public:
  explicit IdentityOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    if (IsRefType(context->input_dtype(0))) {
      printf("yo\n");
      context->forward_ref_input_to_ref_output(0, 0);
    } else {
      // std::cout << name() << " " << context->input(0).SummarizeValue(4) << std::endl;
      context->set_output(0, context->input(0));
    }
  }

  bool IsExpensive() override { return false; }
};

}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_IDENTITY_OP_H_
