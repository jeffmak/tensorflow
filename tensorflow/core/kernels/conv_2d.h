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

#ifndef TENSORFLOW_KERNELS_CONV_2D_H_
#define TENSORFLOW_KERNELS_CONV_2D_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/eigen_backward_spatial_convolutions.h"
#include "tensorflow/core/kernels/eigen_spatial_convolutions.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {
namespace functor {

// TODO(yangke): revisit these operations and in particular, see if we can
// combine all of them into just one operation without causing nvcc to
// timeout.
template <typename Device, typename T, int Dims, typename IndexType>
struct ShuffleAndReverse {
  void operator()(const Device& d,
                  typename TTypes<T, Dims, IndexType>::ConstTensor input,
                  const Eigen::DSizes<IndexType, Dims>& order,
                  const Eigen::array<bool, Dims>& reverse_dims,
                  typename TTypes<T, Dims, IndexType>::Tensor output) {
    output.device(d) = input.shuffle(order).reverse(reverse_dims);
  }
};

template <typename Device, typename T, int Dims, typename IndexType>
struct InflatePadAndShuffle {
  void operator()(
      const Device& d, typename TTypes<T, Dims, IndexType>::ConstTensor input,
      const Eigen::DSizes<IndexType, Dims>& strides,
      const Eigen::array<Eigen::IndexPair<IndexType>, Dims>& pad_dims,
      const Eigen::DSizes<IndexType, Dims>& order,
      typename TTypes<T, Dims, IndexType>::Tensor output) {
    output.device(d) = input.inflate(strides).pad(pad_dims).shuffle(order);
  }
};

template <typename Device, typename Input, typename Filter, typename Output>
void SpatialConvolutionFunc(const Device& d, Output output, Input input,
                            Filter filter, int row_stride, int col_stride,
                            const Eigen::PaddingType& padding) {
  // Need to swap row/col when calling Eigen.
  output.device(d) =
      Eigen::SpatialConvolution(input, filter, col_stride, row_stride, padding);
  // #ifdef TENSORFLOW_USE_SYCL
  // if(std::is_same<Device, Eigen::SyclDevice>::value){
  //   auto buf = d.get_sycl_buffer(&output);
  //   using namespace cl::sycl;
  //   auto hA =
  //     buf.template get_access<access::mode::read, access::target::host_buffer>();
  //   std::cout << hA[0][0] << std::endl;
  // }
  // #endif
  std::cout << "bonjour" << std::endl;
}

template <typename Device, typename TensorType>
struct PrintValues {
  static void print(const Device& device, const TensorType& tensor, string name, int length){
    std::cout << name << ":"<< std::endl;
    const float* ptr = tensor.data();
    std::cout.precision(20);
    for(int i=0; i<length; ++i){
      std::cout << ptr[i] << std::endl;
    }
    std::cout << std::endl;
  }
};

template <typename TensorType>
struct PrintValues<Eigen::SyclDevice, TensorType> {
  static void print(const Eigen::SyclDevice& device, const TensorType& tensor, string name, int length){
    std::cout << name << ":"<< std::endl;
    auto buf_input = device.get_sycl_buffer(tensor.data());
    using namespace cl::sycl;
    auto hA_i =
      buf_input.template get_access<access::mode::read, access::target::host_buffer>();
    auto src_ptr = ConvertToActualTypeSycl(float, hA_i);
    std::cout.precision(20);
    for(int i=0; i<length; ++i){
      if(i%16==0) std::cout << std::endl;
      std::cout << src_ptr[i] << ", ";
    }
    std::cout << std::endl;
  }
};

template <typename Device, typename Input, typename InterC, typename KernelR, typename Filter, typename Output, typename Inter>
void SpatialConvolutionFuncModified(const Device& d,
      Inter inter, InterC interC, KernelR kernelR, Output output, Input input,
                            Filter kernel, int row_stride, int col_stride,
                            const Eigen::PaddingType& padding) {
  // Need to swap row/col when calling Eigen.
  // inter.device(d) = Eigen::SpatialConvolutionModified(d, inter, input, kernel, col_stride, row_stride, padding);
  typedef typename Eigen::internal::traits<Input>::Index TensorIndex;
  // PrintValues<Device,Input>::print(d,input,"Input",32);
  // PrintValues<Device,Filter>::print(d,kernel,"Filter",32);
  inter.device(d) =
      Eigen::SpatialConvolutionModified(d, inter, input, kernel, col_stride, row_stride, padding);
  PrintValues<Device,Inter>::print(d,inter,"Inter",128);
  Eigen::DSizes<TensorIndex, 2> kernel_dims;
  kernel_dims[0] = 16;
  kernel_dims[1] = 2;
  kernelR.device(d) = kernel.reshape(kernel_dims);
  PrintValues<Device,KernelR>::print(d,kernelR,"KernelR",32);
  interC.device(d) =
      Eigen::SpatialConvolutionModified2(d, interC, input, kernel, col_stride, row_stride, padding);
  PrintValues<Device,InterC>::print(d,interC,"InterC",16);
  output.device(d) =
      Eigen::SpatialConvolution(input, kernel, col_stride, row_stride, padding);
  // PrintValues<Device,Output>::print(d,output,"Output",16);

  // row_stride = col_stride = 1
  // int64_t kernelRows = 2, kernelCols = 2, row_in_stride = 1, col_in_stride = 1;
  //
  // typedef typename Eigen::internal::traits<Input>::Index TensorIndex;
  // Eigen::DSizes<TensorIndex, 2> kernel_dims;
  // kernel_dims[0] = 16;
  // kernel_dims[1] = 2;
  // Eigen::DSizes<TensorIndex, 2> pre_contract_dims;
  // pre_contract_dims[0] = 8;
  // pre_contract_dims[1] = 16;
  // Eigen::DSizes<TensorIndex, 4> post_contract_dims;
  // pre_contract_dims[0] = 1;
  // pre_contract_dims[1] = 2;
  // pre_contract_dims[2] = 4;
  // pre_contract_dims[3] = 2;
  // Eigen::array<Eigen::IndexPair<TensorIndex>, 1> contract_dims;
  // contract_dims[0] = Eigen::IndexPair<TensorIndex>(1, 0);
  //
  // output.device(d) = input
  //     .extract_image_patches(kernelRows, kernelCols, row_stride, col_stride,
  //                            row_in_stride, col_in_stride, padding)
  //     .reshape(pre_contract_dims)
  //     .contract(kernel.reshape(kernel_dims), contract_dims)
  //     .reshape(post_contract_dims);

  std::cout << "bonjour" << std::endl;
}


template <typename Device, typename T>
struct SpatialConvolution {
  void operator()(const Device& d, typename TTypes<T, 4>::Tensor output,
                  typename TTypes<T, 4>::ConstTensor input,
                  typename TTypes<T, 4>::ConstTensor filter, int row_stride,
                  int col_stride, const Eigen::PaddingType& padding) {
    SpatialConvolutionFunc(d, output, input, filter, row_stride, col_stride,
                           padding);
  }
};

template <typename Device, typename T>
struct SpatialConvolutionDebug {
  void operator()(const Device& d, typename TTypes<T, 2>::Tensor inter,
                  typename TTypes<T, 4>::Tensor interC,
                  typename TTypes<T, 2>::Tensor kernelR,
                  typename TTypes<T, 4>::Tensor output,
                  typename TTypes<T, 4>::ConstTensor input,
                  typename TTypes<T, 4>::ConstTensor filter, int row_stride,
                  int col_stride, const Eigen::PaddingType& padding) {
    SpatialConvolutionFuncModified(d, inter, interC, kernelR, output, input, filter, row_stride, col_stride,
                           padding);
  }
};

// #ifdef TENSORFLOW_USE_SYCL
// template <typename T>
// struct SpatialConvolutionDebug<Eigen::SyclDevice, T>{
//   void operator()(const Eigen::SyclDevice& d, typename TTypes<T, 2>::Tensor inter,
//                   typename TTypes<T, 4>::Tensor interC,
//                   typename TTypes<T, 4>::Tensor output,
//                   typename TTypes<T, 4>::ConstTensor input,
//                   typename TTypes<T, 4>::ConstTensor filter, int row_stride,
//                   int col_stride, const Eigen::PaddingType& padding) {
//     SpatialConvolutionFuncModified(d, inter, interC, output, input, filter, row_stride, col_stride,
//                            padding);
//   }
// };
// #endif

template <typename Device>
struct SpatialConvolutionDebug<Device, Eigen::half> {
  void operator()(const Device& d,
                  typename TTypes<Eigen::half, 2>::Tensor inter,
                  typename TTypes<Eigen::half, 4>::Tensor interC,
                  typename TTypes<Eigen::half, 2>::Tensor kernelR,
                  typename TTypes<Eigen::half, 4>::Tensor output,
                  typename TTypes<Eigen::half, 4>::ConstTensor input,
                  typename TTypes<Eigen::half, 4>::ConstTensor filter,
                  int row_stride, int col_stride,
                  const Eigen::PaddingType& padding) {
    output.device(d) =
        Eigen::SpatialConvolution(input.cast<float>(), filter.cast<float>(),
                                  col_stride, row_stride, padding)
            .cast<Eigen::half>();
  }
};

template <typename Device>
struct SpatialConvolution<Device, Eigen::half> {
  void operator()(const Device& d,
                  typename TTypes<Eigen::half, 4>::Tensor output,
                  typename TTypes<Eigen::half, 4>::ConstTensor input,
                  typename TTypes<Eigen::half, 4>::ConstTensor filter,
                  int row_stride, int col_stride,
                  const Eigen::PaddingType& padding) {
    output.device(d) =
        Eigen::SpatialConvolution(input.cast<float>(), filter.cast<float>(),
                                  col_stride, row_stride, padding)
            .cast<Eigen::half>();
  }
};

template <typename Device, typename T>
struct SpatialConvolutionBackwardInput {
  void operator()(const Device& d, typename TTypes<T, 4>::Tensor input_backward,
                  typename TTypes<T, 4>::ConstTensor kernel,
                  typename TTypes<T, 4>::ConstTensor output_backward,
                  int input_rows, int input_cols, int row_stride,
                  int col_stride) {
    // Need to swap row/col when calling Eigen.
    input_backward.device(d) = Eigen::SpatialConvolutionBackwardInput(
        kernel, output_backward, input_cols, input_rows, col_stride,
        row_stride);
  }
};

template <typename Device, typename T>
struct SpatialConvolutionBackwardKernel {
  void operator()(const Device& d,
                  typename TTypes<T, 4>::Tensor kernel_backward,
                  typename TTypes<T, 4>::ConstTensor input,
                  typename TTypes<T, 4>::ConstTensor output_backward,
                  int kernel_rows, int kernel_cols, int row_stride,
                  int col_stride) {
    // Need to swap row/col when calling Eigen.
    kernel_backward.device(d) = Eigen::SpatialConvolutionBackwardKernel(
        input, output_backward, kernel_cols, kernel_rows, col_stride,
        row_stride);
  }
};

// TODO(vrv): Figure out how to use the MatMulFunctor in matmul_op.h.
// My initial attempt to do this compiled but failed in the pytest
// due to a swigdeps error.
template <typename Device, typename T>
struct MatMulConvFunctor {
  // Computes on device "d": out = in0 * in1, where * is matrix
  // multiplication.
  void operator()(
      const Device& d, typename TTypes<T, 2>::Tensor out,
      typename TTypes<T, 2>::ConstTensor in0,
      typename TTypes<T, 2>::ConstTensor in1,
      const Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1>& dim_pair) {
    out.device(d) = in0.contract(in1, dim_pair);
  }
};

// Shuffles a filter tensor from:
//   [<spatial_dims>, in, out]
// to:
//   [out, in, <spatial_dims>]
template <typename Device, typename T, typename IndexType, int NDIMS>
struct TransformFilter {
  void operator()(const Device& d,
                  typename TTypes<T, NDIMS, IndexType>::ConstTensor in,
                  typename TTypes<T, NDIMS, IndexType>::Tensor out) {
    // We want a 3, 2, 0, 1 shuffle. Merge the spatial dimensions together
    // to speed up the shuffle operation.
    Eigen::DSizes<IndexType, 3> merged_dims;
    merged_dims[0] = in.dimension(0);  // spatial dimensions
    for (int i = 1; i < NDIMS - 2; ++i) {
      merged_dims[0] *= in.dimension(i);
    }
    merged_dims[1] = in.dimension(NDIMS - 2);  // input filters
    merged_dims[2] = in.dimension(NDIMS - 1);  // output filters

    Eigen::DSizes<IndexType, NDIMS> expanded_dims;
    expanded_dims[0] = in.dimension(NDIMS - 1);  // output filters
    expanded_dims[1] = in.dimension(NDIMS - 2);  // input filters
    for (int i = 0; i < NDIMS; ++i) {            // spatial dimensions
      expanded_dims[i + 2] = in.dimension(i);
    }

    out.device(d) = in.reshape(merged_dims)
                        .shuffle(Eigen::DSizes<IndexType, 3>(2, 1, 0))
                        .reshape(expanded_dims);
  }
};

template <typename Device, typename T, typename IndexType>
struct TransformDepth {
  void operator()(const Device& d,
                  typename TTypes<T, 4, IndexType>::ConstTensor in,
                  const Eigen::DSizes<IndexType, 4>& shuffle,
                  typename TTypes<T, 4, IndexType>::Tensor out) {
    Eigen::DSizes<IndexType, 3> merged_dims;
    Eigen::DSizes<IndexType, 4> expanded_dims;
    Eigen::DSizes<IndexType, 3> new_shuffle;

    // Merge dimensions that won't be shuffled together to speed things up.
    if (shuffle[1] == 2 && shuffle[2] == 3) {
      merged_dims[0] = in.dimension(0);
      merged_dims[1] = in.dimension(1);
      merged_dims[2] = in.dimension(2) * in.dimension(3);
      new_shuffle[0] = shuffle[0];
      new_shuffle[1] = 2;
      new_shuffle[2] = shuffle[3];
      expanded_dims[0] = in.dimension(shuffle[0]);
      expanded_dims[1] = in.dimension(2);
      expanded_dims[2] = in.dimension(3);
      expanded_dims[3] = in.dimension(shuffle[3]);
    } else if (shuffle[0] == 2 && shuffle[1] == 3) {
      merged_dims[0] = in.dimension(0);
      merged_dims[1] = in.dimension(1);
      merged_dims[2] = in.dimension(2) * in.dimension(3);
      new_shuffle[0] = 2;
      new_shuffle[1] = shuffle[2];
      new_shuffle[2] = shuffle[3];
      expanded_dims[0] = in.dimension(2);
      expanded_dims[1] = in.dimension(3);
      expanded_dims[2] = in.dimension(shuffle[2]);
      expanded_dims[3] = in.dimension(shuffle[3]);
    } else if (shuffle[0] == 0 && shuffle[1] == 3 && shuffle[2] == 1 &&
               shuffle[3] == 2) {
      merged_dims[0] = in.dimension(0);
      merged_dims[1] = in.dimension(1) * in.dimension(2);
      merged_dims[2] = in.dimension(3);
      new_shuffle[0] = 0;
      new_shuffle[1] = 2;
      new_shuffle[2] = 1;
      expanded_dims[0] = in.dimension(0);
      expanded_dims[1] = in.dimension(3);
      expanded_dims[2] = in.dimension(1);
      expanded_dims[3] = in.dimension(2);
    } else {
      assert(false && "unexpected shuffle");
    }

    out.device(d) =
        in.reshape(merged_dims).shuffle(new_shuffle).reshape(expanded_dims);
  }
};

template <typename Device, typename T, typename IndexType, int NDIMS>
struct PadInput {
  void operator()(const Device& d,
                  typename TTypes<T, NDIMS, IndexType>::ConstTensor in,
                  const std::array<int, NDIMS - 2>& padding_left,
                  const std::array<int, NDIMS - 2>& padding_right,
                  typename TTypes<T, NDIMS, IndexType>::Tensor out,
                  TensorFormat format) {
    Eigen::array<std::pair<IndexType, IndexType>, NDIMS> padding;
    padding[GetTensorDimIndex<NDIMS - 2>(format, 'N')] = std::make_pair(0, 0);
    for (int i = 0; i < NDIMS - 2; ++i) {
      padding[GetTensorDimIndex<NDIMS - 2>(format, '0' + i)] =
          std::make_pair(padding_left[i], padding_right[i]);
    }
    padding[GetTensorDimIndex<NDIMS - 2>(format, 'C')] = std::make_pair(0, 0);
    out.device(d) = in.pad(padding);
  }
};

// Converts a tensor from:
//   [batch, <spatial>, filters]
// to:
//   [batch, filters, <spatial>]
template <typename Device, typename T, int NDIMS>
struct NHWCToNCHW {
  void operator()(const Device& d, typename TTypes<T, NDIMS>::ConstTensor in,
                  typename TTypes<T, NDIMS>::Tensor out);
};

// Converts a tensor from:
//   [batch, filters, <spatial>]
// to:
//   [batch, <spatial>, filters]
template <typename Device, typename T, int NDIMS>
struct NCHWToNHWC {
  void operator()(const Device& d, typename TTypes<T, NDIMS>::ConstTensor in,
                  typename TTypes<T, NDIMS>::Tensor out);
};

// Converts a tensor from:
//   [dim0, dim1, dim2]
// to:
//   [dim0, dim2, dim1]
template <typename Device, typename T>
struct SwapDimension1And2InTensor3 {
  void operator()(const Device& d, const T* in,
                  const gtl::ArraySlice<int64>& input_dims, T* out);
};

// Converts a tensor from:
//   [dim0, dim1, dim2]
// to:
//   [dim2, dim1, dim0]
template <typename Device, typename T>
struct SwapDimension0And2InTensor3 {
  void operator()(const Device& d, const T* in,
                  const gtl::ArraySlice<int64>& input_dims, T* out);
};

// Reverses the effect of TransformFilter above.
template <typename Device, typename T, int NDIMS>
struct ReverseTransformFilter {
  void operator()(const Device& d, typename TTypes<T, NDIMS>::ConstTensor in,
                  typename TTypes<T, NDIMS>::Tensor out);
};

}  // namespace functor

template <class T>
class ConvAlgorithmMap;

template <>
class ConvAlgorithmMap<Eigen::ThreadPoolDevice> {};
}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_CONV_2D_H_
