//
// Created by fss on 2023/3/20.
//
/**
 * @file tensor_utils.cpp
 * @brief 张量工具函数的实现文件
 * @details 该文件实现了张量的各种操作工具函数，包括张量比较、元素级运算、广播等
 * 提供了张量计算的基础功能支持
 */

#include <glog/logging.h>
#include "data/tensor.hpp"
#include "data/tensor_util.hpp"

namespace kuiper_infer {

/**
 * @brief 比较两个张量是否相同
 * @param a 第一个张量
 * @param b 第二个张量
 * @param threshold 比较阈值，用于浮点数比较
 * @return 如果两个张量相同返回true，否则返回false
 * @details 首先比较形状是否相同，然后使用armadillo的approx_equal函数进行数值比较
 */
bool TensorIsSame(const std::shared_ptr<Tensor<float>>& a,
                  const std::shared_ptr<Tensor<float>>& b, float threshold) {
  CHECK(a != nullptr);  // 检查第一个张量是否为空
  CHECK(b != nullptr);  // 检查第二个张量是否为空
  
  // 首先比较形状是否相同
  if (a->shapes() != b->shapes()) {
    return false;
  }
  
  // 使用armadillo的近似相等函数比较数值，考虑浮点误差
  bool is_same = arma::approx_equal(a->data(), b->data(), "absdiff", threshold);
  return is_same;
}

/**
 * @brief 张量元素级加法（就地操作）
 * @param tensor1 第一个输入张量
 * @param tensor2 第二个输入张量
 * @param output_tensor 输出张量
 * @details 将两个张量进行元素级加法，结果存储在output_tensor中
 * 如果形状不同，会进行广播操作
 */
void TensorElementAdd(const std::shared_ptr<Tensor<float>>& tensor1,
                      const std::shared_ptr<Tensor<float>>& tensor2,
                      const std::shared_ptr<Tensor<float>>& output_tensor) {
  CHECK(tensor1 != nullptr && tensor2 != nullptr && output_tensor != nullptr);  // 检查所有张量是否为空
  
  if (tensor1->shapes() == tensor2->shapes()) {
    // 如果形状相同，直接进行加法运算
    CHECK(tensor1->shapes() == output_tensor->shapes());  // 检查输出张量形状
    output_tensor->set_data(tensor1->data() + tensor2->data());
  } else {
    // 如果形状不同，需要进行广播操作
    CHECK(tensor1->channels() == tensor2->channels())
        << "Tensors shape are not adapting";  // 检查通道数是否兼容
    
    // 进行广播操作，使两个张量形状一致
    const auto& [input_tensor1, input_tensor2] =
        TensorBroadcast(tensor1, tensor2);
    
    // 检查广播后的形状是否与输出张量一致
    CHECK(output_tensor->shapes() == input_tensor1->shapes() &&
          output_tensor->shapes() == input_tensor2->shapes());
    
    // 执行加法运算
    output_tensor->set_data(input_tensor1->data() + input_tensor2->data());
  }
}

/**
 * @brief 张量元素级乘法（就地操作）
 * @param tensor1 第一个输入张量
 * @param tensor2 第二个输入张量
 * @param output_tensor 输出张量
 * @details 将两个张量进行元素级乘法，结果存储在output_tensor中
 * 如果形状不同，会进行广播操作
 */
void TensorElementMultiply(
    const std::shared_ptr<Tensor<float>>& tensor1,
    const std::shared_ptr<Tensor<float>>& tensor2,
    const std::shared_ptr<Tensor<float>>& output_tensor) {
  CHECK(tensor1 != nullptr && tensor2 != nullptr && output_tensor != nullptr);  // 检查所有张量是否为空
  
  if (tensor1->shapes() == tensor2->shapes()) {
    // 如果形状相同，直接进行乘法运算
    CHECK(tensor1->shapes() == output_tensor->shapes());  // 检查输出张量形状
    output_tensor->set_data(tensor1->data() % tensor2->data());  // 使用%表示元素级乘法
  } else {
    // 如果形状不同，需要进行广播操作
    CHECK(tensor1->channels() == tensor2->channels())
        << "Tensors shape are not adapting";  // 检查通道数是否兼容
    
    // 进行广播操作，使两个张量形状一致
    const auto& [input_tensor1, input_tensor2] =
        TensorBroadcast(tensor1, tensor2);
    
    // 检查广播后的形状是否与输出张量一致
    CHECK(output_tensor->shapes() == input_tensor1->shapes() &&
          output_tensor->shapes() == input_tensor2->shapes());
    
    // 执行乘法运算
    output_tensor->set_data(input_tensor1->data() % input_tensor2->data());
  }
}

/**
 * @brief 张量元素级加法（返回新张量）
 * @param tensor1 第一个输入张量
 * @param tensor2 第二个输入张量
 * @return 包含加法结果的新张量
 * @details 将两个张量进行元素级加法，返回一个新的张量对象
 * 如果形状不同，会进行广播操作
 */
std::shared_ptr<Tensor<float>> TensorElementAdd(
    const std::shared_ptr<Tensor<float>>& tensor1,
    const std::shared_ptr<Tensor<float>>& tensor2) {
  CHECK(tensor1 != nullptr && tensor2 != nullptr);  // 检查输入张量是否为空
  
  if (tensor1->shapes() == tensor2->shapes()) {
    // 如果形状相同，直接进行加法运算
    sftensor output_tensor = TensorCreate(tensor1->shapes());  // 创建输出张量
    output_tensor->set_data(tensor1->data() + tensor2->data());
    return output_tensor;
  } else {
    // 如果形状不同，需要进行广播操作
    CHECK(tensor1->channels() == tensor2->channels())
        << "Tensors shape are not adapting";  // 检查通道数是否兼容
    
    // 进行广播操作，使两个张量形状一致
    const auto& [input_tensor1, input_tensor2] =
        TensorBroadcast(tensor1, tensor2);
    
    // 检查广播后的形状是否一致
    CHECK(input_tensor1->shapes() == input_tensor2->shapes());
    
    // 创建输出张量并执行加法运算
    sftensor output_tensor = TensorCreate(input_tensor1->shapes());
    output_tensor->set_data(input_tensor1->data() + input_tensor2->data());
    return output_tensor;
  }
}

/**
 * @brief 张量元素级乘法（返回新张量）
 * @param tensor1 第一个输入张量
 * @param tensor2 第二个输入张量
 * @return 包含乘法结果的新张量
 * @details 将两个张量进行元素级乘法，返回一个新的张量对象
 * 如果形状不同，会进行广播操作
 */
std::shared_ptr<Tensor<float>> TensorElementMultiply(
    const std::shared_ptr<Tensor<float>>& tensor1,
    const std::shared_ptr<Tensor<float>>& tensor2) {
  CHECK(tensor1 != nullptr && tensor2 != nullptr);  // 检查输入张量是否为空
  
  if (tensor1->shapes() == tensor2->shapes()) {
    // 如果形状相同，直接进行乘法运算
    sftensor output_tensor = TensorCreate(tensor1->shapes());  // 创建输出张量
    output_tensor->set_data(tensor1->data() % tensor2->data());
    return output_tensor;
  } else {
    // 如果形状不同，需要进行广播操作
    CHECK(tensor1->channels() == tensor2->channels())
        << "Tensors shape are not adapting";  // 检查通道数是否兼容
    
    // 进行广播操作，使两个张量形状一致
    const auto& [input_tensor1, input_tensor2] =
        TensorBroadcast(tensor1, tensor2);
    
    // 检查广播后的形状是否一致
    CHECK(input_tensor1->shapes() == input_tensor2->shapes());
    
    // 创建输出张量并执行乘法运算
    sftensor output_tensor = TensorCreate(input_tensor1->shapes());
    output_tensor->set_data(input_tensor1->data() % input_tensor2->data());
    return output_tensor;
  }
}

/**
 * @brief 创建指定维度的张量
 * @param channels 通道数
 * @param rows 行数
 * @param cols 列数
 * @return 新创建的张量
 * @details 创建一个指定维度的张量，使用智能指针管理内存
 */
std::shared_ptr<Tensor<float>> TensorCreate(uint32_t channels, uint32_t rows,
                                            uint32_t cols) {
  return std::make_shared<Tensor<float>>(channels, rows, cols);
}

std::shared_ptr<Tensor<float>> TensorCreate(
    const std::vector<uint32_t>& shapes) {
  CHECK(shapes.size() == 3);
  return TensorCreate(shapes.at(0), shapes.at(1), shapes.at(2));
}

std::shared_ptr<Tensor<float>> TensorPadding(
    const std::shared_ptr<Tensor<float>>& tensor,
    const std::vector<uint32_t>& pads, float padding_value) {
  CHECK(tensor != nullptr && !tensor->empty());
  CHECK(pads.size() == 4);
  uint32_t pad_rows1 = pads.at(0);  // up
  uint32_t pad_rows2 = pads.at(1);  // bottom
  uint32_t pad_cols1 = pads.at(2);  // left
  uint32_t pad_cols2 = pads.at(3);  // right

  std::shared_ptr<ftensor> output = std::make_shared<ftensor>(
      tensor->channels(), tensor->rows() + pad_rows1 + pad_rows2,
      tensor->cols() + pad_cols1 + pad_cols2);

  const uint32_t channels = tensor->channels();
  for (uint32_t channel = 0; channel < channels; ++channel) {
    const arma::fmat& in_channel = tensor->slice(channel);
    arma::fmat& output_channel = output->slice(channel);
    const uint32_t in_channel_width = in_channel.n_cols;
    const uint32_t in_channel_height = in_channel.n_rows;

    for (uint32_t w = 0; w < in_channel_width; ++w) {
      float* output_channel_ptr =
          const_cast<float*>(output_channel.colptr(w + pad_cols1));
      const float* in_channel_ptr = in_channel.colptr(w);
      for (uint32_t h = 0; h < in_channel_height; ++h) {
        const float value = *(in_channel_ptr + h);
        *(output_channel_ptr + h + pad_rows1) = value;
      }

      for (uint32_t h = 0; h < pad_rows1; ++h) {
        *(output_channel_ptr + h) = padding_value;
      }

      for (uint32_t h = 0; h < pad_rows2; ++h) {
        *(output_channel_ptr + in_channel_height + pad_rows1 + h) =
            padding_value;
      }
    }

    for (uint32_t w = 0; w < pad_cols1; ++w) {
      float* output_channel_ptr = const_cast<float*>(output_channel.colptr(w));
      for (uint32_t h = 0; h < in_channel_height + pad_rows1 + pad_rows2; ++h) {
        *(output_channel_ptr + h) = padding_value;
      }
    }

    for (uint32_t w = 0; w < pad_cols2; ++w) {
      float* output_channel_ptr = const_cast<float*>(
          output_channel.colptr(pad_cols1 + w + in_channel_width));
      for (uint32_t h = 0; h < in_channel_height + pad_rows1 + pad_rows2; ++h) {
        *(output_channel_ptr + h) = padding_value;
      }
    }
  }
  return output;
}

std::tuple<sftensor, sftensor> TensorBroadcast(const sftensor& tensor1,
                                               const sftensor& tensor2) {
  CHECK(tensor1 != nullptr && tensor2 != nullptr);
  if (tensor1->shapes() == tensor2->shapes()) {
    return {tensor1, tensor2};
  } else {
    CHECK(tensor1->channels() == tensor2->channels());
    if (tensor2->rows() == 1 && tensor2->cols() == 1) {
      sftensor new_tensor =
          TensorCreate(tensor2->channels(), tensor1->rows(), tensor1->cols());
      CHECK(tensor2->size() == tensor2->channels());
      for (uint32_t c = 0; c < tensor2->channels(); ++c) {
        new_tensor->slice(c).fill(tensor2->index(c));
      }
      return {tensor1, new_tensor};
    } else if (tensor1->rows() == 1 && tensor1->cols() == 1) {
      sftensor new_tensor =
          TensorCreate(tensor1->channels(), tensor2->rows(), tensor2->cols());
      CHECK(tensor1->size() == tensor1->channels());
      for (uint32_t c = 0; c < tensor1->channels(); ++c) {
        new_tensor->slice(c).fill(tensor1->index(c));
      }
      return {new_tensor, tensor2};
    } else {
      LOG(FATAL) << "Broadcast shape is not adapting!";
      return {tensor1, tensor2};
    }
  }
}

std::shared_ptr<Tensor<float>> TensorClone(
    std::shared_ptr<Tensor<float>> tensor) {
  return std::make_shared<Tensor<float>>(*tensor);
}
}  // namespace kuiper_infer