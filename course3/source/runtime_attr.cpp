//
// Created by fss on 23-3-4.
//
/**
 * @file runtime_attr.cpp
 * @brief 运行时属性的实现文件
 * @details 该文件实现了运行时属性的管理功能，包括权重的清理等操作
 */

#include "runtime/runtime_attr.hpp"

namespace kuiper_infer {

/**
 * @brief 清理权重数据
 * @details 该函数会清空权重数据容器，释放内存空间
 * 使用swap技巧来确保内存被完全释放，这是一种高效的清空vector的方法
 */
void RuntimeAttribute::ClearWeight() {
  if (!this->weight_data.empty()) {
    // 创建一个空的临时vector，然后与当前权重数据交换
    // 这样可以确保原有权重数据被完全释放
    std::vector<char> tmp = std::vector<char>();
    this->weight_data.swap(tmp);
  }
}

}  // namespace kuiper_infer