//
// Created by fss on 23-2-27.
//
/**
 * @file runtime_op.cpp
 * @brief 运行时算子的实现文件
 * @details 该文件实现了运行时算子的析构函数，负责清理算子中的参数资源
 */

#include "runtime/runtime_op.hpp"
#include "data/tensor_util.hpp"

namespace kuiper_infer {

/**
 * @brief 运行时算子的析构函数
 * @details 该函数会在算子对象销毁时自动调用，负责清理算子中所有参数的内存
 * 使用范围for循环遍历所有参数，删除每个参数对象并置空指针
 */
RuntimeOperator::~RuntimeOperator() {
  // 遍历所有参数，释放内存
  for (auto& [_, param] : this->params) {
    if (param != nullptr) {
      delete param;        // 删除参数对象
      param = nullptr;     // 将指针置空，避免悬空指针
    }
  }
}

}  // namespace kuiper_infer
