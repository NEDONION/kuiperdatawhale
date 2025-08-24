//
// Created by fss on 23-6-4.
//
#include "data/tensor.hpp"
#include <glog/logging.h>
#include <gtest/gtest.h>

/**
 * 测试 1：Fill(values)
 *
 * 功能：
 * - 构造一个三维张量 (channels=2, rows=3, cols=4)，即总元素数 = 2*3*4 = 24。
 * - 使用 Fill(values) 将向量中的数据依次填充到 Tensor 里。
 *
 * 说明：
 * - 这里 values 是 [1, 2, 3, ..., 24]。
 * - 默认 Fill(values) 使用内存顺序 (column-major，因为内部是 Armadillo)，
 *   但你也可以传 Fill(values, true) 以 row-major 方式填充。
 */
TEST(test_fill_reshape, fill1) {
  using namespace kuiper_infer;

  // 创建一个 (2,3,4) 的三维张量：2 通道，每通道 3x4
  Tensor<float> f1(2, 3, 4);

  // 准备一个长度为 24 的数组，存放 [1,2,...,24]
  std::vector<float> values(2 * 3 * 4);
  for (int i = 0; i < 24; ++i) {
    values.at(i) = float(i + 1);
  }

  // 将 values 填充到张量中
  f1.Fill(values);

  // 打印张量（每个通道都会打印一块矩阵）
  f1.Show();
}

/**
 * 测试 2：Reshape(shapes, row_major)
 *
 * 功能：
 * - 构造一个 (2,3,4) 的张量，并填充 [1..24]。
 * - 执行 Reshape({4,3,2}, true)，即改变形状。
 *
 * 说明：
 * - 原始 shape = (2,3,4)，即 channels=2, rows=3, cols=4。
 * - 新 shape = (4,3,2)，即 channels=4, rows=3, cols=2。
 * - 注意：Reshape 不会改变元素总数，只是“换一种分组方式”看同样的数据。
 * - 参数 row_major=true 表示保持 **行优先的顺序** 展平后再重排。
 */
TEST(test_fill_reshape, reshape1) {
  using namespace kuiper_infer;

  LOG(INFO) << "-------------------Reshape-------------------";

  // 创建 (2,3,4) 的三维张量
  Tensor<float> f1(2, 3, 4);

  // 准备一个长度为 24 的数组，存放 [1,2,...,24]
  std::vector<float> values(2 * 3 * 4);
  for (int i = 0; i < 24; ++i) {
    values.at(i) = float(i + 1);
  }

  // 填充数据
  f1.Fill(values);
  f1.Show();  // 打印原始张量

  /// 将大小调整为 (4,3,2)
  /// - 原数据总数 = 24
  /// - 新数据总数 = 4*3*2 = 24
  /// - 所以 reshape 合法
  f1.Reshape({4, 3, 2}, true);

  LOG(INFO) << "-------------------After Reshape-------------------";
  f1.Show();  // 打印变形后的张量
}
