//
// Created by fss on 23-6-4.
//

#include "data/tensor.hpp"
#include <glog/logging.h>
#include <gtest/gtest.h>

/**
 * 演示如何读取 Tensor 的数据：
 * 1) 用 Rand() 随机初始化
 * 2) 用 Show() 打印所有通道
 * 3) 用 slice(0) 读取第 0 个通道的二维矩阵
 * 4) 用 at(c, r, col) 读取单个元素（下标均为 0-based）
 */
TEST(test_tensor_values, tensor_values1) {
  using namespace kuiper_infer;

  // 构造一个三维张量：channels=2, rows=3, cols=4（共 24 个元素）
  Tensor<float> f1(2, 3, 4);

  // 随机初始化数据（Armadillo 的 randn，近似正态分布）
  f1.Rand();

  // 打印每个通道的矩阵（内部会循环输出所有通道）
  f1.Show();

  // 打印第 0 个通道（二维矩阵）。slice(0) 返回 arma::fmat&，可直接被 LOG 打印
  LOG(INFO) << "Data in the first channel:\n" << f1.slice(0);

  // 读取单个元素：参数顺序为 (channel, row, col)，全部是 0-based 下标
  // 这里取第 2 个通道（c=1）、第 2 行（r=1）、第 2 列（col=1）的元素
  LOG(INFO) << "Data at (c=1, r=1, col=1): " << f1.at(1, 1, 1);
}
