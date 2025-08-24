//
// Created by fss on 23-6-4.
//
#include "data/tensor.hpp"
#include <glog/logging.h>
#include <gtest/gtest.h>

/**
 * 测试 1：一维张量 (1D Tensor)
 *
 * 构造：Tensor<float> f1(4)
 * - 即 shape = {4}
 * - 内部表示为 (rows=1, cols=4, channels=1)，但 raw_shapes() 存储为 {4}
 *
 * 测试点：
 * - raw_shapes().size() == 1
 * - raw_shapes()[0] == 4
 * - Show() 打印出一行 4 列的矩阵
 */
TEST(test_tensor, tensor_init1D) {
  using namespace kuiper_infer;

  Tensor<float> f1(4);   // 一维向量：长度为 4
  f1.Fill(1.f);          // 填充全 1，方便观察

  const auto &raw_shapes = f1.raw_shapes();
  LOG(INFO) << "-----------------------Tensor1D-----------------------";
  LOG(INFO) << "raw shapes size: " << raw_shapes.size();  // 1
  const uint32_t size = raw_shapes.at(0);  // 向量长度
  LOG(INFO) << "data numbers: " << size;   // 4
  f1.Show();                               // 打印内容
}

/**
 * 测试 2：二维张量 (2D Tensor)
 *
 * 构造：Tensor<float> f1(4, 4)
 * - 即 shape = {4, 4}
 * - 内部表示为 (rows=4, cols=4, channels=1)，raw_shapes() 存储为 {4, 4}
 *
 * 测试点：
 * - raw_shapes().size() == 2
 * - raw_shapes()[0] == 4 (行数)
 * - raw_shapes()[1] == 4 (列数)
 */
TEST(test_tensor, tensor_init2D) {
  using namespace kuiper_infer;

  Tensor<float> f1(4, 4);  // 二维矩阵：4x4
  f1.Fill(1.f);            // 填充全 1

  const auto &raw_shapes = f1.raw_shapes();
  LOG(INFO) << "-----------------------Tensor2D-----------------------";
  LOG(INFO) << "raw shapes size: " << raw_shapes.size();  // 2
  const uint32_t rows = raw_shapes.at(0);
  const uint32_t cols = raw_shapes.at(1);

  LOG(INFO) << "data rows: " << rows;  // 4
  LOG(INFO) << "data cols: " << cols;  // 4
  f1.Show();                           // 打印矩阵
}

/**
 * 测试 3：三维张量 (3D Tensor)
 *
 * 构造：Tensor<float> f1(2, 3, 4)
 * - 即 shape = {2, 3, 4}
 * - 表示有 2 个通道，每个通道是 3x4 的矩阵
 * - raw_shapes() = {2, 3, 4}
 *
 * 测试点：
 * - raw_shapes().size() == 3
 * - raw_shapes()[0] == 2 (通道数)
 * - raw_shapes()[1] == 3 (行数)
 * - raw_shapes()[2] == 4 (列数)
 */
TEST(test_tensor, tensor_init3D_3) {
  using namespace kuiper_infer;

  Tensor<float> f1(2, 3, 4);  // 三维张量：2 个通道，每个通道 3x4
  f1.Fill(1.f);

  const auto &raw_shapes = f1.raw_shapes();
  LOG(INFO) << "-----------------------Tensor3D 3-----------------------";
  LOG(INFO) << "raw shapes size: " << raw_shapes.size();  // 3
  const uint32_t channels = raw_shapes.at(0);
  const uint32_t rows = raw_shapes.at(1);
  const uint32_t cols = raw_shapes.at(2);

  LOG(INFO) << "data channels: " << channels;  // 2
  LOG(INFO) << "data rows: " << rows;          // 3
  LOG(INFO) << "data cols: " << cols;          // 4
  f1.Show();                                   // 打印两个通道的矩阵
}

/**
 * 测试 4：特殊的三维张量 (channels=1)
 *
 * 构造：Tensor<float> f1(1, 2, 3)
 * - 内部是 (C=1, R=2, C=3)
 * - 因为通道=1，所以 raw_shapes() 存储为 {2, 3}，即退化为二维
 *
 * 测试点：
 * - raw_shapes().size() == 2
 * - raw_shapes()[0] == 2 (行数)
 * - raw_shapes()[1] == 3 (列数)
 */
TEST(test_tensor, tensor_init3D_2) {
  using namespace kuiper_infer;

  Tensor<float> f1(1, 2, 3);  // 1 个通道，2x3
  f1.Fill(1.f);

  const auto &raw_shapes = f1.raw_shapes();
  LOG(INFO) << "-----------------------Tensor3D 2-----------------------";
  LOG(INFO) << "raw shapes size: " << raw_shapes.size();  // 2
  const uint32_t rows = raw_shapes.at(0);
  const uint32_t cols = raw_shapes.at(1);

  LOG(INFO) << "data rows: " << rows;  // 2
  LOG(INFO) << "data cols: " << cols;  // 3
  f1.Show();                           // 打印 2x3 矩阵
}

/**
 * 测试 5：更特殊的三维张量 (channels=1, rows=1)
 *
 * 构造：Tensor<float> f1(1, 1, 3)
 * - 内部是 (C=1, R=1, C=3)
 * - 因为通道=1 且行=1，所以 raw_shapes() 存储为 {3}，即退化为一维
 *
 * 测试点：
 * - raw_shapes().size() == 1
 * - raw_shapes()[0] == 3
 */
TEST(test_tensor, tensor_init3D_1) {
  using namespace kuiper_infer;

  Tensor<float> f1(1, 1, 3);  // 实际上一维长度为 3
  f1.Fill(1.f);

  const auto &raw_shapes = f1.raw_shapes();
  LOG(INFO) << "-----------------------Tensor3D 1-----------------------";
  LOG(INFO) << "raw shapes size: " << raw_shapes.size();  // 1
  const uint32_t size = raw_shapes.at(0);

  LOG(INFO) << "data numbers: " << size;  // 3
  f1.Show();                              // 打印一行 3 列
}
