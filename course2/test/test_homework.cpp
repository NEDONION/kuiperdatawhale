//
// Created by fss on 23-6-4.
//
#include "data/tensor.hpp"
#include <glog/logging.h>
#include <gtest/gtest.h>

/**
 * 测试 1：三维张量的 Flatten（行优先 row_major = true）
 *
 * 构造一个形状为 (channels=2, rows=3, cols=4) 的张量：
 * - 总元素数 = 2 * 3 * 4 = 24
 * - Flatten(true) 后应当变成一维向量，raw_shapes() 只剩一个维度 [24]
 *
 * 说明：
 * - 这里我们只 **检查形状是否正确**，不校验元素排列顺序（行优先 vs 内存顺序）。
 * - 如需验证顺序，可以在 Flatten 前按序填充值再比对，这里为入门先聚焦形状。
 */
TEST(test_homework, homework1_flatten1) {
  using namespace kuiper_infer;
  Tensor<float> f1(2, 3, 4);  // (C=2, R=3, C=4)
  LOG(INFO) << "-------------------before Flatten-------------------";
  f1.Show();                  // 打印各通道矩阵（默认未初始化，可能是随机/未定义值）

  f1.Flatten(true);           // 行优先展平为 (1, 24, 1)

  LOG(INFO) << "-------------------after Flatten-------------------";
  f1.Show();

  // 断言：展平后 raw_shapes 只有一个维度
  ASSERT_EQ(f1.raw_shapes().size(), 1);
  // 断言：维度大小为总元素个数 24
  ASSERT_EQ(f1.raw_shapes().at(0), 24);
}

/**
 * 测试 2：二维张量的 Flatten（行优先 row_major = true）
 *
 * 构造一个形状为 (rows=12, cols=24) 的二维张量（等价三维：C=1, R=12, C=24）：
 * - 总元素数 = 12 * 24 = 288
 * - Flatten(true) 后应当变成一维向量，raw_shapes() = [288]
 */
TEST(test_homework, homework1_flatten2) {
  using namespace kuiper_infer;
  Tensor<float> f1(12, 24);   // (R=12, C=24, C=1)
  LOG(INFO) << "-------------------before Flatten-------------------";
  f1.Show();

  f1.Flatten(true);           // 展平为 (1, 288, 1)

  LOG(INFO) << "-------------------after Flatten-------------------";
  f1.Show();

  ASSERT_EQ(f1.raw_shapes().size(), 1);
  ASSERT_EQ(f1.raw_shapes().at(0), 24 * 12);
}

/**
 * 测试 3：Padding 基本功能（非对称 padding）
 *
 * 初始张量：形状 (channels=3, rows=4, cols=5)，全填充为 1。
 * 执行 padding({1, 2, 3, 4}, 0)：
 * - 上 pad 1 行、下 pad 2 行、左 pad 3 列、右 pad 4 列，pad 值为 0。
 * - 新行数 = 4 + 1 + 2 = 7
 * - 新列数 = 5 + 3 + 4 = 12
 * - 中间原位置（r ∈ [1,4], c ∈ [3,7]）应该都是 1，周围 padding 区域为 0。
 */
TEST(test_homework, homework2_padding1) {
  using namespace kuiper_infer;

  // 构造并检查初始形状
  Tensor<float> tensor(3, 4, 5);   // (C=3, R=4, C=5)
  ASSERT_EQ(tensor.channels(), 3);
  ASSERT_EQ(tensor.rows(), 4);
  ASSERT_EQ(tensor.cols(), 5);

  tensor.Fill(1.f);                // 原区域全部填充 1，便于观察 padding 的差异

  LOG(INFO) << "-------------------before padding-------------------";
  tensor.Show();

  // 上 1、下 2、左 3、右 4，pad 值 0
  tensor.Padding({1, 2, 3, 4}, 0.f);

  LOG(INFO) << "-------------------after padding-------------------";
  tensor.Show();

  // 形状发生变化：行 + (1+2)，列 + (3+4)
  ASSERT_EQ(tensor.rows(), 7);
  ASSERT_EQ(tensor.cols(), 12);

  // 校验：中间原数据区域应保持为 1；其他 padding 区域应为 0
  for (int c = 0; c < static_cast<int>(tensor.channels()); ++c) {
    for (int r = 0; r < static_cast<int>(tensor.rows()); ++r) {
      for (int c_ = 0; c_ < static_cast<int>(tensor.cols()); ++c_) {
        // 原数据映射到新张量中的行列范围：
        // 行：原来 0..3 -> 现在 1..4 （上 padding 1）
        // 列：原来 0..4 -> 现在 3..7 （左 padding 3）
        const bool in_original_rows = (r >= 1 && r <= 4);
        const bool in_original_cols = (c_ >= 3 && c_ <= 7);
        if (in_original_rows && in_original_cols) {
          // 原区域仍为 1
          ASSERT_EQ(tensor.at(c, r, c_), 1.f) << "channel=" << c
                                              << " row=" << r
                                              << " col=" << c_;
        } else {
          // 其余 padding 区域为 0
          ASSERT_EQ(tensor.at(c, r, c_), 0.f)  << "padding mismatch at: "
                                               << "channel=" << c
                                               << " row=" << r
                                               << " col=" << c_;
        }
      }
    }
  }
}

/**
 * 测试 4：Padding 对称填充（每侧都 pad 2），且 pad 值为 3.14
 *
 * 初始张量：形状 (channels=3, rows=4, cols=5)，全 1。
 * 执行 padding({2, 2, 2, 2}, 3.14)：
 * - 新行数 = 4 + 2 + 2 = 8
 * - 新列数 = 5 + 2 + 2 = 9
 * - 原数据在新张量中的位置：行 2..5、列 2..6（0-based）
 * - 断言：四周 padding 区域为 3.14，中间原区域为 1。
 */
TEST(test_homework, homework2_padding2) {
  using namespace kuiper_infer;

  // 【修正 1】这里原代码写成了 ftensor，是个笔误；应为 Tensor<float>
  Tensor<float> tensor(3, 4, 5);
  ASSERT_EQ(tensor.channels(), 3);
  ASSERT_EQ(tensor.rows(), 4);
  ASSERT_EQ(tensor.cols(), 5);

  tensor.Fill(1.f);
  tensor.Padding({2, 2, 2, 2}, 3.14f);  // 四周各补 2，pad 值为 3.14

  ASSERT_EQ(tensor.rows(), 8);          // 4 + 2 + 2
  ASSERT_EQ(tensor.cols(), 9);          // 5 + 2 + 2

  const int rows = static_cast<int>(tensor.rows());
  const int cols = static_cast<int>(tensor.cols());

  for (int ch = 0; ch < static_cast<int>(tensor.channels()); ++ch) {
    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {

        // 【修正 2】原代码误用 channel 索引 c 与列数比较：
        //   if (c_ <= 1 || r <= 1) {...}
        //   else if (c >= tensor.cols() - 1 || r >= tensor.rows() - 1) {...}
        // 应该用列索引 c 与 cols 比较，而不是 channel 索引。

        const bool is_top_padding    = (r <= 1);          // 顶部两行
        const bool is_left_padding   = (c <= 1);          // 左边两列
        const bool is_bottom_padding = (r >= rows - 2);   // 底部两行（6、7）
        const bool is_right_padding  = (c >= cols - 2);   // 右边两列（7、8）

        const bool in_original_rows = (r >= 2 && r <= 5); // 原数据行范围
        const bool in_original_cols = (c >= 2 && c <= 6); // 原数据列范围
        const bool in_original_area = in_original_rows && in_original_cols;

        if (is_top_padding || is_left_padding || is_bottom_padding || is_right_padding) {
          // 四周的 padding 区域都应是 3.14f
          ASSERT_EQ(tensor.at(ch, r, c), 3.14f) << "channel=" << ch
                                                << " row=" << r
                                                << " col=" << c;
        }
        if (in_original_area) {
          // 中间原区域应保持为 1
          ASSERT_EQ(tensor.at(ch, r, c), 1.f) << "channel=" << ch
                                              << " row=" << r
                                              << " col=" << c;
        }
      }
    }
  }
}
