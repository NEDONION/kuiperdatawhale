//
// Created by fss on 23-6-4.
//
#include "data/tensor.hpp"
#include <glog/logging.h>
#include <gtest/gtest.h>

// 一个普通的 一元函数 示例：把输入减 1
// 作为函数指针传给 Tensor::Transform（transform 接受任意可调用对象）
float MinusOne(float value) { return value - 1.f; }

TEST(test_transform, transform1) {
  using namespace kuiper_infer;

  // 构造一个三维张量：channels=2, rows=3, cols=4（共 24 个元素）
  Tensor<float> f1(2, 3, 4);

  // 用随机数初始化（Armadillo 的 randn，近似 ~N(0,1) 的正态分布）
  f1.Rand();

  // 变换前打印每个通道的矩阵
  f1.Show();

  // 对张量逐元素应用函数：x -> x - 1（就地修改，无额外拷贝）
  // 实际调用的是 arma::Cube::transform(functor)
  f1.Transform(MinusOne);

  // 变换后再次打印，便于对比
  f1.Show();
}
