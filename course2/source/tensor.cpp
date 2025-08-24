//
// Created by fss on 22-11-12.
//

#include "data/tensor.hpp"
#include <glog/logging.h>  // Google 的日志库，用来做断言和输出日志
#include <memory>
#include <numeric>

namespace kuiper_infer {

/**
 * Tensor 类构造函数
 * 初始化一个三维的 tensor (channels, rows, cols)
 * 内部用 Armadillo 的 fcube（float 三维矩阵）存储数据
 */
Tensor<float>::Tensor(uint32_t channels, uint32_t rows, uint32_t cols) {
  // 创建一个 (rows x cols x channels) 的 3D 矩阵
  data_ = arma::fcube(rows, cols, channels);

  // 设置 raw_shapes_（原始 shape，方便 reshape 和 debug）
  if (channels == 1 && rows == 1) {
    this->raw_shapes_ = std::vector<uint32_t>{cols};  // 一维向量
  } else if (channels == 1) {
    this->raw_shapes_ = std::vector<uint32_t>{rows, cols};  // 二维矩阵
  } else {
    this->raw_shapes_ = std::vector<uint32_t>{channels, rows, cols};  // 三维张量
  }
}

// 构造函数：一维向量
Tensor<float>::Tensor(uint32_t size) {
  data_ = arma::fcube(1, size, 1); // 行=1, 列=size, 通道=1
  this->raw_shapes_ = std::vector<uint32_t>{size};
}

// 构造函数：二维矩阵
Tensor<float>::Tensor(uint32_t rows, uint32_t cols) {
  data_ = arma::fcube(rows, cols, 1);
  this->raw_shapes_ = std::vector<uint32_t>{rows, cols};
}

// 构造函数：传入 shape 向量（支持 1D/2D/3D）
Tensor<float>::Tensor(const std::vector<uint32_t>& shapes) {
  CHECK(!shapes.empty() && shapes.size() <= 3); // 检查合法性

  // 如果不足 3 维，则前面补 1
  uint32_t remaining = 3 - shapes.size();
  std::vector<uint32_t> shapes_(3, 1);
  std::copy(shapes.begin(), shapes.end(), shapes_.begin() + remaining);

  uint32_t channels = shapes_.at(0);
  uint32_t rows = shapes_.at(1);
  uint32_t cols = shapes_.at(2);

  data_ = arma::fcube(rows, cols, channels);

  // 设置 raw_shapes_（恢复输入的真实维度格式）
  if (channels == 1 && rows == 1) {
    this->raw_shapes_ = std::vector<uint32_t>{cols};
  } else if (channels == 1) {
    this->raw_shapes_ = std::vector<uint32_t>{rows, cols};
  } else {
    this->raw_shapes_ = std::vector<uint32_t>{channels, rows, cols};
  }
}

// 拷贝构造函数
Tensor<float>::Tensor(const Tensor& tensor) {
  if (this != &tensor) {
    this->data_ = tensor.data_;
    this->raw_shapes_ = tensor.raw_shapes_;
  }
}

// 移动构造函数
Tensor<float>::Tensor(Tensor<float>&& tensor) noexcept {
  if (this != &tensor) {
    this->data_ = std::move(tensor.data_); // 移动而不是拷贝，提高性能
    this->raw_shapes_ = tensor.raw_shapes_;
  }
}

// 移动赋值运算符
Tensor<float>& Tensor<float>::operator=(Tensor<float>&& tensor) noexcept {
  if (this != &tensor) {
    this->data_ = std::move(tensor.data_);
    this->raw_shapes_ = tensor.raw_shapes_;
  }
  return *this;
}

// 拷贝赋值运算符
Tensor<float>& Tensor<float>::operator=(const Tensor& tensor) {
  if (this != &tensor) {
    this->data_ = tensor.data_;
    this->raw_shapes_ = tensor.raw_shapes_;
  }
  return *this;
}

// 获取行数
uint32_t Tensor<float>::rows() const {
  CHECK(!this->data_.empty());
  return this->data_.n_rows;
}

// 获取列数
uint32_t Tensor<float>::cols() const {
  CHECK(!this->data_.empty());
  return this->data_.n_cols;
}

// 获取通道数
uint32_t Tensor<float>::channels() const {
  CHECK(!this->data_.empty());
  return this->data_.n_slices;
}

// 获取元素总数
uint32_t Tensor<float>::size() const {
  CHECK(!this->data_.empty());
  return this->data_.size();
}

// 设置数据（必须 shape 一致）
void Tensor<float>::set_data(const arma::fcube& data) {
  CHECK(data.n_rows == this->data_.n_rows);
  CHECK(data.n_cols == this->data_.n_cols);
  CHECK(data.n_slices == this->data_.n_slices);
  this->data_ = data;
}

// 判断是否为空
bool Tensor<float>::empty() const { return this->data_.empty(); }

// 一维下标访问（只读）
float Tensor<float>::index(uint32_t offset) const {
  CHECK(offset < this->data_.size()) << "Tensor index out of bound!";
  return this->data_.at(offset);
}

// 一维下标访问（可写）
float& Tensor<float>::index(uint32_t offset) {
  CHECK(offset < this->data_.size()) << "Tensor index out of bound!";
  return this->data_.at(offset);
}

// 返回标准化的 shape（始终是 3 维格式）
std::vector<uint32_t> Tensor<float>::shapes() const {
  CHECK(!this->data_.empty());
  return {this->channels(), this->rows(), this->cols()};
}

// 获取底层数据引用
arma::fcube& Tensor<float>::data() { return this->data_; }
const arma::fcube& Tensor<float>::data() const { return this->data_; }

// 获取某个通道的二维矩阵
arma::fmat& Tensor<float>::slice(uint32_t channel) {
  CHECK_LT(channel, this->channels());
  return this->data_.slice(channel);
}
const arma::fmat& Tensor<float>::slice(uint32_t channel) const {
  CHECK_LT(channel, this->channels());
  return this->data_.slice(channel);
}

// 三维下标访问
float Tensor<float>::at(uint32_t channel, uint32_t row, uint32_t col) const {
  CHECK_LT(row, this->rows());
  CHECK_LT(col, this->cols());
  CHECK_LT(channel, this->channels());
  return this->data_.at(row, col, channel);
}
float& Tensor<float>::at(uint32_t channel, uint32_t row, uint32_t col) {
  CHECK_LT(row, this->rows());
  CHECK_LT(col, this->cols());
  CHECK_LT(channel, this->channels());
  return this->data_.at(row, col, channel);
}

/**
 * Padding：对张量四周填充
 * pads = {上, 下, 左, 右}
 */
void Tensor<float>::Padding(const std::vector<uint32_t>& pads,
                            float padding_value) {
  CHECK(!this->data_.empty());
  CHECK_EQ(pads.size(), 4);

  uint32_t pad_rows1 = pads.at(0);  // 上填充行数
  uint32_t pad_rows2 = pads.at(1);  // 下填充行数
  uint32_t pad_cols1 = pads.at(2);  // 左填充列数
  uint32_t pad_cols2 = pads.at(3);  // 右填充列数

  uint32_t new_rows = this->rows() + pad_rows1 + pad_rows2;
  uint32_t new_cols = this->cols() + pad_cols1 + pad_cols2;
  uint32_t channels = this->channels();

  arma::fcube new_data(new_rows, new_cols, channels);
  new_data.fill(padding_value); // 先填充默认值

  // 把原始数据复制到新区域的中间
  for (uint32_t c = 0; c < channels; ++c) {
    new_data.slice(c).submat(pad_rows1, pad_cols1,
                             pad_rows1 + this->rows() - 1,
                             pad_cols1 + this->cols() - 1) = this->data_.slice(c);
  }

  // 更新 data 和 shape
  this->data_ = new_data;
  this->raw_shapes_ = {channels, new_rows, new_cols};
}

// 将所有元素填充为某个值
void Tensor<float>::Fill(float value) {
  CHECK(!this->data_.empty());
  this->data_.fill(value);
}

// 用向量填充数据
void Tensor<float>::Fill(const std::vector<float>& values, bool row_major) {
  CHECK(!this->data_.empty());
  const uint32_t total_elems = this->data_.size();
  CHECK_EQ(values.size(), total_elems);

  if (row_major) {
    // 按行填充
    const uint32_t rows = this->rows();
    const uint32_t cols = this->cols();
    const uint32_t planes = rows * cols;
    const uint32_t channels = this->data_.n_slices;

    for (uint32_t i = 0; i < channels; ++i) {
      auto& channel_data = this->data_.slice(i);
      const arma::fmat& channel_data_t =
          arma::fmat(values.data() + i * planes, this->cols(), this->rows());
      channel_data = channel_data_t.t();
    }
  } else {
    // 直接拷贝
    std::copy(values.begin(), values.end(), this->data_.memptr());
  }
}

// 打印张量内容
void Tensor<float>::Show() {
  for (uint32_t i = 0; i < this->channels(); ++i) {
    LOG(INFO) << "Channel: " << i;
    LOG(INFO) << "\n" << this->data_.slice(i);
  }
}

/**
 * Flatten：展平张量
 * - row_major = true: 展平成一维向量（行优先）
 * - row_major = false: 直接按内存布局展平
 */
void Tensor<float>::Flatten(bool row_major) {
  CHECK(!this->data_.empty());

  std::vector<float> vals = this->values(row_major); // 先拿到数据
  this->data_.reshape(1, vals.size(), 1); // 变成 (1, N, 1)

  this->raw_shapes_ = {static_cast<uint32_t>(vals.size())};
  this->Fill(vals, true);
}

// 随机初始化
void Tensor<float>::Rand() {
  CHECK(!this->data_.empty());
  this->data_.randn();
}

// 填充为 1
void Tensor<float>::Ones() {
  CHECK(!this->data_.empty());
  this->Fill(1.f);
}

// 应用函数变换
void Tensor<float>::Transform(const std::function<float(float)>& filter) {
  CHECK(!this->data_.empty());
  this->data_.transform(filter);
}

// 获取原始 shape
const std::vector<uint32_t>& Tensor<float>::raw_shapes() const {
  CHECK(!this->raw_shapes_.empty());
  return this->raw_shapes_;
}

// 改变 shape
void Tensor<float>::Reshape(const std::vector<uint32_t>& shapes,
                            bool row_major) {
  CHECK(!this->data_.empty());
  CHECK(!shapes.empty());

  // 原始大小必须一致
  const uint32_t origin_size = this->size();
  const uint32_t current_size =
      std::accumulate(shapes.begin(), shapes.end(), 1, std::multiplies());
  CHECK(current_size == origin_size);

  std::vector<float> values;
  if (row_major) {
    values = this->values(true);
  }

  if (shapes.size() == 3) {
    this->data_.reshape(shapes.at(1), shapes.at(2), shapes.at(0));
    this->raw_shapes_ = {shapes.at(0), shapes.at(1), shapes.at(2)};
  } else if (shapes.size() == 2) {
    this->data_.reshape(shapes.at(0), shapes.at(1), 1);
    this->raw_shapes_ = {shapes.at(0), shapes.at(1)};
  } else {
    this->data_.reshape(1, shapes.at(0), 1);
    this->raw_shapes_ = {shapes.at(0)};
  }

  if (row_major) {
    this->Fill(values, true);
  }
}

// 获取底层指针
float* Tensor<float>::raw_ptr() {
  CHECK(!this->data_.empty());
  return this->data_.memptr();
}

// 带 offset 的底层指针
float* Tensor<float>::raw_ptr(uint32_t offset) {
  CHECK(!this->data_.empty());
  CHECK_LT(offset, this->size());
  return this->data_.memptr() + offset;
}

// 导出为 vector
std::vector<float> Tensor<float>::values(bool row_major) {
  CHECK(!this->data_.empty());
  std::vector<float> values(this->data_.size());

  if (!row_major) {
    std::copy(this->data_.mem, this->data_.mem + this->data_.size(),
              values.begin());
  } else {
    uint32_t index = 0;
    for (uint32_t c = 0; c < this->data_.n_slices; ++c) {
      const arma::fmat& channel = this->data_.slice(c).t();
      std::copy(channel.begin(), channel.end(), values.begin() + index);
      index += channel.size();
    }
  }
  return values;
}

// 获取某个通道的起始指针
float* Tensor<float>::matrix_raw_ptr(uint32_t index) {
  CHECK_LT(index, this->channels());
  uint32_t offset = index * this->rows() * this->cols();
  return this->raw_ptr() + offset;
}

}  // namespace kuiper_infer
