//
// Created by fss on 22-11-21.
//
/**
 * @file load_data.cpp
 * @brief 数据加载器的实现文件
 * @details 该文件实现了CSV格式数据的加载和解析功能，支持从CSV文件中读取矩阵数据
 */

#include "data/load_data.hpp"
#include <glog/logging.h>
#include <armadillo>
#include <fstream>
#include <string>
#include <utility>

namespace kuiper_infer {

/**
 * @brief 从CSV文件中加载数据到矩阵
 * @param file_path CSV文件路径
 * @param split_char 分隔符，默认为逗号
 * @return 包含CSV数据的浮点矩阵
 * @details 该函数会读取CSV文件，解析其中的数值数据，并返回一个armadillo浮点矩阵
 */
arma::fmat CSVDataLoader::LoadData(const std::string& file_path,
                                   const char split_char) {
  arma::fmat data;  // 创建空的浮点矩阵
  
  // 检查文件路径是否为空
  if (file_path.empty()) {
    LOG(ERROR) << "CSV file path is empty: " << file_path;
    return data;
  }

  // 打开CSV文件
  std::ifstream in(file_path);
  if (!in.is_open() || !in.good()) {
    LOG(ERROR) << "File open failed: " << file_path;
    return data;
  }

  std::string line_str;        // 存储每行的字符串
  std::stringstream line_stream; // 用于解析每行的数据流

  // 获取矩阵的尺寸（行数和列数）
  const auto& [rows, cols] = CSVDataLoader::GetMatrixSize(in, split_char);
  data.zeros(rows, cols);  // 初始化矩阵为指定大小，所有元素设为0

  size_t row = 0;  // 当前处理的行索引
  while (in.good()) {
    std::getline(in, line_str);  // 读取一行数据
    if (line_str.empty()) {
      break;  // 如果行为空，结束读取
    }

    std::string token;  // 存储每个分隔的数据项
    line_stream.clear();  // 清空流状态
    line_stream.str(line_str);  // 将行字符串设置到流中

    size_t col = 0;  // 当前处理的列索引
    while (line_stream.good()) {
      std::getline(line_stream, token, split_char);  // 按分隔符分割数据
      try {
        // 将字符串转换为浮点数并存储到矩阵中
        data.at(row, col) = std::stof(token);
      } catch (std::exception& e) {
        // 如果转换失败，记录错误信息
        DLOG(ERROR) << "Parse CSV File meet error: " << e.what()
                    << " row:" << row << " col:" << col;
      }
      col += 1;
      CHECK(col <= cols) << "There are excessive elements on the column";  // 检查列数是否超出预期
    }

    row += 1;
    CHECK(row <= rows) << "There are excessive elements on the row";  // 检查行数是否超出预期
  }
  return data;  // 返回填充好的矩阵
}

/**
 * @brief 获取CSV文件的矩阵尺寸
 * @param file 已打开的文件流
 * @param split_char 分隔符
 * @return 包含行数和列数的pair
 * @details 该函数会扫描整个CSV文件，统计行数和每行的最大列数，用于预分配矩阵内存
 */
std::pair<size_t, size_t> CSVDataLoader::GetMatrixSize(std::ifstream& file,
                                                       char split_char) {
  bool load_ok = file.good();  // 记录文件状态
  file.clear();  // 清除文件流的状态标志
  size_t fn_rows = 0;  // 行数计数器
  size_t fn_cols = 0;  // 最大列数
  const std::ifstream::pos_type start_pos = file.tellg();  // 记录文件开始位置

  std::string token;      // 存储每个分隔的数据项
  std::string line_str;   // 存储每行的字符串
  std::stringstream line_stream;  // 用于解析每行的数据流

  // 扫描整个文件，统计行数和最大列数
  while (file.good() && load_ok) {
    std::getline(file, line_str);  // 读取一行
    if (line_str.empty()) {
      break;  // 如果行为空，结束扫描
    }

    line_stream.clear();  // 清空流状态
    line_stream.str(line_str);  // 将行字符串设置到流中
    size_t line_cols = 0;  // 当前行的列数

    // 统计当前行的列数
    std::string row_token;
    while (line_stream.good()) {
      std::getline(line_stream, row_token, split_char);  // 按分隔符分割
      ++line_cols;  // 列数加1
    }
    
    // 更新最大列数
    if (line_cols > fn_cols) {
      fn_cols = line_cols;
    }

    ++fn_rows;  // 行数加1
  }
  
  // 恢复文件位置，以便后续读取
  file.clear();
  file.seekg(start_pos);
  
  return {fn_rows, fn_cols};  // 返回行数和最大列数
}

}  // namespace kuiper_infer