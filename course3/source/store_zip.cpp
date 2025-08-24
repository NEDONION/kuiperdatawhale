// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License slice
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

/**
 * @file store_zip.cpp
 * @brief ZIP文件存储和解析的实现文件
 * @details 该文件实现了ZIP格式文件的解析功能，包括文件头解析、CRC32校验、文件解压等
 * 主要用于处理PNNX模型文件的存储格式
 */

#include "runtime/store_zip.hpp"

#include <stdio.h>
#include <stdint.h>
#include <map>
#include <string>
#include <vector>

namespace pnnx {

// 跨平台的内存对齐宏定义
// 在MSVC编译器下使用__pragma，在其他编译器下使用__attribute__
// https://stackoverflow.com/questions/1537964/visual-c-equivalent-of-gccs-attribute-packed
#ifdef _MSC_VER
#define PACK(__Declaration__) __pragma(pack(push, 1)) __Declaration__ __pragma(pack(pop))
#else
#define PACK(__Declaration__) __Declaration__ __attribute__((__packed__))
#endif

/**
 * @brief 本地文件头结构体
 * @details 定义了ZIP文件中每个文件的本地文件头信息，包括版本、标志、压缩方式等
 * 使用PACK宏确保结构体按1字节对齐，符合ZIP文件格式规范
 */
PACK(struct local_file_header {
       uint16_t version;           // 版本号
       uint16_t flag;              // 通用位标志
       uint16_t compression;       // 压缩方法
       uint16_t last_modify_time;  // 最后修改时间
       uint16_t last_modify_date;  // 最后修改日期
       uint32_t crc32;             // CRC32校验值
       uint32_t compressed_size;   // 压缩后大小
       uint32_t uncompressed_size; // 未压缩大小
       uint16_t file_name_length;  // 文件名长度
       uint16_t extra_field_length; // 扩展字段长度
     });

/**
 * @brief 中央目录文件头结构体
 * @details 定义了ZIP文件中央目录中每个文件的头信息，包含文件的完整元数据
 * 中央目录位于ZIP文件末尾，用于快速定位文件
 */
PACK(struct central_directory_file_header {
       uint16_t version_made;      // 创建版本
       uint16_t version;           // 版本号
       uint16_t flag;              // 通用位标志
       uint16_t compression;       // 压缩方法
       uint16_t last_modify_time;  // 最后修改时间
       uint16_t last_modify_date;  // 最后修改日期
       uint32_t crc32;             // CRC32校验值
       uint32_t compressed_size;   // 压缩后大小
       uint32_t uncompressed_size; // 未压缩大小
       uint16_t file_name_length;  // 文件名长度
       uint16_t extra_field_length; // 扩展字段长度
       uint16_t file_comment_length; // 文件注释长度
       uint16_t start_disk;        // 开始磁盘号
       uint16_t internal_file_attrs; // 内部文件属性
       uint32_t external_file_attrs; // 外部文件属性
       uint32_t lfh_offset;        // 本地文件头偏移
     });

/**
 * @brief 中央目录结束记录结构体
 * @details 定义了ZIP文件中央目录的结束标记，包含目录的统计信息
 * 用于标识ZIP文件的结束位置
 */
PACK(struct end_of_central_directory_record {
       uint16_t disk_number;       // 当前磁盘号
       uint16_t start_disk;        // 开始磁盘号
       uint16_t cd_records;        // 当前磁盘上的目录记录数
       uint16_t total_cd_records;  // 总目录记录数
       uint32_t cd_size;           // 中央目录大小
       uint32_t cd_offset;         // 中央目录偏移
       uint16_t comment_length;    // 注释长度
     });

// CRC32校验表，用于快速计算CRC32值
static uint32_t CRC32_TABLE[256];

/**
 * @brief 初始化CRC32校验表
 * @details 该函数会预计算所有可能的CRC32值，存储在全局表中
 * 使用查表法可以大大提高CRC32计算的效率
 */
static void CRC32_TABLE_INIT()
{
  for (int i = 0; i < 256; i++)
  {
    uint32_t c = i;
    // 对每个字节值进行8次CRC32计算
    for (int j = 0; j < 8; j++)
    {
      if (c & 1)
        c = (c >> 1) ^ 0xedb88320;  // CRC32多项式
      else
        c >>= 1;
    }
    CRC32_TABLE[i] = c;  // 存储计算结果
  }
}

/**
 * @brief 计算单个字符的CRC32值
 * @param x 当前的CRC32值
 * @param ch 要处理的字符
 * @return 更新后的CRC32值
 * @details 使用查表法快速计算CRC32，这是CRC32算法的核心函数
 */
static uint32_t CRC32(uint32_t x, unsigned char ch)
{
  return (x >> 8) ^ CRC32_TABLE[(x ^ ch) & 0xff];
}

/**
 * @brief 计算数据缓冲区的CRC32值
 * @param data 数据缓冲区指针
 * @param len 数据长度
 * @return 计算得到的CRC32值
 * @details 该函数会遍历整个数据缓冲区，逐字节计算CRC32校验值
 */
static uint32_t CRC32_buffer(const unsigned char* data, int len)
{
  uint32_t x = 0xffffffff;  // CRC32初始值

  for (int i = 0; i < len; i++)
    x = CRC32(x, data[i]);

  return x ^ 0xffffffff;
}

StoreZipReader::StoreZipReader()
{
  fp = 0;
}

StoreZipReader::~StoreZipReader()
{
  close();
}

int StoreZipReader::open(const std::string& path)
{
  close();

  fp = fopen(path.c_str(), "rb");
  if (!fp)
  {
    fprintf(stderr, "open failed\n");
    return -1;
  }

  while (!feof(fp))
  {
    // peek signature
    uint32_t signature;
    int nread = fread((char*)&signature, sizeof(signature), 1, fp);
    if (nread != 1)
      break;

    if (signature == 0x04034b50)
    {
      local_file_header lfh;
      fread((char*)&lfh, sizeof(lfh), 1, fp);

      if (lfh.flag & 0x08)
      {
        fprintf(stderr, "zip file contains data descriptor, this is not supported yet\n");
        return -1;
      }

      if (lfh.compression != 0 || lfh.compressed_size != lfh.uncompressed_size)
      {
        fprintf(stderr, "not stored zip file %d %d\n", lfh.compressed_size, lfh.uncompressed_size);
        return -1;
      }

      // file name
      std::string name;
      name.resize(lfh.file_name_length);
      fread((char*)name.data(), name.size(), 1, fp);

      // skip extra field
      fseek(fp, lfh.extra_field_length, SEEK_CUR);

      StoreZipMeta fm;
      fm.offset = ftell(fp);
      fm.size = lfh.compressed_size;

      filemetas[name] = fm;

      //             fprintf(stderr, "%s = %d  %d\n", name.c_str(), fm.offset, fm.size);

      fseek(fp, lfh.compressed_size, SEEK_CUR);
    }
    else if (signature == 0x02014b50)
    {
      central_directory_file_header cdfh;
      fread((char*)&cdfh, sizeof(cdfh), 1, fp);

      // skip file name
      fseek(fp, cdfh.file_name_length, SEEK_CUR);

      // skip extra field
      fseek(fp, cdfh.extra_field_length, SEEK_CUR);

      // skip file comment
      fseek(fp, cdfh.file_comment_length, SEEK_CUR);
    }
    else if (signature == 0x06054b50)
    {
      end_of_central_directory_record eocdr;
      fread((char*)&eocdr, sizeof(eocdr), 1, fp);

      // skip comment
      fseek(fp, eocdr.comment_length, SEEK_CUR);
    }
    else
    {
      fprintf(stderr, "unsupported signature %x\n", signature);
      return -1;
    }
  }

  return 0;
}

size_t StoreZipReader::get_file_size(const std::string& name)
{
  if (filemetas.find(name) == filemetas.end())
  {
    fprintf(stderr, "no such file %s\n", name.c_str());
    return 0;
  }

  return filemetas[name].size;
}

int StoreZipReader::read_file(const std::string& name, char* data)
{
  if (filemetas.find(name) == filemetas.end())
  {
    fprintf(stderr, "no such file %s\n", name.c_str());
    return -1;
  }

  size_t offset = filemetas[name].offset;
  size_t size = filemetas[name].size;

  fseek(fp, offset, SEEK_SET);
  fread(data, size, 1, fp);

  return 0;
}

int StoreZipReader::close()
{
  if (!fp)
    return 0;

  fclose(fp);
  fp = 0;

  return 0;
}

StoreZipWriter::StoreZipWriter()
{
  fp = 0;

  CRC32_TABLE_INIT();
}

StoreZipWriter::~StoreZipWriter()
{
  close();
}

int StoreZipWriter::open(const std::string& path)
{
  close();

  fp = fopen(path.c_str(), "wb");
  if (!fp)
  {
    fprintf(stderr, "open failed\n");
    return -1;
  }

  return 0;
}

int StoreZipWriter::write_file(const std::string& name, const char* data, size_t size)
{
  int offset = ftell(fp);

  uint32_t signature = 0x04034b50;
  fwrite((char*)&signature, sizeof(signature), 1, fp);

  uint32_t crc32 = CRC32_buffer((const unsigned char*)data, size);

  local_file_header lfh;
  lfh.version = 0;
  lfh.flag = 0;
  lfh.compression = 0;
  lfh.last_modify_time = 0;
  lfh.last_modify_date = 0;
  lfh.crc32 = crc32;
  lfh.compressed_size = size;
  lfh.uncompressed_size = size;
  lfh.file_name_length = name.size();
  lfh.extra_field_length = 0;

  fwrite((char*)&lfh, sizeof(lfh), 1, fp);

  fwrite((char*)name.c_str(), name.size(), 1, fp);

  fwrite(data, size, 1, fp);

  StoreZipMeta szm;
  szm.name = name;
  szm.lfh_offset = offset;
  szm.crc32 = crc32;
  szm.size = size;

  filemetas.push_back(szm);

  return 0;
}

int StoreZipWriter::close()
{
  if (!fp)
    return 0;

  int offset = ftell(fp);

  for (const StoreZipMeta& szm : filemetas)
  {
    uint32_t signature = 0x02014b50;
    fwrite((char*)&signature, sizeof(signature), 1, fp);

    central_directory_file_header cdfh;
    cdfh.version_made = 0;
    cdfh.version = 0;
    cdfh.flag = 0;
    cdfh.compression = 0;
    cdfh.last_modify_time = 0;
    cdfh.last_modify_date = 0;
    cdfh.crc32 = szm.crc32;
    cdfh.compressed_size = szm.size;
    cdfh.uncompressed_size = szm.size;
    cdfh.file_name_length = szm.name.size();
    cdfh.extra_field_length = 0;
    cdfh.file_comment_length = 0;
    cdfh.start_disk = 0;
    cdfh.internal_file_attrs = 0;
    cdfh.external_file_attrs = 0;
    cdfh.lfh_offset = szm.lfh_offset;

    fwrite((char*)&cdfh, sizeof(cdfh), 1, fp);

    fwrite((char*)szm.name.c_str(), szm.name.size(), 1, fp);
  }

  int offset2 = ftell(fp);

  {
    uint32_t signature = 0x06054b50;
    fwrite((char*)&signature, sizeof(signature), 1, fp);

    end_of_central_directory_record eocdr;
    eocdr.disk_number = 0;
    eocdr.start_disk = 0;
    eocdr.cd_records = filemetas.size();
    eocdr.total_cd_records = filemetas.size();
    eocdr.cd_size = offset2 - offset;
    eocdr.cd_offset = offset;
    eocdr.comment_length = 0;

    fwrite((char*)&eocdr, sizeof(eocdr), 1, fp);
  }

  fclose(fp);
  fp = 0;

  return 0;
}

} // namespace pnnx

#if 0
int main()
{
    StoreZipReader sz;

    sz.open("test.zip");

    std::vector<float> data1;
    sz.read_file("pnnx2.py", data1);

    std::vector<float> data2;
    sz.read_file("pnnx2.param", data2);

    sz.close();


    StoreZipWriter szw;

    szw.open("szw.zip");

    szw.write_file("a.py", data1);
    szw.write_file("zzzz.param", data2);

    szw.close();


    return 0;
}
#endif
