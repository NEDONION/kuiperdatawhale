A lightweight deep learning inference framework for learning deep learning inference and computer vision concepts. 

一个轻量级的深度学习推理框架，用于学习深度学习推理和计算机视觉概念的C++项目。

## 🚀 Features

- Modern C++17 implementation 现代C++17实现
- Google Logging (glog) integration Google日志系统集成
- Google Test framework for unit testing Google测试框架
- OpenMP support for parallel processing OpenMP并行处理支持
- Optimized compilation with native architecture support 原生架构优化编译
- Mathematical libraries integration (Armadillo, BLAS, LAPACK) 数学库集成

## 🛠️ Prerequisites

- CMake 3.16 or higher CMake 3.16 或更高版本
- C++17 compatible compiler C++17兼容编译器
- Google Logging (glog) Google日志库
- Google Test (gtest) Google测试库
- Armadillo linear algebra library Armadillo线性代数库
- BLAS and LAPACK libraries BLAS和LAPACK库

## Installation & Build

### Clone the repository
```bash
git clone <repository-url>
cd KuiperCourse
```

### Build the project
```bash
mkdir build && cd build
cmake ..
make -j8
```

### Run the application
```bash
./kuiper_course
```

## Testing

Run the test suite:
```bash
cd build
make test
```

Or run tests individually:
```bash
cd test

g++ test_first.cpp -o test_first \
>     -lgtest -lgtest_main -lglog -larmadillo -lpthread

./test_first 
```

## Project Structure

```
KuiperCourse/
├── bin/                    # 可执行文件输出目录 / Executable output directory
├── cmake-build-*/         # CMake构建目录 / CMake build directories
├── include/                # 头文件目录 / Header files directory
│   ├── data/               # 数据结构头文件 / Data structure headers
│   ├── factory/            # 工厂模式头文件 / Factory pattern headers
│   ├── layer/              # 神经网络层头文件 / Neural network layer headers
│   ├── ops/                # 操作算子头文件 / Operation operator headers
│   ├── parser/             # 解析器头文件 / Parser headers
│   ├── runtime/            # 运行时头文件 / Runtime headers
│   └── status_code.hpp     # 状态码定义 / Status code definitions
├── source/                 # 源代码目录 / Source code directory
│   ├── data/               # 数据结构实现 / Data structure implementations
│   ├── factory/            # 工厂模式实现 / Factory pattern implementations
│   ├── layer/              # 神经网络层实现 / Neural network layer implementations
│   ├── ops/                # 操作算子实现 / Operation operator implementations
│   ├── parser/             # 解析器实现 / Parser implementations
│   └── runtime/            # 运行时实现 / Runtime implementations
├── test/                   # 测试文件目录 / Test files directory
│   ├── test_conv.cpp       # 卷积测试 / Convolution test
│   ├── test_expression.cpp # 表达式测试 / Expression test
│   ├── test_first.cpp      # 基础测试 / Basic test
│   ├── test_init_inoutput.cpp # 输入输出初始化测试 / Input/Output initialization test
│   ├── test_load_data.cpp  # 数据加载测试 / Data loading test
│   ├── test_main.cpp       # 主测试 / Main test
│   ├── test_maxpooling.cpp # 最大池化测试 / Max pooling test
│   ├── test_relu.cpp       # ReLU激活测试 / ReLU activation test
│   ├── test_runtime1.cpp   # 运行时测试1 / Runtime test 1
│   ├── test_sigmoid.cpp    # Sigmoid激活测试 / Sigmoid activation test
│   ├── test_tensor.cpp     # 张量测试 / Tensor test
│   └── CMakeLists.txt      # 测试CMake配置 / Test CMake configuration
├── tmp/                    # 临时文件目录 / Temporary files directory
├── main.cpp                # 主程序入口点 / Main application entry point
├── CMakeLists.txt          # 主CMake配置 / Main CMake configuration
├── Dockerfile              # Docker配置 / Docker configuration
└── start_cpp_dev.sh        # 开发环境设置脚本 / Development environment setup script
```
