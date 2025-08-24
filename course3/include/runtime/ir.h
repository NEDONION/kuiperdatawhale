// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#ifndef PNNX_IR_H
#define PNNX_IR_H

#include <initializer_list>
#include <map>
#include <set>
#include <string>
#include <vector>

/**
 * 当以 BUILD_PNNX=ON 方式构建时，启用与 TorchScript 的对接：
 * - 前置声明 torch::jit::Value / torch::jit::Node / at::Tensor
 * - 便于从 PyTorch IR / 张量构造 Parameter / Attribute 等
 */
#if BUILD_PNNX
namespace torch {
namespace jit {
struct Value;
struct Node;
} // namespace jit
} // namespace torch
namespace at {
class Tensor;
}
#endif // BUILD_PNNX

namespace pnnx {

/**
 * @brief 统一的“参数”容器，支持多种标量/数组/字符串类型。
 *
 * 设计要点：
 * - 用一个整型 @ref type 标识当前保存的实际类型（类似简易 variant）
 * - 值分别存放在 b/i/f/ai/af/s/as 等成员中（仅使用与 type 对应的那一个）
 * - “字符串成员放在最后”是为了跨编译器 ABI 兼容
 *
 * type 取值说明：
 *   0=null
 *   1=b        (bool)
 *   2=i        (int，含 long/long long 也归并到这里)
 *   3=f        (float/double 归并到 float)
 *   4=s        (std::string)
 *   5=ai       (std::vector<int>)
 *   6=af       (std::vector<float>)
 *   7=as       (std::vector<std::string>)
 *   8=others   (保留扩展)
 */
class Parameter
{
public:
    Parameter()
        : type(0)
    {
    }
    // 以下构造函数根据输入类型设置 type 并写入对应的值
    Parameter(bool _b)
        : type(1), b(_b)
    {
    }
    Parameter(int _i)
        : type(2), i(_i)
    {
    }
    Parameter(long _l)
        : type(2), i(_l)
    {
    }
    Parameter(long long _l)
        : type(2), i(_l)
    {
    }
    Parameter(float _f)
        : type(3), f(_f)
    {
    }
    Parameter(double _d)
        : type(3), f(_d)
    {
    }
    Parameter(const char* _s)
        : type(4), s(_s)
    {
    }
    Parameter(const std::string& _s)
        : type(4), s(_s)
    {
    }
    // 支持列表初始化（int / int64_t 会被收敛为 int）
    Parameter(const std::initializer_list<int>& _ai)
        : type(5), ai(_ai)
    {
    }
    Parameter(const std::initializer_list<int64_t>& _ai)
        : type(5)
    {
        for (const auto& x : _ai)
            ai.push_back((int)x);
    }
    Parameter(const std::vector<int>& _ai)
        : type(5), ai(_ai)
    {
    }
    Parameter(const std::initializer_list<float>& _af)
        : type(6), af(_af)
    {
    }
    Parameter(const std::initializer_list<double>& _af)
        : type(6)
    {
        for (const auto& x : _af)
            af.push_back((float)x);
    }
    Parameter(const std::vector<float>& _af)
        : type(6), af(_af)
    {
    }
    Parameter(const std::initializer_list<const char*>& _as)
        : type(7)
    {
        for (const auto& x : _as)
            as.push_back(std::string(x));
    }
    Parameter(const std::initializer_list<std::string>& _as)
        : type(7), as(_as)
    {
    }
    Parameter(const std::vector<std::string>& _as)
        : type(7), as(_as)
    {
    }

#if BUILD_PNNX
    // 从 TorchScript 节点/值构造 Parameter（解析常量、属性等）
    Parameter(const torch::jit::Node* value_node);
    Parameter(const torch::jit::Value* value);
#endif // BUILD_PNNX

    // 从字符串解析 Parameter（如 "3.14"、"[1,2,3]"、"true" 等）
    static Parameter parse_from_string(const std::string& value);

    // 当前参数的类型编码（见类注释中的枚举说明）
    int type;

    // 具体值（仅会使用与 type 对应的成员）
    bool b;
    int i;
    float f;
    std::vector<int> ai;
    std::vector<float> af;

    // 注意：为了跨 cxxabi 兼容，std::string 类型成员放在末尾
    std::string s;
    std::vector<std::string> as;
};

// 值相等比较（type 与对应存储的值相等才算相等）
bool operator==(const Parameter& lhs, const Parameter& rhs);

/**
 * @brief 权重/常量等原始数据块（Attribute），带类型与形状信息。
 *
 * 使用场景：
 * - 存放算子的权重，如卷积 weight、BN 的 running_mean/running_var 等
 * - type 表示底层标量类型（f32/f16/i8/...）
 * - shape 表示张量形状（按约定的维度顺序）
 * - data 以字节存放原始内存（需要按 type/shape 解释）
 */
class Attribute
{
public:
    Attribute()
        : type(0)
    {
    }

#if BUILD_PNNX
    // 从 at::Tensor 直接构造 Attribute（拷贝类型、形状与数据）
    Attribute(const at::Tensor& t);
#endif // BUILD_PNNX

    // 通过形状 + float 向量构造（常用于简单浮点权重）
    Attribute(const std::initializer_list<int>& shape, const std::vector<float>& t);

    // 标量类型编码：
    // 0=null 1=f32 2=f64 3=f16 4=i32 5=i64 6=i16 7=i8 8=u8 9=bool
    int type;
    std::vector<int> shape;

    // 原始字节数据（按 type/shape 解释）
    std::vector<char> data;
};

bool operator==(const Attribute& lhs, const Attribute& rhs);

// 沿第 0 维拼接两个 Attribute（shape[0] 叠加），其余维度需一致
Attribute operator+(const Attribute& a, const Attribute& b);

class Operator; // 前置声明，供 Operand 使用

/**
 * @brief 计算图中的“边/值”（Operand）：由一个 producer 输出，被多个 consumers 消费。
 *
 * - producer：产生该值的算子
 * - consumers：使用该值作为输入的算子列表
 * - type/shape：该值的推理数据类型与形状
 * - name：值的名称（用于查找/匹配）
 * - params：附加到该值的参数（可选，按需要扩展）
 */
class Operand
{
public:
    // 从 consumers 列表中移除某个使用者（用于图重写/删除算子）
    void remove_consumer(const Operator* c);

    Operator* producer;
    std::vector<Operator*> consumers;

    // 推理时的数据类型编码（与 Attribute 的编码类似，额外含复数等）
    // 0=null 1=f32 2=f64 3=f16 4=i32 5=i64 6=i16 7=i8 8=u8 9=bool 10=cp64 11=cp128 12=cp32
    int type;
    std::vector<int> shape;

    // 注意 ABI：string 成员置于末尾
    std::string name;

    // 该 Operand 的额外参数（可用于携带标记、属性）
    std::map<std::string, Parameter> params;
};

/**
 * @brief 计算图中的“节点/算子”（Operator）
 *
 * - inputs/outputs：输入输出边
 * - type/name：算子类型与唯一名称（如 "Conv2d", "ReLU", "bn1" 等）
 * - inputnames：原始输入名称（便于从外部 IR 映射）
 * - params：算子超参数（kernel/stride/padding/...）
 * - attrs：算子权重/常量（如卷积 weight、BN 的统计量等）
 */
class Operator
{
public:
    std::vector<Operand*> inputs;
    std::vector<Operand*> outputs;

    // 注意 ABI：string 成员置于末尾
    std::string type;
    std::string name;

    std::vector<std::string> inputnames;
    std::map<std::string, Parameter> params;
    std::map<std::string, Attribute> attrs;
};

/**
 * @brief 计算图容器：负责加载/保存/解析图，创建与组织 Operator 和 Operand。
 *
 * 典型流程：
 *   Graph g;
 *   g.load(param, bin);      // 从 .param/.bin 载入
 *   遍历 g.ops / g.operands  // 分析或转换
 *   g.save(param, bin);      // 保存
 */
class Graph
{
public:
    Graph();
    ~Graph();

    /**
     * @brief 从 pnnx 文本（.param）与二进制（.bin）文件加载图
     * @return 0 表示成功；非 0 表示失败（路径错误/格式错误等）
     */
    int load(const std::string& parampath, const std::string& binpath);

    /**
     * @brief 将当前图导出为 pnnx 的 .param/.bin 文件
     * @return 0 表示成功
     */
    int save(const std::string& parampath, const std::string& binpath);

    /**
     * @brief 调用 python 脚本进行处理（如通过 py 侧生成 .bin）
     * @note 具体约定取决于 pnnx 的实现
     */
    int python(const std::string& pypath, const std::string& binpath);

    /**
     * @brief 解析 pnnx .param 文本（字符串形式），构建 Graph
     * @note 常与 @ref load 联用
     */
    int parse(const std::string& param);

    /**
     * @brief 新建算子（追加到图中）
     * @param type 算子类型名
     * @param name 唯一名称
     */
    Operator* new_operator(const std::string& type, const std::string& name);

    /**
     * @brief 在指定算子 cur 之前插入新算子（用于图重写/插桩）
     */
    Operator* new_operator_before(const std::string& type, const std::string& name, const Operator* cur);

    /**
     * @brief 在指定算子 cur 之后插入新算子
     */
    Operator* new_operator_after(const std::string& type, const std::string& name, const Operator* cur);

#if BUILD_PNNX
    /**
     * @brief 由 TorchScript 的 Value 创建图中的 Operand（连接 PyTorch IR 与本 IR）
     */
    Operand* new_operand(const torch::jit::Value* v);
#endif

    /**
     * @brief 通过名称创建一个 Operand（常用于占位/常量）
     */
    Operand* new_operand(const std::string& name);

    /**
     * @brief 按名称查找 Operand（可返回 const 或非 const 指针；未找到返回 nullptr）
     */
    Operand* get_operand(const std::string& name);
    const Operand* get_operand(const std::string& name) const;

    // 图中全部算子与操作数（一般保持拓扑顺序或插入顺序）
    std::vector<Operator*> ops;
    std::vector<Operand*> operands;

private:
    // 禁止拷贝（仅允许按指针/引用管理）
    Graph(const Graph& rhs);
    Graph& operator=(const Graph& rhs);
};

} // namespace pnnx

#endif // PNNX_IR_H
