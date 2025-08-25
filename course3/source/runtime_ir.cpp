/**
 * @file runtime_ir.cpp
 * @brief 运行时中间表示(IR)的实现文件
 * @details
 *  - 负责将 PNNX 导出的静态计算图（param/bin）解析为运行时可用的图结构
 *  - 完成算子(RuntimeOperator)的构建、输入输出连线、参数与权重装载
 *  - 该实现仅做“图的装配”，不涉及执行计划/调度/内存复用等优化
 *  - 线程安全：当前实现 **非线程安全**；如需并发初始化，请在外部加锁或改造为局部对象
 */

#include "runtime/runtime_ir.hpp"
#include "status_code.hpp"
#include <deque>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

namespace kuiper_infer {

    /**
     * @brief 运行时图的构造函数
     * @param param_path 参数文件路径（.param，描述算子/边/参数）
     * @param bin_path 二进制文件路径（.bin，存放权重等二进制数据）
     * @details
     *  - 仅保存路径，不做实际加载；真正加载在 Init() 中完成
     *  - 所有权：路径字符串在 RuntimeGraph 生命周期内保持有效
     */
    RuntimeGraph::RuntimeGraph(std::string param_path, std::string bin_path)
            : param_path_(std::move(param_path)), bin_path_(std::move(bin_path)) {}

    /**
     * @brief 设置二进制文件路径
     * @param bin_path 二进制文件路径
     * @note 仅更新成员，不触发重新解析；若路径变更后需生效，请再次调用 Init()
     */
    void RuntimeGraph::set_bin_path(const std::string &bin_path) {
        this->bin_path_ = bin_path;
    }

    /**
     * @brief 设置参数文件路径
     * @param param_path 参数文件路径
     * @note 同上，仅更新成员，不触发重新解析
     */
    void RuntimeGraph::set_param_path(const std::string &param_path) {
        this->param_path_ = param_path;
    }

    /**
     * @brief 获取参数文件路径
     * @return 参数文件路径的常量引用（零拷贝）
     */
    const std::string &RuntimeGraph::param_path() const {
        return this->param_path_;
    }

    /**
     * @brief 获取二进制文件路径
     * @return 二进制文件路径的常量引用（零拷贝）
     */
    const std::string &RuntimeGraph::bin_path() const { return this->bin_path_; }

    /**
     * @brief 初始化运行时图
     * @return true：成功；false：失败（已通过 LOG 输出原因）
     * @details
     *  加载流程（关键步骤）：
     *   1) 基本校验：param/bin 路径非空
     *   2) 解析 PNNX Graph（graph_->load）
     *   3) 遍历 PNNX Operator 列表，逐个构建 RuntimeOperator：
     *      - name/type
     *      - 输入：根据 Operand->producer 连接拓扑
     *      - 输出：记录下游消费者名字（仅保存名字，实际指针连线可延后）
     *      - 属性 attrs：多为权重（float32/shape/data）
     *      - 参数 params：标量/数组/字符串等
     *
     *  错误处理：
     *   - 任一步失败均返回 false 并写 ERROR/FATAL 日志
     *
     *  复杂度：
     *   - 假设算子数为 N，边数为 E，整体为 O(N + E)
     *
     *  注意：
     *   - 当前未执行“拓扑排序”，operators_ 顺序为 PNNX 的原始顺序
     *   - 若后续执行阶段需要拓扑序，请在 Init 末尾新增排序步骤
     */
    bool RuntimeGraph::Init() {
        // 1) 基本路径校验
        if (this->bin_path_.empty() || this->param_path_.empty()) {
            LOG(ERROR) << "The bin path or param path is empty";
            return false;
        }

        // 2) 创建并加载 PNNX Graph（所有权：std::unique_ptr 自动管理）
        this->graph_ = std::make_unique<pnnx::Graph>();
        int load_result = this->graph_->load(param_path_, bin_path_);
        if (load_result != 0) {
            LOG(ERROR) << "Can not find the param path or bin path: " << param_path_
                       << " " << bin_path_;
            return false;
        }

        // 3) 取出算子列表并做基本校验
        std::vector<pnnx::Operator *> operators = this->graph_->ops;
        if (operators.empty()) {
            LOG(ERROR) << "Can not read the layers' define";
            return false;
        }

        // 4) 清空旧数据（保证可重复 Init）
        this->operators_.clear();
        this->operators_maps_.clear();

        // 5) 遍历 PNNX Operator，构建 RuntimeOperator
        for (const pnnx::Operator *op: operators) {
            if (!op) {
                // 防守式：跳过空指针节点
                LOG(ERROR) << "Meet the empty node";
                continue;
            } else {
                // 5.1 创建运行时算子（共享所有权，便于图内多处引用）
                std::shared_ptr<RuntimeOperator> runtime_operator =
                        std::make_shared<RuntimeOperator>();

                // 5.2 基本元数据
                runtime_operator->name = op->name;   // 唯一标识
                runtime_operator->type = op->type;   // 算子类型（如 Conv/Relu 等）

                // 5.3 解析输入（根据 Operand->producer 建立“上游到我”的连线信息）
                const std::vector<pnnx::Operand *> &inputs = op->inputs;
                if (!inputs.empty()) {
                    InitGraphOperatorsInput(inputs, runtime_operator);
                }

                // 5.4 解析输出（记录下游消费者名称，便于后续连线或调度）
                const std::vector<pnnx::Operand *> &outputs = op->outputs;
                if (!outputs.empty()) {
                    InitGraphOperatorsOutput(outputs, runtime_operator);
                }

                // 5.5 解析属性 attrs（通常是权重）：类型、shape、原始二进制
                const std::map<std::string, pnnx::Attribute> &attrs = op->attrs;
                if (!attrs.empty()) {
                    InitGraphAttrs(attrs, runtime_operator);
                }

                // 5.6 解析参数 params（标量/数组/字符串）：执行期的可配置超参数
                const std::map<std::string, pnnx::Parameter> &params = op->params;
                if (!params.empty()) {
                    InitGraphParams(params, runtime_operator);
                }

                // 5.7 注册到容器与索引（名字->指针）
                this->operators_.push_back(runtime_operator);
                this->operators_maps_.insert({runtime_operator->name, runtime_operator});
            }
        }

        // 提示：如需在此处做拓扑排序，可新增对 operators_ 的重排
        return true;
    }

    /**
     * @brief 初始化图算子的输入
     * @param inputs 输入操作数列表（PNNX 视角）
     * @param runtime_operator 运行时算子（当前被构建的“我”）
     * @details
     *  - 将 PNNX Operand 转换为 RuntimeOperand，并记录在：
     *      - input_operands：键值=上游算子名 -> 该输入
     *      - input_operands_seq：保留 PNNX 的顺序，便于按序访问
     *  - 仅做“描述层”的装配：不分配实际 Tensor 缓冲，执行阶段再绑定
     *  - 类型映射：
     *      input->type == 1 -> kTypeFloat32
     *      input->type == 0 -> kTypeUnknown（占位，待执行期确定）
     *  - 边界条件：
     *      - producer 可能为 nullptr（表示图的外部输入）；当前实现假定存在 producer
     *        若有外部输入，可在此处做判空并为其分配特定占位名（如 "__input_n"）
     */
    void RuntimeGraph::InitGraphOperatorsInput(
            const std::vector<pnnx::Operand *> &inputs,
            const std::shared_ptr<RuntimeOperator> &runtime_operator) {
        for (const pnnx::Operand *input: inputs) {
            if (!input) {
                // 空输入直接跳过（防御）
                continue;
            }
            // 获取产生该输入的上游算子（可能为空：图输入/常量节点）
            const pnnx::Operator *producer = input->producer;
            // 构建运行时操作数的“壳子”（延迟绑定数据）
            std::shared_ptr<RuntimeOperand> runtime_operand =
                    std::make_shared<RuntimeOperand>();

            // 注意：若 producer 为空，这里访问 producer->name 将崩溃
            // 当前实现假设 producer 不为空；如要支持外部输入，请先判空并设定占位名
            runtime_operand->name = producer->name;

            // 形状直接拷贝 PNNX 的 shape，通常为 {N,C,H,W} 或更通用的维度序
            runtime_operand->shapes = input->shape;

            // 简单的类型映射（可按需扩展半精度、整型等）
            switch (input->type) {
                case 1: {
                    runtime_operand->type = RuntimeDataType::kTypeFloat32;
                    break;
                }
                case 0: {
                    runtime_operand->type = RuntimeDataType::kTypeUnknown;
                    break;
                }
                default: {
                    // 未知类型直接 FATAL，避免后续无定义行为
                    LOG(FATAL) << "Unknown input operand type: " << input->type;
                }
            }

            // 以“上游算子名”为键登记，便于后续根据名字快速查找连线
            runtime_operator->input_operands.insert({producer->name, runtime_operand});
            // 同时保留一个顺序容器，满足按照 PNNX 输入序遍历的需求
            runtime_operator->input_operands_seq.push_back(runtime_operand);
        }
    }

    /**
     * @brief 初始化图算子的输出
     * @param outputs 输出操作数列表
     * @param runtime_operator 运行时算子
     * @details
     *  - 仅记录“我的输出被哪些消费者使用”的下游算子名列表 output_names
     *  - 实际的指针连线/句柄仍可在后续阶段完成
     *  - 若同一输出被多个消费者使用，则会记录多个名字（扇出）
     */
    void RuntimeGraph::InitGraphOperatorsOutput(
            const std::vector<pnnx::Operand *> &outputs,
            const std::shared_ptr<RuntimeOperator> &runtime_operator) {
        for (const pnnx::Operand *output: outputs) {
            if (!output) {
                continue;
            }
            const auto &consumers = output->consumers;
            for (const auto &c: consumers) {
                // 仅保存消费者名字（轻量），避免此处强绑定指针导致生命周期复杂化
                runtime_operator->output_names.push_back(c->name);
            }
        }
    }

    /**
     * @brief 初始化图算子的参数
     * @param params 参数列表（PNNX 的参数字典）
     * @param runtime_operator 运行时算子
     * @details
     *  - 将 pnnx::Parameter 的多种类型映射到 RuntimeParameter 层级
     *  - 所有权说明：参数对象通过 new 分配，存入 runtime_operator->params
     *    的 value_type* 指针，释放时机由 RuntimeOperator/Graph 的析构统一处理
     *  - 类型覆盖：
     *      Unknown / Bool / Int / Float / String / IntArray / FloatArray / StringArray
     *  - 扩展性：如需支持更多类型（如形状、复合结构），可在此处新增 case
     */
    void RuntimeGraph::InitGraphParams(
            const std::map<std::string, pnnx::Parameter> &params,
            const std::shared_ptr<RuntimeOperator> &runtime_operator) {
        for (const auto &[name, parameter]: params) {
            const int type = parameter.type;
            switch (type) {
                case int(RuntimeParameterType::kParameterUnknown): {
                    RuntimeParameter *runtime_parameter = new RuntimeParameter;
                    runtime_operator->params.insert({name, runtime_parameter});
                    break;
                }

                case int(RuntimeParameterType::kParameterBool): {
                    RuntimeParameterBool *runtime_parameter = new RuntimeParameterBool;
                    runtime_parameter->value = parameter.b;
                    runtime_operator->params.insert({name, runtime_parameter});
                    break;
                }

                case int(RuntimeParameterType::kParameterInt): {
                    RuntimeParameterInt *runtime_parameter = new RuntimeParameterInt;
                    runtime_parameter->value = parameter.i;
                    runtime_operator->params.insert({name, runtime_parameter});
                    break;
                }

                case int(RuntimeParameterType::kParameterFloat): {
                    RuntimeParameterFloat *runtime_parameter = new RuntimeParameterFloat;
                    runtime_parameter->value = parameter.f;
                    runtime_operator->params.insert({name, runtime_parameter});
                    break;
                }

                case int(RuntimeParameterType::kParameterString): {
                    RuntimeParameterString *runtime_parameter = new RuntimeParameterString;
                    runtime_parameter->value = parameter.s;
                    runtime_operator->params.insert({name, runtime_parameter});
                    break;
                }

                case int(RuntimeParameterType::kParameterIntArray): {
                    RuntimeParameterIntArray *runtime_parameter =
                            new RuntimeParameterIntArray;
                    runtime_parameter->value = parameter.ai;
                    runtime_operator->params.insert({name, runtime_parameter});
                    break;
                }

                case int(RuntimeParameterType::kParameterFloatArray): {
                    RuntimeParameterFloatArray *runtime_parameter =
                            new RuntimeParameterFloatArray;
                    runtime_parameter->value = parameter.af;
                    runtime_operator->params.insert({name, runtime_parameter});
                    break;
                }
                case int(RuntimeParameterType::kParameterStringArray): {
                    RuntimeParameterStringArray *runtime_parameter =
                            new RuntimeParameterStringArray;
                    runtime_parameter->value = parameter.as;
                    runtime_operator->params.insert({name, runtime_parameter});
                    break;
                }
                default: {
                    // 未知参数类型直接 FATAL，便于及早暴露导出/解析不一致的问题
                    LOG(FATAL) << "Unknown parameter type: " << type;
                }
            }
        }
    }

    /**
     * @brief 初始化图算子的属性（通常是权重/常量）
     * @param attrs 属性列表（PNNX 的 Attribute 字典）
     * @param runtime_operator 运行时算子
     * @details
     *  - 目前仅支持 attr.type == 1 (float32 权重)，并保存：
     *      type(kTypeFloat32) / weight_data(原始字节) / shape(维度信息)
     *  - 若未来需要支持 INT8/FP16/BF16/稀疏权重等，请在此扩展类型映射
     *  - 所有权：权重数据拷贝到 RuntimeAttribute::weight_data（std::vector<uint8_t>）
     */
    void RuntimeGraph::InitGraphAttrs(
            const std::map<std::string, pnnx::Attribute> &attrs,
            const std::shared_ptr<RuntimeOperator> &runtime_operator) {
        for (const auto &[name, attr]: attrs) {
            switch (attr.type) {
                case 1: {
                    std::shared_ptr<RuntimeAttribute> runtime_attribute =
                            std::make_shared<RuntimeAttribute>();
                    runtime_attribute->type = RuntimeDataType::kTypeFloat32;
                    runtime_attribute->weight_data = attr.data; // 二进制拷贝
                    runtime_attribute->shape = attr.shape;      // 维度信息
                    runtime_operator->attribute.insert({name, runtime_attribute});
                    break;
                }
                default: {
                    // 其它类型暂不支持；若 PNNX 侧新增类型，需在此同步实现
                    LOG(FATAL) << "Unknown attribute type: " << attr.type;
                }
            }
        }
    }

    /**
     * @brief 获取所有运行时算子
     * @return 运行时算子列表的常量引用
     * @note
     *  - 返回只读视图，避免外部随意改动内部拓扑
     *  - 遍历顺序与 PNNX 原始顺序一致；如需拓扑序，请在 Init 中进行排序
     */
    const std::vector<std::shared_ptr<RuntimeOperator>> &
    RuntimeGraph::operators() const {
        return this->operators_;
    }

} // namespace kuiper_infer
