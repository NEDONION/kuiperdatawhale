/**
 * @file runtime_ir.cpp
 * @brief 运行时中间表示(IR)的实现文件
 * @details 该文件实现了运行时图结构的管理，包括图的初始化、算子管理、输入输出处理等
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
     * @param param_path 参数文件路径
     * @param bin_path 二进制文件路径
     * @details 初始化运行时图，设置参数文件和二进制文件的路径
     */
    RuntimeGraph::RuntimeGraph(std::string param_path, std::string bin_path)
            : param_path_(std::move(param_path)), bin_path_(std::move(bin_path)) {}

    /**
     * @brief 设置二进制文件路径
     * @param bin_path 二进制文件路径
     */
    void RuntimeGraph::set_bin_path(const std::string &bin_path) {
        this->bin_path_ = bin_path;
    }

    /**
     * @brief 设置参数文件路径
     * @param param_path 参数文件路径
     */
    void RuntimeGraph::set_param_path(const std::string &param_path) {
        this->param_path_ = param_path;
    }

    /**
     * @brief 获取参数文件路径
     * @return 参数文件路径的常量引用
     */
    const std::string &RuntimeGraph::param_path() const {
        return this->param_path_;
    }

    /**
     * @brief 获取二进制文件路径
     * @return 二进制文件路径的常量引用
     */
    const std::string &RuntimeGraph::bin_path() const { return this->bin_path_; }

    /**
     * @brief 初始化运行时图
     * @return 初始化是否成功
     * @details 该函数会加载PNNX格式的模型文件，解析图结构，初始化所有算子
     */
    bool RuntimeGraph::Init() {
        // 检查文件路径是否为空
        if (this->bin_path_.empty() || this->param_path_.empty()) {
            LOG(ERROR) << "The bin path or param path is empty";
            return false;
        }

        // 创建PNNX图对象并加载模型文件
        this->graph_ = std::make_unique<pnnx::Graph>();
        int load_result = this->graph_->load(param_path_, bin_path_);
        if (load_result != 0) {
            LOG(ERROR) << "Can not find the param path or bin path: " << param_path_
                       << " " << bin_path_;
            return false;
        }

        // 获取所有算子
        std::vector<pnnx::Operator *> operators = this->graph_->ops;
        if (operators.empty()) {
            LOG(ERROR) << "Can not read the layers' define";
            return false;
        }

        // 清空之前的算子数据
        this->operators_.clear();
        this->operators_maps_.clear();
        
        // 遍历所有算子，初始化运行时算子
        for (const pnnx::Operator *op: operators) {
            if (!op) {
                LOG(ERROR) << "Meet the empty node";
                continue;
            } else {
                // 创建运行时算子对象
                std::shared_ptr<RuntimeOperator> runtime_operator =
                        std::make_shared<RuntimeOperator>();
                
                // 初始化算子的名称和类型
                runtime_operator->name = op->name;
                runtime_operator->type = op->type;

                // 初始化算子中的输入操作数
                const std::vector<pnnx::Operand *> &inputs = op->inputs;
                if (!inputs.empty()) {
                    InitGraphOperatorsInput(inputs, runtime_operator);
                }

                // 记录输出操作数中的名称
                const std::vector<pnnx::Operand *> &outputs = op->outputs;
                if (!outputs.empty()) {
                    InitGraphOperatorsOutput(outputs, runtime_operator);
                }

                // 初始化算子中的属性(权重)
                const std::map<std::string, pnnx::Attribute> &attrs = op->attrs;
                if (!attrs.empty()) {
                    InitGraphAttrs(attrs, runtime_operator);
                }

                // 初始化算子中的参数
                const std::map<std::string, pnnx::Parameter> &params = op->params;
                if (!params.empty()) {
                    InitGraphParams(params, runtime_operator);
                }
                
                // 将运行时算子添加到列表中
                this->operators_.push_back(runtime_operator);
                this->operators_maps_.insert({runtime_operator->name, runtime_operator});
            }
        }

        return true;
    }

    /**
     * @brief 初始化图算子的输入
     * @param inputs 输入操作数列表
     * @param runtime_operator 运行时算子
     * @details 该函数会处理算子的输入操作数，建立算子之间的连接关系
     */
    void RuntimeGraph::InitGraphOperatorsInput(
            const std::vector<pnnx::Operand *> &inputs,
            const std::shared_ptr<RuntimeOperator> &runtime_operator) {
        for (const pnnx::Operand *input: inputs) {
            if (!input) {
                continue;
            }
            // 获取产生该输入的算子
            const pnnx::Operator *producer = input->producer;
            // 创建运行时操作数
            std::shared_ptr<RuntimeOperand> runtime_operand =
                    std::make_shared<RuntimeOperand>();
            runtime_operand->name = producer->name;
            runtime_operand->shapes = input->shape;

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
                    LOG(FATAL) << "Unknown input operand type: " << input->type;
                }
            }
            runtime_operator->input_operands.insert({producer->name, runtime_operand});
            runtime_operator->input_operands_seq.push_back(runtime_operand);
        }
    }

    /**
     * @brief 初始化图算子的输出
     * @param outputs 输出操作数列表
     * @param runtime_operator 运行时算子
     * @details 该函数会处理算子的输出操作数，记录算子的输出名称
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
                runtime_operator->output_names.push_back(c->name);
            }
        }
    }

    /**
     * @brief 初始化图算子的参数
     * @param params 参数列表
     * @param runtime_operator 运行时算子
     * @details 该函数会处理算子的参数，根据参数类型创建不同的参数对象
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
                    LOG(FATAL) << "Unknown parameter type: " << type;
                }
            }
        }
    }

    /**
     * @brief 初始化图算子的属性
     * @param attrs 属性列表
     * @param runtime_operator 运行时算子
     * @details 该函数会处理算子的属性，根据属性类型创建不同的属性对象
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
                    runtime_attribute->weight_data = attr.data;
                    runtime_attribute->shape = attr.shape;
                    runtime_operator->attribute.insert({name, runtime_attribute});
                    break;
                }
                default: {
                    LOG(FATAL) << "Unknown attribute type: " << attr.type;
                }
            }
        }
    }

    /**
     * @brief 获取所有运行时算子
     * @return 运行时算子列表的常量引用
     */
    const std::vector<std::shared_ptr<RuntimeOperator>> &
    RuntimeGraph::operators() const {
        return this->operators_;
    }

} // namespace kuiper_infer
