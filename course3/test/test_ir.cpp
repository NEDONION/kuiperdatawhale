//
// Created by fss on 23-6-4.
//
#include "data/tensor.hpp"
#include "runtime/ir.h"
#include "runtime/runtime_ir.hpp"
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <string>

/**
 * 将形状向量 {a,b,c} 转成人类可读的 "a x b x c"
 * - 仅用于日志打印，方便快速查看张量/权重形状
 */
static std::string ShapeStr(const std::vector<int> &shapes) {
  std::ostringstream ss;
  for (int i = 0; i < (int)shapes.size(); ++i) {
    ss << shapes.at(i);
    if (i != (int)shapes.size() - 1) ss << " x ";
  }
  return ss.str();
}

/**
 * 读取 pnnx 的 .param/.bin 文件，验证是否能解析出算子列表
 * - Graph::load() 成功返回 0
 * - 打印每个 Operator 的唯一 name（便于确认拓扑/命名）
 *
 * 常见坑：相对路径以“测试运行时的工作目录(通常是 build/)”为基准
 */
TEST(test_ir, pnnx_graph_ops) {
  using namespace kuiper_infer;

  std::string bin_path("course3/model_file/test_linear.pnnx.bin");
  std::string param_path("course3/model_file/test_linear.pnnx.param");

  std::unique_ptr<pnnx::Graph> graph = std::make_unique<pnnx::Graph>();
  int load_result = graph->load(param_path, bin_path);
  ASSERT_EQ(load_result, 0);  // 失败请先检查相对路径

  const auto &ops = graph->ops;  // 图中的所有算子（按解析顺序）
  for (int i = 0; i < (int)ops.size(); ++i) {
    LOG(INFO) << ops.at(i)->name;
  }
}

/**
 * 遍历每个算子，打印其输入/输出 Operand 的 name 与 shape
 * - inputs：该算子的所有输入边
 * - outputs：该算子的所有输出边
 * 有助于检查图的连边是否正确
 */
TEST(test_ir, pnnx_graph_operands) {
  using namespace kuiper_infer;

  std::string bin_path("course3/model_file/test_linear.pnnx.bin");
  std::string param_path("course3/model_file/test_linear.pnnx.param");

  std::unique_ptr<pnnx::Graph> graph = std::make_unique<pnnx::Graph>();
  int load_result = graph->load(param_path, bin_path);
  ASSERT_EQ(load_result, 0);

  const auto &ops = graph->ops;
  for (int i = 0; i < (int)ops.size(); ++i) {
    const auto &op = ops.at(i);
    LOG(INFO) << "OP Name: " << op->name;

    LOG(INFO) << "OP Inputs";
    for (int j = 0; j < (int)op->inputs.size(); ++j) {
      LOG(INFO) << "  input: " << op->inputs.at(j)->name
                << "  shape: " << ShapeStr(op->inputs.at(j)->shape);
    }

    LOG(INFO) << "OP Outputs";
    for (int j = 0; j < (int)op->outputs.size(); ++j) {
      LOG(INFO) << "  output: " << op->outputs.at(j)->name
                << "  shape: " << ShapeStr(op->outputs.at(j)->shape);
    }
    LOG(INFO) << "---------------------------------------------";
  }
}

/**
 * 仅关注名为 "linear" 的算子：
 * - 打印其输入/输出
 * - 打印 params（标量/小数组等“超参数”，如 in_features、bias）
 * - 打印 attrs（权重等大块二进制数据，含 type/shape，例如 weight、bias）
 *
 * 这有助于区分：“params”描述配置；“attrs”承载实际权重数据
 */
TEST(test_ir, pnnx_graph_operands_and_params) {
  using namespace kuiper_infer;

  std::string bin_path("course3/model_file/test_linear.pnnx.bin");
  std::string param_path("course3/model_file/test_linear.pnnx.param");

  std::unique_ptr<pnnx::Graph> graph = std::make_unique<pnnx::Graph>();
  int load_result = graph->load(param_path, bin_path);
  ASSERT_EQ(load_result, 0);

  const auto &ops = graph->ops;
  for (int i = 0; i < (int)ops.size(); ++i) {
    const auto &op = ops.at(i);
    if (op->name != "linear") continue;  // 只看指定算子

    LOG(INFO) << "OP Name: " << op->name;

    LOG(INFO) << "OP Inputs";
    for (int j = 0; j < (int)op->inputs.size(); ++j) {
      LOG(INFO) << "  input: " << op->inputs.at(j)->name
                << "  shape: " << ShapeStr(op->inputs.at(j)->shape);
    }

    LOG(INFO) << "OP Outputs";
    for (int j = 0; j < (int)op->outputs.size(); ++j) {
      LOG(INFO) << "  output: " << op->outputs.at(j)->name
                << "  shape: " << ShapeStr(op->outputs.at(j)->shape);
    }

    LOG(INFO) << "Params";
    for (const auto &kv : op->params) {
      // kv.first: 参数名；kv.second.type: Parameter 的类型编码（1=b,2=i,3=f,...）
      LOG(INFO) << "  " << kv.first << "  type=" << kv.second.type;
    }

    LOG(INFO) << "Weights (attrs)";
    for (const auto &kv : op->attrs) {
      // kv.first: 权重名；kv.second.shape/type: 张量形状与标量类型编码
      LOG(INFO) << "  " << kv.first << "  shape=" << ShapeStr(kv.second.shape)
                << "  type=" << kv.second.type;
    }
    LOG(INFO) << "---------------------------------------------";
  }
}

/**
 * 遍历每个 Operand，打印其 consumers / producer
 * - producer：产生该值的算子（唯一）
 * - consumers：使用该值的算子们（0~N 个）
 *
 * 用于检查图的连接关系是否正确
 */
TEST(test_ir, pnnx_graph_operands_customer_producer) {
  using namespace kuiper_infer;

  std::string bin_path("course3/model_file/test_linear.pnnx.bin");
  std::string param_path("course3/model_file/test_linear.pnnx.param");

  std::unique_ptr<pnnx::Graph> graph = std::make_unique<pnnx::Graph>();
  int load_result = graph->load(param_path, bin_path);
  ASSERT_EQ(load_result, 0);

  const auto &operands = graph->operands;
  for (int i = 0; i < (int)operands.size(); ++i) {
    const auto &operand = operands.at(i);

    LOG(INFO) << "Operand: " << operand->name;

    LOG(INFO) << "  Consumers:";
    for (const auto &consumer : operand->consumers) {
      LOG(INFO) << "    " << consumer->name;
    }

    // producer 可能为空（例如图的输入），此处视模型而定
    if (operand->producer)
      LOG(INFO) << "  Producer: " << operand->producer->name;
    else
      LOG(INFO) << "  Producer: <graph_input>";
  }
}

/**
 * 使用 RuntimeGraph 封装后的统一视图：
 * - RuntimeGraph::Init() 解析 pnnx 文件并构建运行时描述
 * - 遍历每个 RuntimeOperator：
 *    * attribute：权重集合（名字 -> Attribute）
 *    * input_operands：输入名 -> 形状
 *    * output_names：输出名列表
 */
TEST(test_ir, pnnx_graph_all) {
  using namespace kuiper_infer;

  std::string bin_path("course3/model_file/test_linear.pnnx.bin");
  std::string param_path("course3/model_file/test_linear.pnnx.param");

  RuntimeGraph graph(param_path, bin_path);
  const bool init_success = graph.Init();
  ASSERT_EQ(init_success, true);

  const auto &operators = graph.operators();
  for (const auto &op : operators) {
    LOG(INFO) << "op name: " << op->name << "  type: " << op->type;

    LOG(INFO) << "attributes (weights):";
    for (const auto &[name, attr] : op->attribute) {
      LOG(INFO) << "  " << name
                << "  type=" << int(attr->type)
                << "  shape=" << ShapeStr(attr->shape);
      const auto &weight_data = attr->weight_data;
      ASSERT_EQ(weight_data.empty(), false); // 权重数据应非空
    }

    LOG(INFO) << "inputs:";
    for (const auto &kv : op->input_operands) {
      LOG(INFO) << "  name=" << kv.first
                << "  shape=" << ShapeStr(kv.second->shapes);
    }

    LOG(INFO) << "outputs:";
    for (const auto &out_name : op->output_names) {
      LOG(INFO) << "  name=" << out_name;
    }
    LOG(INFO) << "--------------------------------------";
  }
}

/**
 * 验证作业要求：检查 "linear" 的运行时参数是否符合预期
 * - 共 3 个参数：bias / in_features / out_features
 * - 具体值：bias=true, in_features=32, out_features=128
 *
 * 这里通过动态类型转换拿到具体参数类型并校验其值
 */
TEST(test_ir, pnnx_graph_all_homework) {
  using namespace kuiper_infer;

  std::string bin_path("course3/model_file/test_linear.pnnx.bin");
  std::string param_path("course3/model_file/test_linear.pnnx.param");

  RuntimeGraph graph(param_path, bin_path);
  const bool init_success = graph.Init();
  ASSERT_EQ(init_success, true);

  const auto &operators = graph.operators();
  for (const auto &op : operators) {
    if (op->name == "linear") {
      const auto &params = op->params;
      ASSERT_EQ(params.size(), 3);

      // bias: bool = true
      ASSERT_EQ(params.count("bias"), 1);
      RuntimeParameter *p_bias = params.at("bias");
      ASSERT_NE(p_bias, nullptr);
      ASSERT_EQ((dynamic_cast<RuntimeParameterBool *>(p_bias)->value), true);

      // in_features: int = 32
      ASSERT_EQ(params.count("in_features"), 1);
      RuntimeParameter *p_in = params.at("in_features");
      ASSERT_NE(p_in, nullptr);
      ASSERT_EQ((dynamic_cast<RuntimeParameterInt *>(p_in)->value), 32);

      // out_features: int = 128
      ASSERT_EQ(params.count("out_features"), 1);
      RuntimeParameter *p_out = params.at("out_features");
      ASSERT_NE(p_out, nullptr);
      ASSERT_EQ((dynamic_cast<RuntimeParameterInt *>(p_out)->value), 128);
    }
  }
}
