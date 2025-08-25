//
// Created by fss on 22-11-28.
//
#ifndef KUIPER_INFER_INCLUDE_PARSER_RUNTIME_OPERATOR_HPP_
#define KUIPER_INFER_INCLUDE_PARSER_RUNTIME_OPERATOR_HPP_

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
// #include "layer/abstract/layer.hpp"
#include "runtime/ir.h"
#include "runtime_attr.hpp"
#include "runtime_operand.hpp"
#include "runtime_parameter.hpp"

namespace kuiper_infer {
class Layer;

/// 计算图中的计算节点
/**
 * @brief 运行时图里的“算子节点”（执行单元）的最小封装
 * @details
 *  设计要点：
 *   1) 该结构既承担“拓扑节点”的角色（上下游连线、输入输出操作数），
 *      也承载“执行载体”的角色（Layer + 参数/属性）。
 *   2) 成员里大量使用 shared_ptr 而不是裸对象/unique_ptr，原因：
 *      - 节点与操作数/权重在图内天然存在“多处持有”的需求（例如同一输出被多个下游消费），
 *        用 shared_ptr 可简单表达共享所有权，避免复杂的所有权转移。
 *      - 初始化阶段与执行阶段可能分离，shared_ptr 便于跨阶段/跨模块传递与缓存。
 *   3) 少数位置采用原始指针（params），是基于“多态层级 + 轻量对象池/工厂”的传统写法；
 *      若需更强的异常安全/RAII，建议后续演进为 unique_ptr 或自定义智能指针。
 */
struct RuntimeOperator {
  /**
   * @brief 虚析构函数
   * @details
   *  - 该类型作为“多态基”，外界可能以基类指针持有（如 Operator* 指向派生类RuntimeGraph::Init()）。
   *  - 虚析构确保经基类指针 delete 时，派生类析构能被正确调用，避免资源泄漏。
   *  - 若仅作为“纯 POD 容器”，也可非虚；但考虑到 Layer 为多态、未来可能派生扩展，保留 virtual 更稳妥。
   */
  virtual ~RuntimeOperator();

  /**
   * @brief 节点是否已执行过 forward（前向计算）
   * @details
   *  - 执行期状态位：避免重复计算，或用于跨批次/流水线的简单去重。
   *  - 若执行计划更复杂（多流/并行），应配合调度器中的拓扑就绪计数使用。
   */
  bool has_forward = false;

  /**
   * @brief 计算节点名称（唯一标识）
   * @details
   *  - 用作 operators_maps 的 key、日志可读性、调试定位。
   *  - 唯一性通常由上游图导出工具（PNNX）保证。
   */
  std::string name;

  /**
   * @brief 计算节点类型（如 "Conv", "ReLU", "MatMul"...）
   * @details
   *  - 在构建 Layer 时进行分发（factory/registry）。
   *  - 也可供可视化/统计用途。
   */
  std::string type;

  /**
   * @brief 节点对应的可执行算子实现（策略对象）
   * @details
   *  - shared_ptr 原因：Layer 可能被图的不同视图/执行上下文共享引用；
   *    或不同阶段（初始化/执行/调试）持有同一 Layer 的引用。
   *  - 若确定 Layer 只会被该节点唯一持有，且无跨结构共享，可用 unique_ptr 简化所有权。
   */
  std::shared_ptr<Layer> layer;

  /**
   * @brief 下游消费者（节点名列表）
   * @details
   *  - 仅保存“名字”，而非直接保存指针引用，有两点考虑：
   *    (1) 初始化时可先不强绑定指针，降低构建顺序/生命周期耦合；
   *    (2) 日志/调试更直观（名字可打印）。
   *  - 若在执行期需要 O(1) 访问下游节点指针，可配合 output_operators 做二阶段解析。
   *  - 容器选择：vector 保留原始顺序，遍历代价 O(k)，k 为扇出数。
   */
  std::vector<std::string> output_names;

  /**
   * @brief 节点的“输出操作数”
   * @details
   *  - 大多数算子只有一个主输出，用一个 shared_ptr 即可表达。
   *  - 若未来要支持多输出（如 Tuple/Branch），可以演进为 vector<shared_ptr<RuntimeOperand>>。
   *  - shared_ptr 的原因：
   *    - 一个输出可能被多个下游节点消费（扇出>1），需要共享持有；
   *    - 输出张量/Tensor 可能延迟分配或在图优化中被替换（别名/内存复用）。
   */
  std::shared_ptr<RuntimeOperand> output_operands;

  /**
   * @brief 以“上游节点名”为 key 的输入操作数映射
   * @details
   *  - map 的动机：
   *    - 便于按“上游节点名”直接索引，适合在连线/调试时通过名字查找；
   *    - 插入/查找为 O(log n)，n 为输入个数；输入一般较少，可接受。
   *  - 也可替换为 unordered_map 获得均摊 O(1) 查找（需权衡可重现性与迭代有序性）。
   *  - value 使用 shared_ptr 的原因：
   *    - 输入操作数（RuntimeOperand）可能同时出现在“按名索引 map”
   *      与“按序容器 input_operands_seq”里，需要共享所有权。
   */
  std::map<std::string, std::shared_ptr<RuntimeOperand>> input_operands;

  /**
   * @brief 按 PNNX/模型原始顺序排列的输入操作数序列
   * @details
   *  - vector 的动机：需要位置语义（第 i 个输入），forward 阶段按序取更高效。
   *  - 与 input_operands（map）是同一批 RuntimeOperand 的不同“视图”，
   *    二者共享同一批 shared_ptr，避免重复分配与生命周期分裂。
   */
  std::vector<std::shared_ptr<RuntimeOperand>> input_operands_seq;

  /**
   * @brief 下游“名字 -> 节点指针”的快速映射
   * @details
   *  - 与 output_names 对应的二阶段解析结果（把名字解析为指针），
   *    便于执行期快速迭代下游节点，无需再去全局哈希表查找。
   *  - 使用 map 的原因同上；若更在意均摊 O(1)，可换 unordered_map。
   *  - 使用 shared_ptr 的原因：
   *    - 拓扑里节点彼此交叉引用，shared_ptr 能表达共享所有权；
   *    - 若担心循环引用，可将某一侧改为 weak_ptr（例如下游列表持 weak_ptr）。
   */
  std::map<std::string, std::shared_ptr<RuntimeOperator>> output_operators;

  /**
   * @brief 算子的参数（标量/数组/字符串等超参数）
   * @details
   *  - 使用原始指针（RuntimeParameter*）的历史原因：
   *    - 常见于“轻量参数对象 + 工厂创建 + 统一销毁”的模式；
   *    - 避免为每种派生类型套智能指针模板（老代码里常见）。
   *  - 风险与改进：
   *    - 需要明确谁负责 delete（通常由 RuntimeOperator 或图在析构时遍历释放）；
   *    - 若追求异常安全与资源确定释放，建议升级为
   *      std::unique_ptr<RuntimeParameter> 或使用自定义删除器的 std::shared_ptr。
   *  - 容器选型为 map：
   *    - 参数通常以“字符串键（如 kernel_size, stride）”索引，map 直观且顺序稳定；
   *    - 也可用 unordered_map 换取更快的平均查找。
   */
  std::map<std::string, RuntimeParameter*> params;

  /**
   * @brief 算子的属性（多为权重/常量张量）
   * @details
   *  - shared_ptr 的原因：
   *    - 权重可能被算子副本/不同执行上下文共享引用（如多流/复用）；
   *    - 构图优化（常量折叠/共享权重）时，引用计数能自然管理生命周期。
   *  - key 使用字符串（如 "weight", "bias"），与导出工具/Layer 实现对齐。
   *  - 若权重很大且不会共享，也可 unique_ptr；但一旦发生多处持有场景，需要回到 shared_ptr。
   */
  std::map<std::string, std::shared_ptr<RuntimeAttribute>> attribute;
};

}  // namespace kuiper_infer
#endif  // KUIPER_INFER_INCLUDE_PARSER_RUNTIME_OPERATOR_HPP_
