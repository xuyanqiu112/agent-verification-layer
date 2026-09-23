# Agent Verification Layer

A pluggable verification layer for LLM agent tool calls, built on [Biomni](https://github.com/snap-stanford/Biomni) (a biomedical ReAct agent).

## 问题

Biomni 的 ReAct Agent 直接调用工具时存在三类风险：
- 参数越界被静默忽略
- Agent 私自改写用户参数
- 异常结果不报错，污染后续推理

## 方案

在 LangGraph 图结构中非侵入式插入校验层，在工具调用前后做参数域校验与异常检测。

```
generate → verify_tool_call --'pass'-→ execute → verify_tool_result → generate
              ↓ 'block'                                  ↓ block
          generate (replan)                        generate (replan)
```

## 核心文件

| 文件             | 作用                                                       |
| ---------------- | ---------------------------------------------------------- |
| `__init__.py`    | 公共 API：`apply_verification(agent, config)`              |
| `config.py`      | `VerificationConfig`（enabled / max_retries / audit_path） |
| `rules.py`       | AST 常量传播 + 注册式规则 + 条件边路由器                   |
| `graph_patch.py` | 图改造：插入校验节点，重建 `generate` 条件边               |

## 关键实现

**AST 常量传播**：解决 agent 写 `sequence = "ATGG..."` 后传变量名导致参数丢失的问题。

```python
sequence = "ATGGAGGAG..."      # 顶层赋值
design_primer(sequence=sequence, start_pos=5000)  # 传变量名
```

通过扫描顶层 `ast.Assign` 建立 `{变量名: 字面量}` 表，解析函数调用时回填。

**保守分析策略**：只处理字面量与可折叠表达式（`"ATGC"[:4]`），对函数返回、跨代码块、条件赋值一律跳过。

## 效果

9 个任务 × 2 轮 Eval：

- 6 个正常任务零误拦
- 参数越界任务从基线"静默改参"变为 100% 拦截 + 明确报错
- 枚举违规任务校验层比基线快 18%（提前拦截省掉工具内部 raise）

## 已知边界

规则只覆盖参数层，不解决语义意图漂移。被拦后 agent 可能改写参数绕过。需 prompt 层协同。

## 依赖

Python 3.11+ / LangGraph 0.3.28/ LangChain。核心代码纯标准库（`ast` / `re` / `json`）。
