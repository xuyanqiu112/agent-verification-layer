"""图改造与校验节点（在 ``workflow.compile()`` **之前**插桩）。

改造后（非 self_critic）::

    START → generate ──"execute"──> verify_tool_call ──pass──> execute ──> verify_tool_result ──"pass"──> generate
                     ├──"generate"──> generate                       （parse 错重试）
                     └──"end"───────> verify_tool_result ──> generate / END

self_critic 打开时 ``generate`` 的 ``"end"`` 先到 ``self_critic``，再 ``self_critic`` →
``verify_tool_result``（否则 self_critic 会和它并行写 messages）。

节点签名 ``def node(state: AgentState) -> AgentState``，返回**同一个** state，用
``state["next_step"] = "pass" | "block"`` 表达路由；只写 ``next_step`` /
``validation_errors`` / ``validation_attempts``。失败动作只有 ``pass`` / ``block``。
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Any

from langchain_core.messages import HumanMessage

from .config import VerificationConfig, load_config
from .rules import (
    check_tool_call,
    collect_known_tools,
    collect_tool_parameters,
    extract_execute_code,
    extract_tool_calls,
    is_python_code,
    observation_errors,
)

STAGE_TOOL_CALL = "tool_call"
STAGE_TOOL_RESULT = "tool_result"
MAX_LLM_CALLS = 10
_llm_calls = 0  # 进程内 LLM 调用计数（校验器自身状态，不放进 AgentState）


def node_verify_tool_call(state: dict, agent: Any, config: VerificationConfig | None = None) -> dict:
    """``generate → execute`` 之间：参数域规则 + 可选 LLM 兜底；有错则 block。"""
    started, config = time.time(), config or load_config(agent)
    if not config.enabled:
        state["next_step"] = "pass"
        return state
    code = extract_execute_code(_last_content(state))
    if code is None or not is_python_code(code):  # 非 Python 直接放行
        return _finish(state, agent, config, STAGE_TOOL_CALL, "pass", [], started)
    known = collect_known_tools(agent)
    errors: list[dict] = []
    for tool_name, kwargs in extract_tool_calls(code, known, collect_tool_parameters(agent)) if known else []:
        messages = check_tool_call(tool_name, kwargs)
        if not messages and config.enable_llm:  # 规则通过后才考虑 LLM
            messages = llm_check_tool_call(agent, config, tool_name, kwargs)
        errors += [{"tool_name": tool_name, "message": message} for message in messages]
    decision = "block" if errors and retry_budget_left(state, config) else "pass"
    if decision == "block":
        _block(state, _block_message("⚠️ Tool call 参数校验失败，请修正后重新生成：", "unknown", errors), errors)
    return _finish(state, agent, config, STAGE_TOOL_CALL, decision, errors, started)


def node_verify_tool_result(state: dict, agent: Any, config: VerificationConfig | None = None) -> dict:
    """``execute → generate`` 之间：``<observation>`` 空输出 / 错误标记 / 超长。"""
    started, config = time.time(), config or load_config(agent)
    if not config.enabled:
        state["next_step"] = "pass"
        return state
    messages = state.get("messages") or []
    tool_name = last_executed_tool(messages, agent) or ""
    errors = observation_errors(getattr(messages[-1], "content", "") if messages else "")
    decision = "block" if errors and retry_budget_left(state, config) else "pass"
    if decision == "block":
        _block(state, _block_message("⚠️ Tool 执行结果异常，请重新规划：", tool_name or "unknown", errors), errors)
    return _finish(state, agent, config, STAGE_TOOL_RESULT, decision, errors, started, tool_name)


def route_after_verify_tool_call(state: dict) -> str:
    """``pass`` → ``execute``；否则（``block``）→ ``generate`` 触发 replan。"""
    return "execute" if state.get("next_step") == "pass" else "generate"


def route_after_verify_tool_result(state: dict) -> str:
    """无论如何都回 ``generate``（block 已通过 HumanMessage 把错误喂回模型）。"""
    return "generate"


def patch_workflow(workflow: Any, agent: Any) -> Any:
    """插入两个校验节点并重建出边。由 ``a1.configure()`` 在 ``workflow.compile()`` 之前调用。

    ``enabled=False`` 直接返回，图与原版逐字节一致。

    边责任划分（``agent/a1.py`` 里那 4 行）::

        self._verification_routers = (routing_function, routing_function_self_critic)
        if getattr(self, "_verification_patch", None) is not None:
            self._verification_patch(workflow, self)     # 由本函数加 execute→verify_tool_result
        else:
            workflow.add_edge("execute", "generate")     # 关闭时保持原样

    为什么不能用 ``add_edge`` 接线（本 bug 的根因）
    ----------------------------------------------
    LangGraph 对同一源节点：``add_conditional_edges`` 是**覆盖**，``add_edge`` 是**累加**，
    且没有公开 API 能删掉已加的边。原图 ``a1.py:1643-1665`` 是
    ``add_conditional_edges("generate", ...)`` + ``add_edge("execute", "generate")``，因此

    * ``add_edge("generate", "verify_tool_call")`` 会**并列**加一条边 → ``generate`` 在同一
      super-step 同时激活 ``verify_tool_call`` 与 ``execute``；
    * 同样地，若此时 ``execute`` 还带着 ``add_edge("execute", "generate")``，再加
      ``add_edge("execute", "verify_tool_result")`` 就会让 ``execute`` 同时激活两个节点。

    两者都会让两个节点在同一 super-step 写 ``state["messages"]`` →
    ``InvalidUpdateError("At key 'messages': Can receive only one value per step")``。

    本函数的做法（**只用文档化 API**）：

    1. ``generate`` 本来就是条件节点 → ``add_conditional_edges`` 覆盖重建 ``path_map``：
       ``"execute"`` → ``verify_tool_call``，``"generate"`` → ``generate``，
       ``"end"`` → ``verify_tool_result``（self_critic 打开时 → ``self_critic``）。
    2. ``execute`` 的旧无条件边由 ``a1.py`` 的 ``if/else`` **不再添加**，所以这里可以安全地
       ``add_edge("execute", "verify_tool_result")``（``execute`` 只有这一条出边）。
    3. ``verify_tool_result`` → ``generate`` 用 ``add_conditional_edges``（唯一出口，pass 与
       block 都回 generate）。
    """
    config = load_config(agent)  # 未知配置项会在这里抛出可读错误
    if not config.enabled:
        return workflow

    def verify_tool_call(state: dict) -> dict:
        return node_verify_tool_call(state, agent, config)

    def verify_tool_result(state: dict) -> dict:
        return node_verify_tool_result(state, agent, config)

    workflow.add_node("verify_tool_call", verify_tool_call)
    workflow.add_node("verify_tool_result", verify_tool_result)

   
    # 3) verify_tool_call → execute / generate
    workflow.add_conditional_edges(
        "verify_tool_call", route_after_verify_tool_call, {"execute": "execute", "generate": "generate"}
    )

    # 4) execute → verify_tool_result（a1.py 关闭时才会加 execute → generate，二者互斥）
    workflow.add_edge("execute", "verify_tool_result")

    # 5) verify_tool_result → generate（唯一出口，pass 与 block 都回 generate）
    workflow.add_conditional_edges("verify_tool_result", route_after_verify_tool_result, {"generate": "generate"})
    return workflow


def _has_node(workflow: Any, name: str) -> bool:
    """尽力判断节点是否已存在（不同 LangGraph 版本用 dict 或属性表）。"""
    nodes = getattr(workflow, "nodes", None)
    return name in nodes if isinstance(nodes, dict) else hasattr(nodes, name)


def llm_check_tool_call(agent: Any, config: VerificationConfig, tool_name: str, kwargs: dict) -> list[str]:
    """让 LLM 判断参数是否符合工具描述。任何失败（超上限/超时/非 JSON）都返回 ``[]`` = pass。"""
    global _llm_calls
    if _llm_calls >= MAX_LLM_CALLS or not config.enable_llm:
        return []
    try:
        llm = resolve_llm(agent, config)
        if llm is None:
            return []
        _llm_calls += 1
        payload = parse_llm_json(getattr(llm.invoke(_llm_prompt(agent, tool_name, kwargs)), "content", ""))
        if payload and str(payload.get("verdict", "pass")).strip().lower() == "block":
            return [str(payload.get("reason") or "LLM 判定参数与工具描述不符")]
    except Exception:
        return []
    return []


def resolve_llm(agent: Any, config: VerificationConfig) -> Any:
    """``llm_model`` 非空时用独立模型（带缓存），否则复用 ``agent.llm``。"""
    if not config.llm_model:
        return getattr(agent, "llm", None)
    cached = getattr(agent, "_verification_llm", None)
    if cached is not None:
        return cached
    from biomni.llm import get_llm  # 惰性：只在需要独立模型时 import

    llm = get_llm(config.llm_model, source=None, temperature=0.0)
    agent._verification_llm = llm
    return llm


def _llm_prompt(agent: Any, tool_name: str, kwargs: dict) -> str:
    """构造 LLM 兜底校验 prompt（带工具描述）。"""
    description = ""
    for tools in (getattr(agent, "module2api", None) or {}).values():
        for schema in tools or []:
            if isinstance(schema, dict) and schema.get("name") == tool_name:
                description = str(schema.get("description") or "")
                break
    return (
        "You are a strict scientific API reviewer. Decide whether the tool call arguments are "
        f"consistent with the tool description.\nTool: {tool_name}\nDescription: {description}\n"
        f"Arguments (JSON): {json.dumps(kwargs, ensure_ascii=False, default=str)}\n\n"
        'Reply with ONLY a JSON object: {"verdict": "pass" | "block", "reason": "..."}\n'
    )


def parse_llm_json(content: Any) -> dict | None:
    """解析 LLM 输出，容忍 ```json ... ``` 围栏与前后说明文字。"""
    text = (content if isinstance(content, str) else str(content or "")).strip()
    if text.startswith("```"):
        text = re.sub(r"```\s*$", "", re.sub(r"^```[a-zA-Z]*\s*", "", text)).strip()
    candidates = [text] + ([text[text.find("{") : text.rfind("}") + 1]] if "{" in text and "}" in text else [])
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(payload, dict):
            return payload
    return None


def last_executed_tool(messages: list, agent: Any) -> str | None:
    """从上一轮的 ``<execute>`` 找出实际调用的第一个已知工具名（审计用）。"""
    known = collect_known_tools(agent)
    if not known:
        return None
    for message in reversed(messages[:-1][-3:] if len(messages) > 1 else []):
        code = extract_execute_code(getattr(message, "content", ""))
        if code is not None:
            calls = extract_tool_calls(code, known)
            if calls:
                return calls[0][0]
    return None


def retry_budget_left(state: dict, config: VerificationConfig) -> bool:
    """``validation_attempts < max_retries`` 时才允许 block，否则放行（避免死循环）。"""
    return int(state.get("validation_attempts") or 0) < max(0, int(config.max_retries))


def write_audit(agent: Any, config: VerificationConfig, record: dict) -> None:
    """append 到 ``agent.verification_audit``；``audit_path`` 非空时追加一行 JSONL。

    字段固定 ``stage``/``decision``/``tool_name``/``errors``/``latency_ms``/``timestamp``；
    写盘失败一律吞掉——审计绝不能打断 agent。
    """
    audit = getattr(agent, "verification_audit", None)
    if not isinstance(audit, list):
        audit = []
        try:
            agent.verification_audit = audit
        except Exception:
            pass
    audit.append(record)
    if not config.audit_path:
        return
    try:
        path = str(config.audit_path)
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception:
        pass


def _last_content(state: dict) -> str:
    messages = state.get("messages") or []
    return getattr(messages[-1], "content", "") if messages else ""


def _block(state: dict, message: str, errors: list[dict]) -> None:
    """block：追加 HumanMessage（触发 replan）并累计错误与尝试次数。"""
    state["messages"].append(HumanMessage(content=message))
    state["validation_errors"] = list(state.get("validation_errors") or []) + errors
    state["validation_attempts"] = int(state.get("validation_attempts") or 0) + 1
    state["next_step"] = "block"


def _finish(state, agent, config, stage, decision, errors, started, tool_name: str = "") -> dict:
    """写审计并设置 ``next_step``（``block`` 已由 :func:`_block` 写过则保留）。"""
    record = {
        "stage": stage,
        "decision": decision,
        "tool_name": tool_name or next((e["tool_name"] for e in errors if e.get("tool_name")), None),
        "errors": [dict(error) for error in errors],
        "latency_ms": round((time.time() - started) * 1000.0, 3),
        "timestamp": time.time(),
    }
    write_audit(agent, config, record)
    if errors and decision == "pass":  # 超限放行时也留下错误痕迹
        state["validation_errors"] = list(state.get("validation_errors") or []) + errors
    if state.get("next_step") not in {"pass", "block"}:
        state["next_step"] = "pass"
    return state


def _block_message(header: str, label: str, errors: list[dict]) -> str:
    lines = [header]
    lines += [f"- [{e.get('tool_name') or label}] {e['message']}" for e in errors]
    return "\n".join(lines)
