"""确定性规则 + ``<execute>``/``<observation>`` 解析 + 条件边 router（纯标准库）。

规则契约::

    (tool_name: str, kwargs: dict) -> list[str]

返回空列表表示通过；每条字符串是一条人类可读的错误信息。**每个规则都被 try/except
包裹**（见 :func:`check_tool_call`）：规则自身报错时跳过，绝不阻塞链路。
"""
from __future__ import annotations

import ast
import inspect
import re
from typing import Any, Callable

RuleFunc = Callable[[str, dict], list[str]]
RULES: dict[str, RuleFunc] = {}  # tool_name -> 规则（同名工具只保留最后注册的）
_EXECUTE_RE = re.compile(r"<execute>(.*?)</execute>", re.DOTALL | re.IGNORECASE)
_OBS_RE = re.compile(r"<observation>(.*?)</observation>", re.DOTALL | re.IGNORECASE)
_SIG_CACHE: dict[str, list[str]] = {}
#: ``verify_tool_result`` 判定为错误的关键词（``NoneType`` 对应越界后返回 ``None``）。
ERROR_MARKERS = ("Error:", "Traceback", "NoneType")
#: 观察输出超过该长度即视为超长。
MAX_OBSERVATION_CHARS = 40_000


def register(*tool_names: str) -> Callable[[RuleFunc], RuleFunc]:
    """装饰器：把函数注册为 ``tool_names`` 的规则。"""
    def decorator(func: RuleFunc) -> RuleFunc:
        for name in tool_names:
            RULES[name] = func
        return func
    return decorator


def get_rule(tool_name: str) -> RuleFunc | None:
    """取某工具的规则（没有则 ``None``）。"""
    return RULES.get(tool_name)


@register("design_primer")
def rule_design_primer(tool_name: str, kwargs: dict) -> list[str]:
    """``design_primer``：``start_pos`` 必须是 int 且严格小于 ``len(sequence)``。

    背景：``tool/molecular_biology.py:1277`` 用
    ``sequence[start_pos : min(start_pos + search_window, len(sequence))]`` 切片，越界时
    **不报错**，只是区域为空并静默 ``return None``（``molecular_biology.py:1309-1310``）；
    agent 于是会把 5000 偷偷改成 400 绕过去。这条规则在执行前把它拦住。
    """
    errors: list[str] = []
    start_pos, sequence = kwargs.get("start_pos"), kwargs.get("sequence")
    if start_pos is not None:
        if isinstance(start_pos, bool) or not isinstance(start_pos, int):
            errors.append(f"start_pos={start_pos!r} 必须是 int（建议：传入整数位置）")
        elif isinstance(sequence, str):
            if start_pos >= len(sequence):
                errors.append(f"start_pos={start_pos} 超出序列长度 {len(sequence)} bp（建议：缩短位置或提供更长序列）")
            elif start_pos < 0:
                errors.append(f"start_pos={start_pos} 不能为负数（建议：使用 0 <= start_pos < {len(sequence)}）")
    primer_length = kwargs.get("primer_length")
    if isinstance(primer_length, int) and not isinstance(primer_length, bool) and primer_length <= 0:
        errors.append(f"primer_length={primer_length} 必须为正整数")
    return errors


@register("get_golden_gate_assembly_protocol")
def rule_golden_gate_enzyme(tool_name: str, kwargs: dict) -> list[str]:
    """``get_golden_gate_assembly_protocol``:``enzyme_name`` 必须是受支持的 Type IIS 酶。

    背景:``tool/molecular_biology.py:1030-1032`` 写死 ``supported_enzymes = ["BsaI", "BsmBI",
    "BbsI", "Esp3I", "BtgZI", "SapI"]``,``enzyme_name not in supported_enzymes`` 时直接
    ``raise ValueError``。agent 常见错误:把经典 Type II 酶(如 ``EcoRI``)当 Type IIS 用,
    或大小写/数字拼错(``bsai``/``Bsa1``)。基线会跑到 raise 后反思改酶名,绕过用户意图;
    本规则在执行前拦住。

    ``enzyme_name`` 默认 ``None`` 时工具也会 raise,但 ``None`` 不是 str,静态无法判断意图,
    规则跳过(宁可漏检不误判)。
    """
    errors: list[str] = []
    enzyme_name = kwargs.get("enzyme_name")
    if enzyme_name is None:
        return errors
    supported = ("BsaI", "BsmBI", "BbsI", "Esp3I", "BtgZI", "SapI")
    if isinstance(enzyme_name, str) and enzyme_name not in supported:
        errors.append(
            f"enzyme_name={enzyme_name!r} 不在支持列表 {supported}(建议:使用上述 Type IIS 酶之一)"
        )
    return errors


@register("design_verification_primers")
def rule_design_verification_primers(tool_name: str, kwargs: dict) -> list[str]:
    """``design_verification_primers``:``target_region`` 端点顺序约束。

    背景:``tool/molecular_biology.py:1465-1468`` 解构 ``start, end = target_region`` 后
    ``region_length = end - start + 1``;``if region_length <= 0: raise ValueError``。
    即 ``end`` 必须 > ``start``(0-based,docstring 明示)。``target_region`` 还隐式要求是
    2 元素 tuple/list(否则解构失败)。本规则在执行前拦住 ``end <= start`` 的调用。

    注:代码**不**单独 ``raise`` ``start < 0``,故本规则也不检查 ``start >= 0``,与工具行为一致。
    AST 提取只抓 agent 代码里的顶层 ``design_verification_primers(...)`` 调用,不会进入工具源码
    内部对 ``design_primer`` 的调用(见 :func:`extract_tool_calls`),故不会误抓嵌套。
    """
    errors: list[str] = []
    target_region = kwargs.get("target_region")
    if target_region is None:
        return errors
    if not isinstance(target_region, (tuple, list)):
        return errors  # 非静态可解形态,跳过(不误判)
    if len(target_region) != 2:
        errors.append(
            f"target_region 必须是 (start, end) 2 元素,当前 {len(target_region)} 个(建议:传 (start, end))"
        )
        return errors
    start, end = target_region
    if isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, int) or not isinstance(end, int):
        errors.append(f"target_region 端点必须是 int,当前 start={start!r} end={end!r}")
        return errors
    if end <= start:
        errors.append(
            f"target_region end={end} 必须 > start={start}(region_length={end - start + 1}<=0 时工具 raise ValueError)"
        )
    return errors


def extract_execute_code(message_content: str) -> str | None:
    """取消息里 ``<execute>...</execute>`` 的代码；没有则 ``None``（容忍未闭合标签）。"""
    if not isinstance(message_content, str) or "<execute>" not in message_content:
        return None
    match = _EXECUTE_RE.search(message_content)
    return match.group(1) if match else message_content.split("<execute>", 1)[1]


def extract_observation(message_content: str) -> str | None:
    """取消息里 ``<observation>...</observation>`` 的内容；没有则 ``None``。"""
    if not isinstance(message_content, str):
        return None
    match = _OBS_RE.search(message_content)
    return match.group(1) if match else None


def is_python_code(code: str) -> bool:
    """``<execute>`` 块是不是 Python（R / Bash 直接放行）。"""
    return not code.strip().startswith(("#!R", "# R code", "# R script", "#!BASH", "# Bash script", "#!CLI"))


def literal(node: ast.AST) -> tuple[bool, Any]:
    """尽力把 AST 节点求值成**常量**；失败返回 ``(False, None)``（调用方应跳过规则）。

    先试 ``ast.literal_eval``，再退到 CPython 常量折叠——这样 ``"ATGC"[:596]``、``100 + 20``
    也能拿到值（task_007 里 agent 正是写 ``sequence="..."[:596]``，不折叠就漏检）。
    两条路径都**不执行用户代码**：折叠只在 code object 不引用任何自由名字时才接受结果。
    """
    try:
        return True, ast.literal_eval(node)
    except Exception:
        pass
    try:
        expression = ast.fix_missing_locations(ast.Expression(body=node))
        code = compile(expression, "<verification>", "eval", dont_inherit=True)
        if code.co_names:  # 引用了名字/调用 -> 不是纯常量，拒绝
            return False, None
        return True, eval(code, {"__builtins__": {}}, {})  # noqa: S307 - 已确认无自由变量
    except Exception:
        return False, None


def _call_name(func: ast.AST) -> str | None:
    """``f(...)`` → ``"f"``；``mod.f(...)`` → ``"f"``（工具是扁平 import 的）。"""
    if isinstance(func, ast.Name):
        return func.id
    return func.attr if isinstance(func, ast.Attribute) else None

def collect_constants(tree: ast.AST) -> dict:
    """收集顶层 Assign 里的常量：{变量名: 字面量值}。"""
    constants = {}
    for statement in getattr(tree, "body", []):
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        target = statement.targets[0]
        if not isinstance(target, ast.Name):
            continue
        ok, value = literal(statement.value)
        if ok:
            constants[target.id] = value
    return constants


def _resolve_argument(node: ast.AST, constants: dict):
    """先常量折叠，再查常量表。拿不到返回 (False, None)。"""
    ok, value = literal(node)
    if ok:
        return True, value
    if isinstance(node, ast.Name):
        try:
            if node.id in constants:
                return True, constants[node.id]
        except (TypeError, KeyError):
            return False, None
    return False, None
    
def extract_tool_calls(
    code: str, known_tools: set[str] | None = None, param_names: dict[str, list[str]] | None = None
) -> list[tuple[str, dict]]:
    """从 Python 代码提取**已知工具**的调用，返回 ``[(tool_name, kwargs), ...]``。

    位置参数按 ``param_names``（通常来自 :func:`collect_tool_parameters`）或真实签名绑定到
    形参名；无法静态求值的实参不进入 kwargs（对应规则自然跳过，宁可漏检不误判）。
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    constants = collect_constants(tree)
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node.func)
        if not name or (known_tools is not None and name not in known_tools):
            continue
        kwargs: dict[str, Any] = {}
        for keyword in node.keywords:
            if keyword.arg is None:  # **kwargs 展开，静态不可知
                continue
            ok, value = _resolve_argument(keyword.value, constants)
            if ok:
                kwargs[keyword.arg] = value
        if node.args and not any(isinstance(arg, ast.Starred) for arg in node.args):
            names = (param_names or {}).get(name) or _parameter_names(name)
            for index, arg in enumerate(node.args):
                if index >= len(names):
                    break
                ok, value = _resolve_argument(arg, constants)
                if ok:
                    kwargs[names[index]] = value
        calls.append((name, kwargs))
    return calls


def collect_tool_parameters(agent: Any) -> dict[str, list[str]]:
    """``{tool_name: [形参名, ...]}``，来自 ``agent.module2api`` 的 schema 顺序。

    位置参数绑定需要它：``inspect.signature`` 要求工具模块已被 import（可能带 Biopython
    等重依赖），而 schema 里的 ``required_parameters + optional_parameters`` 顺序与真实
    签名一致（见 ``tool/tool_description/*.py``）。
    """
    params: dict[str, list[str]] = {}
    for tools in (getattr(agent, "module2api", None) or {}).values():
        for schema in tools or []:
            if not isinstance(schema, dict) or not schema.get("name"):
                continue
            ordered = list(schema.get("required_parameters") or []) + list(schema.get("optional_parameters") or [])
            params[schema["name"]] = [p.get("name") for p in ordered if isinstance(p, dict) and p.get("name")]
    return params


def _parameter_names(tool_name: str) -> list[str]:
    """形参名顺序：真实签名（工具模块须已 import）→ 空。结果缓存。"""
    if tool_name in _SIG_CACHE:
        return _SIG_CACHE[tool_name]
    names: list[str] = []
    try:
        func = _resolve_tool_function(tool_name)
        if func is not None:
            names = [p.name for p in inspect.signature(func).parameters.values()]
    except Exception:
        names = []
    _SIG_CACHE[tool_name] = list(names)
    return _SIG_CACHE[tool_name]


def _resolve_tool_function(tool_name: str) -> Any:
    """在已 import 的工具模块里找同名函数（找不到返回 ``None``）。"""
    import sys

    for module_name, module in list(sys.modules.items()):
        if module_name.startswith("biomni.tool.") and not module_name.startswith("biomni.tool.tool_description"):
            func = getattr(module, tool_name, None)
            if callable(func):
                return func
    return None


def collect_known_tools(agent: Any) -> set[str]:
    """``agent.module2api`` + ``agent._custom_functions`` 里的全部工具名。"""
    known: set[str] = set()
    for tools in (getattr(agent, "module2api", None) or {}).values():
        for schema in tools or []:
            name = schema.get("name") if isinstance(schema, dict) else None
            if name:
                known.add(name)
    known.update((getattr(agent, "_custom_functions", None) or {}).keys())
    return known


def check_tool_call(tool_name: str, kwargs: dict) -> list[str]:
    """跑 ``tool_name`` 的规则；无规则或规则出错都返回 ``[]``。"""
    rule = get_rule(tool_name)
    if rule is None:
        return []
    try:
        return [str(item) for item in (rule(tool_name, kwargs) or [])]
    except Exception:
        return []


def observation_errors(observation: str) -> list[dict]:
    """检查 ``<observation>``：空输出 / 错误关键词 / 超长。返回错误列表（空=通过）。"""
    body = extract_observation(observation)
    if body is None:
        return [{"tool_name": None, "message": "未找到 <observation> 内容，无法确认执行结果"}]
    if not body.strip():
        return [{"tool_name": None, "message": "返回空输出，无法解析结果"}]
    errors: list[dict] = []
    if body.strip() == "None":
        errors.append({"tool_name": None, "message": "返回 None，无法解析结果"})
    for marker in ERROR_MARKERS:
        if marker in body:
            errors.append({"tool_name": None, "message": f"输出包含错误标记 {marker!r}，执行疑似失败"})
    if len(body) > MAX_OBSERVATION_CHARS:
        errors.append(
            {"tool_name": None, "message": f"输出长度 {len(body)} 字符超过上限 {MAX_OBSERVATION_CHARS}，请分步处理"}
        )
    return errors

