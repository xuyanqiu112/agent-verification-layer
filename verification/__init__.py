"""Biomni 极简可插拔校验层（minimal pluggable verification layer）。

硬约束（与本文件的存在理由一一对应）
------------------------------------
* 只有 4 个文件：``__init__`` / ``config`` / ``rules`` / ``graph_patch``。
* 只有 2 个校验点：``verify_tool_call``（generate → execute 之间）、
  ``verify_tool_result``（execute → generate 之间）。
* 只有 2 个失败动作：``pass`` / ``block``。
* **human-in-the-loop 未实现**（批量评测场景不适合阻塞），代码里没有任何人工校验。
* 默认关闭：``enabled=False`` 时 ``A1`` 的行为与改动前完全一致。

为什么是惰性导入
----------------
``graph_patch`` 需要 ``langgraph`` 与 ``langchain_core``；``rules``/``config`` 只需标准库。
惰性导入让 ``import biomni.verification`` 在窄环境里也能成功，并且让
``biomni.config`` ↔ ``biomni.verification`` 之间不存在循环导入。

用法::

    from biomni.agent import A1
    from biomni.verification import VerificationConfig, apply_verification

    agent = A1(...)
    apply_verification(agent, config=VerificationConfig(enabled=True, max_retries=2))
    log, answer = agent.go(task)       # go()/go_stream() 的签名与行为不变
"""

from __future__ import annotations

from typing import Any

from .config import ENV_PREFIX, VerificationConfig, load_config

__all__ = [
    "ENV_PREFIX",
    "VerificationConfig",
    "apply_verification",
    "disable_verification",
    "load_config",
    "rules",
    "graph_patch",
]

__version__ = "2.1.0"

#: 配置归一化过程中发现的问题（由 ``graph_patch.load_config`` 带图重建时抛出）。
_apply_errors: list[str] = []


def apply_verification(agent: Any, config: VerificationConfig | dict | None = None) -> VerificationConfig:
    """把校验层挂到 ``A1`` 实例上，并**立即重建图**。

    为什么必须重建图
    ----------------
    ``A1.__init__`` 末尾已经调用过 ``configure()``，那时校验层还没挂上。所以仅记录配置
    是不够的——必须再触发一次 ``configure()``，让它在 ``workflow.compile()`` 之前调用
    ``patch_workflow``。本函数用"设钩子 + 调 configure"的方案（不包装 ``configure``）：

    1. ``agent._verification_config = config``（节点在运行时读取，支持中途改配置）；
    2. ``agent._verification_patch = patch_workflow``（``configure`` 里的钩子）；
    3. ``agent.verification_audit = []``（审计用内存 list）；
    4. 调用 ``agent.configure()`` 触发重建。

    ``biomni/agent/a1.py`` 的 ``configure()`` 内只有 3 行改动::

        if getattr(self, "_verification_patch", None) is not None:
            self._verification_patch(workflow, self)

    幂等：重复调用只更新配置并再重建一次图，以最后一次为准；``enabled=False`` 时
    ``patch_workflow`` 直接返回，图与原版一致。

    Args:
        agent: ``biomni.agent.a1.A1`` 实例（需要 ``configure`` 方法）。
        config: :class:`VerificationConfig`、覆盖用 ``dict``（键必须是 5 个配置项之一），或 ``None``。

    Returns:
        归一化后的 :class:`VerificationConfig`。
    """
    effective = _coerce_config(config)

    agent._verification_config = effective
    if not isinstance(getattr(agent, "verification_audit", None), list):
        agent.verification_audit = []

    from .graph_patch import patch_workflow  # 惰性：避免 import 期拉入 langgraph

    agent._verification_patch = patch_workflow

    configure = getattr(agent, "configure", None)
    if callable(configure):
        configure()  # 在 compile() 之前重新插桩，图只编译这一次

    return effective


def disable_verification(agent: Any) -> None:
    """关闭校验层并重建图（回到未插桩状态）。"""
    agent._verification_config = VerificationConfig(enabled=False)
    agent._verification_patch = None
    configure = getattr(agent, "configure", None)
    if callable(configure):
        configure()


def _coerce_config(config: VerificationConfig | dict | None) -> VerificationConfig:
    """把 ``None`` / ``dict`` / 任何"像 VerificationConfig"的对象归一化。

    刻意用鸭子类型而不是 ``isinstance``：同一个模块被加载两次（notebook ``%reload_ext``、
    多份 ``sys.path``）会产生**两个** ``VerificationConfig`` 类，``isinstance`` 会假失败。
    未知配置项不在这里抛异常（``graph_patch`` 在重建图时会带着可读信息抛出），但会记录到
    ``_apply_errors``，因为静默忽略配置错误比报错更危险。
    """
    if isinstance(config, VerificationConfig):
        return config.validate()
    if isinstance(config, dict):
        return VerificationConfig.from_env(**config)
    if config is not None and hasattr(config, "to_dict"):
        payload = dict(config.to_dict())
        unknown = [k for k in payload if k not in VerificationConfig.__dataclass_fields__]
        if unknown:
            _apply_errors.append(f"未知配置项 {sorted(unknown)}（只支持 enabled/enable_llm/max_retries/llm_model/audit_path）")
        merged = VerificationConfig.from_env()
        known = {k: v for k, v in payload.items() if k in merged.to_dict() and (v is not None or k == "enabled")}
        return merged.replace(**known)
    return VerificationConfig.from_env()


def __getattr__(name: str) -> Any:
    """PEP 562：``biomni.verification.rules`` / ``.graph_patch`` 惰性可用。"""
    if name in {"rules", "graph_patch"}:
        import importlib

        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
