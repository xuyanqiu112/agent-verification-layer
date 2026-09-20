"""校验层配置（只有 5 个配置项）。

设计边界
--------
* 只用标准库（``dataclasses`` / ``os``），**不引入 pydantic**。
* 字段严格限定为 5 个：``enabled`` / ``enable_llm`` / ``max_retries`` / ``llm_model`` / ``audit_path``。
* ``enabled`` 默认 ``False``：关闭时 ``A1`` 行为与改动前完全一致。
* 本模块**不 import** ``biomni.verification`` 的其他子模块，也不 import ``biomni.agent``，
  因此不存在循环导入。

配置优先级（低 → 高）::

    dataclass 默认值  <  BIOMNI_VERIFY_* 环境变量  <  biomni.config.default_config.verification
                      <  apply_verification(agent, config=...)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

ENV_PREFIX = "BIOMNI_VERIFY_"

_TRUTHY = {"1", "true", "yes", "y", "on"}


@dataclass
class VerificationConfig:
    """校验层配置。

    Attributes:
        enabled: 总开关。``False``（默认）时校验层完全不介入。
        enable_llm: 是否启用 LLM 兜底校验。只在规则校验**通过**后才调用，且最多 10 次。
        max_retries: ``block`` 后允许的重试上限；累计到上限后放行，避免死循环。
        llm_model: LLM 兜底校验使用的模型；``None`` 表示复用 agent 自己的 ``self.llm``。
        audit_path: 审计 JSONL 路径；``None`` 表示只写 ``agent.verification_audit``（内存 list）。
    """

    enabled: bool = False
    enable_llm: bool = False
    max_retries: int = 2
    llm_model: str | None = None
    audit_path: str | None = None

    def to_dict(self) -> dict:
        """转成普通 dict（写入 ``biomni.agent.a1`` 的配置展示、或日志用）。"""
        return {
            "enabled": self.enabled,
            "enable_llm": self.enable_llm,
            "max_retries": self.max_retries,
            "llm_model": self.llm_model,
            "audit_path": self.audit_path,
        }

    def replace(self, **overrides) -> VerificationConfig:
        """返回一份带覆盖值的副本（未知键直接报错，避免静默拼错字段）。"""
        payload = self.to_dict()
        for key, value in overrides.items():
            if value is None:
                continue
            if key not in payload:
                raise ValueError(f"未知的校验层配置项: {key!r}（只支持 {sorted(payload)}）")
            payload[key] = value
        return VerificationConfig(**payload).validate()

    def validate(self) -> VerificationConfig:
        """就地纠正常见错误值（不抛异常，避免打断 agent 启动）。"""
        self.enabled = bool(self.enabled)
        self.enable_llm = bool(self.enable_llm)
        try:
            self.max_retries = max(0, int(self.max_retries))
        except (TypeError, ValueError):
            self.max_retries = 2
        if self.llm_model is not None:
            self.llm_model = str(self.llm_model) or None
        if self.audit_path is not None:
            self.audit_path = str(self.audit_path) or None
        return self

    @classmethod
    def from_env(cls, **overrides) -> VerificationConfig:
        """从 ``BIOMNI_VERIFY_*`` 环境变量构建，``overrides`` 优先。"""
        cfg = cls(
            enabled=_env_bool("ENABLED", False),
            enable_llm=_env_bool("ENABLE_LLM", False),
            max_retries=_env_int("MAX_RETRIES", 2),
            llm_model=os.getenv(ENV_PREFIX + "LLM_MODEL") or None,
            audit_path=os.getenv(ENV_PREFIX + "AUDIT_PATH") or None,
        )
        return cfg.replace(**overrides) if overrides else cfg.validate()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(ENV_PREFIX + name)
    if raw is None:
        return default
    return raw.strip().lower() in _TRUTHY


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(ENV_PREFIX + name)
    if raw is None:
        return default
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return default


def load_config(agent=None) -> VerificationConfig:
    """按优先级解析出最终配置。

    读取顺序：``BIOMNI_VERIFY_*`` → ``biomni.config.default_config.verification``
    → ``agent._verification_config``（``apply_verification`` 写入，优先级最高）。
    后者若不是 5 个配置项之一（例如传入了 ``strict``），这里会抛 ``ValueError``——
    静默忽略配置错误比报错危险得多。

    ``default_config.verification`` 是普通 ``dict``（见 ``biomni/config.py``），
    因此这里不需要 import 校验层即可解出配置。
    """
    from . import _apply_errors

    cfg = VerificationConfig.from_env()

    if agent is not None:
        try:
            from biomni.config import default_config  # noqa: PLC0415 - 惰性，避免早绑定

            payload = getattr(default_config, "verification", None)
            if isinstance(payload, dict):
                cfg = cfg.replace(**payload)
            elif payload is not None and hasattr(payload, "to_dict"):
                cfg = cfg.replace(**payload.to_dict())
        except ValueError:
            raise
        except Exception:
            pass

        own = getattr(agent, "_verification_config", None)
        if own is not None and hasattr(own, "to_dict"):
            # 鸭子类型：模块被重复加载时 isinstance 会失败（见 apply_verification 的说明）
            cfg = cfg.replace(**{k: v for k, v in dict(own.to_dict()).items() if v is not None or k == "enabled"})

    if _apply_errors:
        raise ValueError("；".join(_apply_errors))
    return cfg.validate()


__all__ = ["ENV_PREFIX", "VerificationConfig", "load_config"]
