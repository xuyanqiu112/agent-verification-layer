# 对 Biomni `a1.py` 的改动

## 1. `AgentState` 加 2 个字段

```python
class AgentState(TypedDict):
    messages: list[BaseMessage]
    next_step: str | None
+   validation_errors: list
+   validation_attempts: int
```

## 2. `configure()` 中，`workflow.compile()` 之前插入钩子

```python
+       self._verification_routers = (routing_function, routing_function_self_critic)
+       if getattr(self, "_verification_patch", None) is not None:
+           self._verification_patch(workflow, self)
+       else:
+           workflow.add_edge("execute", "generate")
```

将原代码的 `workflow.add_edge("execute", "generate")`，改成 if/else，把加边责任交给校验层。

## 3. `go()` 初始化新字段

```python
        inputs = {"messages": [HumanMessage(content=prompt)], "next_step": None}
+       inputs["validation_errors"] = []
+       inputs["validation_attempts"] = 0
```

## 4. `go_stream()` 同上
