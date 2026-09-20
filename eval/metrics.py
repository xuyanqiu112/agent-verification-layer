# eval/metrics.py
import time
import json
import re
from pathlib import Path
from datetime import datetime


def _content_str(m):
    c = getattr(m, "content", "")
    if isinstance(c, list):
        return "".join(
            str(b.get("text", "")) if isinstance(b, dict) else str(b)
            for b in c
        )
    return str(c)


def count_tags(messages, tag):
    pattern = re.compile(rf"<{tag}>(.*?)</{tag}>", re.DOTALL | re.IGNORECASE)
    return sum(len(pattern.findall(_content_str(m))) for m in messages)


def extract_token_usage(messages):
    in_tok = out_tok = 0
    found = False
    for m in messages:
        um = getattr(m, "usage_metadata", None)
        if um:
            in_tok += um.get("input_tokens", 0) or 0
            out_tok += um.get("output_tokens", 0) or 0
            found = True
        rm = getattr(m, "response_metadata", None) or {}
        if isinstance(rm, dict):
            tu = rm.get("token_usage") or rm.get("usage") or {}
            if tu:
                in_tok += tu.get("prompt_tokens", 0) or 0
                out_tok += tu.get("completion_tokens", 0) or 0
                found = True
    if not found:
        return None
    return {"input_tokens": in_tok, "output_tokens": out_tok, "total_tokens": in_tok + out_tok}


def estimate_tokens(messages):
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
    except Exception as e:
        return {"error": f"tiktoken unavailable: {e}"}
    text = "\n".join(_content_str(m) for m in messages)
    return {"estimated_tokens": len(enc.encode(text)), "source": "tiktoken_cl100k_approx"}


def extract_tool_calls(agent, messages):
    tool_calls = []
    for m in messages:
        for block in re.findall(r"<execute>(.*?)</execute>", _content_str(m), re.DOTALL):
            parsed = []
            try:
                if hasattr(agent, "_parse_tool_calls_with_modules"):
                    parsed = agent._parse_tool_calls_with_modules(block)
            except Exception:
                parsed = []
            tool_calls.append({"code_len": len(block), "parsed_tools": parsed})
    return tool_calls


def evaluate_run(agent, task_id, run_index, query, out_dir="baseline/runs"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # 关键：重置跨 run 状态
    agent._execution_results = []
    agent._conversation_state = None
    agent.critic_count = 0

    start = time.time()
    try:
        log, final_content = agent.go(query)
        error = ""
    except Exception as e:
        log, final_content, error = [], "", repr(e)

    elapsed = time.time() - start
    final_content = final_content if final_content is not None else ""

    state = getattr(agent, "_conversation_state", None) or {}
    messages = state.get("messages", []) if isinstance(state, dict) else []

    token_real = extract_token_usage(messages)
    token_est = estimate_tokens(messages) if token_real is None else None
    tool_calls = extract_tool_calls(agent, messages)
    exec_results = getattr(agent, "_execution_results", []) or []

    raw_path = Path(out_dir) / f"{task_id}_run{run_index}.txt"
    raw_path.write_text(
        "\n".join(str(x) for x in log) + "\n\n=== FINAL ===\n" + final_content,
        encoding="utf-8",
    )

    row = {
        "task_id": task_id,
        "run_index": run_index,
        "error": error,
        "elapsed_sec": round(elapsed, 1),
        "messages_count": len(messages),
        "execute_blocks": count_tags(messages, "execute"),
        "observation_blocks": count_tags(messages, "observation"),
        "solution_blocks": count_tags(messages, "solution"),
        "tool_calls_count": len(tool_calls),
        "execution_results_count": len(exec_results),
        "token_real": json.dumps(token_real, ensure_ascii=False) if token_real else "",
        "token_estimated": json.dumps(token_est, ensure_ascii=False) if token_est else "",
        "final_output_len": len(final_content),
        "final_output_preview": final_content[:200],
        "raw_output_path": str(raw_path),
        "timestamp": datetime.now().isoformat(),
    }
    return row, final_content