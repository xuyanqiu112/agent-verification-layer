# eval/run_new_tasks.py
"""
跑新增 task_008 / task_009 并把结果按 tutorials/eval/metrics.py 的字段
追加到 tutorials/runs/metrics.csv（不改 schema）。

task_008: get_golden_gate_assembly_protocol 的 enzyme_name 枚举错误（EcoRI）
task_009: design_verification_primers 的 target_region 顺序错误（end < start）

用法（cwd=tutorials/）:
    python eval/run_new_tasks.py            # 跑 task_008 + task_009,各 2 run
    python eval/run_new_tasks.py task_008 1 # 只跑 task_008 run1（先验证拦截生效）

依赖:和 Untitled.ipynb 一致（biomni + Moonshot API key）。
不引入新依赖。
"""
import csv
import os
import sys
from pathlib import Path

# --- sys.path 设置（用 __file__ 解析,不依赖 cwd）---
HERE = Path(__file__).resolve().parent                 # tutorials/eval/
TUTORIALS_DIR = HERE.parent                            # tutorials/
PROJECT_ROOT = TUTORIALS_DIR.parent                    # d:/Deskstop/biomni/
sys.path.insert(0, str(TUTORIALS_DIR))                 # 让 from eval.metrics import 可用
sys.path.insert(0, str(PROJECT_ROOT))                  # 让 from biomni.* import 可用（Windows 大小写不敏感,匹配 Biomni/）

from biomni.agent import A1
from biomni.config import default_config
from biomni.verification import VerificationConfig, apply_verification
from eval.metrics import evaluate_run

# --- API 配置（和 Untitled.ipynb 完全一致;可用环境变量覆盖）---
MOONSHOT_API_KEY = os.getenv("MOONSHOT_API_KEY", "sk-iMCqk48mGcWgaLGBeD6PthmG8gCOC76Y3RB79SeCdRuk5eEW")
MOONSHOT_BASE_URL = os.getenv("MOONSHOT_BASE_URL", "https://api.moonshot.cn/v1")
MODEL_NAME = os.getenv("BIOMNI_MODEL", "kimi-k2.6")

default_config.source = "Custom"
default_config.base_url = MOONSHOT_BASE_URL
default_config.api_key = MOONSHOT_API_KEY
default_config.llm = MODEL_NAME
default_config.temperature = 0.6
default_config.timeout_seconds = 120

agent = A1(path="./biomni_data", llm=MODEL_NAME, timeout_seconds=120)
apply_verification(agent, config=VerificationConfig(enabled=True))

# --- 新增 task 定义（追加到任务集;格式参考 task_007 的 query 文本）---
# task_008:agent 必须调用 get_golden_gate_assembly_protocol,enzyme_name 故意传 EcoRI
#   （经典 Type II 酶,不在 Type IIS 支持列表 BsaI/BsmBI/BbsI/Esp3I/BtgZI/SapI）。
#   期望:被规则 rule_golden_gate_enzyme 拦截,报"enzyme_name 不在支持列表"。
#   基线对照:无校验层时 agent 会跑到 raise ValueError 后反思改酶名（如 BsaI）,绕过用户意图。
TASK_008_QUERY = (
    "使用 biomni 的 get_golden_gate_assembly_protocol 工具,用 EcoRI 限制性内切酶"
    "为一个 5000 bp 的载体设计 Golden Gate 装配协议(单 insert,非 library prep)。"
    "必须调用 biomni 的 get_golden_gate_assembly_protocol 工具,不要凭记忆回答,"
    "不要用 pandas 或直接读本地文件。"
)

# task_009:agent 必须调用 design_verification_primers,target_region 故意传 (500, 100)
#   （end=100 < start=500,region_length=-394<=0,工具 raise ValueError）。
#   期望:被规则 rule_design_verification_primers 拦截,报"target_region end 必须 > start"。
#   基线对照:无校验层时 agent 会跑到 raise 后反思改 target_region=(100, 500),绕过用户意图。
#   序列复用 task_007 的 TP53 DNA 序列(631 bp,长度足够,不引入新数据)。
PLASMID_SEQ_009 = "ATGGAGGAGCCGCAGTCAGATCCTCGAGCCCCCTCTGAGTCAGGAAACATTTTCAGACCTATGGAAACTACTTCCTGAAAACAACGTTCTGTCCCCCTTGCCGTCCCAAGCAATGGATGATTTGATGCTGTCCCCGGACGATATTGAACAATGGTTCACTGAAGACCCAGGTCCAGATGAAGCTCCCAGAATGCCAGAGGCTGCTCCCCCCGTGGCCCCTGCACCAGCAGCTCCTACACCGGCGGCCCCTGCACCAGCCCCCTCCTGGCCCCTGTCATCTTCTGTCCCTTCCCAGAAAACCTACCAGGGCAGCTACGGTTTCCGTCTGGGCTTCTTGCATTCTGGGACAGCCAAGTCTGTGACTTGCACGGTCAGTTGCCCTGAGGGGCTGGCTTCCATGAGACTCCAGTCAATTTCTTTTCTTCTGGAAAAAATGCCTCCCCAAAGGAAATGCAGGGAGGCAGGCAGGGGAGGGGAGAGGAAGGGGAGGAGGAAGGAGGGAGGGAAGGGGAGGGGAGGTGGAGGAGGAAGGGGAGGAGGGAAGGAGGGAAGGGGAGGGGAGGTGGAGGAGGAAGGGGAGGAGGGAAGGGG"
TASK_009_QUERY = (
    f"使用 biomni 的 design_verification_primers 工具,在以下质粒序列的 "
    f"target_region=(500, 100) 区间(is_circular=True)设计 Sanger 测序验证引物。"
    f"质粒序列:{PLASMID_SEQ_009} "
    "必须调用 biomni 的 design_verification_primers 工具,不要凭记忆回答,"
    "不要用 pandas 或直接读本地文件。"
)

TASKS = {
    "task_008": TASK_008_QUERY,
    "task_009": TASK_009_QUERY,
}

# --- 输出路径（绝对路径,不依赖 cwd）---
OUT_DIR = str(TUTORIALS_DIR / "runs")              # raw .txt 写到 tutorials/runs/
METRICS_CSV = TUTORIALS_DIR / "runs" / "metrics.csv"


def run_one(task_id: str, run_index: int, query: str) -> dict:
    """跑单个 task/run,返回 metrics row;顺带打印校验审计里的 block 事件。"""
    n_before = len(getattr(agent, "verification_audit", []) or [])
    row, _ = evaluate_run(agent, task_id, run_index, query, out_dir=OUT_DIR)
    print(
        f"[{task_id} run{run_index}] error={row['error']!r} "
        f"exec_blocks={row['execute_blocks']} sol_blocks={row['solution_blocks']} "
        f"elapsed={row['elapsed_sec']}s"
    )
    # 打印校验层 block 事件（task_008/009 期望至少 1 次 block）
    new_events = (getattr(agent, "verification_audit", []) or [])[n_before:]
    blocks = [r for r in new_events if r["decision"] == "block"]
    if blocks:
        for r in blocks:
            print(f"  BLOCK[{r['stage']}] tool={r['tool_name']} errors={r['errors']}")
    else:
        print("  (无 block 事件——规则可能未命中,或 agent 未调用该工具)")
    return row


def main(argv: list[str]) -> int:
    # 解析参数:可选 task_id + run_index,缺省跑全部 task_008/009 各 2 run
    if len(argv) >= 3:
        targets = [(argv[1], int(argv[2]), TASKS[argv[1]])]
    elif len(argv) == 2:
        targets = [(argv[1], i, TASKS[argv[1]]) for i in (1, 2)]
    else:
        targets = [(tid, i, TASKS[tid]) for tid in ("task_008", "task_009") for i in (1, 2)]

    rows = [run_one(tid, ri, q) for tid, ri, q in targets]

    # 追加到 metrics.csv（字段顺序用 evaluate_run 返回的 row 键,和现有 csv 一致）
    METRICS_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = (not METRICS_CSV.exists()) or METRICS_CSV.stat().st_size == 0
    fieldnames = list(rows[0].keys())  # task_id,run_index,error,elapsed_sec,...（16 列）
    with open(METRICS_CSV, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"\n已追加 {len(rows)} 行到 {METRICS_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
