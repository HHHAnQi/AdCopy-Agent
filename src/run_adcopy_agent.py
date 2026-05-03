import argparse
import json
from pathlib import Path

import sys
sys.path.append("src")

from agent.workflow import AdCopyReActAgent, state_to_report


DEFAULT_INSTRUCTION = (
    "请根据商品属性生成一段中文电商广告文案，要求突出核心卖点，"
    "表达自然，有购买吸引力，不得编造未提供的信息。"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="商品属性：类型=裙；材质=雪纺；颜色=黑色；风格=优雅；图案=印花。",
    )
    parser.add_argument("--output_file", default="outputs/agent/adcopy_agent_trace.json")
    parser.add_argument("--max_iterations", type=int, default=8)
    parser.add_argument("--generator_mode", choices=["mock", "model"], default="mock")
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--adapter_path", default="outputs/models/qwen2_5_7b_adgen_dpo_v2")
    parser.add_argument("--enable_judge", action="store_true")
    parser.add_argument("--force_rewrite_demo", action="store_true")
    args = parser.parse_args()

    agent = AdCopyReActAgent(
        generator_mode=args.generator_mode,
        base_model=args.base_model,
        adapter_path=args.adapter_path,
        enable_judge=args.enable_judge,
        force_rewrite_demo=args.force_rewrite_demo,
    )
    state = agent.run(
        user_input=args.input,
        instruction=DEFAULT_INSTRUCTION,
        max_iterations=args.max_iterations,
    )

    report = state_to_report(state)

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 80)
    print("FINAL COPY:")
    print(report["final_copy"])
    print("=" * 80)
    print("FINAL REPORT:")
    print(json.dumps(report["final_report"], ensure_ascii=False, indent=2))
    print("=" * 80)
    print("TRACE:")
    for t in report["trace"]:
        print(f"[Step {t['step']}]")
        print("Thought:", t["thought"])
        print("Action:", t["action"])
        print("Observation:", json.dumps(t["observation"], ensure_ascii=False)[:500])
        print("-" * 80)

    print(f"[完成] trace saved to: {out_path}")


if __name__ == "__main__":
    main()
