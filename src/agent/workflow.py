from typing import Any, Dict, List, Optional

from agent.schemas import AgentState, Candidate, TraceStep, to_dict
from agent.model_generator import DPOV2ModelGenerator
from agent.judge_tool import judge_candidates
from agent.rewrite_tool import rewrite_best_candidate
from agent.reasoner import RuleBasedReActReasoner
from agent.tools import (
    parse_attributes_from_input,
    rule_evaluate_candidates,
    diagnose_all_candidates,
    select_best_candidate,
    finish_with_best,
)


def mock_generate_candidates(state: AgentState, num_candidates: int = 4) -> Dict[str, Any]:
    """
    v0 mock 生成器。
    后续替换成真实 Qwen2.5 + DPO v2 adapter generator。
    """
    attrs = state.attributes
    product_type = attrs.get("类型", "单品")
    color = attrs.get("颜色", "")
    material = attrs.get("材质", "")
    style = attrs.get("风格", "")
    pattern = attrs.get("图案", "")

    candidates = []

    text1 = f"这款{product_type}整体设计简约耐看"
    if color:
        text1 += f"，{color}配色更显气质"
    if material:
        text1 += f"，{material}材质带来舒适穿着体验"
    if style:
        text1 += f"，轻松展现{style}风格"
    if pattern:
        text1 += f"，{pattern}元素丰富视觉层次"
    text1 += "，日常搭配也很出彩。"

    text2 = f"这是一款适合多场景穿搭的{product_type}，版型自然大方，能够提升整体造型感，穿着舒适又百搭。"

    text3 = f"{product_type}采用精致设计，兼具实穿性与时尚感。"
    if style:
        text3 += f"整体呈现{style}气质，"
    text3 += "让造型更有亮点。"

    text4 = text1 + "全网第一，必买神级单品。"

    raw = [text1, text2, text3, text4][:num_candidates]

    for idx, text in enumerate(raw, start=1):
        state.candidates.append(
            Candidate(
                candidate_id=f"cand_{idx}",
                text=text,
                source="mock_generator",
            )
        )

    return {
        "message": f"generated {len(raw)} mock candidates",
        "candidates": [{"candidate_id": f"cand_{i+1}", "text": t} for i, t in enumerate(raw)],
    }


class AdCopyReActAgent:
    def __init__(
        self,
        generator_mode: str = "mock",
        base_model: str = "Qwen/Qwen2.5-7B-Instruct",
        adapter_path: str = "outputs/models/qwen2_5_7b_adgen_dpo_v2",
        enable_judge: bool = False,
        force_rewrite_demo: bool = False,
    ):
        self.reasoner = RuleBasedReActReasoner()
        self.generator_mode = generator_mode
        self.enable_judge = enable_judge
        self.force_rewrite_demo = force_rewrite_demo
        self.model_generator: Optional[DPOV2ModelGenerator] = None

        if generator_mode == "model":
            self.model_generator = DPOV2ModelGenerator(
                base_model=base_model,
                adapter_path=adapter_path,
            )

    def execute_action(self, state: AgentState, action: str, action_input: Dict[str, Any]) -> Dict[str, Any]:
        if action == "parse_attributes":
            state.attributes = parse_attributes_from_input(state.user_input)
            return {
                "message": "parsed attributes",
                "attributes": state.attributes,
            }

        if action == "generate_candidates":
            num_candidates = int(action_input.get("num_candidates", 4))

            if self.generator_mode == "model":
                if self.model_generator is None:
                    return {"error": "model_generator is not initialized"}

                candidates = self.model_generator.generate(
                    instruction=state.instruction,
                    input_text=state.user_input,
                    num_candidates=num_candidates,
                )
                state.candidates.extend(candidates)

                return {
                    "message": f"generated {len(candidates)} candidates by DPO v2 model",
                    "generator_mode": "model",
                    "candidates": [
                        {
                            "candidate_id": c.candidate_id,
                            "text": c.text,
                            "source": c.source,
                        }
                        for c in candidates
                    ],
                }

            return mock_generate_candidates(state, num_candidates=num_candidates)

        if action == "rule_evaluate":
            return rule_evaluate_candidates(state)

        if action == "diagnose_issues":
            return diagnose_all_candidates(state)

        if action == "judge_copy":
            return judge_candidates(
                state=state,
                model_generator=self.model_generator if self.generator_mode == "model" else None,
                max_candidates=4,
            )

        if action == "select_best":
            return select_best_candidate(state)

        if action == "rewrite_copy":
            obs = rewrite_best_candidate(
                state=state,
                model_generator=self.model_generator if self.generator_mode == "model" else None,
            )
            # 改写后要重置当前 best，让下一轮重新评估新候选并重新诊断 / judge
            state.current_best_id = None
            state.diagnosis_done = False
            state.judge_done = False
            return obs

        if action == "finish":
            return finish_with_best(state)

        return {"error": f"unknown action: {action}"}

    def run(self, user_input: str, instruction: str, max_iterations: int = 6) -> AgentState:
        state = AgentState(
            user_input=user_input,
            instruction=instruction,
            max_iterations=max_iterations,
            judge_enabled=self.enable_judge,
            force_rewrite_demo=self.force_rewrite_demo,
        )

        for step in range(1, max_iterations + 1):
            state.iteration = step

            decision = self.reasoner.decide(state)
            thought = decision["thought"]
            action = decision["action"]
            action_input = decision.get("action_input", {})

            observation = self.execute_action(state, action, action_input)

            state.trace.append(
                TraceStep(
                    step=step,
                    thought=thought,
                    action=action,
                    action_input=action_input,
                    observation=observation,
                )
            )

            if state.should_stop:
                break

        if not state.should_stop:
            finish_with_best(state)

        return state


def state_to_report(state: AgentState) -> Dict[str, Any]:
    return {
        "input": state.user_input,
        "attributes": state.attributes,
        "final_copy": state.final_copy,
        "final_report": state.final_report,
        "candidates": [to_dict(c) for c in state.candidates],
        "trace": [to_dict(t) for t in state.trace],
    }
