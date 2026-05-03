from typing import Any, Dict

from agent.schemas import AgentState, Candidate


class RuleBasedReActReasoner:
    """
    v0/v1: Rule-based ReAct Reasoner.
    It produces Thought-Action decisions based on the current state.
    Later this can be replaced by an LLM reasoner.
    """

    def _needs_rewrite(self, cand: Candidate) -> bool:
        m = cand.rule_metrics or {}
        j = cand.judge_scores or {}

        coverage = float(m.get("coverage", 0.0))
        forbidden = float(m.get("forbidden_count", 0.0))
        conflicts = float(m.get("conflict_count", 0.0))
        repetition = float(m.get("repetition_ratio", 0.0))
        length = float(m.get("length", len(cand.text)))

        judge_total = float(j.get("total", 0.0))
        factual = float(j.get("factual_consistency", 5.0))
        compliance = float(j.get("compliance", 5.0))
        attr_cov = float(j.get("attribute_coverage", 5.0))

        # Hard rule issues
        if forbidden > 0:
            return True
        if conflicts > 0:
            return True
        if coverage < 0.75:
            return True
        if repetition > 0.12:
            return True
        if length < 45 or length > 170:
            return True

        # Judge hard issues
        if judge_total > 0:
            if judge_total < 20:
                return True
            if factual < 4:
                return True
            if compliance < 5:
                return True
            if attr_cov < 4:
                return True

        # Only rewrite rule-critical issues.
        # Do not rewrite merely because of mild judge_issue comments.
        critical_prefixes = (
            "missing_attributes:",
            "forbidden_word:",
            "low_coverage:",
            "too_short:",
            "too_long:",
            "high_repetition:",
            "unsupported_color_description:",
        )
        for issue in cand.issues:
            if str(issue).startswith(critical_prefixes):
                return True

        return False

    def decide(self, state: AgentState) -> Dict[str, Any]:
        if not state.attributes:
            return {
                "thought": "当前还没有解析商品属性，需要先感知输入并提取关键属性。",
                "action": "parse_attributes",
                "action_input": {},
            }

        if not state.candidates:
            return {
                "thought": "当前还没有候选文案，需要先调用生成工具生成多个候选。",
                "action": "generate_candidates",
                "action_input": {"num_candidates": 4},
            }

        if any(not c.rule_metrics for c in state.candidates):
            return {
                "thought": "当前已有候选文案，但还没有规则评分，需要调用 rule_evaluate 工具观察候选质量。",
                "action": "rule_evaluate",
                "action_input": {},
            }

        if not state.diagnosis_done:
            return {
                "thought": "当前已有规则评分，需要诊断候选是否存在属性遗漏、违规词、冲突、过长或过短等问题。",
                "action": "diagnose_issues",
                "action_input": {},
            }

        if state.judge_enabled and not state.judge_done:
            return {
                "thought": "当前候选已经完成规则评分和问题诊断，但还需要从语义层面评估属性覆盖、事实一致性、自然度、吸引力和合规性，因此调用 LLM-as-Judge 工具。",
                "action": "judge_copy",
                "action_input": {},
            }

        if state.current_best_id is None:
            return {
                "thought": "当前候选已经评估、诊断并完成必要的 judge，需要选择综合分最高的候选作为当前最佳文案。",
                "action": "select_best",
                "action_input": {},
            }

        best = next((c for c in state.candidates if c.candidate_id == state.current_best_id), None)

        # Demo mode: force one rewrite after selecting the first best candidate.
        # This is only for demonstrating the Agent's self-revision loop.
        if (
            best
            and state.force_rewrite_demo
            and not state.force_rewrite_used
            and best.source != "llm_rewrite"
            and state.iteration < state.max_iterations
        ):
            if "demo_force_rewrite: improve copy while preserving attributes" not in best.issues:
                best.issues.append("demo_force_rewrite: improve copy while preserving attributes")
            state.force_rewrite_used = True
            return {
                "thought": "当前处于 rewrite demo 模式。虽然最佳候选基本合格，但为了展示 Agent 的自我优化能力，需要调用 rewrite 工具进行一次受约束改写，然后重新评估。",
                "action": "rewrite_copy",
                "action_input": {
                    "candidate_id": best.candidate_id,
                    "issues": best.issues,
                },
            }

        if best and self._needs_rewrite(best) and state.iteration < state.max_iterations:
            if best.source != "rewrite" and best.source != "llm_rewrite":
                return {
                    "thought": "当前最佳候选存在会影响最终质量的严重问题，需要调用 rewrite 工具进行修正，然后重新评估。",
                    "action": "rewrite_copy",
                    "action_input": {
                        "candidate_id": best.candidate_id,
                        "issues": best.issues,
                    },
                }

        return {
            "thought": "当前最佳候选已经满足主要约束，可以停止并输出最终文案。",
            "action": "finish",
            "action_input": {},
        }
