from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class Candidate:
    candidate_id: str
    text: str
    source: str = "generated"
    rule_metrics: Dict[str, Any] = field(default_factory=dict)
    judge_scores: Dict[str, Any] = field(default_factory=dict)
    issues: List[str] = field(default_factory=list)
    combined_score: float = 0.0


@dataclass
class TraceStep:
    step: int
    thought: str
    action: str
    action_input: Dict[str, Any]
    observation: Dict[str, Any]


@dataclass
class AgentState:
    user_input: str
    instruction: str
    attributes: Dict[str, str] = field(default_factory=dict)
    candidates: List[Candidate] = field(default_factory=list)
    current_best_id: Optional[str] = None
    diagnosis_done: bool = False
    judge_enabled: bool = False
    judge_done: bool = False
    force_rewrite_demo: bool = False
    force_rewrite_used: bool = False
    final_copy: Optional[str] = None
    final_report: Dict[str, Any] = field(default_factory=dict)
    trace: List[TraceStep] = field(default_factory=list)
    iteration: int = 0
    max_iterations: int = 4
    should_stop: bool = False


def to_dict(obj: Any) -> Dict[str, Any]:
    return asdict(obj)
