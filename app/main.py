"""
启动命令：
uvicorn app.main:app --reload --port 8000
"""

import random
import sys
from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI
from pydantic import BaseModel, Field

# 兼容从项目根目录启动服务时导入 src 模块
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))

from rule_eval import rule_score  # noqa: E402


app = FastAPI(title="E-commerce Ad Generator API", version="0.1.0")


class GenerateAdRequest(BaseModel):
    attributes: Dict[str, str] = Field(..., description="商品属性字典")
    num_candidates: int = Field(default=3, ge=1, le=10, description="返回候选条数")


class CandidateItem(BaseModel):
    text: str
    score: float
    rule_detail: Dict[str, object]


class GenerateAdResponse(BaseModel):
    candidates: List[CandidateItem]
    best_text: str


def _format_input_text(attributes: Dict[str, str]) -> str:
    if not attributes:
        return "商品属性：无。"
    kv = "；".join(f"{k}={v}" for k, v in attributes.items())
    return f"商品属性：{kv}。"


def _mock_generate_texts(attributes: Dict[str, str], num_candidates: int) -> List[str]:
    """
    生成多风格 mock 广告文案（不依赖大模型）。
    """
    clean_attrs: Dict[str, str] = {}
    for k, v in attributes.items():
        key = str(k).strip()
        value = str(v).strip()
        if key and value:
            clean_attrs[key] = value

    product_type = clean_attrs.get("类型", "")
    material = clean_attrs.get("材质", "")
    color = clean_attrs.get("颜色", "")
    style = clean_attrs.get("风格", "")

    # 自然描述优先：颜色 + 材质 + 类型 + 风格
    core_phrase = f"{color}{material}{product_type}".strip() or "这款单品"
    style_phrase = f"{style}风格" if style else "整体风格"

    # 其它属性仅使用 value 参与描述，避免出现“字段名硬拼接”
    extra_values = [
        v for k, v in clean_attrs.items() if k not in {"类型", "材质", "颜色", "风格"}
    ]
    extra_hint = "、".join(extra_values[:3]) if extra_values else "版型与细节"
    attr_values = list(clean_attrs.values())
    attr_hint = "、".join(attr_values[:3]) if attr_values else "核心卖点"

    templates = [
        (
            f"{core_phrase}清爽百搭，{style_phrase}耐看不挑人。"
            f"结合{extra_hint}的设计思路，日常通勤与休闲场景都能轻松驾驭，"
            "上身舒适自然，穿搭省心。"
        ),
        (
            f"如果你偏爱{style_phrase}，这款{product_type or '单品'}会是不错选择。"
            f"{core_phrase}在视觉上干净利落，同时通过{extra_hint}增强层次感，"
            "既保留实用性，也兼顾时尚感。"
        ),
        (
            f"从细节到气质都很在线的一款{product_type or '单品'}，重点突出{attr_hint}。"
            f"{material or '面料'}质感与版型相互平衡，搭配{color or '经典'}色调，"
            "日常上身不挑人，轻松穿出利落感。"
        ),
        (
            f"这款{core_phrase}以{attr_hint}为灵感，视觉上简洁大方。"
            f"再加上{extra_hint}等元素，能够在保持舒适体验的同时，提升整体穿搭质感。"
        ),
        (
            f"一件真正好搭配的{product_type or '单品'}，往往在细节上足够克制。"
            f"这款围绕{attr_hint}展开设计，{style_phrase}自然不张扬，"
            "日常出门随手搭配就很出彩。"
        ),
    ]

    if num_candidates <= len(templates):
        # 轻微随机，避免总是固定前几条
        random.seed(42)
        return random.sample(templates, k=num_candidates)

    texts = templates[:]
    while len(texts) < num_candidates:
        texts.append(
            f"这款{core_phrase}围绕{attr_hint}进行设计，整体风格自然百搭，"
            "兼顾舒适与实用，能够满足通勤和日常休闲等多种场景需求。"
        )
    return texts


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/generate_ad", response_model=GenerateAdResponse)
def generate_ad(req: GenerateAdRequest) -> GenerateAdResponse:
    input_text = _format_input_text(req.attributes)
    candidates_text = _mock_generate_texts(req.attributes, req.num_candidates)

    scored: List[CandidateItem] = []
    for text in candidates_text:
        detail = rule_score(input_text, text)
        scored.append(
            CandidateItem(
                text=text,
                score=float(detail.get("total_score", 0.0)),
                rule_detail=detail,
            )
        )

    scored.sort(key=lambda x: x.score, reverse=True)
    best_text = scored[0].text if scored else ""

    # TODO: 后续替换为 vLLM / transformers 推理，并保留当前规则打分重排逻辑。
    return GenerateAdResponse(candidates=scored, best_text=best_text)

