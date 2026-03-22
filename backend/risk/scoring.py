from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .thresholds import RISK_THRESHOLDS


@dataclass(frozen=True)
class RiskScore:
    score: float
    category: str
    hotspot: float
    dhw: float
    color: str
    components: dict[str, float]


def _component_score(hotspot: float, dhw: float) -> dict[str, float]:
    hotspot_points = 0.0
    if hotspot >= 1.5:
        hotspot_points = 2.0
    elif hotspot >= 1.0:
        hotspot_points = 1.5
    elif hotspot >= 0.5:
        hotspot_points = 1.0
    elif hotspot > 0:
        hotspot_points = 0.5

    dhw_points = 0.0
    if dhw >= 8:
        dhw_points = 2.0
    elif dhw >= 4:
        dhw_points = 1.5
    elif dhw >= 1:
        dhw_points = 1.0
    elif dhw > 0:
        dhw_points = 0.5

    return {
        "hotspot_points": hotspot_points,
        "dhw_points": dhw_points,
        "combined_score": hotspot_points + dhw_points,
    }


def score_environmental_risk(hotspot: float, dhw: float) -> RiskScore:
    components = _component_score(float(hotspot), float(dhw))
    score = float(components["combined_score"])

    chosen = RISK_THRESHOLDS[0]
    for threshold in RISK_THRESHOLDS:
        if score >= threshold.min_score:
            chosen = threshold

    return RiskScore(
        score=score,
        category=chosen.category,
        hotspot=float(hotspot),
        dhw=float(dhw),
        color=chosen.color,
        components=components,
    )


def score_from_record(record: dict[str, Any]) -> RiskScore:
    hotspot = float(record.get("hotspot_like", record.get("hotspot", record.get("tsa", 0.0))) or 0.0)
    dhw = float(record.get("dhw_like", record.get("dhw", record.get("tsa_dhw", 0.0))) or 0.0)
    return score_environmental_risk(hotspot=hotspot, dhw=dhw)
