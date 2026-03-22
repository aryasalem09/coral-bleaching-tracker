from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RiskThreshold:
    category: str
    min_score: float
    min_hotspot: float
    min_dhw: float
    color: str


RISK_THRESHOLDS = [
    RiskThreshold("low", 0.0, 0.0, 0.0, "#0f8b6d"),
    RiskThreshold("moderate", 1.0, 0.5, 1.0, "#b9971c"),
    RiskThreshold("high", 2.0, 1.0, 4.0, "#db7c1f"),
    RiskThreshold("severe", 3.0, 1.5, 8.0, "#c34b32"),
]

