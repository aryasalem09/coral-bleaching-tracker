from __future__ import annotations

from .scoring import RiskScore


def explain_risk(score: RiskScore) -> str:
    if score.category == "severe":
        return (
            "Thermal stress is in the severe range: both the hotspot signal and accumulated "
            "heat stress are elevated enough to indicate strong bleaching pressure."
        )
    if score.category == "high":
        return (
            "Thermal stress is high: current hotspot values are elevated and heat has been "
            "accumulating long enough to raise bleaching concern."
        )
    if score.category == "moderate":
        return (
            "Thermal stress is moderate: temperatures are above the usual threshold or recent "
            "heat accumulation is noticeable, but not yet in the highest warning range."
        )
    return (
        "Thermal stress is low: hotspot and accumulated heat stress are both limited, so the "
        "environmental signal alone does not suggest strong bleaching pressure."
    )
