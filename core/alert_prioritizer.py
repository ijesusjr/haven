"""
core/alert_prioritizer.py
--------------------------
Crosses all three risk signals × inventory gaps → ranked, actionable alert list.

Option C architecture — three independent signals feed into the prioritizer:
    WEATHER     — fast-moving, primary signal (OWM One Call)
    GEO         — slow-moving, contextual signal (ACLED)
    HEALTH      — medium cadence, epidemic signal (ECDC CDTR)

Alert categories:
    WEATHER     — weather risk alone is noteworthy
    COMBINED    — weather risk amplifies a kit gap (highest value alerts)
    EXPIRY      — item expiring within warning threshold
    KIT_GAP     — item below EU recommended quantity
    GEO         — geopolitical trend is increasing
    HEALTH      — active health threat in EU/EEA
    HEALTH_KIT  — health threat makes a specific kit gap more urgent
                  e.g. respiratory threat + no meds in kit

Sorting order within equal priority scores:
    COMBINED > HEALTH_KIT > EXPIRY > WEATHER > HEALTH > GEO > KIT_GAP
"""

from dataclasses import dataclass
from typing import List, Optional

from core.risk_engine import RiskResult
from core.inventory_analyzer import InventoryReport


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class PrioritizedAlert:
    priority_score: int     # 0-100, higher = more urgent
    urgency:        str     # IMMEDIATE / SOON / ROUTINE
    category:       str     # WEATHER / COMBINED / EXPIRY / KIT_GAP /
                            # GEO / HEALTH / HEALTH_KIT
    message:        str
    detail:         str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _urgency(score: int) -> str:
    if score >= 70: return "IMMEDIATE"
    if score >= 40: return "SOON"
    return "ROUTINE"


_CATEGORY_ORDER = {
    "COMBINED":   0,
    "HEALTH_KIT": 1,
    "EXPIRY":     2,
    "WEATHER":    3,
    "HEALTH":     4,
    "GEO":        5,
    "KIT_GAP":    6,
}

# Kit categories relevant to health threats
_HEALTH_RELEVANT_CATEGORIES = {"meds", "hygiene"}


# ---------------------------------------------------------------------------
# Weather alerts
# ---------------------------------------------------------------------------

def _weather_alerts(risk: RiskResult) -> List[PrioritizedAlert]:
    """Alert when weather risk is ELEVATED or above."""
    if risk.risk_level == "LOW" or risk.risk_score == 0:
        return []
    return [PrioritizedAlert(
        priority_score=risk.risk_score,
        urgency=_urgency(risk.risk_score),
        category="WEATHER",
        message=f"Weather risk is {risk.risk_level} ({risk.risk_score}/100).",
        detail=(
            f"Weather severity: {risk.weather_severity} | "
            f"Alert severity: {risk.alert_severity} | "
            f"Wind: {risk.wind_bonus} | Rain: {risk.rain_bonus}"
        ),
    )]


# ---------------------------------------------------------------------------
# Combined: weather × kit gaps
# ---------------------------------------------------------------------------

def _combined_alerts(
    risk: RiskResult,
    report: InventoryReport,
) -> List[PrioritizedAlert]:
    """
    Weather risk amplifies HIGH-priority kit gaps.
    Most actionable alert type — tells user why they need to act now.
    """
    if risk.risk_level == "LOW":
        return []

    amplifier = {"ELEVATED": 1.2, "HIGH": 1.5, "CRITICAL": 1.8}.get(risk.risk_level, 1.0)
    high_gaps  = [g for g in report.gaps if g.priority == "HIGH"]

    alerts = []
    for gap in high_gaps:
        score = min(int(((risk.risk_score + gap.gap_pct) / 2) * amplifier), 100)
        alerts.append(PrioritizedAlert(
            priority_score=score,
            urgency=_urgency(score),
            category="COMBINED",
            message=(
                f"Weather {risk.risk_level} + low {gap.name} stock "
                f"({gap.current:.1f}/{gap.recommended:.1f} {gap.unit}): "
                f"restock before conditions worsen."
            ),
            detail=(
                f"Risk score: {risk.risk_score} | "
                f"Gap: {gap.gap_pct:.0f}% | "
                f"Amplifier: {amplifier:.1f}x"
            ),
        ))
    return alerts


# ---------------------------------------------------------------------------
# Expiry alerts
# ---------------------------------------------------------------------------

def _expiry_alerts(report: InventoryReport) -> List[PrioritizedAlert]:
    """Alert for items expiring within warning thresholds."""
    alerts = []
    for item in report.expiring:
        score = 80 if item.urgency == "CRITICAL" else 45
        alerts.append(PrioritizedAlert(
            priority_score=score,
            urgency=_urgency(score),
            category="EXPIRY",
            message=(
                f"Replace {item.name}: expires in "
                f"{item.days_remaining} day(s) ({item.expiry_date})."
            ),
            detail=f"Urgency: {item.urgency} | Category: {item.category}",
        ))
    return alerts


# ---------------------------------------------------------------------------
# Kit gap alerts
# ---------------------------------------------------------------------------

def _gap_alerts(report: InventoryReport) -> List[PrioritizedAlert]:
    """Alert for items below EU recommended quantity."""
    weight_map = {"HIGH": 1.0, "MEDIUM": 0.65, "LOW": 0.35}
    alerts = []
    for gap in report.gaps:
        score = min(int(gap.gap_pct * weight_map.get(gap.priority, 0.5)), 100)
        alerts.append(PrioritizedAlert(
            priority_score=score,
            urgency=_urgency(score),
            category="KIT_GAP",
            message=(
                f"Replenish {gap.name}: "
                f"{gap.current:.1f}/{gap.recommended:.1f} {gap.unit} "
                f"({gap.gap_pct:.0f}% missing)."
            ),
            detail=f"Category: {gap.category} | Priority: {gap.priority}",
        ))
    return alerts


# ---------------------------------------------------------------------------
# Geopolitical alerts
# ---------------------------------------------------------------------------

def _geo_alerts(
    geo_score: int,
    geo_trend: str,
    geo_country: str,
) -> List[PrioritizedAlert]:
    """
    Alert when geopolitical risk is noteworthy (score >= 4).
    Normalises 0-30 → 0-100 for consistent urgency mapping.
    Trend INCREASING adds a 10-point boost.
    """
    if geo_score < 4:
        return []

    normalised = min(int((geo_score / 30) * 100), 100)
    trend_note = ""
    if geo_trend == "INCREASING":
        normalised = min(normalised + 10, 100)
        trend_note = " Trend is worsening."
    elif geo_trend == "DECREASING":
        trend_note = " Trend is improving."

    country_str = f" in {geo_country}" if geo_country else ""
    return [PrioritizedAlert(
        priority_score=normalised,
        urgency=_urgency(normalised),
        category="GEO",
        message=(
            f"Geopolitical risk{country_str}: {geo_score}/30.{trend_note} "
            f"Ensure documents, cash, and spare keys are accessible."
        ),
        detail=f"Geo score: {geo_score}/30 | Trend: {geo_trend}",
    )]


# ---------------------------------------------------------------------------
# Health alerts
# ---------------------------------------------------------------------------

def _health_alerts(
    health_score: int,
    health_level: str,
    top_threats: List[str],
) -> List[PrioritizedAlert]:
    """
    Alert when a health threat is ELEVATED or above.
    Normalises 0-50 → 0-100 for consistent urgency mapping.
    """
    if health_level == "ROUTINE":
        return []

    normalised  = min(int((health_score / 50) * 100), 100)
    threat_str  = ", ".join(top_threats[:3]) if top_threats else "unknown"

    return [PrioritizedAlert(
        priority_score=normalised,
        urgency=_urgency(normalised),
        category="HEALTH",
        message=(
            f"Health alert: {health_level} ({health_score}/50). "
            f"Active threats: {threat_str}. "
            f"Check medication and hygiene supplies."
        ),
        detail=f"Source: ECDC CDTR | Threats: {threat_str}",
    )]


def _health_kit_alerts(
    health_score: int,
    health_level: str,
    top_threats: List[str],
    report: InventoryReport,
) -> List[PrioritizedAlert]:
    """
    Health threat amplifies gaps in health-relevant kit categories (meds, hygiene).
    Only fires at ELEVATED or above and when score >= 10.
    """
    if health_level == "ROUTINE" or health_score < 10:
        return []

    amplifier  = {"ELEVATED": 1.1, "HIGH": 1.3, "CRITICAL": 1.6}.get(health_level, 1.0)
    threat_str = ", ".join(top_threats[:2]) if top_threats else "active health threat"
    health_gaps = [g for g in report.gaps if g.category in _HEALTH_RELEVANT_CATEGORIES]

    alerts = []
    for gap in health_gaps:
        # health_score * 2 normalises 0-50 → 0-100 before averaging with gap_pct
        base  = (health_score * 2 + gap.gap_pct) / 2
        score = min(int(base * amplifier), 100)
        alerts.append(PrioritizedAlert(
            priority_score=score,
            urgency=_urgency(score),
            category="HEALTH_KIT",
            message=(
                f"{threat_str} active + low {gap.name} stock "
                f"({gap.current:.1f}/{gap.recommended:.1f} {gap.unit}): "
                f"restock {gap.name} while supplies are available."
            ),
            detail=(
                f"Health level: {health_level} | "
                f"Score: {health_score}/50 | "
                f"Kit gap: {gap.gap_pct:.0f}%"
            ),
        ))
    return alerts


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def prioritize(
    risk: RiskResult,
    inventory_report: InventoryReport,
    geo_score: int = 0,
    geo_trend: str = "STABLE",
    geo_country: str = "",
    health_score: int = 0,
    health_level: str = "ROUTINE",
    top_health_threats: Optional[List[str]] = None,
) -> List[PrioritizedAlert]:
    """
    Cross all three risk signals × inventory state → ranked action list.

    Args:
        risk:               Weather RiskResult from compute_risk_score().
        inventory_report:   InventoryReport from analyze_inventory().
        geo_score:          0-30 from GeopoliticalSnapshot.geo_score.
        geo_trend:          STABLE / INCREASING / DECREASING.
        geo_country:        Country name for message context.
        health_score:       0-50 from HealthSnapshot.health_score.
        health_level:       ROUTINE / ELEVATED / HIGH / CRITICAL.
        top_health_threats: Active threat names from HealthSnapshot.top_threats.

    Returns:
        List of PrioritizedAlert sorted by priority_score descending.
        On ties: COMBINED > HEALTH_KIT > EXPIRY > WEATHER > HEALTH > GEO > KIT_GAP.
    """
    if top_health_threats is None:
        top_health_threats = []

    all_alerts: List[PrioritizedAlert] = []

    # Weather dimension
    all_alerts += _combined_alerts(risk, inventory_report)
    all_alerts += _weather_alerts(risk)

    # Expiry and gaps
    all_alerts += _expiry_alerts(inventory_report)
    all_alerts += _gap_alerts(inventory_report)

    # Health dimension
    all_alerts += _health_kit_alerts(health_score, health_level, top_health_threats, inventory_report)
    all_alerts += _health_alerts(health_score, health_level, top_health_threats)

    # Geopolitical dimension
    all_alerts += _geo_alerts(geo_score, geo_trend, geo_country)

    all_alerts.sort(
        key=lambda a: (-a.priority_score, _CATEGORY_ORDER.get(a.category, 9))
    )

    return all_alerts
