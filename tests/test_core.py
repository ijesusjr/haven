"""
tests/test_core.py
------------------
Unit tests for:
    - core.risk_engine
    - core.inventory_analyzer
    - core.alert_prioritizer

Run with:  pytest tests/test_core.py -v
"""

import pytest
from datetime import date, timedelta

from core.risk_engine import (
    WeatherSnapshot,
    Alert,
    RiskResult,
    weather_id_to_severity,
    alert_severity_to_score,
    score_to_level,
    compute_risk_score,
)
from core.inventory_analyzer import (
    KitItem,
    GapItem,
    ExpiryItem,
    InventoryReport,
    analyze_gaps,
    analyze_expiry,
    analyze_inventory,
    EXPIRY_CRITICAL_DAYS,
    EXPIRY_WARNING_DAYS,
)
from core.alert_prioritizer import (
    PrioritizedAlert,
    prioritize,
)


# ===========================================================================
# Helpers
# ===========================================================================

def make_weather(weather_id=800, wind=0.0, rain=0.0) -> WeatherSnapshot:
    return WeatherSnapshot(weather_id=weather_id, wind_speed_ms=wind, rain_1h_mm=rain)


def make_alert(severity="Moderate", event="Test alert") -> Alert:
    return Alert(event=event, severity=severity)


def make_item(
    name="Drinking water",
    category="water",
    quantity=4.0,
    eu_recommended=9.0,
    unit="liters",
    expiry_date=None,
) -> KitItem:
    return KitItem(
        name=name,
        category=category,
        quantity=quantity,
        eu_recommended=eu_recommended,
        unit=unit,
        expiry_date=expiry_date,
    )


def make_risk(score=50, level="HIGH") -> RiskResult:
    return RiskResult(
        risk_score=score,
        risk_level=level,
        weather_severity=20,
        alert_severity=25,
        wind_bonus=5,
        rain_bonus=0,
    )


def make_report(gaps=None, expiring=None) -> InventoryReport:
    return InventoryReport(gaps=gaps or [], expiring=expiring or [])


# ===========================================================================
# Tests: risk_engine
# ===========================================================================

class TestWeatherIdToSeverity:

    def test_clear_sky_returns_zero(self):
        assert weather_id_to_severity(800) == 0

    def test_cloudy_returns_zero(self):
        assert weather_id_to_severity(803) == 0

    def test_light_rain(self):
        assert weather_id_to_severity(500) == 15

    def test_moderate_rain(self):
        assert weather_id_to_severity(501) == 20

    def test_heavy_rain(self):
        assert weather_id_to_severity(502) == 30

    def test_extreme_rain_is_max(self):
        assert weather_id_to_severity(504) == 40

    def test_unknown_rain_id_fallback(self):
        assert weather_id_to_severity(520) == 20  # fallback default

    def test_thunderstorm(self):
        assert weather_id_to_severity(200) == 35

    def test_tornado_is_maximum(self):
        assert weather_id_to_severity(781) == 40

    def test_snow(self):
        assert weather_id_to_severity(601) == 25

    def test_fog(self):
        assert weather_id_to_severity(741) == 15

    def test_drizzle(self):
        assert weather_id_to_severity(300) == 10

    def test_unknown_id_returns_zero(self):
        assert weather_id_to_severity(999) == 0


class TestAlertSeverityToScore:

    def test_minor(self):
        assert alert_severity_to_score("Minor") == 10

    def test_moderate(self):
        assert alert_severity_to_score("Moderate") == 25

    def test_severe(self):
        assert alert_severity_to_score("Severe") == 45

    def test_extreme(self):
        assert alert_severity_to_score("Extreme") == 60

    def test_unknown(self):
        assert alert_severity_to_score("Unknown") == 0

    def test_unrecognised_label_returns_zero(self):
        assert alert_severity_to_score("Apocalyptic") == 0


class TestScoreToLevel:

    def test_low(self):
        assert score_to_level(0)  == "LOW"
        assert score_to_level(19) == "LOW"

    def test_medium(self):
        assert score_to_level(20) == "MEDIUM"
        assert score_to_level(44) == "MEDIUM"

    def test_high(self):
        assert score_to_level(45) == "HIGH"
        assert score_to_level(69) == "HIGH"

    def test_critical(self):
        assert score_to_level(70)  == "CRITICAL"
        assert score_to_level(100) == "CRITICAL"


class TestComputeRiskScore:

    def test_calm_conditions_no_alerts(self):
        result = compute_risk_score(make_weather(800, wind=2, rain=0), [])
        assert result.risk_score == 0
        assert result.risk_level == "LOW"

    def test_heavy_rain_with_moderate_alert(self):
        w = make_weather(502, wind=6, rain=0)
        a = [make_alert("Moderate")]
        result = compute_risk_score(w, a)
        # weather=30, alert=25, wind=0, rain=0 → 55
        assert result.risk_score == 55
        assert result.risk_level == "HIGH"

    def test_wind_bonus_applied(self):
        result = compute_risk_score(make_weather(800, wind=20, rain=0), [])
        assert result.wind_bonus == 15

    def test_rain_bonus_applied(self):
        result = compute_risk_score(make_weather(800, wind=0, rain=20), [])
        assert result.rain_bonus == 10

    def test_score_capped_at_100(self):
        w = make_weather(781, wind=35, rain=40)
        a = [make_alert("Extreme")]
        result = compute_risk_score(w, a)
        assert result.risk_score == 100

    def test_multiple_alerts_uses_highest(self):
        alerts = [make_alert("Minor"), make_alert("Extreme"), make_alert("Moderate")]
        result = compute_risk_score(make_weather(800), alerts)
        assert result.alert_severity == 60  # Extreme

    def test_empty_alerts_list(self):
        result = compute_risk_score(make_weather(501), [])
        assert result.alert_severity == 0

    def test_breakdown_sums_to_score(self):
        w = make_weather(502, wind=10, rain=5)
        a = [make_alert("Minor")]
        result = compute_risk_score(w, a)
        total = (
            result.weather_severity
            + result.alert_severity
            + result.wind_bonus
            + result.rain_bonus
        )
        assert result.risk_score == min(total, 100)

    def test_result_has_breakdown(self):
        result = compute_risk_score(make_weather(501), [make_alert("Minor")])
        bd = result.breakdown()
        assert "weather_severity" in bd
        assert "alert_severity"   in bd
        assert "wind_bonus"       in bd
        assert "rain_bonus"       in bd


# ===========================================================================
# Tests: inventory_analyzer
# ===========================================================================

class TestAnalyzeGaps:

    def test_full_kit_no_gaps(self):
        item = make_item(quantity=9.0, eu_recommended=9.0)
        assert analyze_gaps([item]) == []

    def test_above_recommended_no_gap(self):
        item = make_item(quantity=12.0, eu_recommended=9.0)
        assert analyze_gaps([item]) == []

    def test_gap_detected(self):
        item = make_item(quantity=4.0, eu_recommended=9.0)
        gaps = analyze_gaps([item])
        assert len(gaps) == 1
        assert gaps[0].gap == pytest.approx(5.0)
        assert gaps[0].gap_pct == pytest.approx(55.6, abs=0.1)

    def test_high_priority_category(self):
        item = make_item(category="water", quantity=0, eu_recommended=9.0)
        gaps = analyze_gaps([item])
        assert gaps[0].priority == "HIGH"

    def test_low_priority_category(self):
        item = make_item(category="hygiene", quantity=0, eu_recommended=1.0)
        gaps = analyze_gaps([item])
        assert gaps[0].priority == "LOW"

    def test_sorted_high_priority_first(self):
        water = make_item(name="Water", category="water",  quantity=0, eu_recommended=9.0)
        soap  = make_item(name="Soap",  category="hygiene", quantity=0, eu_recommended=1.0)
        gaps = analyze_gaps([soap, water])
        assert gaps[0].name == "Water"

    def test_zero_recommended_excluded(self):
        item = make_item(quantity=0, eu_recommended=0)
        assert analyze_gaps([item]) == []

    def test_multiple_items(self):
        items = [
            make_item(name="Water", category="water", quantity=4.0, eu_recommended=9.0),
            make_item(name="Food",  category="food",  quantity=3.0, eu_recommended=3.0),
            make_item(name="Meds",  category="meds",  quantity=0.0, eu_recommended=7.0),
        ]
        gaps = analyze_gaps(items)
        names = [g.name for g in gaps]
        assert "Food" not in names
        assert "Water" in names
        assert "Meds" in names


class TestAnalyzeExpiry:

    def _ref(self) -> date:
        return date(2026, 4, 10)

    def test_no_expiry_date_excluded(self):
        item = make_item(expiry_date=None)
        assert analyze_expiry([item], self._ref()) == []

    def test_expires_beyond_threshold_excluded(self):
        item = make_item(expiry_date=self._ref() + timedelta(days=31))
        assert analyze_expiry([item], self._ref()) == []

    def test_warning_within_30_days(self):
        item = make_item(expiry_date=self._ref() + timedelta(days=20))
        result = analyze_expiry([item], self._ref())
        assert len(result) == 1
        assert result[0].urgency == "WARNING"
        assert result[0].days_remaining == 20

    def test_critical_within_7_days(self):
        item = make_item(expiry_date=self._ref() + timedelta(days=5))
        result = analyze_expiry([item], self._ref())
        assert result[0].urgency == "CRITICAL"

    def test_expired_item_is_critical(self):
        item = make_item(expiry_date=self._ref() - timedelta(days=1))
        result = analyze_expiry([item], self._ref())
        assert result[0].urgency == "CRITICAL"
        assert result[0].days_remaining == -1

    def test_sorted_most_urgent_first(self):
        items = [
            make_item(name="A", expiry_date=self._ref() + timedelta(days=25)),
            make_item(name="B", expiry_date=self._ref() + timedelta(days=3)),
        ]
        result = analyze_expiry(items, self._ref())
        assert result[0].name == "B"

    def test_defaults_to_today(self):
        item = make_item(expiry_date=date.today() + timedelta(days=5))
        result = analyze_expiry([item])
        assert len(result) == 1


class TestInventoryReport:

    def test_has_critical_gaps(self):
        gap = GapItem("Water", "water", 0, 9, "liters", 9, 100.0, "HIGH")
        report = make_report(gaps=[gap])
        assert report.has_critical_gaps is True

    def test_no_critical_gaps(self):
        gap = GapItem("Soap", "hygiene", 0, 1, "units", 1, 100.0, "LOW")
        report = make_report(gaps=[gap])
        assert report.has_critical_gaps is False

    def test_total_gap_score_capped_at_100(self):
        gaps = [
            GapItem(f"Item{i}", "water", 0, 9, "liters", 9, 100.0, "HIGH")
            for i in range(20)
        ]
        report = make_report(gaps=gaps)
        assert report.total_gap_score <= 100

    def test_full_kit_gap_score_zero(self):
        report = make_report(gaps=[])
        assert report.total_gap_score == 0


# ===========================================================================
# Tests: alert_prioritizer
# ===========================================================================

class TestPrioritize:

    def _full_kit_report(self) -> InventoryReport:
        return make_report(gaps=[], expiring=[])

    def _water_gap_report(self) -> InventoryReport:
        gap = GapItem("Drinking water", "water", 4.0, 9.0, "liters", 5.0, 55.6, "HIGH")
        return make_report(gaps=[gap])

    def _meds_gap_report(self) -> InventoryReport:
        gap = GapItem("Regular medication", "meds", 2.0, 7.0, "days", 5.0, 71.4, "HIGH")
        return make_report(gaps=[gap])

    def test_no_risk_no_gaps_returns_empty(self):
        risk   = compute_risk_score(make_weather(800), [])
        report = self._full_kit_report()
        result = prioritize(risk, report)
        assert result == []

    def test_weather_alert_generated(self):
        risk   = make_risk(score=60, level="HIGH")
        report = self._full_kit_report()
        result = prioritize(risk, report)
        assert any(a.category == "WEATHER" for a in result)

    def test_kit_gap_alert_generated(self):
        risk   = make_risk(score=0, level="LOW")
        report = self._water_gap_report()
        result = prioritize(risk, report)
        assert any(a.category == "KIT_GAP" for a in result)

    def test_combined_alert_when_risk_medium_and_gap_exists(self):
        risk   = make_risk(score=50, level="HIGH")
        report = self._water_gap_report()
        result = prioritize(risk, report)
        assert any(a.category == "COMBINED" for a in result)

    def test_no_combined_alert_when_risk_low(self):
        risk   = make_risk(score=5, level="LOW")
        report = self._water_gap_report()
        result = prioritize(risk, report)
        assert not any(a.category == "COMBINED" for a in result)

    def test_sorted_by_priority_score_descending(self):
        risk   = make_risk(score=80, level="CRITICAL")
        report = self._water_gap_report()
        result = prioritize(risk, report)
        scores = [a.priority_score for a in result]
        assert scores == sorted(scores, reverse=True)

    def test_combined_before_kit_gap_on_equal_score(self):
        risk   = make_risk(score=50, level="HIGH")
        report = self._water_gap_report()
        result = prioritize(risk, report)
        combined_idx = next(i for i, a in enumerate(result) if a.category == "COMBINED")
        kit_gap_idx  = next((i for i, a in enumerate(result) if a.category == "KIT_GAP"), None)
        if kit_gap_idx is not None:
            assert combined_idx <= kit_gap_idx

    def test_expiry_alert_generated(self):
        exp    = ExpiryItem("Food", "food", date.today() + timedelta(days=3), 3, "CRITICAL")
        report = make_report(expiring=[exp])
        risk   = make_risk(score=0, level="LOW")
        result = prioritize(risk, report)
        assert any(a.category == "EXPIRY" for a in result)

    def test_immediate_urgency_for_critical_score(self):
        risk   = make_risk(score=80, level="CRITICAL")
        report = self._full_kit_report()
        result = prioritize(risk, report)
        assert result[0].urgency == "IMMEDIATE"

    def test_all_alert_messages_are_non_empty(self):
        risk   = make_risk(score=60, level="HIGH")
        report = self._water_gap_report()
        result = prioritize(risk, report)
        for alert in result:
            assert len(alert.message) > 0

    def test_critical_weather_amplifies_water_gap(self):
        risk   = make_risk(score=90, level="CRITICAL")
        gap    = GapItem("Drinking water", "water", 1.0, 9.0, "liters", 8.0, 88.9, "HIGH")
        report = make_report(gaps=[gap])
        result = prioritize(risk, report)
        combined = [a for a in result if a.category == "COMBINED"]
        assert combined[0].priority_score >= 80

    # --- Geopolitical alerts ---

    def test_geo_alert_generated_when_score_above_threshold(self):
        risk   = make_risk(score=0, level="LOW")
        report = self._full_kit_report()
        result = prioritize(risk, report, geo_score=15, geo_trend="STABLE", geo_country="Spain")
        assert any(a.category == "GEO" for a in result)

    def test_geo_alert_not_generated_below_threshold(self):
        risk   = make_risk(score=0, level="LOW")
        report = self._full_kit_report()
        result = prioritize(risk, report, geo_score=2, geo_trend="STABLE")
        assert not any(a.category == "GEO" for a in result)

    def test_geo_increasing_trend_raises_score(self):
        risk   = make_risk(score=0, level="LOW")
        report = self._full_kit_report()
        stable    = prioritize(risk, report, geo_score=15, geo_trend="STABLE")
        increasing = prioritize(risk, report, geo_score=15, geo_trend="INCREASING")
        geo_stable    = next(a for a in stable    if a.category == "GEO")
        geo_increasing = next(a for a in increasing if a.category == "GEO")
        assert geo_increasing.priority_score > geo_stable.priority_score

    def test_geo_message_contains_country(self):
        risk   = make_risk(score=0, level="LOW")
        report = self._full_kit_report()
        result = prioritize(risk, report, geo_score=10, geo_country="Spain")
        geo = next(a for a in result if a.category == "GEO")
        assert "Spain" in geo.message

    # --- Health alerts ---

    def test_health_alert_generated_when_medium(self):
        risk   = make_risk(score=0, level="LOW")
        report = self._full_kit_report()
        result = prioritize(
            risk, report,
            health_score=20, health_level="MEDIUM",
            top_health_threats=["Influenza"]
        )
        assert any(a.category == "HEALTH" for a in result)

    def test_health_alert_not_generated_when_routine(self):
        risk   = make_risk(score=0, level="LOW")
        report = self._full_kit_report()
        result = prioritize(
            risk, report,
            health_score=5, health_level="ROUTINE",
            top_health_threats=["Influenza"]
        )
        assert not any(a.category == "HEALTH" for a in result)

    def test_health_kit_alert_generated_for_meds_gap(self):
        risk   = make_risk(score=0, level="LOW")
        report = self._meds_gap_report()
        result = prioritize(
            risk, report,
            health_score=25, health_level="HIGH",
            top_health_threats=["Avian Influenza"]
        )
        assert any(a.category == "HEALTH_KIT" for a in result)

    def test_health_kit_not_generated_for_water_gap(self):
        """Water is not a health-relevant category."""
        risk   = make_risk(score=0, level="LOW")
        report = self._water_gap_report()
        result = prioritize(
            risk, report,
            health_score=25, health_level="HIGH",
            top_health_threats=["Avian Influenza"]
        )
        assert not any(a.category == "HEALTH_KIT" for a in result)

    def test_health_kit_message_contains_threat_name(self):
        risk   = make_risk(score=0, level="LOW")
        report = self._meds_gap_report()
        result = prioritize(
            risk, report,
            health_score=25, health_level="HIGH",
            top_health_threats=["Mpox"]
        )
        hk = next(a for a in result if a.category == "HEALTH_KIT")
        assert "Mpox" in hk.message

    def test_all_six_categories_can_appear(self):
        """Full scenario: all signals active, kit has gaps and expiring items."""
        risk = make_risk(score=60, level="HIGH")
        gap  = GapItem("Drinking water", "water", 1.0, 9.0, "liters", 8.0, 88.9, "HIGH")
        meds_gap = GapItem("Regular medication", "meds", 1.0, 7.0, "days", 6.0, 85.7, "HIGH")
        exp  = ExpiryItem("Food", "food", date.today() + timedelta(days=3), 3, "CRITICAL")
        report = make_report(gaps=[gap, meds_gap], expiring=[exp])

        result = prioritize(
            risk, report,
            geo_score=15, geo_trend="INCREASING", geo_country="Spain",
            health_score=25, health_level="HIGH",
            top_health_threats=["Avian Influenza"]
        )
        categories = {a.category for a in result}
        assert "COMBINED"   in categories
        assert "WEATHER"    in categories
        assert "EXPIRY"     in categories
        assert "KIT_GAP"    in categories
        assert "GEO"        in categories
        assert "HEALTH"     in categories
        assert "HEALTH_KIT" in categories
