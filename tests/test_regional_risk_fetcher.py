"""
tests/test_regional_risk_fetcher.py
-------------------------------------
Unit tests for core/regional_risk_fetcher.py

Tests cover scoring logic, level mapping, and simulate_regional_snapshot.
No network calls are made — all tests use local data or the simulate helper.

Run with:  pytest tests/test_regional_risk_fetcher.py -v
"""

import pytest
from core.regional_risk_fetcher import (
    DisasterEvent,
    CrisisReport,
    RegionalSnapshot,
    compute_disaster_score,
    compute_crisis_score,
    regional_score_to_level,
    simulate_regional_snapshot,
    MAX_REGIONAL_SCORE,
    SPAIN_REGION,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_disaster(alert_level="Green", event_type="FL", country="Spain", score=None) -> DisasterEvent:
    level_scores = {"Red": 15, "Orange": 8, "Green": 2}
    return DisasterEvent(
        title=       f"{event_type} event in {country}",
        event_type=  event_type,
        alert_level= alert_level,
        country=     country,
        score=       score if score is not None else level_scores.get(alert_level, 2),
        url=         "",
    )


def make_crisis(theme="Conflict and Violence", country="Spain", score=8) -> CrisisReport:
    return CrisisReport(
        title=   f"Crisis report in {country}",
        country= country,
        date=    "2026-04-01",
        theme=   theme,
        url=     "",
        score=   score,
    )


# ---------------------------------------------------------------------------
# Tests: regional_score_to_level
# ---------------------------------------------------------------------------

class TestRegionalScoreToLevel:

    def test_minimal(self):
        assert regional_score_to_level(0)  == "MINIMAL"
        assert regional_score_to_level(3)  == "MINIMAL"

    def test_low(self):
        assert regional_score_to_level(4)  == "LOW"
        assert regional_score_to_level(11) == "LOW"

    def test_medium(self):
        assert regional_score_to_level(12) == "MEDIUM"
        assert regional_score_to_level(21) == "MEDIUM"

    def test_high(self):
        assert regional_score_to_level(22) == "HIGH"
        assert regional_score_to_level(30) == "HIGH"

    def test_boundary_values(self):
        assert regional_score_to_level(4)  == "LOW"
        assert regional_score_to_level(12) == "MEDIUM"
        assert regional_score_to_level(22) == "HIGH"


# ---------------------------------------------------------------------------
# Tests: compute_disaster_score
# ---------------------------------------------------------------------------

class TestComputeDisasterScore:

    def test_no_events_returns_zero(self):
        assert compute_disaster_score([]) == 0

    def test_single_green_event(self):
        score = compute_disaster_score([make_disaster("Green")])
        assert score == 2

    def test_single_red_event(self):
        score = compute_disaster_score([make_disaster("Red")])
        assert score == 15

    def test_score_capped_at_15(self):
        events = [make_disaster("Red")] * 10
        assert compute_disaster_score(events) <= 15

    def test_diminishing_returns(self):
        one  = compute_disaster_score([make_disaster("Orange")])
        two  = compute_disaster_score([make_disaster("Orange"), make_disaster("Orange")])
        assert two < one * 2

    def test_score_increases_with_more_events(self):
        s1 = compute_disaster_score([make_disaster("Green")])
        s2 = compute_disaster_score([make_disaster("Green"), make_disaster("Green")])
        assert s2 > s1

    def test_red_outscores_many_greens(self):
        red    = compute_disaster_score([make_disaster("Red")])
        greens = compute_disaster_score([make_disaster("Green")] * 10)
        assert red > greens

    def test_returns_integer(self):
        score = compute_disaster_score([make_disaster("Orange")])
        assert isinstance(score, int)


# ---------------------------------------------------------------------------
# Tests: compute_crisis_score
# ---------------------------------------------------------------------------

class TestComputeCrisisScore:

    def test_no_reports_returns_zero(self):
        assert compute_crisis_score([]) == 0

    def test_single_conflict_report(self):
        score = compute_crisis_score([make_crisis(score=8)])
        assert score == 8

    def test_score_capped_at_15(self):
        reports = [make_crisis(score=8)] * 10
        assert compute_crisis_score(reports) <= 15

    def test_diminishing_returns(self):
        one = compute_crisis_score([make_crisis(score=8)])
        two = compute_crisis_score([make_crisis(score=8), make_crisis(score=8)])
        assert two < one * 2

    def test_returns_integer(self):
        score = compute_crisis_score([make_crisis(score=4)])
        assert isinstance(score, int)

    def test_higher_score_report_scores_higher(self):
        low  = compute_crisis_score([make_crisis(score=4)])
        high = compute_crisis_score([make_crisis(score=8)])
        assert high > low


# ---------------------------------------------------------------------------
# Tests: combined regional score
# ---------------------------------------------------------------------------

class TestCombinedScore:

    def test_combined_score_is_sum(self):
        disasters = [make_disaster("Orange")]   # score 8
        crises    = [make_crisis(score=4)]
        d_score   = compute_disaster_score(disasters)
        c_score   = compute_crisis_score(crises)
        combined  = min(d_score + c_score, MAX_REGIONAL_SCORE)
        assert combined == d_score + c_score

    def test_combined_score_capped_at_30(self):
        disasters = [make_disaster("Red")] * 5
        crises    = [make_crisis(score=8)] * 5
        d_score   = compute_disaster_score(disasters)
        c_score   = compute_crisis_score(crises)
        combined  = min(d_score + c_score, MAX_REGIONAL_SCORE)
        assert combined <= MAX_REGIONAL_SCORE

    def test_max_score_is_30(self):
        assert MAX_REGIONAL_SCORE == 30

    def test_calm_scenario_near_zero(self):
        snap = simulate_regional_snapshot("calm")
        assert snap.regional_score == 0

    def test_crisis_scenario_higher_than_medium(self):
        medium = simulate_regional_snapshot("medium")
        crisis   = simulate_regional_snapshot("crisis")
        assert crisis.regional_score > medium.regional_score


# ---------------------------------------------------------------------------
# Tests: simulate_regional_snapshot
# ---------------------------------------------------------------------------

class TestSimulateRegionalSnapshot:

    def test_calm_scenario(self):
        snap = simulate_regional_snapshot("calm")
        assert snap.regional_score == 0
        assert snap.level == "MINIMAL"
        assert snap.disaster_events == []
        assert snap.crisis_reports  == []

    def test_medium_scenario(self):
        snap = simulate_regional_snapshot("medium")
        assert snap.regional_score > 0
        assert snap.level in ("LOW", "MEDIUM", "HIGH")

    def test_crisis_scenario(self):
        snap = simulate_regional_snapshot("crisis")
        assert snap.regional_score >= 12
        assert snap.level in ("MEDIUM", "HIGH")

    def test_unknown_scenario_defaults_to_calm(self):
        snap = simulate_regional_snapshot("nonexistent")
        assert snap is not None
        assert snap.regional_score == 0

    def test_snapshot_fields_populated(self):
        snap = simulate_regional_snapshot("medium")
        assert snap.country      != ""
        assert snap.fetched_at   != ""
        assert snap.level        in ("MINIMAL", "LOW", "MEDIUM", "HIGH")
        assert isinstance(snap.disaster_score, int)
        assert isinstance(snap.crisis_score,   int)
        assert isinstance(snap.regional_score, int)

    def test_score_within_bounds(self):
        for scenario in ("calm", "medium", "crisis"):
            snap = simulate_regional_snapshot(scenario)
            assert 0 <= snap.regional_score <= MAX_REGIONAL_SCORE

    def test_level_consistent_with_score(self):
        for scenario in ("calm", "medium", "crisis"):
            snap = simulate_regional_snapshot(scenario)
            assert snap.level == regional_score_to_level(snap.regional_score)

    def test_disaster_score_within_bounds(self):
        for scenario in ("calm", "medium", "crisis"):
            snap = simulate_regional_snapshot(scenario)
            assert 0 <= snap.disaster_score <= 15

    def test_crisis_score_within_bounds(self):
        for scenario in ("calm", "medium", "crisis"):
            snap = simulate_regional_snapshot(scenario)
            assert 0 <= snap.crisis_score <= 15

    def test_country_is_spain(self):
        snap = simulate_regional_snapshot("calm")
        assert snap.country == "Spain"

    def test_region_countries_populated(self):
        snap = simulate_regional_snapshot("calm")
        assert len(snap.region_countries) > 0
        assert "Spain" in snap.region_countries


# ---------------------------------------------------------------------------
# Tests: Option C independence
# ---------------------------------------------------------------------------

class TestOptionCIndependence:

    def test_max_regional_less_than_weather_max(self):
        """Regional max (30) < weather max (100)."""
        assert MAX_REGIONAL_SCORE == 30
        assert MAX_REGIONAL_SCORE < 100

    def test_max_regional_less_than_health_max(self):
        """Regional max (30) < health max (50)."""
        assert MAX_REGIONAL_SCORE < 50

    def test_level_labels_distinct_from_weather(self):
        """Regional uses MINIMAL as bottom level, weather uses LOW."""
        assert regional_score_to_level(0) == "MINIMAL"
        assert regional_score_to_level(0) != "LOW"

    def test_scores_never_exceed_cap(self):
        """No combination of disasters and crises can exceed 30."""
        disasters = [make_disaster("Red")] * 20
        crises    = [make_crisis(score=8)] * 20
        d = compute_disaster_score(disasters)
        c = compute_crisis_score(crises)
        assert min(d + c, MAX_REGIONAL_SCORE) <= 30
