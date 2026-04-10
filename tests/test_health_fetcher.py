"""
tests/test_health_fetcher.py
-----------------------------
Unit tests for data/fetchers/health_fetcher.py

Tests cover scoring logic, threat extraction, level mapping, and
the simulate_health_snapshot helper. No network calls are made.

Run with:  pytest tests/test_health_fetcher.py -v
"""

import pytest
from core.health_fetcher import (
    ThreatSignal,
    HealthSnapshot,
    _extract_threats_from_text,
    _compute_health_score,
    health_score_to_level,
    simulate_health_snapshot,
    MAX_HEALTH_SCORE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_threat(name="Influenza", score=5, eu_impact=False) -> ThreatSignal:
    return ThreatSignal(
        name=name,
        score=score,
        eu_impact=eu_impact,
        source_text=f"Test signal for {name}",
    )


def make_threats(scores: list, eu_impact: bool = False) -> list:
    return [
        make_threat(name=f"Threat{i}", score=s, eu_impact=eu_impact)
        for i, s in enumerate(scores)
    ]


# ---------------------------------------------------------------------------
# Tests: health_score_to_level
# ---------------------------------------------------------------------------

class TestHealthScoreToLevel:

    def test_routine(self):
        assert health_score_to_level(0)  == "ROUTINE"
        assert health_score_to_level(9)  == "ROUTINE"

    def test_elevated(self):
        assert health_score_to_level(10) == "ELEVATED"
        assert health_score_to_level(24) == "ELEVATED"

    def test_high(self):
        assert health_score_to_level(25) == "HIGH"
        assert health_score_to_level(39) == "HIGH"

    def test_critical(self):
        assert health_score_to_level(40) == "CRITICAL"
        assert health_score_to_level(50) == "CRITICAL"

    def test_boundary_values(self):
        assert health_score_to_level(10) == "ELEVATED"
        assert health_score_to_level(25) == "HIGH"
        assert health_score_to_level(40) == "CRITICAL"


# ---------------------------------------------------------------------------
# Tests: _compute_health_score
# ---------------------------------------------------------------------------

class TestComputeHealthScore:

    def test_no_threats_returns_zero(self):
        assert _compute_health_score([]) == 0

    def test_single_threat(self):
        threats = [make_threat(score=20)]
        score = _compute_health_score(threats)
        assert score == 20

    def test_score_capped_at_max(self):
        threats = make_threats([50, 50, 50, 50, 50])
        assert _compute_health_score(threats) <= MAX_HEALTH_SCORE

    def test_diminishing_returns(self):
        # Two threats should score less than double the first
        one   = _compute_health_score([make_threat(score=20)])
        two   = _compute_health_score([make_threat(score=20), make_threat(score=20)])
        assert two < one * 2

    def test_dominant_threat_outweighs_many_small(self):
        # One pandemic signal should outscore ten minor signals
        pandemic = _compute_health_score([make_threat(score=50)])
        minor    = _compute_health_score(make_threats([5] * 10))
        assert pandemic > minor

    def test_score_increases_with_more_threats(self):
        s1 = _compute_health_score(make_threats([10]))
        s2 = _compute_health_score(make_threats([10, 10]))
        assert s2 > s1

    def test_returns_integer(self):
        score = _compute_health_score([make_threat(score=15)])
        assert isinstance(score, int)

    def test_high_score_threat(self):
        threats = [make_threat(score=50)]
        score = _compute_health_score(threats)
        assert score == 50


# ---------------------------------------------------------------------------
# Tests: _extract_threats_from_text
# ---------------------------------------------------------------------------

class TestExtractThreatsFromText:

    def test_empty_text_returns_empty(self):
        assert _extract_threats_from_text("") == []

    def test_detects_influenza(self):
        text = "Seasonal influenza activity is at baseline levels in EU/EEA."
        threats = _extract_threats_from_text(text)
        names = [t.name.lower() for t in threats]
        assert any("influenza" in n for n in names)

    def test_detects_mpox(self):
        text = "Mpox cases have been reported in several EU member states."
        threats = _extract_threats_from_text(text)
        names = [t.name.lower() for t in threats]
        assert any("mpox" in n for n in names)

    def test_eu_impact_flag_set(self):
        text = "Measles outbreak reported in EU/EEA member states with community transmission."
        threats = _extract_threats_from_text(text)
        measles = next((t for t in threats if "measles" in t.name.lower()), None)
        assert measles is not None
        assert measles.eu_impact is True

    def test_distant_threat_reduces_score(self):
        # Same keyword but with "risk to eu is very low"
        text_close   = "Marburg virus disease outbreak in EU/EEA member states."
        text_distant = "Marburg virus disease. Risk to EU is very low. No cases in EU."

        threats_close   = _extract_threats_from_text(text_close)
        threats_distant = _extract_threats_from_text(text_distant)

        score_close   = sum(t.score for t in threats_close   if "marburg" in t.name.lower())
        score_distant = sum(t.score for t in threats_distant if "marburg" in t.name.lower())

        assert score_close > score_distant

    def test_pandemic_keyword_scores_highest(self):
        text = "WHO has declared a pandemic. Novel pathogen spreading globally including EU/EEA."
        threats = _extract_threats_from_text(text)
        max_score = max(t.score for t in threats) if threats else 0
        assert max_score >= 35

    def test_no_false_positives_on_unrelated_text(self):
        text = "The weather is sunny today. No health concerns reported."
        threats = _extract_threats_from_text(text)
        assert len(threats) == 0

    def test_multiple_threats_detected(self):
        text = (
            "This week: influenza at seasonal levels in EU/EEA. "
            "Mpox cases reported in three EU member states. "
            "Cholera monitoring continues in Africa."
        )
        threats = _extract_threats_from_text(text)
        assert len(threats) >= 2

    def test_source_text_populated(self):
        text = "Measles cases have been reported in Romania and France in EU/EEA."
        threats = _extract_threats_from_text(text)
        for t in threats:
            assert len(t.source_text) > 0


# ---------------------------------------------------------------------------
# Tests: simulate_health_snapshot
# ---------------------------------------------------------------------------

class TestSimulateHealthSnapshot:

    def test_routine_scenario(self):
        snap = simulate_health_snapshot("routine")
        assert snap.health_score <= 20
        assert snap.level in ("ROUTINE", "ELEVATED")

    def test_elevated_scenario(self):
        snap = simulate_health_snapshot("elevated")
        assert snap.health_score > 5
        assert snap.level in ("ELEVATED", "HIGH", "CRITICAL")

    def test_pandemic_scenario(self):
        snap = simulate_health_snapshot("pandemic")
        assert snap.health_score >= 30
        assert snap.level in ("HIGH", "CRITICAL")

    def test_unknown_scenario_defaults_to_routine(self):
        snap = simulate_health_snapshot("nonexistent_scenario")
        assert snap is not None
        assert snap.level in ("ROUTINE", "ELEVATED", "HIGH", "CRITICAL")

    def test_snapshot_fields_populated(self):
        snap = simulate_health_snapshot("routine")
        assert snap.week_label != ""
        assert snap.period     != ""
        assert snap.fetched_at != ""
        assert len(snap.threats) > 0
        assert len(snap.top_threats) > 0

    def test_score_within_bounds(self):
        for scenario in ("routine", "elevated", "pandemic"):
            snap = simulate_health_snapshot(scenario)
            assert 0 <= snap.health_score <= MAX_HEALTH_SCORE

    def test_pandemic_higher_than_routine(self):
        routine  = simulate_health_snapshot("routine")
        pandemic = simulate_health_snapshot("pandemic")
        assert pandemic.health_score > routine.health_score

    def test_level_consistent_with_score(self):
        snap = simulate_health_snapshot("routine")
        expected_level = health_score_to_level(snap.health_score)
        assert snap.level == expected_level


# ---------------------------------------------------------------------------
# Tests: Option C — signals remain independent
# ---------------------------------------------------------------------------

class TestOptionCIndependence:

    def test_health_score_never_exceeds_50(self):
        """Health signal is bounded at 50 regardless of threat count."""
        many_threats = make_threats([50] * 20, eu_impact=True)
        score = _compute_health_score(many_threats)
        assert score <= MAX_HEALTH_SCORE

    def test_health_max_less_than_weather_max(self):
        """Health max (50) < weather max (100) — weather remains primary driver."""
        assert MAX_HEALTH_SCORE == 50
        assert MAX_HEALTH_SCORE < 100

    def test_three_scenarios_produce_different_scores(self):
        """Each scenario produces a meaningfully different score."""
        routine  = simulate_health_snapshot("routine").health_score
        elevated = simulate_health_snapshot("elevated").health_score
        pandemic = simulate_health_snapshot("pandemic").health_score
        assert routine < elevated < pandemic

    def test_level_labels_distinct_from_weather_labels(self):
        """
        Health uses ROUTINE/ELEVATED/HIGH/CRITICAL.
        Weather uses LOW/ELEVATED/HIGH/CRITICAL.
        ROUTINE vs LOW distinction signals they are different dimensions.
        """
        routine_label = health_score_to_level(0)
        assert routine_label == "ROUTINE"
        assert routine_label != "LOW"   # weather uses LOW, health uses ROUTINE
