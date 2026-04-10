"""
tests/test_geopolitical_fetcher.py
------------------------------------
Unit tests for data/fetchers/geopolitical_fetcher.py

Tests cover:
    - Score computation logic (no API calls needed)
    - Trend detection
    - Snapshot building
    - Edge cases: empty data, zero fatalities, neighbour spill-over

Run with:  pytest tests/test_geopolitical_fetcher.py -v
"""

import pytest
from datetime import date, timedelta

from core.geopolitical_fetcher import (
    ConflictEvent,
    GeopoliticalSnapshot,
    compute_geo_score,
    build_snapshot,
    _compute_trend,
    MAX_GEO_SCORE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_event(
    event_type="Battles",
    fatalities=0,
    days_ago=10,
    country="Spain",
) -> ConflictEvent:
    event_date = str(date.today() - timedelta(days=days_ago))
    return ConflictEvent(
        event_date=event_date,
        event_type=event_type,
        sub_event_type="Armed clash",
        country=country,
        admin1="Cataluña",
        location="Barcelona",
        fatalities=fatalities,
        source="Test source",
        notes="",
    )


def make_events(n: int, event_type="Battles", fatalities=0) -> list:
    return [make_event(event_type=event_type, fatalities=fatalities, days_ago=i+1)
            for i in range(n)]


# ---------------------------------------------------------------------------
# Tests: compute_geo_score
# ---------------------------------------------------------------------------

class TestComputeGeoScore:

    def test_no_events_returns_zero(self):
        assert compute_geo_score([]) == 0

    def test_score_capped_at_max(self):
        events = make_events(100, fatalities=50)
        assert compute_geo_score(events) <= MAX_GEO_SCORE

    def test_score_increases_with_event_count(self):
        s1 = compute_geo_score(make_events(2))
        s2 = compute_geo_score(make_events(20))
        assert s2 > s1

    def test_fatalities_increase_score(self):
        s_no_fatalities   = compute_geo_score(make_events(5, fatalities=0))
        s_with_fatalities = compute_geo_score(make_events(5, fatalities=100))
        assert s_with_fatalities > s_no_fatalities

    def test_neighbour_events_increase_score(self):
        events    = make_events(5)
        s_alone   = compute_geo_score(events, neighbour_events=[])
        s_with_nb = compute_geo_score(events, neighbour_events=make_events(20))
        assert s_with_nb > s_alone

    def test_no_neighbours_no_neighbour_score(self):
        events = make_events(3, fatalities=0)
        score  = compute_geo_score(events, neighbour_events=None)
        assert score >= 0

    def test_single_event_no_fatalities(self):
        score = compute_geo_score([make_event(fatalities=0)])
        assert 0 < score <= MAX_GEO_SCORE

    def test_score_is_integer(self):
        score = compute_geo_score(make_events(10, fatalities=5))
        assert isinstance(score, int)

    def test_high_fatality_event_scores_higher(self):
        low  = compute_geo_score([make_event(fatalities=1)])
        high = compute_geo_score([make_event(fatalities=1000)])
        assert high > low


# ---------------------------------------------------------------------------
# Tests: _compute_trend
# ---------------------------------------------------------------------------

class TestComputeTrend:

    def test_empty_events_is_stable(self):
        assert _compute_trend([]) == "STABLE"

    def test_all_recent_events_is_increasing(self):
        # All events in the second half of the window (days_ago < 45 for 90d window)
        events = [make_event(days_ago=5) for _ in range(20)]
        trend = _compute_trend(events, lookback_days=90)
        assert trend == "INCREASING"

    def test_all_old_events_is_decreasing(self):
        # All events in the first half of the window
        events = [make_event(days_ago=80) for _ in range(20)]
        trend = _compute_trend(events, lookback_days=90)
        assert trend == "DECREASING"

    def test_evenly_distributed_is_stable(self):
        # Mix of old and recent events in similar rates
        events = (
            [make_event(days_ago=80) for _ in range(5)] +
            [make_event(days_ago=10) for _ in range(5)]
        )
        trend = _compute_trend(events, lookback_days=90)
        assert trend == "STABLE"

    def test_returns_valid_string(self):
        events = make_events(5)
        result = _compute_trend(events)
        assert result in ("STABLE", "INCREASING", "DECREASING")


# ---------------------------------------------------------------------------
# Tests: build_snapshot
# ---------------------------------------------------------------------------

class TestBuildSnapshot:

    def test_returns_snapshot(self):
        events   = make_events(5, fatalities=2)
        snapshot = build_snapshot("Spain", events)
        assert isinstance(snapshot, GeopoliticalSnapshot)

    def test_snapshot_country(self):
        snapshot = build_snapshot("Spain", make_events(3))
        assert snapshot.country == "Spain"

    def test_total_events_correct(self):
        events   = make_events(7)
        snapshot = build_snapshot("Spain", events)
        assert snapshot.total_events == 7

    def test_total_fatalities_correct(self):
        events   = [make_event(fatalities=5) for _ in range(4)]
        snapshot = build_snapshot("Spain", events)
        assert snapshot.total_fatalities == 20

    def test_empty_events_snapshot(self):
        snapshot = build_snapshot("Spain", [])
        assert snapshot.total_events == 0
        assert snapshot.geo_score == 0
        assert snapshot.trend == "STABLE"

    def test_event_breakdown_populated(self):
        events = (
            make_events(3, event_type="Battles") +
            make_events(2, event_type="Explosions/Remote violence")
        )
        snapshot = build_snapshot("Spain", events)
        assert snapshot.event_breakdown.get("Battles") == 3
        assert snapshot.event_breakdown.get("Explosions/Remote violence") == 2

    def test_geo_score_within_bounds(self):
        events   = make_events(50, fatalities=10)
        snapshot = build_snapshot("Spain", events)
        assert 0 <= snapshot.geo_score <= MAX_GEO_SCORE

    def test_period_dates_set(self):
        snapshot = build_snapshot("Spain", make_events(2), lookback_days=90)
        assert snapshot.period_start != ""
        assert snapshot.period_end   != ""

    def test_fetched_at_set(self):
        snapshot = build_snapshot("Spain", make_events(2))
        assert snapshot.fetched_at != ""

    def test_neighbour_events_influence_score(self):
        events    = make_events(2)
        s_without = build_snapshot("Spain", events)
        s_with    = build_snapshot("Spain", events, neighbour_events=make_events(30))
        assert s_with.geo_score >= s_without.geo_score

    def test_trend_field_valid(self):
        snapshot = build_snapshot("Spain", make_events(5))
        assert snapshot.trend in ("STABLE", "INCREASING", "DECREASING")
