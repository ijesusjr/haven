"""
data/fetchers/geopolitical_fetcher.py
--------------------------------------
Fetches conflict event data from ACLED (Armed Conflict Location & Event Data)
and computes a slow-moving geopolitical risk score (0-30) for a given country.

ACLED registration (free, non-commercial):
    https://developer.acleddata.com/

Authentication: email + access key passed as parameters or via .env:
    ACLED_EMAIL=your@email.com
    ACLED_KEY=your_access_key

API reference:
    https://acleddata.com/api-documentation/acled-endpoint

Design notes:
    - Called weekly (not daily) — conflict trends change slowly.
    - Lookback window: 90 days by default.
    - Filters to political violence only (excludes pure protest/demonstration).
    - Score is additive to the weather risk score in the risk engine.
    - Max contribution is capped at 30 to keep weather the primary driver.

Event types used:
    Battles                — armed clashes between organised groups
    Explosions/Remote violence — IEDs, airstrikes, artillery
    Violence against civilians — direct targeting of non-combatants
    (Protests and Riots excluded — too common in democracies to be informative)
"""

import os
import requests
from dataclasses import dataclass
from datetime import date, timedelta
from typing import List, Optional


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ACLED_BASE_URL = "https://acleddata.com/api/acled/read"

LOOKBACK_DAYS  = 90    # how far back to look for events
MAX_GEO_SCORE  = 30    # cap on contribution to overall risk score

# Event types that signal genuine conflict risk
VIOLENCE_EVENT_TYPES = [
    "Battles",
    "Explosions/Remote violence",
    "Violence against civilians",
]

# Neighbouring countries to monitor alongside the user's country
# These are used to compute a regional proximity score
SPAIN_NEIGHBOURS = ["France", "Portugal", "Morocco", "Andorra"]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ConflictEvent:
    event_date:     str
    event_type:     str
    sub_event_type: str
    country:        str
    admin1:         str
    location:       str
    fatalities:     int
    source:         str
    notes:          str


@dataclass
class GeopoliticalSnapshot:
    country:          str
    period_start:     str
    period_end:       str
    total_events:     int
    total_fatalities: int
    event_breakdown:  dict    # event_type → count
    geo_score:        int     # 0-30
    trend:            str     # STABLE / INCREASING / DECREASING
    fetched_at:       str


# ---------------------------------------------------------------------------
# Fetcher
# ---------------------------------------------------------------------------

def get_acled_token(email: str, password: str, timeout: int = 15) -> str:
    """Get OAuth Bearer token from ACLED. Valid for 24 hours."""
    response = requests.post(
        "https://acleddata.com/oauth/token",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data={
            "username":   email,
            "password":   password,
            "grant_type": "password",
            "client_id":  "acled",
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()["access_token"]


def fetch_conflict_events(
    email: str,
    password: str,
    country: str,
    lookback_days: int = LOOKBACK_DAYS,
    event_types: Optional[List[str]] = None,
) -> List[ConflictEvent]:
    if event_types is None:
        event_types = VIOLENCE_EVENT_TYPES

    end_date   = date.today()
    start_date = end_date - timedelta(days=lookback_days)

    token = get_acled_token(email, password)

    params = {
        "country":           country,
        "event_date":        f"{start_date}|{end_date}",
        "event_date_where":  "BETWEEN",
        "event_type":        "|".join(event_types),
        "fields":            "event_date|event_type|sub_event_type|country|admin1|location|fatalities|source|notes",
        "_format":           "json",
        "limit":             500,
    }

    response = requests.get(
        ACLED_BASE_URL,
        params=params,
        headers={"Authorization": f"Bearer {token}"},
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()

    events = []
    for row in data.get("data", []):
        events.append(ConflictEvent(
            event_date=     row.get("event_date", ""),
            event_type=     row.get("event_type", ""),
            sub_event_type= row.get("sub_event_type", ""),
            country=        row.get("country", ""),
            admin1=         row.get("admin1", ""),
            location=       row.get("location", ""),
            fatalities=     int(row.get("fatalities", 0) or 0),
            source=         row.get("source", ""),
            notes=          row.get("notes", ""),
        ))
    return events

# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------

def compute_geo_score(
    events: List[ConflictEvent],
    neighbour_events: Optional[List[ConflictEvent]] = None,
) -> int:
    """
    Compute geopolitical risk score (0-30) from conflict event data.

    Scoring components:
        Event volume   (0-15): number of violent events in the period
        Fatality load  (0-10): total fatalities, log-scaled
        Neighbour proximity (0-5): spill-over risk from neighbouring countries

    Args:
        events:           Conflict events for the primary country.
        neighbour_events: Conflict events for neighbouring countries (optional).

    Returns:
        Integer score between 0 and MAX_GEO_SCORE.
    """
    import math

    # Component 1: event volume (0-15)
    n = len(events)
    if n == 0:
        volume_score = 0
    elif n <= 5:
        volume_score = 3
    elif n <= 15:
        volume_score = 7
    elif n <= 30:
        volume_score = 11
    else:
        volume_score = 15

    # Component 2: fatality load (0-10), log-scaled
    total_fatalities = sum(e.fatalities for e in events)
    if total_fatalities == 0:
        fatality_score = 0
    else:
        fatality_score = min(int(math.log1p(total_fatalities) * 2.5), 10)

    # Component 3: neighbour proximity (0-5)
    if neighbour_events:
        neighbour_count = len(neighbour_events)
        neighbour_score = min(int(neighbour_count / 5), 5)
    else:
        neighbour_score = 0

    total = volume_score + fatality_score + neighbour_score
    return min(total, MAX_GEO_SCORE)


def _compute_trend(
    events: List[ConflictEvent],
    lookback_days: int = LOOKBACK_DAYS,
) -> str:
    """
    Compare first half vs second half of the period to detect trend.

    Returns: STABLE / INCREASING / DECREASING
    """
    if not events:
        return "STABLE"

    mid = date.today() - timedelta(days=lookback_days // 2)
    mid_str = str(mid)

    first_half  = [e for e in events if e.event_date < mid_str]
    second_half = [e for e in events if e.event_date >= mid_str]

    # Normalise to events-per-day rate
    rate_first  = len(first_half)  / (lookback_days / 2)
    rate_second = len(second_half) / (lookback_days / 2)

    if rate_second > rate_first * 1.3:
        return "INCREASING"
    if rate_second < rate_first * 0.7:
        return "DECREASING"
    return "STABLE"


def build_snapshot(
    country: str,
    events: List[ConflictEvent],
    neighbour_events: Optional[List[ConflictEvent]] = None,
    lookback_days: int = LOOKBACK_DAYS,
) -> GeopoliticalSnapshot:
    """
    Build a full GeopoliticalSnapshot from fetched events.

    Args:
        country:          Primary country name.
        events:           Events for the primary country.
        neighbour_events: Events for neighbouring countries.
        lookback_days:    Period used for the fetch.

    Returns:
        GeopoliticalSnapshot ready to store in the DB.
    """
    from datetime import datetime, timezone

    breakdown: dict = {}
    for e in events:
        breakdown[e.event_type] = breakdown.get(e.event_type, 0) + 1

    return GeopoliticalSnapshot(
        country=country,
        period_start=str(date.today() - timedelta(days=lookback_days)),
        period_end=str(date.today()),
        total_events=len(events),
        total_fatalities=sum(e.fatalities for e in events),
        event_breakdown=breakdown,
        geo_score=compute_geo_score(events, neighbour_events),
        trend=_compute_trend(events, lookback_days),
        fetched_at=datetime.now(timezone.utc).isoformat(),
    )


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------

def get_geopolitical_snapshot(
    country: str,
    email: Optional[str] = None,
    password: Optional[str] = None,
    include_neighbours: bool = True,
    neighbour_countries: Optional[List[str]] = None,
    lookback_days: int = LOOKBACK_DAYS,
) -> GeopoliticalSnapshot:
    email    = email    or os.getenv("ACLED_EMAIL", "")
    password = password or os.getenv("ACLED_PASSWORD", "")

    if not email or not password:
        raise ValueError(
            "ACLED credentials required. "
            "Set ACLED_EMAIL and ACLED_PASSWORD environment variables "
            "or pass them explicitly."
        )

    events = fetch_conflict_events(
        email, password, country, lookback_days
    )

    neighbour_events = []
    if include_neighbours:
        neighbours = neighbour_countries or SPAIN_NEIGHBOURS
        for neighbour in neighbours:
            try:
                n_events = fetch_conflict_events(
                    email, password, neighbour, lookback_days
                )
                neighbour_events.extend(n_events)
            except Exception:
                pass

    return build_snapshot(country, events, neighbour_events, lookback_days)