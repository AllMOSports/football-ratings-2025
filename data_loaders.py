"""
data_loaders.py
================
Turns AllMOSports' real football data files into the inputs
strength_of_record.py needs: a {team_id: overall_rating} dict and
{team_id: [GameResult, ...]} schedules, grouped by classification.

INPUT FILES (from AllMOSports/football-ratings-2025)
-----------------------------------------------------
- classifications.json          {"teams": [{"school", "classification", "district"}, ...]}
- football_ratings_2025.json     {"league_average", "teams": [{"school", "ovr_rating", ...}, ...]}
                                  (the GLOBAL file, not a per-class one -- you need every
                                   team's rating available, since teams play cross-class
                                   opponents)
- football_scoreboard_2025.csv    Date, Home Team, Home Score, Away Team, Away Score

NAME MATCHING
-------------
No alias/co-op resolution step is needed here: football_ratings_2025.py already
resolves every scoreboard name to the canonical classifications.json "school"
string at scrape time (see resolve_name() in that script), so team strings
already match exactly across all three files.

HOME/AWAY
---------
Not modeled right now (by design -- see strength_of_record.py). The "Home
Team"/"Away Team" columns in the scoreboard CSV are simply ignored here; only
which two teams played and the final score matter.
"""

from __future__ import annotations

import csv
import json
from typing import Dict, List

from strength_of_record import GameResult


def load_ratings(ratings_json_path: str) -> Dict[str, float]:
    """{team_id: overall_rating}, from the GLOBAL ratings file (not a per-class one)."""
    with open(ratings_json_path) as f:
        data = json.load(f)
    return {team["school"]: team["ovr_rating"] for team in data["teams"]}


def load_classifications(classifications_json_path: str) -> Dict[str, dict]:
    """{team_id: {"classification": int, "district": int}}"""
    with open(classifications_json_path) as f:
        data = json.load(f)
    return {
        t["school"]: {"classification": t["classification"], "district": t["district"]}
        for t in data["teams"]
    }


def load_schedules(scoreboard_csv_path: str) -> Dict[str, List[GameResult]]:
    """
    Explodes each scoreboard row into two GameResults (one per team's
    perspective), bucketed into {team_id: [GameResult, ...]}. Site is
    ignored entirely.
    """
    schedules: Dict[str, List[GameResult]] = {}

    with open(scoreboard_csv_path, newline="") as f:
        for row in csv.DictReader(f):
            home, away = row["Home Team"], row["Away Team"]
            home_score, away_score = int(row["Home Score"]), int(row["Away Score"])

            schedules.setdefault(home, []).append(
                GameResult(home, away, home_score, away_score)
            )
            schedules.setdefault(away, []).append(
                GameResult(away, home, away_score, home_score)
            )

    return schedules


def all_games_flat(schedules: Dict[str, List[GameResult]]) -> List[GameResult]:
    """
    Flattened list of every GameResult across every team -- this is what
    calibrate_margin_model() wants. Each physical game appears twice (once
    per team's perspective); that's fine for calibration since it just
    doubles the sample symmetrically.
    """
    return [g for games in schedules.values() for g in games]


def group_schedules_by_class(
    schedules: Dict[str, List[GameResult]],
    classifications: Dict[str, dict],
) -> Dict[int, Dict[str, List[GameResult]]]:
    """
    {classification: {team_id: [GameResult, ...]}}. Opponents from other
    classes are left untouched inside each team's schedule -- this only
    controls which pool a team gets percentile-ranked against.
    """
    by_class: Dict[int, Dict[str, List[GameResult]]] = {}
    skipped = []
    for team, games in schedules.items():
        cls = classifications.get(team, {}).get("classification")
        if cls is None:
            skipped.append(team)
            continue
        by_class.setdefault(cls, {})[team] = games
    if skipped:
        print(
            f"  [group_schedules_by_class] {len(skipped)} teams had games but no "
            f"classification entry -- likely out-of-state opponents. Skipped: "
            f"{skipped[:10]}{'...' if len(skipped) > 10 else ''}"
        )
    return by_class


# ---------------------------------------------------------------------------
# End-to-end demo on the REAL 2025 files
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from strength_of_record import calibrate_margin_model, build_classification_sor

    DATA_DIR = "/home/claude/allmosports_data"

    ratings = load_ratings(f"{DATA_DIR}/football_ratings_2025.json")
    classifications = load_classifications(f"{DATA_DIR}/classifications.json")
    schedules = load_schedules(f"{DATA_DIR}/football_scoreboard_2025.csv")

    print(f"Loaded {len(ratings)} rated teams, {len(schedules)} teams with schedules.\n")

    # NOTE: calibrating against the SAME season's final ratings that were
    # themselves fit to these same games is circular (lookahead bias) -- your
    # rating engine already minimized error against this exact data. The
    # clean version of this calibration uses 2025's FINAL ratings to predict
    # 2026 games once those exist, or uses weekly rating snapshots from this
    # season instead of the final one.
    model = calibrate_margin_model(all_games_flat(schedules), ratings)
    print(
        f"Calibrated model (illustrative, not yet out-of-sample): "
        f"a={model.a:.3f}  c={model.c:.3f}  sigma={model.sigma:.2f}\n"
    )

    by_class = group_schedules_by_class(schedules, classifications)

    TARGET_CLASS = 6
    class_schedules = by_class[TARGET_CLASS]
    sor_scores = build_classification_sor(class_schedules, ratings, model)

    print(f"=== Class {TARGET_CLASS} Strength of Record ===")
    print(f"{'Team':<28}{'Record':<10}{'OVR':>8}{'SOR':>8}")
    for team_id, sor in sorted(sor_scores.items(), key=lambda kv: -kv[1]):
        games = class_schedules[team_id]
        wins = sum(1 for g in games if g.team_score > g.opp_score)
        losses = len(games) - wins
        print(f"{team_id:<28}{wins}-{losses:<8}{ratings[team_id]:>8.2f}{sor:>8.1f}")
