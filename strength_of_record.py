"""
strength_of_record.py
======================
Strength-of-Record (SOR) calculator for AllMOSports football ratings.

WHAT THIS MEASURES
-------------------
SOR answers: "How impressive is this team's actual win-loss record, given
exactly who they played?" It is NOT a power rating -- it's a resume/record-
quality metric, complementary to your offensive/defensive/overall ratings.

METHOD
------
1. Calibrate a margin-prediction model from historical results:
       margin = a * (rating_diff) + c
   and measure the residual standard deviation (sigma) -- how much actual
   margins vary around what the model predicts. This sigma is what lets us
   turn a rating differential into a properly calibrated win probability,
   instead of guessing a logistic constant.

2. Define a classification-relative "reference team" rating (e.g. the 80th
   percentile overall rating within that class) -- a stand-in for an
   above-average, clearly-good team in that class.

3. For each real team, replay their actual schedule (same opponents) as if
   the reference team had played it instead:
     a) WIN-BASED COMPONENT: build the Poisson-binomial distribution of how
        many games the reference team would be expected to win across that
        schedule, then compute how likely the reference team is to match or
        beat the real team's actual win total. A LOW probability means the
        record was hard to produce -- so SOR is reported as 1 minus that
        probability (higher = more impressive).
     b) MARGIN-BASED COMPONENT: compare the team's actual (capped) scoring
        margin in each game to the reference team's expected (capped) margin
        against that same opponent, averaged over the season. This adds
        signal beyond pure win/loss, which is noisy over a 9-10 game
        schedule.

4. Percentile-rank both components within the team's classification, then
   blend them into a single 0-100 SOR score.

NOTE ON HOME/AWAY
------------------
Site (home/away/neutral) is intentionally NOT modeled right now. Earlier
testing on real 2025 data produced a home-field coefficient with the wrong
sign -- almost certainly an artifact of calibrating on a single season
against ratings that were already fit to that same season's games (circular).
Dropping it removes that artifact and simplifies the model. If you want it
back later: add a `site` field to GameResult, add a home_indicator term to
the regression in calibrate_margin_model(), and add it to expected_margin().
That's the entire change -- nothing else in the pipeline depends on it.

IMPORTANT CAVEATS
------------------
- LOOKAHEAD BIAS: for the most defensible SOR, calibrate the margin model
  using ratings AS THEY STOOD BEFORE EACH GAME, not final-of-season ratings.
  Using final ratings for early-season games leaks information the team
  didn't "deserve credit" for yet. This script still works as a retrospective
  approximation if that's all you have, but treat it as the noisier part of
  the metric until you can calibrate out-of-sample (e.g. fit on 2025, then
  validate against 2026 once it exists).
- MARGIN_CAP, the reference-team percentile, and the win/margin blend weight
  are all tunable constants below. Back-test them against last season's
  actual playoff seeding / "who got snubbed" conversations to pick values
  that match how Missouri coaches and fans actually judge resumes.
- This assumes no ties (MSHSAA football has OT rules), and that opponent
  ratings are available for every game. Non-MSHSAA or unrated opponents
  (out-of-state games, etc.) are skipped.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Tunable constants -- back-test these against real seasons before trusting them
# ---------------------------------------------------------------------------
MARGIN_CAP = 28.0          # points; caps blowout influence on the margin component
REFERENCE_PERCENTILE = 80  # "reference team" = this percentile of class overall ratings
WIN_COMPONENT_WEIGHT = 0.5  # blend weight; margin component gets (1 - this)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class GameResult:
    """One game, from team_id's perspective. No site field -- see note above."""
    team_id: str
    opp_id: str
    team_score: int
    opp_score: int


@dataclass
class CalibratedModel:
    a: float       # points of margin per rating point of differential
    c: float       # intercept (systemic bias correction)
    sigma: float   # residual std dev of margin -- the model's real-world noise


# ---------------------------------------------------------------------------
# Step 1: calibrate the margin/win-probability model from historical results
# ---------------------------------------------------------------------------

def calibrate_margin_model(
    games: List[GameResult],
    ratings: Dict[str, float],
) -> CalibratedModel:
    """
    Fit margin = a*(team_rating - opp_rating) + c via least squares, then
    compute the residual standard deviation. `ratings` should ideally be
    pregame snapshots matched to each game's date -- see the lookahead-bias
    caveat above.
    """
    rows, targets = [], []
    for g in games:
        if g.team_id not in ratings or g.opp_id not in ratings:
            continue
        rating_diff = ratings[g.team_id] - ratings[g.opp_id]
        rows.append([rating_diff, 1.0])
        targets.append(g.team_score - g.opp_score)

    if len(rows) < 10:
        raise ValueError("Need a reasonable sample of historical games to calibrate.")

    X = np.array(rows)
    y = np.array(targets, dtype=float)
    coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
    a, c = (float(v) for v in coeffs)

    residuals = y - X @ coeffs
    sigma = float(np.std(residuals, ddof=2))  # ddof = number of fit parameters

    return CalibratedModel(a=a, c=c, sigma=sigma)


def expected_margin(model: CalibratedModel, rating_diff: float) -> float:
    return model.a * rating_diff + model.c


def win_probability(model: CalibratedModel, rating_diff: float) -> float:
    """P(margin > 0), treating margin as Normal(expected_margin, sigma)."""
    margin = expected_margin(model, rating_diff)
    z = margin / model.sigma
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2)))


def reference_rating(class_ratings: List[float], percentile: float = REFERENCE_PERCENTILE) -> float:
    """The 'above-average, clearly good' benchmark team for this class."""
    return float(np.percentile(class_ratings, percentile))


# ---------------------------------------------------------------------------
# Step 2: Poisson-binomial distribution (DP), for the win-based component
# ---------------------------------------------------------------------------

def poisson_binomial_pmf(probs: List[float]) -> List[float]:
    """
    Returns dist where dist[k] = P(exactly k successes), given independent
    Bernoulli trials with (possibly all different) success probabilities.
    O(n^2) DP -- trivial at n = 9-12 games.
    """
    n = len(probs)
    dist = [1.0] + [0.0] * n
    for p in probs:
        new_dist = [0.0] * (n + 1)
        for k, mass in enumerate(dist):
            if mass == 0.0:
                continue
            new_dist[k] += mass * (1.0 - p)
            if k + 1 <= n:
                new_dist[k + 1] += mass * p
        dist = new_dist
    return dist


# ---------------------------------------------------------------------------
# Step 3: per-team SOR components
# ---------------------------------------------------------------------------

def _clip(x: float, cap: float = MARGIN_CAP) -> float:
    return max(-cap, min(cap, x))


def compute_team_sor_components(
    schedule: List[GameResult],
    ratings: Dict[str, float],
    model: CalibratedModel,
    ref_rating: float,
) -> Dict[str, Optional[float]]:
    """
    Computes the raw (pre-percentile) win-based and margin-based SOR
    components for one team, against a fixed reference team rating.
    """
    ref_win_probs: List[float] = []
    margin_deltas: List[float] = []
    actual_wins = 0

    for g in schedule:
        opp_rating = ratings.get(g.opp_id)
        if opp_rating is None:
            continue  # unrated opponent -- consider substituting a league-average rating instead

        ref_diff = ref_rating - opp_rating
        ref_win_probs.append(win_probability(model, ref_diff))

        ref_expected = _clip(expected_margin(model, ref_diff))
        actual_margin = _clip(g.team_score - g.opp_score)
        margin_deltas.append(actual_margin - ref_expected)

        if g.team_score > g.opp_score:
            actual_wins += 1

    if not ref_win_probs:
        return {"win_sor_raw": None, "margin_sor_raw": None}

    dist = poisson_binomial_pmf(ref_win_probs)
    p_ref_matches_or_beats = sum(dist[actual_wins:])
    win_sor_raw = 1.0 - p_ref_matches_or_beats  # higher = harder for an avg-good team to replicate

    margin_sor_raw = sum(margin_deltas) / len(margin_deltas)  # avg pts/game above reference expectation

    return {"win_sor_raw": win_sor_raw, "margin_sor_raw": margin_sor_raw}


# ---------------------------------------------------------------------------
# Step 4: percentile-normalize within classification and blend
# ---------------------------------------------------------------------------

def percentile_rank(value: float, all_values: List[float]) -> float:
    sorted_vals = sorted(all_values)
    idx = bisect.bisect_right(sorted_vals, value)
    return idx / len(sorted_vals) * 100.0


def build_classification_sor(
    schedules: Dict[str, List[GameResult]],
    ratings: Dict[str, float],
    model: CalibratedModel,
    win_weight: float = WIN_COMPONENT_WEIGHT,
) -> Dict[str, float]:
    """
    schedules: {team_id: [GameResult, ...]} for every team IN ONE CLASSIFICATION.
    ratings: {team_id: overall_rating} -- should include every team in this class
             plus any cross-class opponents played.
    Returns {team_id: SOR (0-100)}.
    """
    class_ratings = [ratings[t] for t in schedules if t in ratings]
    ref_rating = reference_rating(class_ratings)

    raw = {
        team_id: compute_team_sor_components(games, ratings, model, ref_rating)
        for team_id, games in schedules.items()
    }

    win_vals = [v["win_sor_raw"] for v in raw.values() if v["win_sor_raw"] is not None]
    margin_vals = [v["margin_sor_raw"] for v in raw.values() if v["margin_sor_raw"] is not None]

    final: Dict[str, float] = {}
    for team_id, v in raw.items():
        if v["win_sor_raw"] is None:
            continue
        win_pct = percentile_rank(v["win_sor_raw"], win_vals)
        margin_pct = percentile_rank(v["margin_sor_raw"], margin_vals)
        final[team_id] = round(win_weight * win_pct + (1 - win_weight) * margin_pct, 1)

    return final


# ---------------------------------------------------------------------------
# Demo with synthetic data -- swap in your real schools.json / ratings / results
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random

    random.seed(7)

    true_strength = {f"T{i}": random.uniform(-15, 15) for i in range(1, 9)}

    def simulate_game(t1, t2) -> GameResult:
        diff = true_strength[t1] - true_strength[t2]
        margin = round(random.gauss(diff, 14))
        t1_score = max(0, 21 + margin // 2)
        t2_score = max(0, 21 - margin // 2)
        return GameResult(t1, t2, t1_score, t2_score)

    hist_games = []
    teams = list(true_strength)
    for _ in range(150):
        t1, t2 = random.sample(teams, 2)
        hist_games.append(simulate_game(t1, t2))

    model = calibrate_margin_model(hist_games, true_strength)
    print(f"Calibrated model: a={model.a:.3f} c={model.c:.3f} sigma={model.sigma:.2f}\n")

    schedules: Dict[str, List[GameResult]] = {t: [] for t in teams}
    for t1, t2 in [(teams[i], teams[(i + 1) % 8]) for i in range(8)] + \
                  [(teams[i], teams[(i + 3) % 8]) for i in range(8)]:
        g = simulate_game(t1, t2)
        schedules[t1].append(g)
        schedules[t2].append(GameResult(t2, t1, g.opp_score, g.team_score))

    sor_scores = build_classification_sor(schedules, true_strength, model)

    print("Team   Record   SOR")
    for team_id, sor in sorted(sor_scores.items(), key=lambda kv: -kv[1]):
        wins = sum(1 for g in schedules[team_id] if g.team_score > g.opp_score)
        losses = len(schedules[team_id]) - wins
        print(f"{team_id:6} {wins}-{losses:<6} {sor:5.1f}")
