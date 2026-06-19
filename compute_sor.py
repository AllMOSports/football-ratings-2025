"""
compute_sor.py
===============
End-to-end script: loads your real football data, calibrates the SOR model
once, computes Strength of Record for every team in every classification,
and writes output JSON files in the same style as football_ratings_2025.py's
output -- so this can slot into the same pipeline/GitHub Action.

USAGE
-----
    python3 compute_sor.py

Expects classifications.json, football_ratings_2025.json, and
football_scoreboard_2025.csv in DATA_DIR (defaults to the current directory --
change below if your files live elsewhere).

OUTPUT
------
    football_sor_2025.json              -- every team, all classes, with sor_rank
    football_sor_2025_class{1-6}.json   -- per-class, same shape as your existing
                                            per-class rating files
"""

import json
from datetime import datetime
from pathlib import Path

from strength_of_record import calibrate_margin_model, build_classification_sor
from data_loaders import (
    load_ratings,
    load_classifications,
    load_schedules,
    all_games_flat,
    group_schedules_by_class,
)

DATA_DIR = Path(".")  # <-- change this if your data files live elsewhere
OUTPUT_PATH = "football_sor_2025.json"


def main():
    ratings = load_ratings(DATA_DIR / "football_ratings_2025.json")
    classifications = load_classifications(DATA_DIR / "classifications.json")
    schedules = load_schedules(DATA_DIR / "football_scoreboard_2025.csv")
    print(f"Loaded {len(ratings)} rated teams, {len(schedules)} teams with schedules.")

    model = calibrate_margin_model(all_games_flat(schedules), ratings)
    print(f"Calibrated model: a={model.a:.3f} c={model.c:.3f} sigma={model.sigma:.2f}\n")

    by_class = group_schedules_by_class(schedules, classifications)

    all_entries = []
    timestamp = datetime.now().strftime("%B %d, %Y at %I:%M %p")

    for cls in sorted(by_class):
        class_schedules = by_class[cls]
        sor_scores = build_classification_sor(class_schedules, ratings, model)
        ranked = sorted(sor_scores.items(), key=lambda kv: -kv[1])

        class_entries = []
        for rank, (team, sor) in enumerate(ranked, start=1):
            games = class_schedules[team]
            wins = sum(1 for g in games if g.team_score > g.opp_score)
            losses = len(games) - wins
            class_entries.append({
                "school": team,
                "classification": cls,
                "district": classifications[team]["district"],
                "wins": wins,
                "losses": losses,
                "ovr_rating": ratings[team],
                "sor": sor,
                "sor_rank": rank,
            })

        class_path = f"football_sor_2025_class{cls}.json"
        with open(class_path, "w") as f:
            json.dump(
                {"last_updated": timestamp, "classification": cls, "teams": class_entries},
                f, indent=2,
            )
        print(f"  Class {cls}: {len(class_entries)} teams -> {class_path}")
        all_entries.extend(class_entries)

    with open(OUTPUT_PATH, "w") as f:
        json.dump({"last_updated": timestamp, "teams": all_entries}, f, indent=2)
    print(f"\nSaved {len(all_entries)} teams total -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
