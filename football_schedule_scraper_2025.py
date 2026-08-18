import requests
from bs4 import BeautifulSoup
import json
import csv
import re
import pandas as pd
from datetime import date, timedelta
import time
 
# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
 
SEASON_YEAR   = 2025
SEASON_START  = date(2025, 8, 1)
SEASON_END    = date(2025, 12, 15)
BASE_URL      = "https://www.mshsaa.org/activities/scoreboard.aspx?alg=19&date={}"
MAX_POINTS    = 100
CSV_PATH      = f"football_scoreboard_{SEASON_YEAR}.csv"
CLASSIFICATIONS_PATH  = "classifications.json"
SCHOOLS_CSV           = "mshsaa_schools.csv"
 
# ---------------------------------------------------------------------------
# MANUAL GAMES (not listed on MSHSAA Scoreboard)
# ---------------------------------------------------------------------------
# Add any games missing from the MSHSAA scoreboard here.
# Format: ("YYYY-MM-DD", "Team 1 Name", score1, "Team 2 Name", score2)
 
MANUAL_GAMES = [
    ("2025-08-29", "Cole Camp", 27, "Russellville", 26),
    ("2025-09-05", "Van-Far", 18, "Russellville", 32),
    ("2025-09-12", "Russellville", 41, "Midway", 6),
    ("2025-09-19", "Russellville", 51, "Affton", 16),
    ("2025-09-26", "South Callaway", 0, "Russellville", 12),
    ("2025-10-03", "Linn", 12, "Russellville", 49),
    ("2025-10-10", "Russellville", 44, "Harrisburg", 28),
    ("2025-10-17", "Russellville", 0, "Tipton", 52),
    ("2025-10-24", "Fayette", 26, "Russellville", 41),
    ("2025-11-07", "Russellville", 35, "Cole Camp", 21),
    ("2025-11-14", "Tipton", 53, "Russellville", 18),
]
 
# ---------------------------------------------------------------------------
# SCORE CORRECTIONS (from Suspicious_Scores_-_Football.xlsx review)
# ---------------------------------------------------------------------------
# Format: ("YYYY-MM-DD", "Team A", correct_score_A, "Team B", correct_score_B)
 
SCORE_CORRECTIONS = [
]
 
# ---------------------------------------------------------------------------
# EXCLUDED GAMES (from Suspicious_Scores_-_Football.xlsx review)
# ---------------------------------------------------------------------------
# Format: ("YYYY-MM-DD", "Team A", "Team B")
 
EXCLUDED_GAMES = [
    ("2025-10-31", "Nevada", "Southeast"),
    ("2025-10-31", "Ste. Genevieve", "Normandy Collaborative"),
    ("2025-10-30", "Lee's Summit West", "Park Hill South"),
]
 
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": "https://www.mshsaa.org/"
}
 
# ---------------------------------------------------------------------------
# HTTP SESSION (connection reuse + retry on transient failures)
# ---------------------------------------------------------------------------
 
def build_session():
    from requests.adapters import HTTPAdapter
    try:
        from urllib3.util.retry import Retry
    except ImportError:
        from requests.packages.urllib3.util.retry import Retry
 
    session = requests.Session()
    retry = Retry(
        total=1,
        connect=1,
        read=1,
        backoff_factor=1.5,
        status_forcelist=[500, 502, 503, 504],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=1, pool_maxsize=1)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session
 
# ---------------------------------------------------------------------------
# CLASSIFICATIONS (used only to normalize names when possible -- NOT used
# to restrict which games get scraped)
# ---------------------------------------------------------------------------
 
def load_classifications(path=CLASSIFICATIONS_PATH):
    """Return team_to_class dict keyed by school name (used for name
    normalization only)."""
    with open(path) as f:
        data = json.load(f)
    team_to_class = {}
    for entry in data["teams"]:
        team_to_class[entry["school"]] = entry["classification"]
    return team_to_class
 
 
def build_id_to_classname(team_to_class, schools_csv=SCHOOLS_CSV):
    """
    Build { school_id_str : classification_name } by exact-matching
    mshsaa_schools.csv names to classifications.json names after stripping
    the ' High School' suffix. No fuzzy matching used.
 
    MANUAL_OVERRIDES covers schools whose mshsaa_schools.csv name doesn't
    match their classifications.json name. IDs were looked up directly
    from the MSHSAA scoreboard pages.
    """
    MANUAL_OVERRIDES = {
        "271": "Clopton with Elsberry",
        "331": "King City with Pattonsburg",
        "126": "Lockwood with Golden City",
        "421": "Princeton with Mercer",
        "424": "Rich Hill with Hume",
        "431": "Salisbury",
        "435": "Scott City",
        "443": "Skyline",
        "193": "Slater",
        "194": "Smith-Cotton",
        "197": "South Callaway",
        "549": "St. Mary's South Side",
        "463": "Stockton",
        "207": "Sullivan",
        "208": "Sumner",
        "469": "Sweet Springs with Malta Bend",
        "198": "Truman",
        "479": "University Academy Charter",
        "204": "Van Horn",
        "206": "Vashon",
        "20": "Appleton City with Montrose",
        "275": "Drexel with Miami (Amoret)",
        "575": "Renaissance Academy Charter",
        "172": "St. James",
        "35": "DeSoto with Kingston",
        "917": "Father Tolton with Calvary Lutheran",
        "342": "Liberal with Bronaugh",
        "776": "Transportation and Law with Beaumont",
        "483": "Van-Far with Community",
    }
 
    df = pd.read_csv(schools_csv)
    known_class_names = set(team_to_class.keys())
 
    id_to_classname = {}
    for _, row in df.iterrows():
        full_name = row["school_name"]
        sid       = str(row["school_id"])
        stripped  = full_name.replace(" High School", "").strip()
 
        if stripped in known_class_names:
            id_to_classname[sid] = stripped
        elif full_name in known_class_names:
            id_to_classname[sid] = full_name
 
    id_to_classname.update(MANUAL_OVERRIDES)
 
    print(f"  [name-resolve] {len(id_to_classname)} schools mapped by ID "
          f"({len(MANUAL_OVERRIDES)} via manual overrides)")
    return id_to_classname
 
 
# ---------------------------------------------------------------------------
# NAME RESOLUTION
# ---------------------------------------------------------------------------
 
def resolve_name(cell, id_to_classname, known_teams):
    """
    Resolve a scoreboard table cell to a team name.
 
    Step 1: Extract s= ID from href -> look up in id_to_classname.
            Handles renamed/merged schools (e.g. 'Scott City with Chaffee'
            -> 'Scott City') because the ID in the href never changes.
    Step 2: Exact match of display text against known_teams
            (classifications.json).
    Step 3 (fallback): If the team isn't in classifications.json at all --
            an out-of-state opponent, non-MSHSAA/unclassified school, etc.
            -- just use whatever name is on the scoreboard page. The game
            is still captured either way; this function no longer causes
            games to be skipped.
    Returns None only if the cell has no usable text at all.
    """
    a = cell.find("a", href=lambda h: h and "/MySchool/Schedule.aspx" in h)
 
    if a:
        href  = a.get("href", "")
        match = re.search(r"[?&]s=(\d+)", href, re.IGNORECASE)
        if match:
            sid = match.group(1)
            if sid in id_to_classname:
                return id_to_classname[sid]
 
        display_text = a.get_text(strip=True)
        if display_text in known_teams:
            return display_text
 
        return display_text or None
 
    display_text = cell.get_text(strip=True)
    return display_text or None
 
 
# ---------------------------------------------------------------------------
# SCRAPING
# ---------------------------------------------------------------------------
 
def parse_score(text):
    text = text.strip()
    if not text:
        return None
    try:
        score = int(text)
    except ValueError:
        return None
    return score if 0 <= score <= MAX_POINTS else None
 
 
def is_forfeit(c1, c2):
    return "forfeit" in (c1.get_text() + c2.get_text()).lower()
 
 
def scrape_date(target_date, id_to_classname, known_teams, session):
    url = BASE_URL.format(target_date.strftime("%m%d%Y"))
    try:
        resp = session.get(url, timeout=(10, 25), headers=HEADERS)
        resp.raise_for_status()
    except requests.exceptions.Timeout as e:
        print(f"  TIMEOUT {target_date}: {e}")
        return [], "timeout"
    except requests.RequestException as e:
        print(f"  Failed {target_date}: {e}")
        return [], "error"
 
    soup  = BeautifulSoup(resp.text, "html.parser")
    games = []
 
    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        if len(rows) < 3:
            continue
        if "final" not in rows[-1].get_text().lower():
            continue
 
        t1c = rows[1].find_all("td")
        t2c = rows[2].find_all("td")
        if len(t1c) < 3 or len(t2c) < 3:
            continue
        if is_forfeit(t1c[1], t2c[1]):
            continue
 
        name1 = resolve_name(t1c[1], id_to_classname, known_teams)
        name2 = resolve_name(t2c[1], id_to_classname, known_teams)
 
        if name1 is None or name2 is None:
            continue
 
        s1 = parse_score(t1c[2].get_text())
        s2 = parse_score(t2c[2].get_text())
        if s1 is None or s2 is None:
            continue
 
        games.append((
            target_date.strftime("%Y-%m-%d"),
            name1, s1,
            name2, s2
        ))
 
    return games, None
 
 
def scrape_full_season(id_to_classname, known_teams):
    all_games     = []
    current       = SEASON_START
    scrape_t0     = time.perf_counter()
    slow_days     = []
    failed_days   = []
    session       = build_session()
 
    while current <= min(SEASON_END, date.today()):
        day_t0 = time.perf_counter()
        print(f"  Scraping {current}...", end=" ", flush=True)
        day_games, fail_reason = scrape_date(current, id_to_classname, known_teams, session)
        all_games.extend(day_games)
        day_elapsed = time.perf_counter() - day_t0
        print(f"{len(day_games)} games ({day_elapsed:.1f}s)")
        if day_elapsed > 3.0:
            slow_days.append((current, day_elapsed))
        if fail_reason is not None:
            failed_days.append((current, fail_reason))
        current += timedelta(days=1)
        time.sleep(0.5)
 
    scrape_elapsed = time.perf_counter() - scrape_t0
    print(f"\n  [TIMING] Scraping took {scrape_elapsed:.1f}s total "
          f"for {len(all_games)} games.")
    if slow_days:
        print(f"  [TIMING] {len(slow_days)} slow day(s) (>3s each):")
        for d, secs in slow_days:
            print(f"    {d}: {secs:.1f}s")
    if failed_days:
        print(f"\n  *** {len(failed_days)} date(s) NEVER returned data, "
              f"even after retry -- these dates may be missing real "
              f"games. Check them manually against MSHSAA and add via "
              f"MANUAL_GAMES if needed: ***")
        for d, reason in failed_days:
            print(f"    {d} ({reason})")
    else:
        print("  All dates returned successfully -- no known data gaps "
              "from scraping failures.")
    return all_games
 
 
def apply_score_corrections(all_games, corrections=SCORE_CORRECTIONS):
    """
    Fix known-bad scores in place. Matches each game by date + the two team
    names (order-independent), then overwrites each named team's score with
    the corrected value.
    """
    lookup = {}
    for date_str, team_a, score_a, team_b, score_b in corrections:
        lookup[(date_str, frozenset([team_a, team_b]))] = {team_a: score_a, team_b: score_b}
 
    corrected = 0
    fixed_games = []
    for date_str, t1, s1, t2, s2 in all_games:
        key = (date_str, frozenset([t1, t2]))
        fix = lookup.get(key)
        if fix is not None:
            new_s1 = fix.get(t1, s1)
            new_s2 = fix.get(t2, s2)
            if (new_s1, new_s2) != (s1, s2):
                corrected += 1
            fixed_games.append((date_str, t1, new_s1, t2, new_s2))
        else:
            fixed_games.append((date_str, t1, s1, t2, s2))
 
    if corrected:
        print(f"  Corrected {corrected} game score(s) via SCORE_CORRECTIONS.")
    else:
        print("  No SCORE_CORRECTIONS matched (nothing changed).")
 
    return fixed_games
 
 
def apply_exclusions(all_games, exclusions=EXCLUDED_GAMES):
    """
    Drop games confirmed bad/unverifiable. Matches by date + the two team
    names (order-independent).
    """
    exclude_keys = {(date_str, frozenset([team_a, team_b]))
                     for date_str, team_a, team_b in exclusions}
 
    filtered_games = [
        g for g in all_games
        if (g[0], frozenset([g[1], g[3]])) not in exclude_keys
    ]
 
    removed = len(all_games) - len(filtered_games)
    if removed:
        print(f"  Removed {removed} excluded game(s) via EXCLUDED_GAMES.")
    else:
        print("  No EXCLUDED_GAMES matched (nothing removed).")
 
    return filtered_games
 
 
def deduplicate_games(all_games):
    """
    Remove duplicate games where the same two teams played on the same date
    with the same scores, regardless of which team is listed as home or away.
    """
    seen         = set()
    unique_games = []
    duplicates   = 0
 
    for game in all_games:
        date_str, t1, s1, t2, s2 = game
        key = (date_str, frozenset([t1, t2]))
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        unique_games.append(game)
 
    if duplicates:
        print(f"  Removed {duplicates} duplicate game(s). "
              f"{len(unique_games)} unique games remain.")
    else:
        print(f"  No duplicates found. {len(unique_games)} games.")
 
    return unique_games
 
 
# ---------------------------------------------------------------------------
# CSV OUTPUT
# ---------------------------------------------------------------------------
 
def save_csv(all_games):
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Date", "Home Team", "Home Score", "Away Team", "Away Score"])
        for date_str, t1, s1, t2, s2 in all_games:
            writer.writerow([date_str, t1, s1, t2, s2])
    print(f"Saved {len(all_games)} games to {CSV_PATH}")
 
 
# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
 
if __name__ == "__main__":
    print(f"=== MSHSAA Football Schedule Scraper {SEASON_YEAR} ===")
 
    print("\nLoading classifications (used for name normalization only)...")
    team_to_class = load_classifications()
    known_teams = set(team_to_class.keys())
    print(f"  Loaded {len(team_to_class)} teams from {CLASSIFICATIONS_PATH}")
 
    print("\nBuilding school ID -> classification name lookup...")
    id_to_classname = build_id_to_classname(team_to_class, SCHOOLS_CSV)
 
    print("\nScraping season scoreboard (ALL games, no MSHSAA/classification restriction)...")
    all_games = scrape_full_season(id_to_classname, known_teams)
    print(f"\nTotal games scraped (before dedup): {len(all_games)}")
    if not all_games:
        print("No games found -- exiting.")
        exit(1)
 
    if MANUAL_GAMES:
        print(f"\nAdding {len(MANUAL_GAMES)} manual game(s)...")
        all_games.extend(MANUAL_GAMES)
 
    print("\nApplying score corrections...")
    all_games = apply_score_corrections(all_games)
 
    print("\nApplying game exclusions...")
    all_games = apply_exclusions(all_games)
 
    print("\nDeduplicating games...")
    all_games = deduplicate_games(all_games)
 
    print("\nSaving scoreboard CSV...")
    save_csv(all_games)
 
    print("\nDone. No ratings were calculated -- this script only scrapes and saves schedule data.")
