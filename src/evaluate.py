"""
Offline evaluation of the relocation recommender.

Metrics
-------
NDCG@10     — normalised discounted cumulative gain; measures how high
              the "ideal" country ranks in the returned top-10 list.
              Score = 1/log2(rank+1) if ideal in top-10, else 0.
              Mean across 50 synthetic profiles.

Coverage    — fraction of dataset countries that appear at least once
              in any recommendation list (top-10 across all profiles).

Diversity@10 — mean pairwise cosine distance between the feature vectors
               of the 10 recommended countries, averaged over profiles.
               Higher → more varied suggestions.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances

# ── path setup ────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from recommender import (
    UserProfile,
    _BUDGET_BREAKS,
    _LANG_COUNTRIES,
    _WARM_CLIMATE,
    recommend,
)

PROCESSED_PATH = _ROOT / "data" / "processed" / "countries.csv"
METRICS_PATH   = _ROOT / "data" / "processed" / "metrics.json"

random.seed(42)
np.random.seed(42)

# Feature columns used for diversity calculation
_DIVERSITY_COLS = [
    "safety_index", "healthcare_index", "life_expectancy",
    "gdp_per_capita", "cost_of_living_index", "purchasing_power_index",
    "residency_difficulty", "pollution_index",
]

# 50 target countries spanning all clusters (hand-picked for geographic diversity)
_TARGET_ISO3: list[str] = [
    # Western/Northern Europe
    "DEU", "NLD", "SWE", "NOR", "FIN", "CHE", "AUT", "BEL",
    # Southern Europe
    "PRT", "ESP", "ITA", "GRC", "MLT", "CYP",
    # Central/Eastern Europe
    "POL", "CZE", "HUN", "ROU", "BGR", "EST", "HRV",
    # Balkans / Caucasus
    "SRB", "GEO", "ARM",
    # Anglosphere
    "GBR", "CAN", "AUS", "NZL",
    # North America
    "USA", "MEX",
    # Latin America
    "BRA", "ARG", "COL", "CHL", "URY", "PAN",
    # Middle East / Gulf
    "ARE", "ISR",
    # East Asia
    "JPN", "KOR", "TWN", "SGP",
    # Southeast Asia
    "THA", "MYS", "IDN", "VNM",
    # South Asia / Africa
    "IND", "MAR", "ZAF",
]


# ── helpers ───────────────────────────────────────────────────────────────────

def _min_budget_for_cost(cost_index: float) -> float:
    """Return the smallest budget_usd whose ceiling covers cost_index."""
    for budget_thresh, max_cost in _BUDGET_BREAKS:
        if cost_index <= max_cost:
            return float(min(budget_thresh, 10_000))
    return 10_000.0


def _clamp(v: float, lo: int = 1, hi: int = 5) -> int:
    return int(max(lo, min(hi, round(v))))


def _languages_for(iso3: str) -> list[str]:
    return [lang for lang, ctries in _LANG_COUNTRIES.items() if iso3 in ctries]


def _make_profile(
    iso3: str,
    df: pd.DataFrame,
    noise: int = 0,
    rng: np.random.Generator | None = None,
) -> UserProfile:
    """
    Synthesise a UserProfile that strongly favours `iso3`.

    Parameters
    ----------
    noise : int
        If > 0, add ±noise random perturbation to each integer priority.
        Used to generate harder (noisier) test cases.
    """
    if rng is None:
        rng = np.random.default_rng(abs(hash(iso3)) % (2**31))

    row = df.loc[iso3] if iso3 in df.index else pd.Series(dtype=float)

    def _f(col: str, default: float = 0.5) -> float:
        v = row.get(col, default)
        return float(v) if pd.notna(v) else default

    cost       = _f("cost_of_living_index")
    safety     = _f("safety_index")
    health     = _f("healthcare_index")
    res_dif    = _f("residency_difficulty")   # 0=easy, 1=hard after scaling

    # Budget: just enough to pass the hard cost filter for this country
    budget = _min_budget_for_cost(cost)

    # Priorities: map scaled feature [0,1] → integer [1,5]
    def _pri(val: float) -> int:
        raw = _clamp(val * 5)
        if noise:
            raw = _clamp(raw + int(rng.integers(-noise, noise + 1)))
        return raw

    safety_pri  = _pri(safety)
    health_pri  = _pri(health)
    visa_pri    = _pri(1.0 - res_dif)        # invert: lower difficulty → higher priority

    langs = _languages_for(iso3)[:2]          # at most 2 languages
    warm  = iso3 in _WARM_CLIMATE

    return UserProfile(
        budget_usd=budget,
        safety_priority=safety_pri,
        climate_warm=warm,
        visa_easy_priority=visa_pri,
        healthcare_priority=health_pri,
        language=langs,
        user_is_russian=False,       # base profiles are passport-agnostic
    )


def _generate_profiles(
    df: pd.DataFrame,
    n: int = 50,
) -> list[tuple[UserProfile, str]]:
    """
    Return a list of (UserProfile, ideal_iso3) pairs.

    Strategy
    --------
    - First half (n//2)   : "clean" profiles — directly derived from target features.
    - Second half (n//2)  : "noisy" profiles — ±1 random perturbation on priorities
      (simulates imprecise user input; tests robustness).
    """
    # Keep only targets that exist in the dataset
    targets = [iso3 for iso3 in _TARGET_ISO3 if iso3 in df.index]

    # Fill up to n by cycling targets if needed
    all_targets: list[str] = []
    while len(all_targets) < n:
        all_targets.extend(targets)
    all_targets = all_targets[:n]

    profiles: list[tuple[UserProfile, str]] = []
    half = n // 2

    for i, iso3 in enumerate(all_targets):
        noise = 0 if i < half else 1
        profile = _make_profile(iso3, df, noise=noise)
        profiles.append((profile, iso3))

    return profiles


# 10 Russian-specific target countries (popular Russian emigration destinations)
_RU_TARGET_ISO3: list[str] = [
    "GEO", "ARM", "SRB", "TUR", "ARE",
    "THA", "MNE", "IDN", "ARG", "MEX",
]


def _make_russian_profile(
    iso3: str,
    df: pd.DataFrame,
    rng: np.random.Generator | None = None,
) -> UserProfile:
    """
    Synthesise a UserProfile that represents a Russian citizen targeting `iso3`.
    Always sets user_is_russian=True, needs_banking_access=True.
    """
    if rng is None:
        rng = np.random.default_rng(abs(hash(iso3 + "_ru")) % (2**31))

    row = df.loc[iso3] if iso3 in df.index else pd.Series(dtype=float)

    def _f(col: str, default: float = 0.5) -> float:
        v = row.get(col, default)
        return float(v) if pd.notna(v) else default

    cost    = _f("cost_of_living_index")
    safety  = _f("safety_index")
    health  = _f("healthcare_index")
    res_dif = _f("residency_difficulty")

    budget = _min_budget_for_cost(cost)
    safety_pri  = _clamp(safety * 5)
    health_pri  = _clamp(health * 5)
    visa_pri    = _clamp((1.0 - res_dif) * 5)

    langs = ["russian"] + _languages_for(iso3)[:1]
    warm  = iso3 in _WARM_CLIMATE

    return UserProfile(
        budget_usd=budget,
        safety_priority=safety_pri,
        climate_warm=warm,
        visa_easy_priority=visa_pri,
        healthcare_priority=health_pri,
        language=langs,
        user_is_russian=True,
        needs_banking_access=True,
        has_eu_visa=False,
    )


def _generate_russian_profiles(
    df: pd.DataFrame,
) -> list[tuple[UserProfile, str]]:
    """
    Return 10 (UserProfile, ideal_iso3) pairs for Russian emigrants.
    """
    targets = [iso3 for iso3 in _RU_TARGET_ISO3 if iso3 in df.index]
    return [(
        _make_russian_profile(iso3, df),
        iso3,
    ) for iso3 in targets]


# ── metrics ───────────────────────────────────────────────────────────────────

def _ndcg_at_k(ranked_iso3s: list[str], ideal_iso3: str, k: int = 10) -> float:
    """
    Binary-relevance NDCG@k.

    DCG@k  = 1 / log2(rank + 1)  if ideal found at position `rank` (1-based)
    IDCG@k = 1 / log2(2) = 1.0   (ideal at position 1)
    NDCG@k = DCG@k / IDCG@k
    """
    top_k = ranked_iso3s[:k]
    if ideal_iso3 in top_k:
        rank = top_k.index(ideal_iso3) + 1   # 1-based
        return 1.0 / np.log2(rank + 1)
    return 0.0


def _diversity_at_k(
    ranked_iso3s: list[str],
    df: pd.DataFrame,
    cols: list[str],
    k: int = 10,
) -> float:
    """
    Mean pairwise cosine distance between the feature vectors of the top-k countries.
    Returns 0 if fewer than 2 countries are present.
    """
    top_k  = [iso3 for iso3 in ranked_iso3s[:k] if iso3 in df.index]
    if len(top_k) < 2:
        return 0.0

    available = [c for c in cols if c in df.columns]
    X = df.loc[top_k, available].fillna(0.5).values.astype(float)
    dists = cosine_distances(X)
    # Upper triangle (no diagonal)
    n = len(top_k)
    upper = dists[np.triu_indices(n, k=1)]
    return float(upper.mean()) if len(upper) > 0 else 0.0


# ── evaluation loop ───────────────────────────────────────────────────────────

def evaluate(df: pd.DataFrame, n_profiles: int = 50) -> dict:
    """
    Run the full offline evaluation pipeline.

    Returns a dict with aggregate metrics and per-profile details.
    """
    profiles = _generate_profiles(df, n=n_profiles)
    half     = n_profiles // 2

    # Append Russian-specific profiles
    ru_profiles = _generate_russian_profiles(df)
    profiles.extend(ru_profiles)
    n_total_profiles = len(profiles)

    per_profile: list[dict] = []
    all_recommended: set[str] = set()

    print(f"\nEvaluating {n_total_profiles} synthetic profiles "
          f"({n_profiles} base + {len(ru_profiles)} russian) …\n")
    print(f"  {'#':>3}  {'Target':>6}  {'Profile type':<10}  "
          f"{'Budget':>7}  {'Rank':>6}  {'NDCG@10':>8}  {'Div@10':>7}")
    print("  " + "─" * 62)

    for idx, (profile, ideal_iso3) in enumerate(profiles):
        if idx < half:
            profile_type = "clean"
        elif idx < n_profiles:
            profile_type = "noisy"
        else:
            profile_type = "russian"

        try:
            rec = recommend(profile, df, top_n=10)
        except ValueError:
            # Budget too tight → no candidates; treat as rank=∞
            per_profile.append({
                "idx": idx, "ideal": ideal_iso3, "type": profile_type,
                "budget": profile.budget_usd, "rank": None,
                "ndcg": 0.0, "diversity": 0.0, "in_top10": False,
            })
            print(f"  {idx+1:>3}  {ideal_iso3:>6}  {profile_type:<10}  "
                  f"  {profile.budget_usd:>6.0f}  {'–':>6}  {'0.0000':>8}  {'0.0000':>7}")
            continue

        ranked = list(rec.index)
        all_recommended.update(ranked)

        rank    = ranked.index(ideal_iso3) + 1 if ideal_iso3 in ranked else None
        ndcg    = _ndcg_at_k(ranked, ideal_iso3, k=10)
        div     = _diversity_at_k(ranked, df, _DIVERSITY_COLS, k=10)
        rank_str = str(rank) if rank else "–"

        per_profile.append({
            "idx": idx, "ideal": ideal_iso3, "type": profile_type,
            "budget": profile.budget_usd, "rank": rank,
            "ndcg": round(ndcg, 4), "diversity": round(div, 4),
            "in_top10": rank is not None,
        })

        print(f"  {idx+1:>3}  {ideal_iso3:>6}  {profile_type:<10}  "
              f"  {profile.budget_usd:>6.0f}  {rank_str:>6}  {ndcg:>8.4f}  {div:>7.4f}")

    # ── aggregate ──────────────────────────────────────────────────────────
    df_res   = pd.DataFrame(per_profile)
    n_total  = len(df)
    n_profiles = len(profiles)  # includes Russian profiles

    mean_ndcg     = float(df_res["ndcg"].mean())
    mean_diversity= float(df_res["diversity"].mean())
    coverage      = len(all_recommended) / n_total
    hit_rate      = float(df_res["in_top10"].mean())   # % profiles where ideal in top-10

    # Split by profile type
    clean_ndcg = float(df_res[df_res["type"] == "clean"]["ndcg"].mean())
    noisy_ndcg = float(df_res[df_res["type"] == "noisy"]["ndcg"].mean())
    ru_rows = df_res[df_res["type"] == "russian"]
    ru_ndcg = float(ru_rows["ndcg"].mean()) if len(ru_rows) else 0.0
    ru_hit  = float(ru_rows["in_top10"].mean()) if len(ru_rows) else 0.0

    # Rank distribution among hits
    hits = df_res[df_res["rank"].notna()]["rank"].astype(int)
    rank_dist = hits.value_counts().sort_index().to_dict() if len(hits) else {}

    metrics = {
        "n_profiles":    n_profiles,
        "n_countries":   n_total,
        "ndcg_at_10":    round(mean_ndcg, 4),
        "ndcg_clean":    round(clean_ndcg, 4),
        "ndcg_noisy":    round(noisy_ndcg, 4),
        "ndcg_russian":  round(ru_ndcg, 4),
        "hit_rate":      round(hit_rate, 4),
        "hit_rate_russian": round(ru_hit, 4),
        "coverage":      round(coverage, 4),
        "diversity_at_10": round(mean_diversity, 4),
        "rank_distribution": {str(k): int(v) for k, v in rank_dist.items()},
        "per_profile":   per_profile,
    }
    return metrics


# ── report ────────────────────────────────────────────────────────────────────

def _print_report(m: dict) -> None:
    print("\n")
    width = 56
    print("═" * width)
    print(f"  {'ОТЧЁТ ОБ ОЦЕНКЕ РЕКОМЕНДАТЕЛЬНОЙ СИСТЕМЫ':^{width-4}}")
    print("═" * width)

    rows = [
        ("Профилей",                    m["n_profiles"]),
        ("Стран в датасете",             m["n_countries"]),
        ("Hit Rate (идеал в топ-10)",    f"{m['hit_rate']:.1%}"),
        ("NDCG@10 (все профили)",        f"{m['ndcg_at_10']:.4f}"),
        ("  └ чистые профили",          f"{m['ndcg_clean']:.4f}"),
        ("  └ шумные профили",          f"{m['ndcg_noisy']:.4f}"),
        ("  └ российские профили",     f"{m['ndcg_russian']:.4f}"),
        ("Hit Rate (российские)",       f"{m['hit_rate_russian']:.1%}"),
        ("Coverage (охват датасета)",    f"{m['coverage']:.1%}"),
        ("Diversity@10 (косинус. дист.)",f"{m['diversity_at_10']:.4f}"),
    ]

    for label, value in rows:
        print(f"  {label:<34} {str(value):>14}")

    print("─" * width)
    print("  Распределение по рангу (где идеал найден):")

    rank_dist = m["rank_distribution"]
    for rank_str in sorted(rank_dist, key=int):
        count = rank_dist[rank_str]
        bar   = "█" * count
        print(f"    Rank {rank_str:>2}: {bar:<26} {count}")

    print("─" * width)

    # Best / worst profiles
    df_res = pd.DataFrame(m["per_profile"])
    hits   = df_res[df_res["in_top10"]].sort_values("ndcg", ascending=False)
    misses = df_res[~df_res["in_top10"]]

    if len(hits):
        print(f"\n  Лучшие результаты (топ-5 по NDCG):")
        for _, row in hits.head(5).iterrows():
            print(f"    {row['ideal']:>6}  rank={int(row['rank']):<3}  "
                  f"NDCG={row['ndcg']:.4f}  [{row['type']}]")

    if len(misses):
        print(f"\n  Промахи (идеал не попал в топ-10) — {len(misses)} профилей:")
        missed = misses["ideal"].value_counts()
        for iso3, cnt in missed.head(8).items():
            print(f"    {iso3}  (×{cnt})")

    print("═" * width)


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Loading {PROCESSED_PATH} …")
    df = pd.read_csv(PROCESSED_PATH, index_col="iso3")
    print(f"  {len(df)} countries, {df.shape[1]} columns")

    metrics = evaluate(df, n_profiles=50)
    _print_report(metrics)

    # Save
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Remove per_profile from JSON to keep it readable (add back below)
    out = {k: v for k, v in metrics.items() if k != "per_profile"}
    out["per_profile"] = metrics["per_profile"]
    with open(METRICS_PATH, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(f"\nMetrics saved → {METRICS_PATH}")
