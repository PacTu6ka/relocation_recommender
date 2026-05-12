"""
Recommendation engine for relocation_recommender.

Scoring pipeline
----------------
1. Hard filter  — drop countries whose cost_of_living_index exceeds the
                  user's monthly budget ceiling.
2. Feature prep — invert "lower-is-better" features so all axes point
                  "higher = better".
3. Ideal vector — build a desired-country vector from the user profile.
4. Weights      — scale each dimension by the user's priorities.
5. Cosine sim   — similarity between weighted ideal vector and each country.
6. Bonuses      — +0.10 for the best-matching cluster, +0.05 for language
                  match, +0.03 for warm-climate match when requested.
7. Return top-N ranked rows with a per-criterion breakdown.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

PROCESSED_PATH = Path(__file__).parent.parent / "data" / "processed" / "countries.csv"

# ── domain knowledge ──────────────────────────────────────────────────────────

# ISO3 codes with reliably warm / sunny climate
_WARM_CLIMATE: set[str] = {
    "THA", "MYS", "IDN", "VNM", "PHL",          # SE Asia
    "ARE", "ISR",                                 # Middle East
    "PRT", "ESP", "ITA", "GRC", "MLT", "CYP",   # Southern Europe
    "MEX", "PAN", "CRI", "COL", "BRA", "ECU",   # Latin America
    "MAR", "ZAF",                                 # Africa
    "GEO", "ARM", "TUR",                          # Caucasus / Turkey
}

# Language → set of ISO3 countries where that language is widely used / official
_LANG_COUNTRIES: dict[str, set[str]] = {
    "english":    {"USA", "GBR", "CAN", "AUS", "NZL", "IRL", "SGP",
                   "PHL", "MLT", "ZAF", "HKG"},
    "spanish":    {"ESP", "MEX", "ARG", "COL", "CHL", "PAN", "CRI",
                   "PER", "ECU", "URY", "PRY"},
    "portuguese": {"PRT", "BRA"},
    "french":     {"FRA", "BEL", "LUX", "CHE"},
    "german":     {"DEU", "AUT", "CHE"},
    "italian":    {"ITA", "CHE"},
    "russian":    {"GEO", "ARM", "EST", "LVA", "LTU"},   # widely understood
    "arabic":     {"ARE", "MAR", "ISR"},
    "thai":       {"THA"},
    "malay":      {"MYS", "IDN"},
    "japanese":   {"JPN"},
    "korean":     {"KOR"},
    "chinese":    {"CHN", "TWN", "SGP", "HKG"},
    "turkish":    {"TUR"},
    "georgian":   {"GEO"},
    "armenian":   {"ARM"},
    "romanian":   {"ROU", "MDA"},
    "serbian":    {"SRB", "MNE", "BIH"},
    "albanian":   {"ALB", "MNE"},
}

# Features where a lower raw value is better (will be inverted: 1 - x)
_LOWER_IS_BETTER: set[str] = {
    "residency_difficulty",
    "cost_of_living_index",
    "pollution_index",
    "unemployment_rate",
    "gini_index",
    "traffic_index",
    "ru_sanctions_risk",
}

# Mapping monthly budget USD → maximum acceptable cost_of_living_index (0-1 scale)
# Thresholds are approximate and assume Numbeo index normalised across ~100 countries.
_BUDGET_BREAKS: list[tuple[float, float]] = [
    (1_500,  0.28),
    (2_500,  0.42),
    (3_500,  0.57),
    (5_000,  0.72),
    (8_000,  0.88),
    (float("inf"), 1.00),
]

# Features included in scoring (must exist after preprocessing)
_SCORE_FEATURES: list[str] = [
    "safety_index",
    "healthcare_index",
    "life_expectancy",
    "gdp_per_capita",
    "purchasing_power_index",
    "cost_of_living_index",
    "residency_difficulty",
    "digital_nomad_visa",
    "visa_free_count",
    "pollution_index",
    "eu_member",
    "schengen",
    "ru_banking_access",
    "ru_sanctions_risk",
]

# Human-readable labels for the breakdown output
_FEATURE_LABELS: dict[str, str] = {
    "safety_index":          "Безопасность",
    "healthcare_index":      "Здравоохранение",
    "life_expectancy":       "Продолжительность жизни",
    "gdp_per_capita":        "ВВП на душу",
    "purchasing_power_index":"Покупательная способность",
    "cost_of_living_index":  "Стоимость жизни",
    "residency_difficulty":  "Простота эмиграции",
    "digital_nomad_visa":    "Виза цифрового кочевника",
    "visa_free_count":       "Безвизовых стран у паспорта",
    "pollution_index":       "Загрязнение (ниже = лучше)",
    "eu_member":             "Член ЕС",
    "schengen":              "Шенген",
    "ru_banking_access":     "Банкинг для россиян",
    "ru_sanctions_risk":     "Санкционный риск (ниже = лучше)",
}


# ── data classes ──────────────────────────────────────────────────────────────

@dataclass
class UserProfile:
    """Represents a relocation seeker's preferences and constraints."""
    budget_usd:          float        # monthly disposable budget in USD
    safety_priority:     int          # 1 (low) … 5 (critical)
    climate_warm:        bool         # prefers warm / sunny climate
    visa_easy_priority:  int          # 1 … 5; importance of easy residency
    healthcare_priority: int          # 1 … 5
    language:            list[str] = field(default_factory=list)  # known languages (lowercase)
    # ── Russia-centric fields ────────────────────────────────────────────
    user_is_russian:        bool = False   # enable Russian-passport bonuses/penalties
    needs_banking_access:   bool = False   # prioritise countries where Russians can open accounts
    has_eu_visa:            bool = False   # already holds a Schengen / national EU visa

    def __post_init__(self) -> None:
        for attr in ("safety_priority", "visa_easy_priority", "healthcare_priority"):
            v = getattr(self, attr)
            if not (1 <= v <= 5):
                raise ValueError(f"{attr} must be between 1 and 5, got {v}")
        self.language = [lang.lower().strip() for lang in self.language]


# ── helpers ───────────────────────────────────────────────────────────────────

def _budget_to_max_cost(budget: float) -> float:
    """Map monthly USD budget to max acceptable cost_of_living_index."""
    for threshold, max_idx in _BUDGET_BREAKS:
        if budget <= threshold:
            return max_idx
    return 1.0


def _available_features(df: pd.DataFrame) -> list[str]:
    """Return _SCORE_FEATURES present in df."""
    return [c for c in _SCORE_FEATURES if c in df.columns]


def _invert_lower_is_better(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Return a copy of df where lower-is-better columns are replaced by (1 - x)."""
    df2 = df[cols].copy()
    for c in cols:
        if c in _LOWER_IS_BETTER:
            df2[c] = 1.0 - df2[c]
    return df2


def profile_to_weights(profile: UserProfile) -> dict[str, float]:
    """
    Convert a UserProfile into a feature-weight dictionary.

    Weights are non-negative floats; higher means the dimension contributes
    more to the cosine similarity.  Budget is expressed as a derived cost
    weight: cheaper budget → higher cost-feature weight.
    """
    s   = profile.safety_priority     / 5.0
    h   = profile.healthcare_priority / 5.0
    v   = profile.visa_easy_priority  / 5.0

    # Budget urgency: tighter budget → more weight on cost index
    budget_urgency = 1.0 - (_budget_to_max_cost(profile.budget_usd) ** 0.5)
    budget_urgency = max(0.2, min(1.0, budget_urgency + 0.2))

    weights: dict[str, float] = {
        "safety_index":          round(s * 2.5,  3),
        "healthcare_index":      round(h * 2.0,  3),
        "life_expectancy":       0.6,
        "gdp_per_capita":        0.5,
        "purchasing_power_index":0.6,
        "cost_of_living_index":  round(budget_urgency * 2.0, 3),
        "residency_difficulty":  round(v * 1.8,  3),
        "digital_nomad_visa":    round(v * 0.9,  3),
        "visa_free_count":       round(v * 0.7,  3),
        "pollution_index":       0.4,
        "eu_member":             round(v * 0.5,  3),
        "schengen":              round(v * 0.4,  3),
        # Russian-specific weights — only meaningful when user_is_russian=True
        "ru_banking_access":     round(1.5 if profile.needs_banking_access else 0.3, 3),
        "ru_sanctions_risk":     round(1.8 if profile.user_is_russian else 0.0, 3),
    }
    return weights


def _ideal_vector(profile: UserProfile, cols: list[str]) -> np.ndarray:
    """
    Build the user's ideal country as a 1-D array aligned to `cols`.
    All values are on a [0, 1] "higher = better" scale (after inversion).
    """
    # Start with a neutral 0.5 baseline
    ideal: dict[str, float] = {c: 0.5 for c in cols}

    # Safety and healthcare → user's priority as fraction of max
    ideal["safety_index"]     = profile.safety_priority / 5.0
    ideal["healthcare_index"] = profile.healthcare_priority / 5.0
    ideal["life_expectancy"]  = 0.8        # always somewhat desirable

    # GDP / purchasing power — want high, weighted by budget
    gdp_desire = min(1.0, profile.budget_usd / 5000.0)
    ideal["gdp_per_capita"]        = gdp_desire
    ideal["purchasing_power_index"] = gdp_desire

    # Cost — lower budget → want cheaper country (after inversion: higher ideal)
    max_cost = _budget_to_max_cost(profile.budget_usd)
    ideal["cost_of_living_index"] = 1.0 - max_cost   # inverted: want low cost

    # Visa / residency — want easy (after inversion: high)
    ideal["residency_difficulty"] = profile.visa_easy_priority / 5.0
    ideal["digital_nomad_visa"]   = 1.0 if profile.visa_easy_priority >= 3 else 0.0
    ideal["visa_free_count"]      = profile.visa_easy_priority / 5.0

    # Pollution — always want low (after inversion: high ideal)
    ideal["pollution_index"] = 0.75

    # EU / Schengen — valuable when visa_easy >= 3
    ideal["eu_member"] = 1.0 if profile.visa_easy_priority >= 3 else 0.3
    ideal["schengen"]  = 1.0 if profile.visa_easy_priority >= 3 else 0.3

    # Russian-specific ideal values
    ideal["ru_banking_access"] = 1.0 if profile.needs_banking_access else 0.5
    ideal["ru_sanctions_risk"] = 1.0 if profile.user_is_russian else 0.5  # inverted: want LOW risk → high after inversion

    return np.array([ideal.get(c, 0.5) for c in cols], dtype=float)


def _find_best_cluster(
    profile: UserProfile,
    df: pd.DataFrame,
    cols: list[str],
) -> int | None:
    """
    Return the cluster id whose centroid is closest to the user's ideal vector.
    Returns None if no cluster column is present.
    """
    cluster_col = next(
        (c for c in ("cluster_label", "cluster") if c in df.columns), None
    )
    if cluster_col is None:
        return None

    df_inv  = _invert_lower_is_better(df, cols).fillna(0.5)
    ideal   = _ideal_vector(profile, cols).reshape(1, -1)
    weights = profile_to_weights(profile)
    W       = np.array([weights.get(c, 1.0) for c in cols])

    best_cluster, best_sim = None, -1.0
    for cid, grp in df_inv.groupby(df[cluster_col]):
        centroid = grp.mean().values.reshape(1, -1)
        sim = cosine_similarity(ideal * W, centroid * W)[0, 0]
        if sim > best_sim:
            best_sim, best_cluster = sim, cid

    return best_cluster


# ── public API ────────────────────────────────────────────────────────────────

def recommend(
    profile: UserProfile,
    df: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Rank countries by fit to a UserProfile.

    Parameters
    ----------
    profile : UserProfile
    df      : Processed countries DataFrame (iso3-indexed).
    top_n   : Number of results to return.

    Returns
    -------
    pd.DataFrame with columns: country, score, cluster_label (if present),
    + one column per scoring feature showing the inverted value used.
    """
    cols = _available_features(df)
    if not cols:
        raise ValueError("No scoring features found in DataFrame.")

    # ── 1. hard budget filter ─────────────────────────────────────────────
    max_cost = _budget_to_max_cost(profile.budget_usd)
    if "cost_of_living_index" in df.columns:
        candidates = df[df["cost_of_living_index"] <= max_cost].copy()
    else:
        candidates = df.copy()

    if candidates.empty:
        raise ValueError(
            f"No countries within budget ${profile.budget_usd:,.0f}/month "
            f"(max cost index {max_cost:.2f}). Try raising budget."
        )

    # ── 2. invert lower-is-better features ───────────────────────────────
    feat_df  = _invert_lower_is_better(candidates, cols).fillna(0.5)
    X        = feat_df.values                             # shape (n, d)

    # ── 3. ideal vector + weights ─────────────────────────────────────────
    ideal   = _ideal_vector(profile, cols).reshape(1, -1) # shape (1, d)
    weights = profile_to_weights(profile)
    W       = np.array([weights.get(c, 1.0) for c in cols])

    X_w     = X     * W
    ideal_w = ideal * W

    # ── 4. cosine similarity ──────────────────────────────────────────────
    scores = cosine_similarity(ideal_w, X_w)[0]           # shape (n,)

    # ── 5. cluster bonus ──────────────────────────────────────────────────
    cluster_col = next(
        (c for c in ("cluster_label", "cluster") if c in candidates.columns), None
    )
    bonus = np.zeros(len(candidates))
    if cluster_col is not None:
        best_cluster = _find_best_cluster(profile, df, cols)
        bonus = np.where(candidates[cluster_col] == best_cluster, 0.10, 0.0)

    # ── 6. climate & language bonuses ─────────────────────────────────────
    iso3s = np.array(candidates.index)

    climate_bonus = np.array([
        0.03 if (profile.climate_warm and iso3 in _WARM_CLIMATE) else 0.0
        for iso3 in iso3s
    ])

    lang_iso3s: set[str] = set()
    for lang in profile.language:
        lang_iso3s |= _LANG_COUNTRIES.get(lang, set())
    language_bonus = np.array([
        0.05 if iso3 in lang_iso3s else 0.0
        for iso3 in iso3s
    ])

    # ── 6b. Russian-passport bonuses ─────────────────────────────────
    ru_visa_bonus = np.zeros(len(candidates))
    ru_sanctions_penalty = np.zeros(len(candidates))
    ru_banking_bonus = np.zeros(len(candidates))

    if profile.user_is_russian:
        # Bonus for visa-free entry with Russian passport
        if "ru_visa_free" in candidates.columns:
            ru_visa_bonus = np.where(
                candidates["ru_visa_free"].astype(float) > 0.5, 0.08, 0.0
            )
            # Extra penalty if no EU visa and country requires Schengen
            if not profile.has_eu_visa and "schengen" in candidates.columns:
                eu_visa_penalty = np.where(
                    (candidates["schengen"].astype(float) > 0.5) &
                    (candidates["ru_visa_free"].astype(float) < 0.5),
                    -0.04, 0.0
                )
                ru_visa_bonus = ru_visa_bonus + eu_visa_penalty

        # Sanctions penalty — scaled by ru_sanctions_risk (already 0-1 after MinMax)
        if "ru_sanctions_risk" in candidates.columns:
            # Higher risk → bigger penalty (risk is raw 0-1, higher=worse)
            ru_sanctions_penalty = -0.06 * candidates["ru_sanctions_risk"].astype(float).values

        # Banking access bonus
        if profile.needs_banking_access and "ru_banking_access" in candidates.columns:
            ru_banking_bonus = 0.04 * candidates["ru_banking_access"].astype(float).values

    final_scores = (scores + bonus + climate_bonus + language_bonus
                    + ru_visa_bonus + ru_sanctions_penalty + ru_banking_bonus)

    # ── 7. assemble result ────────────────────────────────────────────────
    result = candidates[["country"] if "country" in candidates.columns else []].copy()
    result["score"]          = np.round(final_scores, 4)
    result["base_sim"]       = np.round(scores, 4)
    result["cluster_bonus"]  = bonus.round(3)
    result["climate_bonus"]  = climate_bonus.round(3)
    result["language_bonus"] = language_bonus.round(3)
    result["ru_visa_bonus"]       = np.round(ru_visa_bonus, 3)
    result["ru_sanctions_penalty"]= np.round(ru_sanctions_penalty, 3)
    result["ru_banking_bonus"]    = np.round(ru_banking_bonus, 3)

    # Per-feature inverted values (for breakdown / explain)
    for c in cols:
        result[f"_f_{c}"] = feat_df[c].values.round(4)

    if cluster_col in candidates.columns:
        result["cluster_label"] = candidates[cluster_col]

    return result.sort_values("score", ascending=False).head(top_n)


def explain(
    country_iso3: str,
    profile: UserProfile,
    df: pd.DataFrame,
    rec_df: pd.DataFrame | None = None,
) -> str:
    """
    Return a human-readable Russian explanation of why a country was recommended.

    Parameters
    ----------
    country_iso3 : ISO3 code, e.g. "PRT".
    profile      : The user profile used for recommendation.
    df           : Full processed countries DataFrame.
    rec_df       : Optional pre-computed recommend() result; avoids recomputing.
    """
    if rec_df is None:
        rec_df = recommend(profile, df, top_n=len(df))

    if country_iso3 not in rec_df.index:
        return f"Страна {country_iso3} не прошла фильтры (скорее всего, бюджет)."

    row    = rec_df.loc[country_iso3]
    raw    = df.loc[country_iso3] if country_iso3 in df.index else pd.Series(dtype=float)
    cols   = _available_features(df)
    weights = profile_to_weights(profile)

    cname  = raw.get("country", country_iso3)
    score  = row["score"]
    rank   = int((rec_df["score"] > score).sum()) + 1

    lines: list[str] = [
        f"{'═' * 56}",
        f"  {cname} ({country_iso3})  —  итоговый score: {score:.4f}  (место #{rank})",
        f"{'═' * 56}",
        "",
        "  Составляющие оценки:",
        f"    Базовое косинусное сходство : {row['base_sim']:.4f}",
        f"    Бонус за кластер            : {row['cluster_bonus']:.3f}",
        f"    Бонус за климат             : {row['climate_bonus']:.3f}",
        f"    Бонус за язык               : {row['language_bonus']:.3f}",
        f"    Безвиз РФ                   : {row.get('ru_visa_bonus', 0):.3f}",
        f"    Санкционный штраф            : {row.get('ru_sanctions_penalty', 0):.3f}",
        f"    Бонус за банкинг (РФ)        : {row.get('ru_banking_bonus', 0):.3f}",
        "",
        "  По критериям (0 = плохо для вас, 1 = идеально):",
    ]

    # Build per-feature breakdown sorted by weighted contribution
    contributions: list[tuple[float, str, float, float]] = []
    for c in cols:
        inv_val = row.get(f"_f_{c}", np.nan)
        w       = weights.get(c, 0.0)
        contrib = float(inv_val) * w if pd.notna(inv_val) else 0.0
        label   = _FEATURE_LABELS.get(c, c)
        raw_val = raw.get(c, np.nan)
        contributions.append((contrib, label, float(inv_val) if pd.notna(inv_val) else 0.0, raw_val))

    contributions.sort(reverse=True)

    for contrib, label, inv_val, raw_val in contributions:
        bar     = "█" * int(inv_val * 10) + "░" * (10 - int(inv_val * 10))
        raw_str = f"{raw_val:.3f}" if pd.notna(raw_val) and isinstance(raw_val, float) else str(raw_val)
        lines.append(f"    {label:<30} [{bar}]  {inv_val:.2f}  (raw={raw_str})")

    # Climate compatibility
    warm_tag = "тёплый" if country_iso3 in _WARM_CLIMATE else "умеренный/холодный"
    lines += [
        "",
        f"  Климат: {warm_tag}"
        + (" ✓ соответствует вашему предпочтению" if (
            profile.climate_warm and country_iso3 in _WARM_CLIMATE
        ) else ""),
    ]

    # Language compatibility
    spoken: list[str] = []
    for lang in profile.language:
        if country_iso3 in _LANG_COUNTRIES.get(lang, set()):
            spoken.append(lang)
    if spoken:
        lines.append(f"  Языки: {', '.join(spoken)} — знакомы вам ✓")
    else:
        lines.append(f"  Языки: знакомых вам языков не обнаружено")

    # Cluster
    if "cluster_label" in row.index:
        lines.append(f"  Кластер: {row['cluster_label']}")

    # Pros / cons summary
    pros  = [_FEATURE_LABELS.get(c, c) for _, c, v, _ in
             sorted(contributions, reverse=True)[:3] if v >= 0.65]
    cons  = [_FEATURE_LABELS.get(c, c) for _, c, v, _ in
             sorted(contributions)[:3] if v <= 0.35]

    lines += ["", "  Плюсы:"]
    lines += [f"    + {p}" for p in pros] if pros else ["    (нет явных плюсов по приоритетам)"]
    lines += ["  Минусы:"]
    lines += [f"    - {c}" for c in cons] if cons else ["    (нет явных минусов)"]
    lines.append("═" * 56)

    return "\n".join(lines)


def _print_recommendations(
    rec: pd.DataFrame,
    profile: UserProfile,
    profile_name: str,
) -> None:
    """Pretty-print recommendation table to stdout."""
    sep = "─" * 72
    print(f"\n{'═' * 72}")
    print(f"  Профиль: {profile_name}")
    print(f"  Бюджет: ${profile.budget_usd:,.0f}/мес | "
          f"Безопасность: {profile.safety_priority}/5 | "
          f"Здравоохр.: {profile.healthcare_priority}/5 | "
          f"Виза: {profile.visa_easy_priority}/5 | "
          f"Климат: {'тёплый' if profile.climate_warm else 'любой'} | "
          f"Языки: {', '.join(profile.language) or '—'}")
    print("═" * 72)
    print(f"  {'#':<3} {'ISO3':<6} {'Страна':<26} {'Score':>7} "
          f"{'Клим':>5} {'Яз':>5} {'Клас':>5}")
    print(sep)

    country_col = "country" in rec.columns
    for rank, (iso3, row) in enumerate(rec.iterrows(), start=1):
        name    = row["country"][:24] if country_col else iso3
        climate = "✓" if iso3 in _WARM_CLIMATE and profile.climate_warm else ""
        lang    = "✓" if any(iso3 in _LANG_COUNTRIES.get(l, set()) for l in profile.language) else ""
        cluster = str(row.get("cluster_label", row.get("cluster", "")))[:18]
        print(f"  {rank:<3} {iso3:<6} {name:<26} {row['score']:>7.4f} "
              f"{climate:>5} {lang:>5}   {cluster}")

    print(sep)


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Loading {PROCESSED_PATH} …")
    df = pd.read_csv(PROCESSED_PATH, index_col="iso3")
    print(f"  {len(df)} countries, {df.shape[1]} columns\n")

    # ── Profile 1: Remote worker, tight budget, warm climate, SE Asia / LatAm
    p1 = UserProfile(
        budget_usd=2_000,
        safety_priority=3,
        climate_warm=True,
        visa_easy_priority=5,
        healthcare_priority=2,
        language=["english", "spanish"],
    )

    # ── Profile 2: Family, comfortable budget, Europe, healthcare first
    p2 = UserProfile(
        budget_usd=5_000,
        safety_priority=5,
        climate_warm=False,
        visa_easy_priority=3,
        healthcare_priority=5,
        language=["english", "german"],
    )

    # ── Profile 3: Retiree, medium budget, safety & healthcare, English-speaking
    p3 = UserProfile(
        budget_usd=3_500,
        safety_priority=5,
        climate_warm=True,
        visa_easy_priority=4,
        healthcare_priority=4,
        language=["english"],
    )

    profiles = [
        (p1, "Цифровой кочевник (бюджет $2k, тёплый климат, испанский/английский)"),
        (p2, "Семья с детьми (бюджет $5k, Европа, приоритет — здравоохранение)"),
        (p3, "Пенсионер (бюджет $3.5k, безопасность, английский язык)"),
    ]

    all_recs: list[tuple[pd.DataFrame, UserProfile, str]] = []
    for profile, name in profiles:
        weights = profile_to_weights(profile)
        print(f"\nВеса признаков — {name}:")
        for feat, w in sorted(weights.items(), key=lambda x: -x[1]):
            bar = "▓" * int(w * 5)
            print(f"  {feat:<30} {w:>5.2f}  {bar}")

        rec = recommend(profile, df, top_n=10)
        _print_recommendations(rec, profile, name)
        all_recs.append((rec, profile, name))

    # ── Detailed explanation for top-1 of each profile ───────────────────
    print("\n\n" + "═" * 56)
    print("  ДЕТАЛЬНЫЕ ОБЪЯСНЕНИЯ (топ-1 по каждому профилю)")
    print("═" * 56)
    for rec, profile, name in all_recs:
        top_iso3 = rec.index[0]
        print(f"\n>>> Профиль: {name}")
        print(explain(top_iso3, profile, df, rec_df=rec))
