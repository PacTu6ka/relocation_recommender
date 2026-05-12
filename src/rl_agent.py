"""
Contextual-bandit weight optimisation for the relocation recommender.

Idea
----
The recommender derives a feature-weight vector from each user's profile via
`profile_to_weights()`. These weights are reasonable defaults, but different
user *types* benefit from different emphases — e.g. low-budget digital nomads
may want cost-of-living weighted even more, while families may want safety
amplified beyond the default.

We frame this as a contextual bandit:
  • Context  x   — 9-dim user-profile embedding
  • Action   a   — one of K discrete weight-multiplier templates (Δw)
  • Reward   r   — NDCG@10 measured against the user's *known* ideal country

LinUCB
------
Per-arm parameters:  A_a ∈ ℝ^{d×d},  b_a ∈ ℝ^d  (init: I, 0).
Action selection:    â = argmax_a  θ_a·x + α √(x·A_a⁻¹·x),  θ_a = A_a⁻¹ b_a
Update:              A_a ← A_a + xxᵀ,  b_a ← b_a + r·x.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# ── import recommender internals ──────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from recommender import (
    UserProfile,
    _BUDGET_BREAKS,
    _LANG_COUNTRIES,
    _WARM_CLIMATE,
    _available_features,
    _budget_to_max_cost,
    _ideal_vector,
    _invert_lower_is_better,
    profile_to_weights,
)

PROCESSED_PATH = _ROOT / "data" / "processed" / "countries.csv"
HISTORY_PATH   = _ROOT / "data" / "processed" / "rl_training.json"


# ── action space: 8 weight-multiplier templates ───────────────────────────────
_ARMS: list[tuple[str, dict[str, float]]] = [
    ("baseline",        {}),                                                                 # leave defaults
    ("boost_safety",    {"safety_index": 2.5}),
    ("boost_health",    {"healthcare_index": 2.5, "life_expectancy": 1.8}),
    ("boost_cost",      {"cost_of_living_index": 2.5, "purchasing_power_index": 1.5}),
    ("boost_visa",      {"residency_difficulty": 2.5, "digital_nomad_visa": 1.8,
                         "visa_free_count": 1.5}),
    ("boost_eu",        {"eu_member": 3.0, "schengen": 3.0}),
    ("damp_gdp",        {"gdp_per_capita": 0.3, "purchasing_power_index": 0.5}),
    ("boost_clean",     {"pollution_index": 2.5}),
]

# Country pool for synthetic training (geographically diverse)
_TARGETS: list[str] = [
    "DEU", "NLD", "SWE", "NOR", "FIN", "CHE", "AUT", "BEL", "LUX", "DNK",
    "PRT", "ESP", "ITA", "GRC", "MLT", "CYP",
    "POL", "CZE", "HUN", "ROU", "BGR", "EST", "LVA", "LTU", "HRV", "SVN",
    "SRB", "MNE", "GEO", "ARM", "ALB",
    "GBR", "IRL", "CAN", "AUS", "NZL", "USA",
    "MEX", "PAN", "CRI", "BRA", "ARG", "COL", "CHL", "URY", "PER", "ECU",
    "ARE", "ISR", "TUR",
    "JPN", "KOR", "TWN", "SGP", "HKG",
    "THA", "MYS", "IDN", "VNM", "PHL",
    "MAR", "ZAF",
]


# ── helpers ───────────────────────────────────────────────────────────────────

def _context_vector(profile: UserProfile) -> np.ndarray:
    """9-dim embedding of a UserProfile (bias term included)."""
    return np.array([
        profile.budget_usd / 10_000.0,
        profile.safety_priority    / 5.0,
        float(profile.climate_warm),
        profile.visa_easy_priority / 5.0,
        profile.healthcare_priority / 5.0,
        float("english" in profile.language),
        float(any(l in profile.language for l in ("spanish", "portuguese", "italian", "french"))),
        float(any(l in profile.language for l in ("german", "russian", "turkish"))),
        1.0,                                # bias
    ], dtype=float)


def _apply_arm(base_weights: dict[str, float], arm_idx: int) -> dict[str, float]:
    """Multiply base weights by the arm's adjustment template."""
    _, mult = _ARMS[arm_idx]
    return {k: v * mult.get(k, 1.0) for k, v in base_weights.items()}


def _rank_with_weights(
    profile: UserProfile,
    df: pd.DataFrame,
    weights: dict[str, float],
    top_n: int = 10,
) -> list[str]:
    """Same scoring pipeline as recommend() but with caller-supplied weights."""
    cols = _available_features(df)
    if not cols:
        return []

    max_cost = _budget_to_max_cost(profile.budget_usd)
    candidates = (
        df[df["cost_of_living_index"] <= max_cost]
        if "cost_of_living_index" in df.columns else df
    )
    if candidates.empty:
        return []

    feat_df = _invert_lower_is_better(candidates, cols).fillna(0.5)
    X       = feat_df.values
    ideal   = _ideal_vector(profile, cols).reshape(1, -1)
    W       = np.array([weights.get(c, 1.0) for c in cols])

    sims  = cosine_similarity(ideal * W, X * W)[0]
    iso3s = list(candidates.index)
    ranked = [iso3 for _, iso3 in sorted(zip(sims, iso3s), key=lambda p: -p[0])]
    return ranked[:top_n]


def _ndcg_at_10(ranked: list[str], ideal_iso3: str) -> float:
    top10 = ranked[:10]
    if ideal_iso3 in top10:
        rank = top10.index(ideal_iso3) + 1
        return 1.0 / np.log2(rank + 1)
    return 0.0


def _make_random_profile(
    df: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[UserProfile, str]:
    """Sample (profile, ideal_iso3) where profile is biased toward ideal_iso3."""
    targets = [c for c in _TARGETS if c in df.index]
    iso3    = rng.choice(targets)
    row     = df.loc[iso3]

    def _f(col: str) -> float:
        v = row.get(col, 0.5)
        return float(v) if pd.notna(v) else 0.5

    # Budget high enough to pass the hard filter, with ±20% jitter
    cost = _f("cost_of_living_index")
    base_budget = next(
        (b for b, c in _BUDGET_BREAKS if cost <= c), 10_000
    )
    budget = float(np.clip(base_budget * rng.uniform(0.9, 1.3), 500, 10_000))

    def _pri(val: float) -> int:
        return int(np.clip(round(val * 5 + rng.integers(-1, 2)), 1, 5))

    profile = UserProfile(
        budget_usd          = budget,
        safety_priority     = _pri(_f("safety_index")),
        climate_warm        = iso3 in _WARM_CLIMATE,
        visa_easy_priority  = _pri(1.0 - _f("residency_difficulty")),
        healthcare_priority = _pri(_f("healthcare_index")),
        language            = [
            lang for lang, c in _LANG_COUNTRIES.items() if iso3 in c
        ][:2],
    )
    return profile, str(iso3)


# ── LinUCB (≈ 50 lines) ───────────────────────────────────────────────────────

class LinUCB:
    """
    Disjoint LinUCB (one independent linear model per arm).

    Parameters
    ----------
    n_arms      : number of discrete actions
    context_dim : dimensionality of context vector x
    alpha       : exploration coefficient (higher → more exploration)
    """

    def __init__(self, n_arms: int, context_dim: int, alpha: float = 0.6):
        self.K     = n_arms
        self.d     = context_dim
        self.alpha = alpha
        # Per-arm precision matrix A and reward-feature vector b
        self.A     = [np.eye(context_dim) for _ in range(n_arms)]
        self.b     = [np.zeros(context_dim) for _ in range(n_arms)]
        # Cache of A_inv (recomputed on every update for simplicity)
        self.A_inv = [np.eye(context_dim) for _ in range(n_arms)]
        self.counts = np.zeros(n_arms, dtype=int)

    def select(self, x: np.ndarray) -> int:
        """Pick the arm with the highest upper-confidence-bound payoff."""
        ucb = np.empty(self.K)
        for a in range(self.K):
            theta = self.A_inv[a] @ self.b[a]
            expected = float(theta @ x)
            variance = float(x @ self.A_inv[a] @ x)
            ucb[a]   = expected + self.alpha * np.sqrt(max(variance, 1e-9))
        return int(np.argmax(ucb))

    def update(self, arm: int, x: np.ndarray, reward: float) -> None:
        """Bayesian-style ridge update with the observed (x, r) pair."""
        self.A[arm]    += np.outer(x, x)
        self.b[arm]    += reward * x
        # Recompute the inverse for the touched arm only
        self.A_inv[arm] = np.linalg.inv(self.A[arm])
        self.counts[arm] += 1

    def greedy(self, x: np.ndarray) -> int:
        """Exploitation-only action (no UCB bonus); used at evaluation time."""
        return int(np.argmax([
            float((self.A_inv[a] @ self.b[a]) @ x) for a in range(self.K)
        ]))


# ── training & evaluation ─────────────────────────────────────────────────────

def train(
    df: pd.DataFrame,
    n_iters: int = 1000,
    alpha: float = 0.6,
    seed: int = 42,
) -> tuple[LinUCB, list[dict]]:
    """Online training loop. Returns trained agent + per-iteration history."""
    rng   = np.random.default_rng(seed)
    agent = LinUCB(n_arms=len(_ARMS), context_dim=9, alpha=alpha)

    history: list[dict] = []
    rolling_reward = []
    rolling_baseline = []

    print(f"\nTraining LinUCB for {n_iters} iterations (α={alpha}) …")
    print(f"  {'iter':>5}  {'arm':<12}  {'reward':>7}  "
          f"{'rolling agent':>13}  {'rolling baseline':>16}")
    print("  " + "─" * 66)

    for t in range(n_iters):
        profile, ideal = _make_random_profile(df, rng)
        x              = _context_vector(profile)
        base_weights   = profile_to_weights(profile)

        # Agent action
        arm     = agent.select(x)
        weights = _apply_arm(base_weights, arm)
        ranked  = _rank_with_weights(profile, df, weights, top_n=10)
        reward  = _ndcg_at_10(ranked, ideal)
        agent.update(arm, x, reward)

        # Baseline for comparison (no weight tweak)
        base_ranked = _rank_with_weights(profile, df, base_weights, top_n=10)
        base_reward = _ndcg_at_10(base_ranked, ideal)

        rolling_reward.append(reward)
        rolling_baseline.append(base_reward)

        history.append({
            "iter": t, "ideal": ideal, "arm": _ARMS[arm][0],
            "reward": round(reward, 4), "baseline": round(base_reward, 4),
        })

        if (t + 1) % 100 == 0:
            ra = np.mean(rolling_reward[-100:])
            rb = np.mean(rolling_baseline[-100:])
            print(f"  {t+1:>5}  {_ARMS[arm][0]:<12}  {reward:>7.4f}  "
                  f"{ra:>13.4f}  {rb:>16.4f}")

    print("\n  Arm pull counts:")
    for i, (name, _) in enumerate(_ARMS):
        bar = "█" * int(agent.counts[i] / max(agent.counts.max(), 1) * 30)
        print(f"    {i}  {name:<12} {bar:<30}  {agent.counts[i]:>4}")

    return agent, history


def evaluate(
    df: pd.DataFrame,
    agent: LinUCB,
    n_test: int = 200,
    seed: int = 99,
) -> dict:
    """Compare agent vs baseline weights on a held-out batch of users."""
    rng = np.random.default_rng(seed)

    ndcg_base:  list[float] = []
    ndcg_agent: list[float] = []
    arm_hist:   list[str]   = []

    for _ in range(n_test):
        profile, ideal = _make_random_profile(df, rng)
        x              = _context_vector(profile)
        base_weights   = profile_to_weights(profile)

        # Baseline
        ndcg_base.append(_ndcg_at_10(
            _rank_with_weights(profile, df, base_weights), ideal
        ))

        # Agent (greedy — no UCB exploration)
        arm     = agent.greedy(x)
        arm_hist.append(_ARMS[arm][0])
        weights = _apply_arm(base_weights, arm)
        ndcg_agent.append(_ndcg_at_10(
            _rank_with_weights(profile, df, weights), ideal
        ))

    base_mean    = float(np.mean(ndcg_base))
    agent_mean   = float(np.mean(ndcg_agent))
    improvement  = (agent_mean - base_mean) / max(base_mean, 1e-9) * 100

    # Win/lose/tie counts
    wins   = sum(1 for a, b in zip(ndcg_agent, ndcg_base) if a > b)
    losses = sum(1 for a, b in zip(ndcg_agent, ndcg_base) if a < b)
    ties   = n_test - wins - losses

    # Arm usage distribution at test time
    arm_dist: dict[str, int] = {}
    for name in arm_hist:
        arm_dist[name] = arm_dist.get(name, 0) + 1

    return {
        "n_test":         n_test,
        "baseline_ndcg":  round(base_mean, 4),
        "agent_ndcg":     round(agent_mean, 4),
        "improvement_pct":round(improvement, 2),
        "wins":           wins,
        "losses":         losses,
        "ties":           ties,
        "arm_distribution": arm_dist,
    }


# ── report ────────────────────────────────────────────────────────────────────

def _print_evaluation(m: dict) -> None:
    width = 56
    print("\n" + "═" * width)
    print(f"  {'СРАВНЕНИЕ: BASELINE vs LinUCB AGENT':^{width-4}}")
    print("═" * width)
    print(f"  Тестовых профилей           : {m['n_test']}")
    print(f"  NDCG@10 baseline            : {m['baseline_ndcg']:.4f}")
    print(f"  NDCG@10 LinUCB agent        : {m['agent_ndcg']:.4f}")
    print(f"  Δ улучшение                 : {m['improvement_pct']:+.2f}%")
    print("─" * width)
    print(f"  Поарёмные исходы:")
    print(f"    Wins  (agent > base) : {m['wins']:>4} ({m['wins']/m['n_test']:.0%})")
    print(f"    Losses(agent < base) : {m['losses']:>4} ({m['losses']/m['n_test']:.0%})")
    print(f"    Ties  (agent = base) : {m['ties']:>4} ({m['ties']/m['n_test']:.0%})")
    print("─" * width)
    print("  Использование рук агентом на тесте:")
    for name, count in sorted(m["arm_distribution"].items(), key=lambda x: -x[1]):
        bar = "█" * int(count / m["n_test"] * 30)
        pct = count / m["n_test"] * 100
        print(f"    {name:<14} {bar:<30}  {count:>3} ({pct:>5.1f}%)")
    print("═" * width)


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Loading {PROCESSED_PATH} …")
    df = pd.read_csv(PROCESSED_PATH, index_col="iso3")
    print(f"  {len(df)} countries, {df.shape[1]} columns")

    agent, history = train(df, n_iters=1000, alpha=0.6, seed=42)
    metrics        = evaluate(df, agent, n_test=200, seed=99)
    _print_evaluation(metrics)

    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_PATH, "w", encoding="utf-8") as fh:
        json.dump({
            "arms":     [name for name, _ in _ARMS],
            "history":  history,
            "evaluation": metrics,
            "arm_pulls": {name: int(c) for (name, _), c in zip(_ARMS, agent.counts)},
        }, fh, ensure_ascii=False, indent=2)
    print(f"\nTraining log saved → {HISTORY_PATH}")
