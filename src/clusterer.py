import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import umap
from hdbscan import HDBSCAN
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

PROCESSED_PATH = Path(__file__).parent.parent / "data" / "processed" / "countries.csv"
FIGURES_DIR    = Path(__file__).parent.parent / "notebooks" / "figures"

# Columns that are metadata, not features
_META_COLS = {"country", "cluster", "cluster_hdbscan", "cluster_label", "umap_x", "umap_y"}

# Boolean feature columns (already 0/1 — kept as features but not re-scaled)
_BOOL_COLS = {"digital_nomad_visa", "investor_visa", "eu_member", "schengen"}

# ISO3 → macro-region (used by label_clusters for geographic inference)
_ISO3_REGION: dict[str, str] = {
    **dict.fromkeys(
        ["DEU", "FRA", "NLD", "BEL", "AUT", "CHE", "LUX",
         "SWE", "NOR", "DNK", "FIN", "ISL"],
        "Западная/Северная Европа"),
    **dict.fromkeys(
        ["GBR", "IRL", "PRI"],           # Puerto Rico → anglosphere
        "Британские острова"),
    **dict.fromkeys(
        ["PRT", "ESP", "ITA", "GRC", "MLT", "CYP"],
        "Южная Европа"),
    **dict.fromkeys(
        ["POL", "CZE", "SVK", "HUN", "ROU", "BGR",
         "HRV", "SVN", "EST", "LVA", "LTU",
         "BIH", "MKD", "BLR", "RUS"],    # added Balkans + post-Soviet EU-adjacent
        "Центральная/Восточная Европа"),
    **dict.fromkeys(
        ["SRB", "MNE", "ALB", "GEO", "ARM", "AZE", "KAZ", "UZB"],
        "Балканы/Кавказ"),
    **dict.fromkeys(
        ["TUR", "ARE", "ISR", "SAU", "QAT", "KWT",
         "OMN", "BHR", "JOR", "IRN"],    # full Gulf + Levant
        "Ближний Восток"),
    **dict.fromkeys(
        ["EGY", "MAR", "TUN", "DZA", "LBY"],
        "Северная Африка"),
    **dict.fromkeys(
        ["USA", "CAN"],
        "Северная Америка"),
    **dict.fromkeys(
        ["MEX", "PAN", "CRI", "GTM", "HND", "SLV", "NIC"],
        "Центральная Америка"),
    **dict.fromkeys(
        ["BRA", "ARG", "COL", "CHL", "URY", "PRY",
         "ECU", "PER", "VEN", "BOL", "GUY"],
        "Латинская Америка"),
    **dict.fromkeys(
        ["JPN", "KOR", "TWN", "HKG", "CHN", "MNG"],
        "Восточная Азия"),
    **dict.fromkeys(
        ["SGP", "THA", "MYS", "IDN", "VNM", "PHL",
         "MMR", "KHM", "LAO", "BRN"],
        "Юго-Восточная Азия"),
    **dict.fromkeys(
        ["IND", "PAK", "BGD", "LKA", "NPL", "AFG"],
        "Южная Азия"),
    **dict.fromkeys(
        ["AUS", "NZL"],
        "Океания"),
    **dict.fromkeys(
        ["ZAF", "NGA", "KEN", "ETH", "GHA", "TZA", "UGA"],
        "Африка к югу от Сахары"),
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _get_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """
    Extract and standardise the numeric feature matrix from df.
    Drops metadata / cluster columns. Returns (X_scaled, feature_names).
    """
    feature_cols = [
        c for c in df.columns
        if c not in _META_COLS and pd.api.types.is_numeric_dtype(df[c])
    ]
    X = df[feature_cols].fillna(df[feature_cols].median()).values.astype(float)
    X_scaled = StandardScaler().fit_transform(X)
    return X_scaled, feature_cols


def _dominant_region(iso3_list: list[str]) -> str:
    """Return the most common macro-region among a list of ISO3 codes."""
    regions = [_ISO3_REGION.get(c, "Другое") for c in iso3_list]
    if not regions:
        return "Другое"
    return max(set(regions), key=regions.count)


def _cluster_name(centroid: pd.Series, iso3_list: list[str]) -> str:
    """
    Build a human-readable Russian label for a cluster based on its scaled
    centroid values and the geographic distribution of member countries.

    Thresholds assume features are MinMax-scaled to [0, 1].
    """
    gdp      = centroid.get("gdp_per_capita",       0.5)
    cost     = centroid.get("cost_of_living_index",  0.5)
    safety   = centroid.get("safety_index",          0.5)
    health   = centroid.get("healthcare_index",      0.5)
    eu_frac  = centroid.get("eu_member",             0.0)
    sch_frac = centroid.get("schengen",              0.0)
    dnv      = centroid.get("digital_nomad_visa",    0.0)
    res_dif  = centroid.get("residency_difficulty",  0.5)

    region = _dominant_region(iso3_list)
    n      = len(iso3_list)

    # Count countries per macro-group for mixed-cluster disambiguation
    region_counts: dict[str, int] = {}
    for iso3 in iso3_list:
        r = _ISO3_REGION.get(iso3, "Другое")
        region_counts[r] = region_counts.get(r, 0) + 1

    # Fractions of key geographic groups in this cluster
    latam_frac   = region_counts.get("Латинская Америка", 0) / n
    s_asia_frac  = region_counts.get("Южная Азия", 0) / n
    e_asia_frac  = region_counts.get("Восточная Азия", 0) / n
    gulf_frac    = region_counts.get("Ближний Восток", 0) / n
    africa_frac  = region_counts.get("Африка к югу от Сахары", 0) / n
    balkan_frac  = region_counts.get("Балканы/Кавказ", 0) / n

    # ── 1. EU / Schengen blocs (highest priority) ─────────────────────────
    if eu_frac >= 0.6 and gdp >= 0.55 and cost >= 0.55:
        return "Богатая Европа (ЕС)"
    if eu_frac >= 0.6 and cost < 0.48:
        return "Доступная Европа (ЕС)"
    if sch_frac >= 0.5 and gdp >= 0.45:
        return "Шенген — высокий уровень жизни"

    # ── 2. English-speaking developed world ───────────────────────────────
    if region in ("Северная Америка", "Океания", "Британские острова") and gdp >= 0.4:
        return "Англосфера — высокий уровень жизни"

    # ── 3. East Asia (Japan / Korea / Taiwan / China) ─────────────────────
    if e_asia_frac >= 0.35 and gulf_frac >= 0.30:
        # Mixed Gulf + East Asia cluster (e.g. ARE, JPN, KOR, QAT, SAU …)
        return "Богатая Азия и Залив"
    if e_asia_frac >= 0.40:
        if gdp >= 0.20 or health >= 0.50:
            return "Развитая Восточная Азия"
        return "Восточная Азия"

    # ── 4. Gulf / Middle East ─────────────────────────────────────────────
    if gulf_frac >= 0.40 and gdp >= 0.20:
        return "Ближний Восток — деловые хабы"

    # ── 5. Southeast Asia — affordable nomad-friendly ─────────────────────
    if region == "Юго-Восточная Азия" and cost < 0.45:
        return "Доступная Юго-Восточная Азия"

    # ── 6. Large diverse developing-world cluster ─────────────────────────
    if n >= 15:
        # Characterise by dominant sub-group
        if latam_frac >= 0.35:
            if balkan_frac >= 0.15:
                return "Латинская Америка, Балканы и развивающийся мир"
            return "Латинская Америка и развивающийся мир"
        if s_asia_frac >= 0.25:
            return "Южная Азия и развивающийся мир"
        if africa_frac >= 0.25:
            return "Африка и развивающийся мир"
        return "Развивающийся мир — доступные страны"

    # ── 7. Single-country or tiny cluster ─────────────────────────────────
    if n == 1:
        return f"Отдельная страна ({iso3_list[0]})"

    # ── 8. Latin America ──────────────────────────────────────────────────
    if region == "Латинская Америка":
        return "Доступная Латинская Америка" if cost < 0.42 else "Латинская Америка"

    # ── 9. Balkans / Caucasus ─────────────────────────────────────────────
    if region == "Балканы/Кавказ":
        return "Простая эмиграция (Балканы/Кавказ)"

    # ── 10. Eastern Europe (non-EU) ───────────────────────────────────────
    if region == "Центральная/Восточная Европа":
        return "Центральная и Восточная Европа"

    # ── 11. Southern Europe ───────────────────────────────────────────────
    if region == "Южная Европа":
        return "Южная Европа (Средиземноморье)"

    # ── 12. North Africa ──────────────────────────────────────────────────
    if region == "Северная Африка":
        return "Северная Африка"

    # ── Feature-profile catch-alls ────────────────────────────────────────
    if gdp >= 0.55 and safety >= 0.55:
        return "Развитые безопасные страны"
    if cost <= 0.30:
        return "Очень доступные страны"
    if safety >= 0.60:
        return "Безопасные страны со средним доходом"
    if dnv >= 0.60 and cost < 0.40:
        return "Цифровые кочевники — доступные страны"

    return f"Смешанный кластер ({region})"


# ── public API ────────────────────────────────────────────────────────────────

def run_kmeans(
    df: pd.DataFrame,
    k_range: tuple[int, int] = (4, 12),
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Fit KMeans for each K in k_range, pick the optimal K by maximising the
    Silhouette Score (elbow inertia curve printed for reference).

    Returns a copy of df with a new integer column 'cluster'.
    """
    X, _ = _get_feature_matrix(df)
    k_values   = range(k_range[0], k_range[1] + 1)
    inertias   : list[float] = []
    silhouettes: list[float] = []

    print("KMeans search:")
    print(f"  {'K':>3}  {'Inertia':>10}  {'Silhouette':>11}")
    print(f"  {'─'*3}  {'─'*10}  {'─'*11}")

    for k in k_values:
        km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        sil = silhouette_score(X, labels)
        silhouettes.append(sil)
        print(f"  {k:>3}  {km.inertia_:>10.1f}  {sil:>11.4f}")

    # Optimal K: highest silhouette score
    best_idx = int(np.argmax(silhouettes))
    best_k   = list(k_values)[best_idx]
    print(f"\n  Best K = {best_k}  (silhouette = {silhouettes[best_idx]:.4f})")

    # Elbow: largest second-derivative of inertia curve
    inertia_arr = np.array(inertias)
    diffs2 = np.diff(inertia_arr, n=2)
    elbow_k = list(k_values)[int(np.argmax(np.abs(diffs2))) + 1]
    print(f"  Elbow  K = {elbow_k}  (largest curvature)")

    km_final = KMeans(n_clusters=best_k, random_state=random_state, n_init="auto")
    result = df.copy()
    result["cluster"] = km_final.fit_predict(X)
    return result


def run_hdbscan(
    df: pd.DataFrame,
    min_cluster_size: int = 4,
    min_samples: int = 2,
) -> pd.DataFrame:
    """
    Cluster with HDBSCAN. Noise points receive label -1 and are kept in the
    returned DataFrame with cluster_hdbscan = -1.

    Returns a copy of df with a new integer column 'cluster_hdbscan'.
    """
    X, _ = _get_feature_matrix(df)

    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(X)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = int((labels == -1).sum())
    print(f"HDBSCAN → {n_clusters} clusters, {n_noise} noise points")

    result = df.copy()
    result["cluster_hdbscan"] = labels
    return result


def visualize_clusters(
    df: pd.DataFrame,
    cluster_col: str = "cluster",
    random_state: int = 42,
) -> go.Figure:
    """
    Compute a 2-D UMAP embedding of the feature matrix, then build an
    interactive Plotly scatter with country labels and cluster colours.
    Saves HTML to notebooks/figures/clusters.html.

    Returns the Plotly Figure object.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    X, _ = _get_feature_matrix(df)

    print("Fitting UMAP (2-D)…")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=12,
        min_dist=0.25,
        metric="euclidean",
        random_state=random_state,
    )
    embedding = reducer.fit_transform(X)

    plot_df = df.copy()
    plot_df["umap_x"] = embedding[:, 0]
    plot_df["umap_y"] = embedding[:, 1]
    plot_df["iso3"]   = plot_df.index

    label_col = "cluster_label" if "cluster_label" in plot_df.columns else cluster_col
    plot_df["cluster_str"] = plot_df[label_col].astype(str)

    country_col = "country" if "country" in plot_df.columns else "iso3"

    # Build hover text
    hover_cols = [
        "gdp_per_capita", "cost_of_living_index", "safety_index",
        "healthcare_index", "life_expectancy", "eu_member",
    ]
    hover_cols = [c for c in hover_cols if c in plot_df.columns]

    def _hover(row: pd.Series) -> str:
        lines = [f"<b>{row.get(country_col, row['iso3'])}</b>  ({row['iso3']})"]
        for c in hover_cols:
            lines.append(f"{c}: {row[c]:.3f}" if pd.notna(row[c]) else f"{c}: —")
        return "<br>".join(lines)

    plot_df["hover"] = plot_df.apply(_hover, axis=1)

    fig = px.scatter(
        plot_df,
        x="umap_x",
        y="umap_y",
        color="cluster_str",
        text=country_col,
        hover_name=country_col,
        hover_data={"hover": True, "umap_x": False, "umap_y": False, "cluster_str": False},
        title="Страны эмиграции — кластеры (UMAP 2D)",
        labels={"cluster_str": "Кластер"},
        template="plotly_white",
        color_discrete_sequence=px.colors.qualitative.Bold,
        width=1100,
        height=750,
    )

    fig.update_traces(
        textposition="top center",
        textfont_size=9,
        marker=dict(size=10, opacity=0.85, line=dict(width=0.5, color="white")),
    )
    fig.update_layout(
        legend=dict(
            title_text="Кластер",
            font_size=11,
            itemsizing="constant",
        ),
        margin=dict(l=20, r=20, t=60, b=20),
    )

    out_path = FIGURES_DIR / "clusters.html"
    fig.write_html(str(out_path))
    print(f"Saved → {out_path}")
    return fig


def label_clusters(
    df: pd.DataFrame,
    cluster_col: str = "cluster",
) -> pd.DataFrame:
    """
    Analyse each cluster's centroid and member countries, assign a readable
    Russian name, and print the cluster composition.

    Returns a copy of df with a new column 'cluster_label'.
    """
    result      = df.copy()
    feature_cols = [
        c for c in df.columns
        if c not in _META_COLS and pd.api.types.is_numeric_dtype(df[c])
    ]
    country_col  = "country" if "country" in df.columns else None

    cluster_ids   = sorted(df[cluster_col].dropna().unique())
    id_to_label   : dict[int | float, str] = {}

    for cid in cluster_ids:
        mask      = df[cluster_col] == cid
        members   = df[mask]
        iso3_list = list(members.index)
        centroid  = members[feature_cols].mean()
        name      = _cluster_name(centroid, iso3_list)

        # Deduplicate: if name already used, append cluster id
        existing_names = list(id_to_label.values())
        if name in existing_names:
            name = f"{name} #{int(cid)}"

        id_to_label[cid] = name

    result["cluster_label"] = result[cluster_col].map(id_to_label)

    # ── print composition ─────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print(f"  СОСТАВ КЛАСТЕРОВ  (col='{cluster_col}')")
    print("═" * 60)
    for cid, label in id_to_label.items():
        mask    = df[cluster_col] == cid
        members = df[mask]
        iso3s   = list(members.index)
        if country_col:
            names = members[country_col].tolist()
            entries = [f"{iso3} ({name})" for iso3, name in zip(iso3s, names)]
        else:
            entries = iso3s

        print(f"\n[{int(cid):>2}] {label}  ({len(iso3s)} стран)")
        # Print in rows of 3
        for i in range(0, len(entries), 3):
            row = entries[i:i + 3]
            print("     " + ",  ".join(f"{e:<28}" for e in row))
    print("═" * 60 + "\n")

    return result


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Loading {PROCESSED_PATH} …")
    df_raw = pd.read_csv(PROCESSED_PATH, index_col="iso3")
    print(f"  {len(df_raw)} countries, {df_raw.shape[1]} columns\n")

    # 1. KMeans
    df_km = run_kmeans(df_raw, k_range=(4, 12))

    # 2. HDBSCAN
    df_hdb = run_hdbscan(df_raw)
    df_km["cluster_hdbscan"] = df_hdb["cluster_hdbscan"]

    # 3. Label clusters
    df_labeled = label_clusters(df_km, cluster_col="cluster")

    # 4. Visualise (UMAP + Plotly HTML)
    fig = visualize_clusters(df_labeled, cluster_col="cluster")

    # 5. Quick HDBSCAN composition
    print("\nHDBSCAN кластеры:")
    for cid in sorted(df_labeled["cluster_hdbscan"].unique()):
        mask   = df_labeled["cluster_hdbscan"] == cid
        tag    = "шум" if cid == -1 else f"кластер {int(cid)}"
        isos   = ", ".join(df_labeled[mask].index.tolist())
        print(f"  {tag:>12} ({mask.sum():>3}): {isos}")
