import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

try:
    import pycountry
    import pycountry_convert as pc
    _PYCOUNTRY_AVAILABLE = True
except ImportError:
    _PYCOUNTRY_AVAILABLE = False

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
OUTPUT_PATH = PROCESSED_DIR / "countries.csv"

MISSING_THRESHOLD = 0.40

# Columns that are already binary (0/1) — excluded from MinMaxScaler
_BOOL_COLS = {"digital_nomad_visa", "investor_visa", "eu_member", "schengen", "ru_visa_free"}

# Manual overrides: lowercase country name → ISO 3166-1 alpha-3
_NAME_OVERRIDES: dict[str, str] = {
    "hong kong":                  "HKG",
    "taiwan":                     "TWN",
    "south korea":                "KOR",
    "korea, rep.":                "KOR",
    "uae":                        "ARE",
    "czech republic":             "CZE",
    "czechia":                    "CZE",
    "russia":                     "RUS",
    "iran":                       "IRN",
    "iran, islamic rep.":         "IRN",
    "syria":                      "SYR",
    "venezuela":                  "VEN",
    "vietnam":                    "VNM",
    "bolivia":                    "BOL",
    "tanzania":                   "TZA",
    "egypt, arab rep.":           "EGY",
    "congo, dem. rep.":           "COD",
    "congo, rep.":                "COG",
    "gambia, the":                "GMB",
    "bahamas, the":               "BHS",
    "kyrgyz republic":            "KGZ",
    "lao pdr":                    "LAO",
    "micronesia, fed. sts.":      "FSM",
    "st. lucia":                  "LCA",
    "st. vincent and the grenadines": "VCT",
    "st. kitts and nevis":        "KNA",
    "são tomé and príncipe":      "STP",
    "timor-leste":                "TLS",
    "west bank and gaza":         "PSE",
    "kosovo":                     "XKX",
    "united states":              "USA",
    "uk":                         "GBR",
    "south africa":               "ZAF",
}

# UN M.49 subregion lookup (iso3 → region).
# Drives median imputation — granular enough to be meaningful.
_ISO3_TO_REGION: dict[str, str] = {
    # Northern Europe
    **dict.fromkeys(["SWE", "NOR", "DNK", "FIN", "ISL", "GBR", "IRL",
                     "EST", "LVA", "LTU"], "Northern Europe"),
    # Western Europe
    **dict.fromkeys(["DEU", "FRA", "NLD", "BEL", "AUT", "CHE", "LUX"], "Western Europe"),
    # Southern Europe
    **dict.fromkeys(["PRT", "ESP", "ITA", "GRC", "MLT", "CYP", "HRV",
                     "SVN", "MNE", "ALB", "SRB", "MKD", "BIH"], "Southern Europe"),
    # Eastern Europe
    **dict.fromkeys(["POL", "CZE", "SVK", "HUN", "ROU", "BGR",
                     "UKR", "BLR", "MDA"], "Eastern Europe"),
    # Caucasus & Central Asia
    **dict.fromkeys(["GEO", "ARM", "AZE", "KAZ", "UZB", "TKM",
                     "KGZ", "TJK"], "Caucasus & Central Asia"),
    # Middle East & North Africa
    **dict.fromkeys(["TUR", "ARE", "ISR", "SAU", "QAT", "KWT", "OMN",
                     "BHR", "JOR", "LBN", "EGY", "MAR", "TUN", "LBY",
                     "DZA"], "Middle East & North Africa"),
    # Sub-Saharan Africa
    **dict.fromkeys(["ZAF", "NGA", "KEN", "ETH", "GHA", "TZA",
                     "COD", "UGA", "ZMB", "ZWE"], "Sub-Saharan Africa"),
    # North America
    **dict.fromkeys(["USA", "CAN", "MEX"], "North America"),
    # Central America & Caribbean
    **dict.fromkeys(["PAN", "CRI", "GTM", "HND", "SLV", "NIC",
                     "BLZ", "CUB", "DOM", "JAM", "HTI"], "Central America & Caribbean"),
    # South America
    **dict.fromkeys(["BRA", "ARG", "COL", "CHL", "URY", "PRY",
                     "ECU", "PER", "VEN", "BOL", "GUY", "SUR"], "South America"),
    # East Asia
    **dict.fromkeys(["JPN", "KOR", "CHN", "TWN", "HKG", "MNG"], "East Asia"),
    # Southeast Asia
    **dict.fromkeys(["SGP", "THA", "MYS", "IDN", "VNM", "PHL",
                     "MMR", "KHM", "LAO", "BRN", "TLS"], "Southeast Asia"),
    # South Asia
    **dict.fromkeys(["IND", "PAK", "BGD", "NPL", "LKA", "AFG",
                     "MDV", "BTN"], "South Asia"),
    # Oceania
    **dict.fromkeys(["AUS", "NZL", "FJI", "PNG", "WSM", "TON"], "Oceania"),
}

# Continent fallback (pycountry_convert codes → label)
_CONTINENT_FALLBACK: dict[str, str] = {
    "AF": "Sub-Saharan Africa",
    "AS": "South Asia",
    "EU": "Eastern Europe",
    "NA": "North America",
    "OC": "Oceania",
    "SA": "South America",
    "AN": "Oceania",
}


# ── helpers ──────────────────────────────────────────────────────────────────

_iso3_cache: dict[str, str | None] = {}


def _to_iso3(name: str) -> str | None:
    """Resolve a country name string to an ISO 3166-1 alpha-3 code."""
    if not isinstance(name, str) or not name.strip():
        return None

    key = name.strip().lower()
    if key in _iso3_cache:
        return _iso3_cache[key]

    # 1. manual overrides
    result = _NAME_OVERRIDES.get(key)

    # 2. exact pycountry lookup
    if result is None and _PYCOUNTRY_AVAILABLE:
        country = pycountry.countries.get(name=name.strip())
        if country:
            result = country.alpha_3

    # 3. fuzzy pycountry search (catches "United States" → "United States of America")
    if result is None and _PYCOUNTRY_AVAILABLE:
        try:
            matches = pycountry.countries.search_fuzzy(name.strip())
            if matches:
                result = matches[0].alpha_3
        except LookupError:
            pass

    _iso3_cache[key] = result
    return result


def _assign_region(iso3: str) -> str:
    """Map an ISO3 code to a geographic subregion for imputation grouping."""
    region = _ISO3_TO_REGION.get(iso3)
    if region:
        return region

    if _PYCOUNTRY_AVAILABLE:
        try:
            country = pycountry.countries.get(alpha_3=iso3)
            if country:
                alpha2 = country.alpha_2
                continent_code = pc.country_alpha2_to_continent_code(alpha2)
                return _CONTINENT_FALLBACK.get(continent_code, "Other")
        except (KeyError, AttributeError):
            pass

    return "Other"


def _load_worldbank() -> pd.DataFrame:
    path = RAW_DIR / "worldbank.csv"
    df = pd.read_csv(path, index_col="iso3")
    df.index = df.index.str.strip().str.upper()
    df = df.rename(columns={"country": "country_wb"})
    return df


def _load_numbeo() -> pd.DataFrame:
    path = RAW_DIR / "numbeo.csv"
    df = pd.read_csv(path)
    df["iso3"] = df["country"].map(_to_iso3)
    df = df.dropna(subset=["iso3"]).set_index("iso3")
    df.index = df.index.str.strip().str.upper()
    df = df.rename(columns={"country": "country_nb"})
    return df


def _load_visa() -> pd.DataFrame:
    path = RAW_DIR / "visa_data.csv"
    df = pd.read_csv(path)
    bool_cols = [c for c in _BOOL_COLS if c in df.columns]
    df[bool_cols] = df[bool_cols].astype(int)  # True/False → 1/0 before merge
    df["iso3"] = df["country"].map(_to_iso3)
    df = df.dropna(subset=["iso3"]).set_index("iso3")
    df.index = df.index.str.strip().str.upper()
    df = df.rename(columns={"country": "country_vd"})
    return df


# ── main ─────────────────────────────────────────────────────────────────────

def build_dataset() -> pd.DataFrame:
    """
    Merge World Bank, Numbeo, and visa data into a single clean feature matrix.

    Pipeline
    --------
    1. Load and ISO3-key all three sources
    2. Outer-join on ISO3 index
    3. Drop rows missing > 40 % of feature columns
    4. Impute remaining NaNs with regional median
    5. MinMaxScale all numeric columns except binary flags
    6. Save to data/processed/countries.csv

    Returns
    -------
    pd.DataFrame  — one row per country, ISO3-indexed
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. load ───────────────────────────────────────────────────────────────
    print("Loading raw files...")
    wb  = _load_worldbank()
    nb  = _load_numbeo()
    vd  = _load_visa()
    print(f"  World Bank : {len(wb):>4} countries")
    print(f"  Numbeo     : {len(nb):>4} countries")
    print(f"  Visa data  : {len(vd):>4} countries")

    # ── 2. merge ──────────────────────────────────────────────────────────────
    df = wb.join(nb, how="outer", rsuffix="_nb")
    df = df.join(vd, how="outer", rsuffix="_vd")

    # Consolidate duplicate country-name columns into one clean 'country' column
    name_cols = [c for c in df.columns if c.startswith("country")]
    df["country"] = df[name_cols].bfill(axis=1).iloc[:, 0]
    df = df.drop(columns=name_cols)

    # Drop pure-metadata / non-feature columns before missing-rate calculation
    feature_cols = [c for c in df.columns if c != "country"]
    total_features = len(feature_cols)

    # ── 2b. fill Russian-specific columns with neutral defaults ──────────
    # Countries outside visa_data get neutral values so they aren't penalised
    # by the missing-rate filter.
    _RU_DEFAULTS = {"ru_visa_free": 0, "ru_banking_access": 3, "ru_sanctions_risk": 1}
    for col, default_val in _RU_DEFAULTS.items():
        if col in df.columns:
            df[col] = df[col].fillna(default_val)

    # ── 3. drop countries with > 40 % missing features ────────────────────────
    missing_rate = df[feature_cols].isna().mean(axis=1)
    before = len(df)
    df = df[missing_rate <= MISSING_THRESHOLD].copy()
    after = len(df)
    print(f"\nDropped {before - after} countries (>{MISSING_THRESHOLD:.0%} missing) "
          f"— {after} remain")

    # ── 4. assign region & impute with regional median ────────────────────────
    df["_region"] = df.index.map(_assign_region)

    numeric_cols = (
        df[feature_cols]
        .select_dtypes(include="number")
        .columns
        .tolist()
    )

    for col in numeric_cols:
        # Regional median first
        regional_median = df.groupby("_region")[col].transform("median")
        # Global median as fallback for regions with all-NaN
        global_median = df[col].median()
        df[col] = df[col].fillna(regional_median).fillna(global_median)

    df = df.drop(columns=["_region"])

    # ── 5. MinMaxScale numeric non-binary columns ─────────────────────────────
    scale_cols = [c for c in numeric_cols if c not in _BOOL_COLS]
    scaler = MinMaxScaler()
    df[scale_cols] = scaler.fit_transform(df[scale_cols])

    # ── 6. save ───────────────────────────────────────────────────────────────
    df.index.name = "iso3"
    df = df.sort_index()
    df.to_csv(OUTPUT_PATH, encoding="utf-8")

    # ── stats ─────────────────────────────────────────────────────────────────
    remaining_na = df[feature_cols].isna().sum().sum()
    print(f"\n{'─' * 45}")
    print(f"  Countries  : {len(df)}")
    print(f"  Features   : {total_features}")
    print(f"  Missing    : {remaining_na} cells remaining after imputation")
    print(f"  Scaled     : {len(scale_cols)} columns via MinMaxScaler")
    print(f"  Binary     : {len(_BOOL_COLS & set(numeric_cols))} columns left as 0/1")
    print(f"  Saved to   : {OUTPUT_PATH}")
    print(f"{'─' * 45}")

    return df


if __name__ == "__main__":
    df = build_dataset()
    print(f"\nSample (first 5 rows):\n{df.head().to_string()}")
