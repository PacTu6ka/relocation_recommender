import requests
import pandas as pd
import time
from pathlib import Path
from bs4 import BeautifulSoup

WORLDBANK_BASE = "https://api.worldbank.org/v2"

INDICATORS = {
    "NY.GDP.PCAP.CD": "gdp_per_capita",
    "SL.UEM.TOTL.ZS": "unemployment_rate",
    "SP.DYN.LE00.IN": "life_expectancy",
    "SE.ADT.LITR.ZS": "literacy_rate",
    "SI.POV.GINI": "gini_index",
}

OUTPUT_PATH = Path(__file__).parent.parent / "data" / "raw" / "worldbank.csv"
NUMBEO_OUTPUT_PATH = Path(__file__).parent.parent / "data" / "raw" / "numbeo.csv"

NUMBEO_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

NUMBEO_PAGES = {
    "cost-of-living": {
        "url": "https://www.numbeo.com/cost-of-living/rankings_by_country.jsp",
        "columns": {
            "Cost of Living Index": "cost_of_living_index",
            "Rent Index": "rent_index",
            "Groceries Index": "groceries_index",
            "Restaurant Price Index": "restaurant_price_index",
            "Local Purchasing Power Index": "purchasing_power_index",
        },
    },
    "quality-of-life": {
        "url": "https://www.numbeo.com/quality-of-life/rankings_by_country.jsp",
        "columns": {
            "Safety Index": "safety_index",
            "Health Care Index": "healthcare_index",
            "Pollution Index": "pollution_index",
            "Traffic Commute Time Index": "traffic_index",
        },
    },
}


def _fetch_indicator(indicator: str, per_page: int = 1000, retries: int = 3) -> list[dict]:
    """Fetch all country values for a single indicator (most recent year)."""
    url = f"{WORLDBANK_BASE}/country/all/indicator/{indicator}"
    params = {
        "format": "json",
        "per_page": per_page,
        "mrv": 1,  # most recent value
    }

    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            payload = response.json()

            # World Bank wraps data in a 2-element list: [metadata, data]
            if not isinstance(payload, list) or len(payload) < 2:
                raise ValueError(f"Unexpected response structure for {indicator}")

            metadata, records = payload
            total_pages = metadata.get("pages", 1)
            results = list(records or [])

            # Paginate if needed
            for page in range(2, total_pages + 1):
                resp = requests.get(url, params={**params, "page": page}, timeout=30)
                resp.raise_for_status()
                _, page_records = resp.json()
                results.extend(page_records or [])

            return results

        except requests.exceptions.RequestException as exc:
            print(f"  [attempt {attempt}/{retries}] Network error for {indicator}: {exc}")
            if attempt < retries:
                time.sleep(2 ** attempt)
        except (ValueError, KeyError, TypeError) as exc:
            print(f"  [attempt {attempt}/{retries}] Parse error for {indicator}: {exc}")
            if attempt < retries:
                time.sleep(1)

    print(f"  Failed to fetch {indicator} after {retries} attempts — skipping.")
    return []


def _records_to_series(records: list[dict], col_name: str) -> pd.Series:
    """Convert raw API records to a country-indexed Series."""
    rows = {}
    for rec in records:
        country      = rec.get("country", {})
        # World Bank puts the real ISO3 in 'countryiso3code'; 'country.id' is alpha-2
        iso3         = rec.get("countryiso3code", "").strip()
        alpha2       = country.get("id", "").strip()
        country_name = country.get("value", "").strip()
        value        = rec.get("value")

        # Skip aggregates: real countries have a standard 2-letter alpha-2 code
        # (aggregates use codes like "1W", "ZH", "ZI" that contain digits or are unusual)
        if not iso3 or len(iso3) != 3:
            continue
        if not alpha2.isalpha() or len(alpha2) != 2:
            continue
        if value is not None:
            rows[iso3] = {"country": country_name, col_name: float(value)}

    return pd.DataFrame(rows).T  # rows: iso3 → {country, col_name}


def collect_worldbank() -> pd.DataFrame:
    """
    Download key development indicators for all countries from the World Bank API
    and save the result to data/raw/worldbank.csv.

    Returns
    -------
    pd.DataFrame
        Indexed by ISO-3 country code with one column per indicator.
    """
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    total = len(INDICATORS)

    for idx, (indicator_code, col_name) in enumerate(INDICATORS.items(), start=1):
        print(f"[{idx}/{total}] Fetching {indicator_code} ({col_name}) ...")
        records = _fetch_indicator(indicator_code)

        if not records:
            print(f"  No data returned for {indicator_code}.")
            continue

        df_indicator = _records_to_series(records, col_name)
        frames.append(df_indicator)
        print(f"  {len(df_indicator)} countries retrieved.")

    if not frames:
        raise RuntimeError("No data was fetched. Check your network connection or the API endpoint.")

    # Merge all indicators on ISO-3 index, keep country name from first frame
    combined = frames[0]
    for frame in frames[1:]:
        # Drop duplicate 'country' column before joining
        combined = combined.join(frame.drop(columns="country", errors="ignore"), how="outer")

    combined.index.name = "iso3"
    combined = combined.sort_index()

    combined.to_csv(OUTPUT_PATH, encoding="utf-8")
    print(f"\nSaved {len(combined)} countries to {OUTPUT_PATH}")

    return combined


def _fetch_numbeo_page(section: str, config: dict, retries: int = 3) -> pd.DataFrame:
    """
    Fetch and parse a single Numbeo rankings page.

    Numbeo renders its ranking tables as plain HTML <table id="t2">.
    Column header text is matched against config['columns'] to rename them.
    """
    url = config["url"]
    col_map = config["columns"]

    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, headers=NUMBEO_HEADERS, timeout=30)
            response.raise_for_status()
            break
        except requests.exceptions.RequestException as exc:
            print(f"  [attempt {attempt}/{retries}] Network error ({section}): {exc}")
            if attempt < retries:
                time.sleep(2 ** attempt)
            else:
                print(f"  Failed to fetch {url} — skipping section.")
                return pd.DataFrame()

    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.find("table", {"id": "t2"})
    if table is None:
        # Fallback: try the first sizeable table on the page
        table = soup.find("table", class_="stripe")
    if table is None:
        print(f"  Could not locate rankings table on {url}")
        return pd.DataFrame()

    # Parse header row
    header_row = table.find("thead")
    if header_row:
        raw_headers = [th.get_text(strip=True) for th in header_row.find_all("th")]
    else:
        raw_headers = [th.get_text(strip=True) for th in table.find("tr").find_all(["th", "td"])]

    # Build index map: raw header text → target column name
    header_index: dict[int, str] = {}
    country_col_idx: int | None = None

    for i, h in enumerate(raw_headers):
        if h.lower() in ("country", "city"):
            country_col_idx = i
        for raw_key, col_name in col_map.items():
            if raw_key.lower() in h.lower():
                header_index[i] = col_name

    if country_col_idx is None:
        # Numbeo usually has "Country" as the second column (after rank)
        country_col_idx = 1

    rows = []
    tbody = table.find("tbody") or table
    for tr in tbody.find_all("tr"):
        cells = [td.get_text(strip=True) for td in tr.find_all("td")]
        if len(cells) < 2:
            continue
        country_name = cells[country_col_idx]
        if not country_name:
            continue

        row: dict[str, object] = {"country": country_name}
        for col_idx, col_name in header_index.items():
            raw_val = cells[col_idx] if col_idx < len(cells) else None
            try:
                row[col_name] = float(raw_val.replace(",", "")) if raw_val else None
            except (ValueError, AttributeError):
                row[col_name] = None
        rows.append(row)

    df = pd.DataFrame(rows)
    # Ensure all expected columns exist even if not found in this table
    for col_name in col_map.values():
        if col_name not in df.columns:
            df[col_name] = None

    return df


def collect_numbeo(delay: float = 2.0) -> pd.DataFrame:
    """
    Scrape country-level indices from Numbeo and save to data/raw/numbeo.csv.

    Fetches two pages:
      - cost-of-living/rankings_by_country.jsp
      - quality-of-life/rankings_by_country.jsp

    Parameters
    ----------
    delay : float
        Seconds to wait between page requests (polite scraping).

    Returns
    -------
    pd.DataFrame
        Indexed by normalised country name with one column per index.
    """
    NUMBEO_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    total = len(NUMBEO_PAGES)
    frames: list[pd.DataFrame] = []

    for idx, (section, config) in enumerate(NUMBEO_PAGES.items(), start=1):
        print(f"[{idx}/{total}] Scraping Numbeo / {section} ...")
        df = _fetch_numbeo_page(section, config)

        if df.empty:
            print(f"  No data parsed for section '{section}'.")
        else:
            print(f"  {len(df)} countries parsed.")
            frames.append(df)

        if idx < total:
            time.sleep(delay)

    if not frames:
        raise RuntimeError("Numbeo scraping returned no data. The page structure may have changed.")

    # Merge on normalised country name
    combined = frames[0]
    for frame in frames[1:]:
        combined = pd.merge(combined, frame, on="country", how="outer")

    combined["country"] = combined["country"].str.strip()
    combined = combined.sort_values("country").reset_index(drop=True)

    combined.to_csv(NUMBEO_OUTPUT_PATH, index=False, encoding="utf-8")
    print(f"\nSaved {len(combined)} countries to {NUMBEO_OUTPUT_PATH}")

    return combined


VISA_OUTPUT_PATH = Path(__file__).parent.parent / "data" / "raw" / "visa_data.csv"

# fmt: off
# Sources (as of 2025):
#   visa_free_count  — Henley Passport Index 2025 (visa-free + visa-on-arrival destinations)
#   residency_difficulty — 1 (very easy) … 5 (very hard); author assessment based on
#                          processing times, language requirements, bureaucracy, income thresholds
#   digital_nomad_visa — country has a dedicated DNV / remote-worker residency programme
#   investor_visa      — golden/investor residency or citizenship-by-investment programme exists
#   eu_member          — EU member state
#   schengen           — full Schengen Area member (RO + BG joined Jan 2025)
_VISA_RECORDS: list[dict] = [
    # ── Western Europe ──────────────────────────────────────────────────────────
    # ru_visa_free      — Russians can enter without a visa or get visa-on-arrival (as of 2025)
    # ru_banking_access  — 1 (impossible) … 5 (easy) — how accessible banking is for Russian citizens post-2022
    # ru_sanctions_risk  — 1 (low/none) … 3 (high) — sanctions pressure / hostile environment for Russian nationals
    # ── Western Europe ──────────────────────────────────────────────────────────
    {"country": "Germany",        "visa_free_count": 191, "residency_difficulty": 3, "digital_nomad_visa": True,  "investor_visa": False, "eu_member": True,  "schengen": True,  "ru_visa_free": False, "ru_banking_access": 2, "ru_sanctions_risk": 3},
    {"country": "Portugal",       "visa_free_count": 188, "residency_difficulty": 2, "digital_nomad_visa": True,  "investor_visa": True,  "eu_member": True,  "schengen": True,  "ru_visa_free": False, "ru_banking_access": 2, "ru_sanctions_risk": 3},
    {"country": "Spain",          "visa_free_count": 190, "residency_difficulty": 3, "digital_nomad_visa": True,  "investor_visa": True,  "eu_member": True,  "schengen": True,  "ru_visa_free": False, "ru_banking_access": 2, "ru_sanctions_risk": 3},
    {"country": "France",         "visa_free_count": 191, "residency_difficulty": 3, "digital_nomad_visa": False, "investor_visa": False, "eu_member": True,  "schengen": True,  "ru_visa_free": False, "ru_banking_access": 2, "ru_sanctions_risk": 3},
    {"country": "Netherlands",    "visa_free_count": 191, "residency_difficulty": 3, "digital_nomad_visa": False, "investor_visa": False, "eu_member": True,  "schengen": True,  "ru_visa_free": False, "ru_banking_access": 2, "ru_sanctions_risk": 3},
    {"country": "Italy",          "visa_free_count": 190, "residency_difficulty": 3, "digital_nomad_visa": True,  "investor_visa": True,  "eu_member": True,  "schengen": True,  "ru_visa_free": False, "ru_banking_access": 2, "ru_sanctions_risk": 3},
    {"country": "Switzerland",    "visa_free_count": 192, "residency_difficulty": 4, "digital_nomad_visa": False, "investor_visa": False, "eu_member": False, "schengen": True,  "ru_visa_free": False, "ru_banking_access": 1, "ru_sanctions_risk": 3},
    {"country": "Austria",        "visa_free_count": 191, "residency_difficulty": 3, "digital_nomad_visa": True,  "investor_visa": False, "eu_member": True,  "schengen": True,  "ru_visa_free": False, "ru_banking_access": 2, "ru_sanctions_risk": 3},
    {"country": "Belgium",        "visa_free_count": 191, "residency_difficulty": 3, "digital_nomad_visa": False, "investor_visa": False, "eu_member": True,  "schengen": True,  "ru_visa_free": False, "ru_banking_access": 2, "ru_sanctions_risk": 3},
    {"country": "Ireland",        "visa_free_count": 190, "residency_difficulty": 3, "digital_nomad_visa": False, "investor_visa": True,  "eu_member": True,  "schengen": False, "ru_visa_free": False, "ru_banking_access": 2, "ru_sanctions_risk": 3},
    {"country": "United Kingdom", "visa_free_count": 190, "residency_difficulty": 3, "digital_nomad_visa": False, "investor_visa": True,  "eu_member": False, "schengen": False, "ru_visa_free": False, "ru_banking_access": 1, "ru_sanctions_risk": 3},
    # ── Northern Europe ──────────────────────────────────────────────────────────
    {"country": "Sweden",         "visa_free_count": 191, "residency_difficulty": 3, "digital_nomad_visa": False, "investor_visa": False, "eu_member": True,  "schengen": True,  "ru_visa_free": False, "ru_banking_access": 2, "ru_sanctions_risk": 3},
    {"country": "Norway",         "visa_free_count": 191, "residency_difficulty": 3, "digital_nomad_visa": True,  "investor_visa": False, "eu_member": False, "schengen": True,  "ru_visa_free": False, "ru_banking_access": 2, "ru_sanctions_risk": 3},
    {"country": "Denmark",        "visa_free_count": 191, "residency_difficulty": 4, "digital_nomad_visa": False, "investor_visa": False, "eu_member": True,  "schengen": True,  "ru_visa_free": False, "ru_banking_access": 2, "ru_sanctions_risk": 3},
    {"country": "Finland",        "visa_free_count": 191, "residency_difficulty": 3, "digital_nomad_visa": False, "investor_visa": False, "eu_member": True,  "schengen": True,  "ru_visa_free": False, "ru_banking_access": 2, "ru_sanctions_risk": 3},
    {"country": "Iceland",        "visa_free_count": 190, "residency_difficulty": 3, "digital_nomad_visa": True,  "investor_visa": False, "eu_member": False, "schengen": True,  "ru_visa_free": False, "ru_banking_access": 2, "ru_sanctions_risk": 3},
    # ── Central & Eastern Europe ─────────────────────────────────────────────────
    {"country": "Poland",         "visa_free_count": 187, "residency_difficulty": 2, "digital_nomad_visa": False, "investor_visa": False, "eu_member": True,  "schengen": True,  "ru_visa_free": False, "ru_banking_access": 2, "ru_sanctions_risk": 3},
    {"country": "Czech Republic", "visa_free_count": 188, "residency_difficulty": 3, "digital_nomad_visa": True,  "investor_visa": False, "eu_member": True,  "schengen": True,  "ru_visa_free": False, "ru_banking_access": 2, "ru_sanctions_risk": 3},
    {"country": "Hungary",        "visa_free_count": 186, "residency_difficulty": 2, "digital_nomad_visa": True,  "investor_visa": True,  "eu_member": True,  "schengen": True,  "ru_visa_free": False, "ru_banking_access": 3, "ru_sanctions_risk": 2},
    {"country": "Slovakia",       "visa_free_count": 187, "residency_difficulty": 3, "digital_nomad_visa": False, "investor_visa": False, "eu_member": True,  "schengen": True,  "ru_visa_free": False, "ru_banking_access": 2, "ru_sanctions_risk": 3},
    {"country": "Slovenia",       "visa_free_count": 187, "residency_difficulty": 3, "digital_nomad_visa": True,  "investor_visa": False, "eu_member": True,  "schengen": True,  "ru_visa_free": False, "ru_banking_access": 2, "ru_sanctions_risk": 3},
    {"country": "Romania",        "visa_free_count": 175, "residency_difficulty": 2, "digital_nomad_visa": True,  "investor_visa": False, "eu_member": True,  "schengen": True,  "ru_visa_free": False, "ru_banking_access": 2, "ru_sanctions_risk": 3},
    {"country": "Bulgaria",       "visa_free_count": 175, "residency_difficulty": 2, "digital_nomad_visa": True,  "investor_visa": True,  "eu_member": True,  "schengen": True,  "ru_visa_free": False, "ru_banking_access": 2, "ru_sanctions_risk": 3},
    {"country": "Croatia",        "visa_free_count": 186, "residency_difficulty": 2, "digital_nomad_visa": True,  "investor_visa": False, "eu_member": True,  "schengen": True,  "ru_visa_free": False, "ru_banking_access": 2, "ru_sanctions_risk": 3},
    # ── Southern Europe & Balkans ────────────────────────────────────────────────
    {"country": "Greece",         "visa_free_count": 188, "residency_difficulty": 2, "digital_nomad_visa": True,  "investor_visa": True,  "eu_member": True,  "schengen": True,  "ru_visa_free": False, "ru_banking_access": 2, "ru_sanctions_risk": 3},
    {"country": "Malta",          "visa_free_count": 188, "residency_difficulty": 2, "digital_nomad_visa": True,  "investor_visa": True,  "eu_member": True,  "schengen": True,  "ru_visa_free": False, "ru_banking_access": 2, "ru_sanctions_risk": 3},
    {"country": "Cyprus",         "visa_free_count": 177, "residency_difficulty": 2, "digital_nomad_visa": True,  "investor_visa": True,  "eu_member": True,  "schengen": False, "ru_visa_free": False, "ru_banking_access": 2, "ru_sanctions_risk": 3},
    {"country": "Estonia",        "visa_free_count": 188, "residency_difficulty": 2, "digital_nomad_visa": True,  "investor_visa": False, "eu_member": True,  "schengen": True,  "ru_visa_free": False, "ru_banking_access": 1, "ru_sanctions_risk": 3},
    {"country": "Latvia",         "visa_free_count": 186, "residency_difficulty": 2, "digital_nomad_visa": False, "investor_visa": True,  "eu_member": True,  "schengen": True,  "ru_visa_free": False, "ru_banking_access": 1, "ru_sanctions_risk": 3},
    {"country": "Lithuania",      "visa_free_count": 186, "residency_difficulty": 2, "digital_nomad_visa": False, "investor_visa": False, "eu_member": True,  "schengen": True,  "ru_visa_free": False, "ru_banking_access": 1, "ru_sanctions_risk": 3},
    {"country": "Serbia",         "visa_free_count": 140, "residency_difficulty": 1, "digital_nomad_visa": True,  "investor_visa": False, "eu_member": False, "schengen": False, "ru_visa_free": True,  "ru_banking_access": 5, "ru_sanctions_risk": 1},
    {"country": "Montenegro",     "visa_free_count": 125, "residency_difficulty": 1, "digital_nomad_visa": True,  "investor_visa": True,  "eu_member": False, "schengen": False, "ru_visa_free": True,  "ru_banking_access": 4, "ru_sanctions_risk": 1},
    {"country": "Albania",        "visa_free_count": 121, "residency_difficulty": 1, "digital_nomad_visa": True,  "investor_visa": False, "eu_member": False, "schengen": False, "ru_visa_free": True,  "ru_banking_access": 3, "ru_sanctions_risk": 1},
    # ── Caucasus & Turkey ────────────────────────────────────────────────────────
    {"country": "Georgia",        "visa_free_count": 118, "residency_difficulty": 1, "digital_nomad_visa": True,  "investor_visa": False, "eu_member": False, "schengen": False, "ru_visa_free": True,  "ru_banking_access": 5, "ru_sanctions_risk": 1},
    {"country": "Armenia",        "visa_free_count":  68, "residency_difficulty": 1, "digital_nomad_visa": True,  "investor_visa": False, "eu_member": False, "schengen": False, "ru_visa_free": True,  "ru_banking_access": 5, "ru_sanctions_risk": 1},
    {"country": "Turkey",         "visa_free_count": 111, "residency_difficulty": 2, "digital_nomad_visa": False, "investor_visa": True,  "eu_member": False, "schengen": False, "ru_visa_free": True,  "ru_banking_access": 4, "ru_sanctions_risk": 1},
    # ── Middle East ──────────────────────────────────────────────────────────────
    {"country": "UAE",            "visa_free_count": 180, "residency_difficulty": 2, "digital_nomad_visa": True,  "investor_visa": True,  "eu_member": False, "schengen": False, "ru_visa_free": True,  "ru_banking_access": 4, "ru_sanctions_risk": 1},
    {"country": "Israel",         "visa_free_count": 162, "residency_difficulty": 4, "digital_nomad_visa": False, "investor_visa": True,  "eu_member": False, "schengen": False, "ru_visa_free": True,  "ru_banking_access": 3, "ru_sanctions_risk": 2},
    # ── North America ────────────────────────────────────────────────────────────
    {"country": "United States",  "visa_free_count": 186, "residency_difficulty": 4, "digital_nomad_visa": False, "investor_visa": True,  "eu_member": False, "schengen": False, "ru_visa_free": False, "ru_banking_access": 1, "ru_sanctions_risk": 3},
    {"country": "Canada",         "visa_free_count": 185, "residency_difficulty": 3, "digital_nomad_visa": False, "investor_visa": True,  "eu_member": False, "schengen": False, "ru_visa_free": False, "ru_banking_access": 1, "ru_sanctions_risk": 3},
    {"country": "Mexico",         "visa_free_count": 162, "residency_difficulty": 2, "digital_nomad_visa": True,  "investor_visa": True,  "eu_member": False, "schengen": False, "ru_visa_free": True,  "ru_banking_access": 3, "ru_sanctions_risk": 1},
    {"country": "Panama",         "visa_free_count": 146, "residency_difficulty": 1, "digital_nomad_visa": True,  "investor_visa": True,  "eu_member": False, "schengen": False, "ru_visa_free": True,  "ru_banking_access": 3, "ru_sanctions_risk": 1},
    {"country": "Costa Rica",     "visa_free_count": 152, "residency_difficulty": 2, "digital_nomad_visa": True,  "investor_visa": True,  "eu_member": False, "schengen": False, "ru_visa_free": True,  "ru_banking_access": 3, "ru_sanctions_risk": 1},
    # ── South America ────────────────────────────────────────────────────────────
    {"country": "Brazil",         "visa_free_count": 170, "residency_difficulty": 2, "digital_nomad_visa": True,  "investor_visa": True,  "eu_member": False, "schengen": False, "ru_visa_free": True,  "ru_banking_access": 3, "ru_sanctions_risk": 1},
    {"country": "Argentina",      "visa_free_count": 171, "residency_difficulty": 2, "digital_nomad_visa": True,  "investor_visa": True,  "eu_member": False, "schengen": False, "ru_visa_free": True,  "ru_banking_access": 3, "ru_sanctions_risk": 1},
    {"country": "Colombia",       "visa_free_count": 152, "residency_difficulty": 2, "digital_nomad_visa": True,  "investor_visa": True,  "eu_member": False, "schengen": False, "ru_visa_free": True,  "ru_banking_access": 3, "ru_sanctions_risk": 1},
    {"country": "Chile",          "visa_free_count": 174, "residency_difficulty": 2, "digital_nomad_visa": True,  "investor_visa": True,  "eu_member": False, "schengen": False, "ru_visa_free": True,  "ru_banking_access": 3, "ru_sanctions_risk": 1},
    {"country": "Uruguay",        "visa_free_count": 155, "residency_difficulty": 2, "digital_nomad_visa": True,  "investor_visa": True,  "eu_member": False, "schengen": False, "ru_visa_free": True,  "ru_banking_access": 3, "ru_sanctions_risk": 1},
    {"country": "Paraguay",       "visa_free_count": 145, "residency_difficulty": 1, "digital_nomad_visa": False, "investor_visa": True,  "eu_member": False, "schengen": False, "ru_visa_free": True,  "ru_banking_access": 3, "ru_sanctions_risk": 1},
    {"country": "Ecuador",        "visa_free_count": 95,  "residency_difficulty": 1, "digital_nomad_visa": True,  "investor_visa": True,  "eu_member": False, "schengen": False, "ru_visa_free": True,  "ru_banking_access": 3, "ru_sanctions_risk": 1},
    {"country": "Peru",           "visa_free_count": 138, "residency_difficulty": 2, "digital_nomad_visa": False, "investor_visa": True,  "eu_member": False, "schengen": False, "ru_visa_free": True,  "ru_banking_access": 3, "ru_sanctions_risk": 1},
    # ── Asia-Pacific ─────────────────────────────────────────────────────────────
    {"country": "Japan",          "visa_free_count": 193, "residency_difficulty": 4, "digital_nomad_visa": True,  "investor_visa": False, "eu_member": False, "schengen": False, "ru_visa_free": False, "ru_banking_access": 2, "ru_sanctions_risk": 3},
    {"country": "South Korea",    "visa_free_count": 192, "residency_difficulty": 4, "digital_nomad_visa": True,  "investor_visa": True,  "eu_member": False, "schengen": False, "ru_visa_free": False, "ru_banking_access": 2, "ru_sanctions_risk": 2},
    {"country": "Singapore",      "visa_free_count": 193, "residency_difficulty": 4, "digital_nomad_visa": False, "investor_visa": True,  "eu_member": False, "schengen": False, "ru_visa_free": False, "ru_banking_access": 2, "ru_sanctions_risk": 2},
    {"country": "Taiwan",         "visa_free_count": 140, "residency_difficulty": 3, "digital_nomad_visa": True,  "investor_visa": True,  "eu_member": False, "schengen": False, "ru_visa_free": False, "ru_banking_access": 2, "ru_sanctions_risk": 2},
    {"country": "Hong Kong",      "visa_free_count": 165, "residency_difficulty": 3, "digital_nomad_visa": False, "investor_visa": True,  "eu_member": False, "schengen": False, "ru_visa_free": True,  "ru_banking_access": 3, "ru_sanctions_risk": 1},
    {"country": "Thailand",       "visa_free_count":  82, "residency_difficulty": 3, "digital_nomad_visa": True,  "investor_visa": True,  "eu_member": False, "schengen": False, "ru_visa_free": True,  "ru_banking_access": 4, "ru_sanctions_risk": 1},
    {"country": "Malaysia",       "visa_free_count": 176, "residency_difficulty": 2, "digital_nomad_visa": True,  "investor_visa": True,  "eu_member": False, "schengen": False, "ru_visa_free": True,  "ru_banking_access": 3, "ru_sanctions_risk": 1},
    {"country": "Indonesia",      "visa_free_count":  75, "residency_difficulty": 3, "digital_nomad_visa": True,  "investor_visa": True,  "eu_member": False, "schengen": False, "ru_visa_free": True,  "ru_banking_access": 4, "ru_sanctions_risk": 1},
    {"country": "Vietnam",        "visa_free_count":  55, "residency_difficulty": 3, "digital_nomad_visa": False, "investor_visa": True,  "eu_member": False, "schengen": False, "ru_visa_free": True,  "ru_banking_access": 3, "ru_sanctions_risk": 1},
    {"country": "Philippines",    "visa_free_count":  67, "residency_difficulty": 2, "digital_nomad_visa": False, "investor_visa": True,  "eu_member": False, "schengen": False, "ru_visa_free": True,  "ru_banking_access": 3, "ru_sanctions_risk": 1},
    {"country": "Australia",      "visa_free_count": 186, "residency_difficulty": 3, "digital_nomad_visa": False, "investor_visa": True,  "eu_member": False, "schengen": False, "ru_visa_free": False, "ru_banking_access": 1, "ru_sanctions_risk": 3},
    {"country": "New Zealand",    "visa_free_count": 186, "residency_difficulty": 3, "digital_nomad_visa": False, "investor_visa": True,  "eu_member": False, "schengen": False, "ru_visa_free": False, "ru_banking_access": 1, "ru_sanctions_risk": 3},
    # ── Africa ───────────────────────────────────────────────────────────────────
    {"country": "Morocco",        "visa_free_count":  68, "residency_difficulty": 2, "digital_nomad_visa": False, "investor_visa": True,  "eu_member": False, "schengen": False, "ru_visa_free": True,  "ru_banking_access": 3, "ru_sanctions_risk": 1},
    {"country": "South Africa",   "visa_free_count": 104, "residency_difficulty": 3, "digital_nomad_visa": True,  "investor_visa": True,  "eu_member": False, "schengen": False, "ru_visa_free": True,  "ru_banking_access": 3, "ru_sanctions_risk": 1},
]
# fmt: on


def collect_visa_data() -> pd.DataFrame:
    """
    Build a curated DataFrame of visa/residency attributes for 60 popular
    emigration destinations and save it to data/raw/visa_data.csv.

    Columns
    -------
    country                : country name
    visa_free_count        : Henley Passport Index 2025 — number of destinations
                             accessible visa-free / visa-on-arrival for that passport
    residency_difficulty   : 1 (very easy) … 5 (very hard) — subjective assessment
                             based on processing time, income threshold, language req.
    digital_nomad_visa     : dedicated remote-worker / digital nomad visa exists
    investor_visa          : golden / investor residency or citizenship-by-investment
    eu_member              : European Union member state
    schengen               : full Schengen Area member (Romania & Bulgaria joined 2025)

    Returns
    -------
    pd.DataFrame  (60 rows × 7 columns)
    """
    VISA_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(_VISA_RECORDS)

    bool_cols = ["digital_nomad_visa", "investor_visa", "eu_member", "schengen", "ru_visa_free"]
    df[bool_cols] = df[bool_cols].astype(bool)
    df["residency_difficulty"] = df["residency_difficulty"].astype(int)
    df["visa_free_count"] = df["visa_free_count"].astype(int)
    df["ru_banking_access"] = df["ru_banking_access"].astype(int)
    df["ru_sanctions_risk"] = df["ru_sanctions_risk"].astype(int)

    df = df.sort_values("country").reset_index(drop=True)

    df.to_csv(VISA_OUTPUT_PATH, index=False, encoding="utf-8")
    print(f"Saved {len(df)} countries to {VISA_OUTPUT_PATH}")

    return df


if __name__ == "__main__":
    df_wb = collect_worldbank()
    print(df_wb.head(10).to_string())
    print()
    df_nb = collect_numbeo()
    print(df_nb.head(10).to_string())
    print()
    df_vd = collect_visa_data()
    print(df_vd.to_string())
