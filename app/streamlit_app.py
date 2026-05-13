"""
Relocation Recommender — Streamlit UI (Cyberpunk 2077 edition, RU)
Запуск:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make src/ importable regardless of working directory
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from recommender import UserProfile, explain, profile_to_weights, recommend
from recommender import _LANG_COUNTRIES, _LOWER_IS_BETTER, _WARM_CLIMATE

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ПЕРЕЕЗД РЕКОМЕНДАТЕЛЬ",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROCESSED_PATH = _ROOT / "data" / "processed" / "countries.csv"

# ── Cyberpunk colour palette ──────────────────────────────────────────────────
C_BG_DARK   = "#08080f"
C_BG_PANEL  = "#12121d"
C_YELLOW    = "#fcee0c"
C_CYAN      = "#00f0ff"
C_MAGENTA   = "#ff003c"
C_TEXT      = "#d0d0d0"
C_MUTED     = "#7a7a8a"

CYBERPUNK_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Exo+2:wght@400;500;600;700;800&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── global ──────────────────────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background:
        radial-gradient(ellipse at 15% 0%, rgba(252,238,12,0.06) 0%, transparent 45%),
        radial-gradient(ellipse at 90% 100%, rgba(0,240,255,0.07) 0%, transparent 45%),
        linear-gradient(180deg, #08080f 0%, #12121d 100%) !important;
    color: #d0d0d0 !important;
    font-family: 'Inter', sans-serif !important;
}

/* scanline overlay */
[data-testid="stAppViewContainer"]::before {
    content: "";
    position: fixed; top: 0; left: 0; right: 0; bottom: 0;
    pointer-events: none;
    background: repeating-linear-gradient(
        0deg,
        rgba(255,255,255,0.015) 0px,
        rgba(255,255,255,0.015) 1px,
        transparent 1px,
        transparent 3px
    );
    z-index: 1;
}
[data-testid="stMain"] { position: relative; z-index: 2; }

/* ── sidebar ─────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #06060d 0%, #0e0e18 100%) !important;
    border-right: 1px solid rgba(252,238,12,0.25) !important;
    box-shadow: inset -2px 0 8px rgba(0,240,255,0.08);
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #fcee0c !important;
    font-family: 'Exo 2', sans-serif !important;
    letter-spacing: 0.05em;
    text-shadow: 0 0 6px rgba(252,238,12,0.5);
}

/* ── headings (main area) ────────────────────────────────────────────── */
h1, h2, h3, h4, h5 {
    font-family: 'Exo 2', sans-serif !important;
    color: #fcee0c !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    text-shadow: 0 0 8px rgba(252,238,12,0.35);
}
h1 {
    border-bottom: 2px solid #fcee0c;
    padding-bottom: 8px;
    margin-bottom: 12px;
}
h2, h3 {
    border-left: 4px solid #fcee0c;
    padding-left: 12px;
}

/* ── paragraphs / lists ──────────────────────────────────────────────── */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span,
[data-testid="stCaption"] {
    font-family: 'Inter', sans-serif !important;
    color: #c0c0c8 !important;
    font-size: 1.02rem !important;
    line-height: 1.45 !important;
}
[data-testid="stCaption"] { color: #7a7a8a !important; font-style: italic; }

/* ── sliders ─────────────────────────────────────────────────────────── */
[data-testid="stSlider"] label {
    font-family: 'Exo 2', sans-serif !important;
    color: #00f0ff !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    font-size: 0.85rem !important;
}
[data-testid="stSlider"] [role="slider"] {
    background-color: #fcee0c !important;
    border: 2px solid #fcee0c !important;
    box-shadow: 0 0 12px rgba(252,238,12,0.7) !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] > div > div {
    background: linear-gradient(90deg, #fcee0c 0%, #ff003c 100%) !important;
}

/* ── multiselect & input ─────────────────────────────────────────────── */
[data-baseweb="select"] > div,
[data-baseweb="input"] > div {
    background: #14141e !important;
    border: 1px solid rgba(0,240,255,0.4) !important;
    border-radius: 0 !important;
    color: #c0c0c8 !important;
}
[data-testid="stMultiSelect"] label,
[data-testid="stSelectbox"] label {
    font-family: 'Exo 2', sans-serif !important;
    color: #00f0ff !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    font-size: 0.85rem !important;
}
[data-baseweb="tag"] {
    background: rgba(252,238,12,0.15) !important;
    border: 1px solid #fcee0c !important;
    border-radius: 0 !important;
    color: #fcee0c !important;
    font-family: 'Exo 2', sans-serif !important;
}

/* ── metrics cards ───────────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: rgba(252,238,12,0.04);
    border: 1px solid rgba(252,238,12,0.3);
    border-left: 4px solid #fcee0c;
    padding: 10px 14px;
    box-shadow: 0 0 12px rgba(252,238,12,0.08);
}
[data-testid="stMetricLabel"] {
    color: #7a7a8a !important;
    font-family: 'Exo 2', sans-serif !important;
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
[data-testid="stMetricValue"] {
    color: #fcee0c !important;
    font-family: 'Exo 2', sans-serif !important;
    text-shadow: 0 0 6px rgba(252,238,12,0.5);
}

/* ── code blocks ─────────────────────────────────────────────────────── */
[data-testid="stCode"], pre, code {
    background: #0a0a14 !important;
    border: 1px solid rgba(0,240,255,0.35) !important;
    border-left: 4px solid #00f0ff !important;
    border-radius: 0 !important;
    color: #c0c0c8 !important;
    font-family: 'JetBrains Mono', monospace !important;
    box-shadow: 0 0 14px rgba(0,240,255,0.08);
}

/* ── dataframe ───────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(252,238,12,0.3) !important;
    background: rgba(8,8,15,0.6);
}
[data-testid="stDataFrame"] [role="columnheader"] {
    background: #1a1a28 !important;
    color: #fcee0c !important;
    font-family: 'Exo 2', sans-serif !important;
    text-transform: uppercase;
    border-bottom: 2px solid #fcee0c !important;
}

/* ── dividers ────────────────────────────────────────────────────────── */
hr {
    border: 0 !important;
    height: 1px !important;
    background: linear-gradient(90deg, transparent 0%, #fcee0c 50%, transparent 100%) !important;
    margin: 18px 0 !important;
}

/* ── expander ────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    background: rgba(0,240,255,0.03);
    border: 1px solid rgba(0,240,255,0.3) !important;
    border-radius: 0 !important;
    box-shadow: 0 0 10px rgba(0,240,255,0.08);
}
[data-testid="stExpander"] summary {
    font-family: 'Exo 2', sans-serif !important;
    color: #00f0ff !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

/* ── alerts ──────────────────────────────────────────────────────────── */
[data-testid="stAlert"] {
    background: rgba(255,0,60,0.08) !important;
    border: 1px solid #ff003c !important;
    border-left: 5px solid #ff003c !important;
    border-radius: 0 !important;
    color: #ff003c !important;
    font-family: 'Exo 2', sans-serif !important;
}

/* ── tooltip help icon ───────────────────────────────────────────────── */
[data-testid="stTooltipIcon"] svg { color: #00f0ff !important; }

/* ── main title ──────────────────────────────────────────────────────── */
.cyberpunk-title {
    font-family: 'Exo 2', sans-serif;
    font-size: 3.4rem;
    color: #fcee0c;
    letter-spacing: 0.08em;
    text-shadow:
        0 0 10px rgba(252,238,12,0.7),
        2px 0 #ff003c,
        -2px 0 #00f0ff;
    margin: 0;
    padding: 0;
    line-height: 1;
}
.cyberpunk-subtitle {
    font-family: 'Exo 2', sans-serif;
    font-size: 0.9rem;
    color: #00f0ff;
    letter-spacing: 0.15em;
    margin-bottom: 18px;
    text-transform: uppercase;
}
.cyberpunk-info {
    font-family: 'Inter', sans-serif;
    border: 1px solid #00f0ff66;
    border-left: 4px solid #00f0ff;
    padding: 14px 18px;
    background: rgba(0,240,255,0.04);
    color: #c0c0c8;
    margin: 10px 0 18px 0;
    box-shadow: 0 0 12px rgba(0,240,255,0.08);
}
.cyberpunk-info b { color: #fcee0c; }
.cyberpunk-info code {
    background: rgba(252,238,12,0.1) !important;
    color: #fcee0c !important;
    border: 1px solid rgba(252,238,12,0.3) !important;
    padding: 1px 6px;
    font-size: 0.85rem;
}

/* hide streamlit branding */
#MainMenu, footer { visibility: hidden; }
header {
    background: transparent !important;
    border: none !important;
}

/* ── sidebar toggle button (collapse / expand) ───────────────────────── */
[data-testid="collapsedControl"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    z-index: 999999 !important;
    background: #12121d !important;
    border: 1px solid rgba(252,238,12,0.5) !important;
    border-radius: 0 4px 4px 0 !important;
    box-shadow: 2px 0 8px rgba(252,238,12,0.2) !important;
    color: #fcee0c !important;
}
[data-testid="collapsedControl"]:hover {
    background: rgba(252,238,12,0.15) !important;
    border-color: #fcee0c !important;
    box-shadow: 2px 0 14px rgba(252,238,12,0.4) !important;
}
[data-testid="collapsedControl"] svg {
    fill: #fcee0c !important;
    color: #fcee0c !important;
}
</style>
"""

# ── Russian country name translations ────────────────────────────────────────
COUNTRY_RU: dict[str, str] = {
    # Europe
    "Germany":            "Германия",
    "Portugal":           "Португалия",
    "Spain":              "Испания",
    "France":             "Франция",
    "Netherlands":        "Нидерланды",
    "Italy":              "Италия",
    "Switzerland":        "Швейцария",
    "Austria":            "Австрия",
    "Belgium":            "Бельгия",
    "Ireland":            "Ирландия",
    "United Kingdom":     "Великобритания",
    "Sweden":             "Швеция",
    "Norway":             "Норвегия",
    "Denmark":            "Дания",
    "Finland":            "Финляндия",
    "Iceland":            "Исландия",
    "Poland":             "Польша",
    "Czech Republic":     "Чехия",
    "Hungary":            "Венгрия",
    "Slovakia":           "Словакия",
    "Slovenia":           "Словения",
    "Romania":            "Румыния",
    "Bulgaria":           "Болгария",
    "Croatia":            "Хорватия",
    "Greece":             "Греция",
    "Malta":              "Мальта",
    "Cyprus":             "Кипр",
    "Estonia":            "Эстония",
    "Latvia":             "Латвия",
    "Lithuania":          "Литва",
    "Serbia":             "Сербия",
    "Montenegro":         "Черногория",
    "Albania":            "Албания",
    "Luxembourg":         "Люксембург",
    "Bosnia and Herzegovina": "Босния и Герцеговина",
    "North Macedonia":    "Северная Македония",
    "Moldova":            "Молдова",
    "Ukraine":            "Украина",
    "Belarus":            "Беларусь",
    # Caucasus & Turkey
    "Georgia":            "Грузия",
    "Armenia":            "Армения",
    "Turkey":             "Турция",
    "Azerbaijan":         "Азербайджан",
    # Middle East
    "United Arab Emirates": "ОАЭ",
    "Israel":             "Израиль",
    "Saudi Arabia":       "Саудовская Аравия",
    "Qatar":              "Катар",
    "Kuwait":             "Кувейт",
    "Bahrain":            "Бахрейн",
    "Jordan":             "Иордания",
    # North America
    "United States":      "США",
    "Canada":             "Канада",
    "Mexico":             "Мексика",
    "Panama":             "Панама",
    "Costa Rica":         "Коста-Рика",
    # South America
    "Brazil":             "Бразилия",
    "Argentina":          "Аргентина",
    "Colombia":           "Колумбия",
    "Chile":              "Чили",
    "Uruguay":            "Уругвай",
    "Paraguay":           "Парагвай",
    "Ecuador":            "Эквадор",
    "Peru":               "Перу",
    "Venezuela":          "Венесуэла",
    "Bolivia":            "Боливия",
    # Asia-Pacific
    "Japan":              "Япония",
    "South Korea":        "Южная Корея",
    "Singapore":          "Сингапур",
    "Taiwan":             "Тайвань",
    "Hong Kong":          "Гонконг",
    "China":              "Китай",
    "Thailand":           "Таиланд",
    "Malaysia":           "Малайзия",
    "Indonesia":          "Индонезия",
    "Vietnam":            "Вьетнам",
    "Philippines":        "Филиппины",
    "Australia":          "Австралия",
    "New Zealand":        "Новая Зеландия",
    "India":              "Индия",
    "Kazakhstan":         "Казахстан",
    "Uzbekistan":         "Узбекистан",
    # Africa
    "Morocco":            "Марокко",
    "South Africa":       "ЮАР",
    "Egypt":              "Египет",
    "Kenya":              "Кения",
    "Nigeria":            "Нигерия",
}


# ── radar axes ────────────────────────────────────────────────────────────────
RADAR_AXES: list[tuple[str, str, bool]] = [
    ("safety_index",          "Безопасность",      False),
    ("healthcare_index",      "Здравоохранение",   False),
    ("cost_of_living_index",  "Доступность",       True),
    ("life_expectancy",       "Долголетие",        False),
    ("purchasing_power_index","Покупат. сила",     False),
    ("residency_difficulty",  "Простота визы",     True),
    ("gdp_per_capita",        "ВВП",               False),
    ("pollution_index",       "Экология",          True),
]

# Russian display label → internal lowercase English key
LANG_DISPLAY_TO_INTERNAL: dict[str, str] = {
    "Английский":   "english",
    "Испанский":    "spanish",
    "Немецкий":     "german",
    "Французский":  "french",
    "Португальский":"portuguese",
    "Итальянский":  "italian",
    "Русский":      "russian",
    "Арабский":     "arabic",
    "Тайский":      "thai",
    "Малайский":    "malay",
    "Японский":     "japanese",
    "Корейский":    "korean",
    "Китайский":    "chinese",
    "Турецкий":     "turkish",
    "Грузинский":   "georgian",
    "Армянский":    "armenian",
    "Румынский":    "romanian",
    "Сербский":     "serbian",
    "Албанский":    "albanian",
}
LANG_OPTIONS = sorted(LANG_DISPLAY_TO_INTERNAL.keys())


# ── caching ───────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="ЗАГРУЗКА ДАННЫХ ...")
def load_data() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_PATH, index_col="iso3")


@st.cache_data(show_spinner="РАСЧЁТ РЕКОМЕНДАЦИЙ ...")
def cached_recommend(
    budget: float,
    safety: int,
    climate: bool,
    visa: int,
    health: int,
    languages: tuple[str, ...],
    user_is_russian: bool = False,
    needs_banking: bool = False,
    has_eu_visa: bool = False,
) -> pd.DataFrame:
    df = load_data()
    profile = UserProfile(
        budget_usd=budget,
        safety_priority=safety,
        climate_warm=climate,
        visa_easy_priority=visa,
        healthcare_priority=health,
        language=list(languages),
        user_is_russian=user_is_russian,
        needs_banking_access=needs_banking,
        has_eu_visa=has_eu_visa,
    )
    return recommend(profile, df, top_n=len(df))


# ── plotly figures (cyberpunk theme) ─────────────────────────────────────────

def _build_choropleth(df: pd.DataFrame, rec: pd.DataFrame) -> go.Figure:
    merged = df[["country"]].copy()
    merged["score"] = rec["score"].reindex(merged.index)
    merged = merged.reset_index(drop=False)

    hover_cols = ["safety_index", "cost_of_living_index",
                  "healthcare_index", "residency_difficulty"]
    for c in hover_cols:
        if c in df.columns:
            merged[c] = df[c].reindex(merged["iso3"]).values

    hover_labels = {
        "safety_index":         "Безопасность",
        "cost_of_living_index": "Стоимость жизни",
        "healthcare_index":     "Здравоохранение",
        "residency_difficulty": "Сложность визы",
    }

    fig = px.choropleth(
        merged,
        locations="iso3",
        locationmode="ISO-3",
        color="score",
        hover_name="country",
        hover_data={
            "iso3": False,
            "score": ":.4f",
            **{c: ":.3f" for c in hover_cols if c in merged.columns},
        },
        color_continuous_scale=[
            (0.0, "#ff003c"),
            (0.4, "#7d1a3e"),
            (0.6, "#3a3a5c"),
            (0.8, "#00bcd4"),
            (1.0, "#fcee0c"),
        ],
        range_color=(
            merged["score"].dropna().quantile(0.05),
            merged["score"].dropna().quantile(0.95),
        ),
        labels={"score": "Оценка", **hover_labels},
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        coloraxis_colorbar=dict(
            title=dict(text="ОЦЕНКА", font=dict(color=C_YELLOW, family="Exo 2", size=12)),
            thickness=14,
            len=0.7,
            tickformat=".2f",
            tickfont=dict(color=C_CYAN, family="Exo 2", size=10),
            outlinecolor=C_YELLOW,
            outlinewidth=1,
        ),
        geo=dict(
            showframe=False,
            showcoastlines=True,
            coastlinecolor=C_CYAN,
            coastlinewidth=0.6,
            showland=True,
            landcolor="#1a1a28",
            showocean=True,
            oceancolor="#08080f",
            bgcolor="rgba(0,0,0,0)",
            projection_type="natural earth",
            lakecolor="#08080f",
            countrycolor="#2a2a40",
            showcountries=True,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=460,
        font=dict(color=C_TEXT, family="Inter"),
    )
    return fig


def _build_radar(iso3: str, df: pd.DataFrame) -> go.Figure:
    row = df.loc[iso3] if iso3 in df.index else None
    if row is None:
        return go.Figure()

    values, labels = [], []
    for col, label, invert in RADAR_AXES:
        if col in df.columns:
            v = float(row[col]) if pd.notna(row[col]) else 0.5
            values.append(round(1 - v if invert else v, 3))
            labels.append(label.upper())

    values_closed = values + [values[0]]
    labels_closed = labels + [labels[0]]
    country_name  = row.get("country", iso3) if hasattr(row, "get") else iso3

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=labels_closed,
        fill="toself",
        fillcolor="rgba(252,238,12,0.18)",
        line=dict(color=C_YELLOW, width=2.5),
        marker=dict(size=8, color=C_YELLOW,
                    line=dict(color=C_MAGENTA, width=1)),
        name=country_name,
        hovertemplate="%{theta}: %{r:.3f}<extra></extra>",
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickfont=dict(size=9, color=C_MUTED, family="Exo 2"),
                gridcolor="rgba(0,240,255,0.2)",
                linecolor="rgba(0,240,255,0.4)",
            ),
            angularaxis=dict(
                tickfont=dict(size=10, color=C_CYAN, family="Exo 2"),
                linecolor="rgba(252,238,12,0.4)",
                gridcolor="rgba(252,238,12,0.15)",
            ),
            bgcolor="rgba(8,8,15,0.6)",
        ),
        showlegend=False,
        title=dict(
            text=f"{country_name.upper()}",
            font=dict(size=18, color=C_YELLOW, family="Exo 2"),
            x=0.5,
        ),
        margin=dict(l=50, r=50, t=80, b=40),
        height=420,
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _build_top10_display(rec: pd.DataFrame, df: pd.DataFrame, russian_mode: bool = False) -> pd.DataFrame:
    """Top-N таблица без эмодзи (ASCII-маркеры)."""
    rows = []
    for iso3, row in rec.iterrows():
        raw = df.loc[iso3] if iso3 in df.index else pd.Series(dtype=float)
        en_name = raw.get("country", iso3)
        entry: dict = {
            "ISO3":              iso3,
            "Страна":            en_name,
            "На русском":        COUNTRY_RU.get(en_name, "—"),
            "Оценка":            round(row["score"], 4),
            "Стоимость жизни":   round(float(raw.get("cost_of_living_index", 0)), 3),
            "Безопасность":      round(float(raw.get("safety_index", 0)), 3),
            "Сложность визы":    round(float(raw.get("residency_difficulty", 0)), 3),
            "ЕС":                "[+]" if raw.get("eu_member", 0) else "[ ]",
            "Климат":            "ТЁПЛЫЙ" if iso3 in _WARM_CLIMATE else "УМЕР.",
        }
        if russian_mode:
            # Visa status for Russian passport
            ru_free = raw.get("ru_visa_free", 0)
            entry["Безвиз РФ"] = "ДА" if (isinstance(ru_free, (bool, int, float)) and float(ru_free) > 0.5) else "НЕТ"
            # Banking access level
            bank_val = raw.get("ru_banking_access", 0)
            bank_int = int(round(float(bank_val) * 5)) if 0 < float(bank_val) <= 1 else int(bank_val)
            bank_int = max(1, min(5, bank_int))
            entry["Банкинг"] = "+" * bank_int
        rows.append(entry)
    return pd.DataFrame(rows)


# ── sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar() -> tuple[UserProfile, int]:
    with st.sidebar:
        st.markdown(
            "<h2 style='margin-top:0'>КОНФИГУРАЦИЯ</h2>",
            unsafe_allow_html=True,
        )

        # ── Russian user preset ──────────────────────────────────────────
        st.markdown("<h3>ПРОФИЛЬ РОССИЯНИНА</h3>", unsafe_allow_html=True)
        user_is_russian = st.checkbox(
            "Я ИЗ РОССИИ",
            value=True,
            help=(
                "Включает учёт российского паспорта: безвизовый въезд, "
                "санкционные ограничения, доступность банков. "
                "Дружественные страны получают бонус, враждебные — штраф."
            ),
        )
        needs_banking = False
        has_eu_visa = False
        if user_is_russian:
            needs_banking = st.checkbox(
                "НУЖЕН БАНКОВСКИЙ СЧЁТ",
                value=False,
                help=(
                    "Увеличивает вес параметра ru_banking_access. "
                    "Страны, где россиянин может открыть счёт, получают бонус."
                ),
            )
            has_eu_visa = st.checkbox(
                "ЕСТЬ ШЕНГЕНСКАЯ / ЕС ВИЗА",
                value=False,
                help=(
                    "Если у вас уже есть действующая виза — снимается дополнительный "
                    "штраф за страны Шенгена, требующие визу для паспорта РФ."
                ),
            )

        st.divider()

        budget = st.slider(
            "МЕСЯЧНЫЙ БЮДЖЕТ (USD)",
            min_value=500, max_value=10_000, value=3_000, step=100,
            help="Максимальная стоимость жизни в месяц. Жёсткий фильтр: "
                 "страны с cost_of_living_index выше порога исключаются.",
        )

        st.markdown("<h3>ПРИОРИТЕТЫ</h3>", unsafe_allow_html=True)

        safety = st.slider(
            "БЕЗОПАСНОСТЬ", 1, 5, 3,
            help="Приоритет безопасности (1 = не важно, 5 = критично).",
        )
        climate = st.slider(
            "ТЁПЛЫЙ КЛИМАТ", 1, 5, 3,
            help="При значении ≥ 3 активируется бонус за тёплые страны.",
        )
        visa = st.slider(
            "ЛЁГКОСТЬ ВИЗЫ", 1, 5, 3,
            help=(
                "Насколько важна простая процедура переезда. "
                "5 = критично (нужны программы DNV / золотая виза / простая ВНЖ); "
                "1 = готов(а) к сложной бюрократии."
            ),
        )
        health = st.slider(
            "МЕДИЦИНА", 1, 5, 3,
            help="Приоритет качества медицины (1 = не важно, 5 = критично).",
        )

        st.markdown("<h3>ЯЗЫКИ</h3>", unsafe_allow_html=True)
        default_langs = ["Русский", "Английский"] if user_is_russian else ["Английский"]
        langs_ru = st.multiselect(
            "ЗНАКОМЫЕ ЯЗЫКИ",
            options=LANG_OPTIONS,
            default=default_langs,
            help="Страны с этими языками получают бонус +0.05 к оценке.",
        )

        st.divider()
        top_n = st.slider("ТОП-N РЕЗУЛЬТАТОВ", 5, 30, 10)

        st.caption(
            "алгоритм: взвешенное косинусное сходство между профилем "
            "и признаками страны + бонусы за кластер / климат / язык"
            + (" / безвиз РФ / санкции / банкинг" if user_is_russian else "")
        )

    # Map Russian language labels back to internal keys
    langs_internal = [LANG_DISPLAY_TO_INTERNAL[l] for l in langs_ru]

    profile = UserProfile(
        budget_usd=float(budget),
        safety_priority=safety,
        climate_warm=climate >= 3,
        visa_easy_priority=visa,
        healthcare_priority=health,
        language=langs_internal,
        user_is_russian=user_is_russian,
        needs_banking_access=needs_banking,
        has_eu_visa=has_eu_visa,
    )
    return profile, top_n


# ── visa difficulty explainer block ──────────────────────────────────────────

VISA_INFO_BLOCK = """
<div class="cyberpunk-info">
<b>ЛЁГКОСТЬ ВИЗЫ И СЛОЖНОСТЬ ВИЗЫ — что это</b><br><br>

В системе есть <b>два связанных параметра</b>:
<br><br>

<b>1. Слайдер «ЛЁГКОСТЬ ВИЗЫ» (1–5)</b> — это ваш приоритет:
насколько вам важно, чтобы получить ВНЖ или гражданство было просто.
Он управляет весом следующих признаков страны:
<code>residency_difficulty</code>, <code>digital_nomad_visa</code>,
<code>visa_free_count</code>, <code>eu_member</code>, <code>schengen</code>.
<br><br>

<b>2. Признак страны «Сложность визы» (residency_difficulty, шкала 1–5)</b>
— объективная оценка того, насколько сложно получить долгосрочное
резидентство в конкретной стране. После MinMax-нормализации хранится в
[0, 1]: <b>0 = легче всего, 1 = сложнее всего</b>.
<br><br>

<b>Расшифровка шкалы 1–5:</b>
<br>
&nbsp;&nbsp;<code>1</code> &nbsp;Очень легко — <b>GEO, PAN, PRY, ALB, MNE, SRB</b>
(длительный безвиз или поселение по факту проживания)
<br>
&nbsp;&nbsp;<code>2</code> &nbsp;Легко — <b>PRT D7, ESP non-lucrative, MEX, MYS MM2H</b>
(понятный путь при наличии пассивного дохода)
<br>
&nbsp;&nbsp;<code>3</code> &nbsp;Средне — <b>DEU, FRA, NLD, CZE, AUT, AUS</b>
(стандартные рабочие / семейные / бизнес-визы)
<br>
&nbsp;&nbsp;<code>4</code> &nbsp;Сложно — <b>USA, CAN, CHE, JPN, KOR, ISR, DNK</b>
(квоты, спонсорство, высокие пороги)
<br>
&nbsp;&nbsp;<code>5</code> &nbsp;Очень сложно — закрытые системы, длинные сроки
<br><br>

В колонке таблицы <b>«Сложность визы»</b> вы видите нормализованное
значение [0, 1]: чем выше — тем сложнее переехать. Чтобы система
рекомендовала простые страны, увеличьте слайдер <b>ЛЁГКОСТЬ ВИЗЫ</b>.
</div>
"""


RU_MODE_INFO_BLOCK = """
<div class="cyberpunk-info">
<b>РЕЖИМ «Я ИЗ РОССИИ» — как он влияет на рекомендации</b><br><br>

Когда флажок <b>«Я ИЗ РОССИИ»</b> активен, система учитывает три дополнительных
фактора, специфичных для граждан РФ:<br><br>

<b>1. Безвизовый въезд с паспортом РФ</b> (<code>ru_visa_free</code>)<br>
Страны, куда россиянин может въехать без визы или по visa-on-arrival,
получают бонус <code>+0.08</code> к оценке. Если у вас <b>нет шенгенской визы</b>,
а страна входит в Шенген и требует визу — дополнительный штраф <code>-0.04</code>.
<br><br>

<b>2. Санкционный риск</b> (<code>ru_sanctions_risk</code>, шкала 1–3)<br>
&nbsp;&nbsp;<code>1</code> — Низкий: нет ограничений для россиян (<b>GEO, ARM, SRB, TUR, ARE, THA</b>)<br>
&nbsp;&nbsp;<code>2</code> — Средний: частичные ограничения (<b>HUN, ISR, KOR, SGP</b>)<br>
&nbsp;&nbsp;<code>3</code> — Высокий: полный пакет санкций, сложности с банками и визами
(<b>USA, GBR, EU, CAN, AUS, JPN, CHE</b>)<br>
Штраф пропорционален уровню: <code>-0.06 * risk</code>.<br><br>

<b>3. Доступность банков для россиян</b> (<code>ru_banking_access</code>, шкала 1–5)<br>
&nbsp;&nbsp;<code>5</code> — Открытие счёта без проблем (<b>ARM, GEO, SRB</b>)<br>
&nbsp;&nbsp;<code>4</code> — Возможно с документами (<b>TUR, ARE, THA, MNE</b>)<br>
&nbsp;&nbsp;<code>3</code> — С трудностями (<b>ISR, MEX, BRA, LatAm</b>)<br>
&nbsp;&nbsp;<code>2</code> — Сложно, отказы часты (<b>EU</b>)<br>
&nbsp;&nbsp;<code>1</code> — Практически невозможно (<b>USA, GBR, CHE, EST, LVA, LTU</b>)<br>
Если включён флажок <b>«Нужен банковский счёт»</b>, бонус <code>+0.04 * access</code>.<br><br>

<b>«Есть шенгенская / ЕС виза»</b> — если у вас уже есть действующая виза,
дополнительный штраф за Шенген-страны снимается.
</div>
"""


# ── main layout ───────────────────────────────────────────────────────────────

def main() -> None:
    st.markdown(CYBERPUNK_CSS, unsafe_allow_html=True)

    # ── hero title ────────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="margin-top:-30px">
            <p class="cyberpunk-title">ПЕРЕЕЗД РЕКОМЕНДАТЕЛЬ</p>
            <p class="cyberpunk-subtitle">
                версия 2.077 &nbsp;|&nbsp; подбор страны по профилю &nbsp;|&nbsp;
                world bank · numbeo · linucb
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not PROCESSED_PATH.exists():
        st.error(
            f"ДАННЫЕ ОТСУТСТВУЮТ: `{PROCESSED_PATH.relative_to(_ROOT)}` не найден.\n\n"
            "Запустите пайплайн предобработки:\n"
            "```\n"
            "python src/data_collector.py\n"
            "python src/preprocessor.py\n"
            "```"
        )
        st.stop()

    df = load_data()
    profile, top_n = render_sidebar()

    # ── compute recommendations ───────────────────────────────────────────
    try:
        rec = cached_recommend(
            budget=profile.budget_usd,
            safety=profile.safety_priority,
            climate=profile.climate_warm,
            visa=profile.visa_easy_priority,
            health=profile.healthcare_priority,
            languages=tuple(sorted(profile.language)),
            user_is_russian=profile.user_is_russian,
            needs_banking=profile.needs_banking_access,
            has_eu_visa=profile.has_eu_visa,
        )
    except ValueError as exc:
        st.warning(f"СБОЙ: {exc}")
        st.stop()

    # ── visa info block ──────────────────────────────────────────────────
    with st.expander("ПОКАЗАТЬ ОБЪЯСНЕНИЕ ПАРАМЕТРА «СЛОЖНОСТЬ ВИЗЫ»", expanded=False):
        st.markdown(VISA_INFO_BLOCK, unsafe_allow_html=True)

    # ── Russian-specific info block ──────────────────────────────────────
    if profile.user_is_russian:
        with st.expander("КАК РАБОТАЕТ РЕЖИМ «Я ИЗ РОССИИ»", expanded=False):
            st.markdown(RU_MODE_INFO_BLOCK, unsafe_allow_html=True)

    # ── choropleth ────────────────────────────────────────────────────────
    st.markdown("## КАРТА СОВМЕСТИМОСТИ СТРАН")
    choropleth_fig = _build_choropleth(df, rec)

    map_event = st.plotly_chart(
        choropleth_fig,
        use_container_width=True,
        key="choropleth_map",
        on_select="rerun",
        selection_mode="points",
    )

    map_selected: str | None = None
    if map_event and map_event.selection and map_event.selection.points:
        pt = map_event.selection.points[0]
        map_selected = pt.get("location") or pt.get("customdata", [None])[0]

    if map_selected and map_selected in df.index:
        st.session_state["selected_iso3"] = map_selected

    # ── top-N table ───────────────────────────────────────────────────────
    st.markdown(f"## ТОП-{top_n} СТРАН ДЛЯ ПЕРЕЕЗДА")

    rec_topn   = rec.head(top_n)
    display_df = _build_top10_display(rec_topn, df, russian_mode=profile.user_is_russian)

    table_event = st.dataframe(
        display_df.set_index("ISO3"),
        use_container_width=True,
        height=min(42 * top_n + 60, 460),
        on_select="rerun",
        selection_mode="single-row",
        key="topn_table",
    )

    if table_event and table_event.selection and table_event.selection.rows:
        row_idx    = table_event.selection.rows[0]
        table_iso3 = display_df.iloc[row_idx]["ISO3"]
        st.session_state["selected_iso3"] = table_iso3

    # ── country detail ────────────────────────────────────────────────────
    selected = st.session_state.get("selected_iso3")
    if not selected or selected not in df.index:
        selected = rec.index[0]
        st.session_state["selected_iso3"] = selected

    country_name = df.loc[selected, "country"] if "country" in df.columns else selected

    st.markdown("---")
    st.markdown(f"## ДЕТАЛЬНЫЙ АНАЛИЗ — {country_name.upper()} [{selected}]")

    col_radar, col_explain = st.columns([1, 1], gap="large")

    with col_radar:
        st.plotly_chart(
            _build_radar(selected, df),
            use_container_width=True,
            key="radar_chart",
        )

        if selected in rec.index:
            row = rec.loc[selected]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("ИТОГ. ОЦЕНКА",   f"{row['score']:.4f}")
            m2.metric("КОСИНУС",        f"{row['base_sim']:.4f}")
            m3.metric("БОНУС КЛАСТЕРА", f"+{row['cluster_bonus']:.3f}")
            m4.metric("КЛИМАТ/ЯЗЫК",
                      f"+{row['climate_bonus'] + row['language_bonus']:.3f}")

            if profile.user_is_russian:
                r1, r2, r3 = st.columns(3)
                r1.metric("БЕЗВИЗ РФ",     f"{row.get('ru_visa_bonus', 0):+.3f}")
                r2.metric("САНКЦИИ",        f"{row.get('ru_sanctions_penalty', 0):+.3f}")
                r3.metric("БАНКИНГ РФ",     f"{row.get('ru_banking_bonus', 0):+.3f}")

    with col_explain:
        st.markdown("### ПОЧЕМУ ЭТА СТРАНА?")
        if selected in rec.index:
            explanation = explain(
                country_iso3=selected,
                profile=profile,
                df=df,
                rec_df=rec,
            )
            st.code(explanation, language=None)
        else:
            st.info("страна отфильтрована по бюджету")

    # ── weight chart (collapsed) ──────────────────────────────────────────
    with st.expander("ВЕСА ПРИЗНАКОВ ДЛЯ ВАШЕГО ПРОФИЛЯ", expanded=False):
        weights = profile_to_weights(profile)
        w_df = (
            pd.DataFrame.from_dict(weights, orient="index", columns=["weight"])
            .sort_values("weight", ascending=True)
        )
        fig_w = px.bar(
            w_df,
            x="weight",
            y=w_df.index,
            orientation="h",
            labels={"weight": "ВЕС", "y": "ПРИЗНАК"},
            color="weight",
            color_continuous_scale=[
                (0.0, "#00f0ff"),
                (0.5, "#fcee0c"),
                (1.0, "#ff003c"),
            ],
            height=380,
        )
        fig_w.update_layout(
            showlegend=False,
            coloraxis_showscale=False,
            margin=dict(l=0, r=20, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(8,8,15,0.6)",
            font=dict(color=C_TEXT, family="Exo 2"),
            xaxis=dict(gridcolor="rgba(0,240,255,0.15)", color=C_CYAN),
            yaxis=dict(gridcolor="rgba(0,240,255,0.15)", color=C_CYAN),
        )
        st.plotly_chart(fig_w, use_container_width=True)


if __name__ == "__main__":
    main()
