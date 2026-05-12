// Build a comprehensive presentation document for the relocation_recommender project.
// Run: node build_doc.js
const fs = require('fs');
const path = require('path');

// Resolve global docx package
const GLOBAL_MODULES = "C:\\Users\\user\\AppData\\Roaming\\npm\\node_modules";
const {
    Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
    Header, Footer, AlignmentType, PageOrientation, LevelFormat,
    HeadingLevel, BorderStyle, WidthType, ShadingType, PageNumber,
    TabStopType, TabStopPosition, PageBreak, TableOfContents,
    Bookmark, InternalHyperlink
} = require(path.join(GLOBAL_MODULES, 'docx'));

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────
const ACCENT_DARK   = "1F4E79";
const ACCENT_LIGHT  = "D5E8F0";
const CODE_BG       = "F2F2F2";
const MUTED         = "595959";

const border = (color="CCCCCC") => ({ style: BorderStyle.SINGLE, size: 4, color });
const cellBorders = (color="CCCCCC") => ({
    top: border(color), bottom: border(color),
    left: border(color), right: border(color),
});

function P(text, opts = {}) {
    return new Paragraph({
        spacing: { after: 120, line: 300 },
        alignment: opts.center ? AlignmentType.CENTER : (opts.justify ? AlignmentType.JUSTIFY : AlignmentType.LEFT),
        ...opts,
        children: Array.isArray(text)
            ? text
            : [new TextRun({ text, size: 22, font: "Calibri", ...opts.run })],
    });
}

function H1(text) {
    return new Paragraph({
        heading: HeadingLevel.HEADING_1,
        pageBreakBefore: true,
        children: [new TextRun({ text, bold: true, size: 36, color: ACCENT_DARK, font: "Calibri" })],
    });
}

function H2(text) {
    return new Paragraph({
        heading: HeadingLevel.HEADING_2,
        spacing: { before: 240, after: 120 },
        children: [new TextRun({ text, bold: true, size: 28, color: ACCENT_DARK, font: "Calibri" })],
    });
}

function H3(text) {
    return new Paragraph({
        heading: HeadingLevel.HEADING_3,
        spacing: { before: 180, after: 80 },
        children: [new TextRun({ text, bold: true, size: 24, color: "2F5496", font: "Calibri" })],
    });
}

function bullet(text, level = 0) {
    return new Paragraph({
        numbering: { reference: "bullets", level },
        spacing: { after: 60 },
        children: Array.isArray(text) ? text : [new TextRun({ text, size: 22, font: "Calibri" })],
    });
}

function num(text, level = 0) {
    return new Paragraph({
        numbering: { reference: "numbers", level },
        spacing: { after: 60 },
        children: Array.isArray(text) ? text : [new TextRun({ text, size: 22, font: "Calibri" })],
    });
}

function code(text) {
    // Monospace-styled block; one paragraph per line
    return text.split("\n").map(line => new Paragraph({
        spacing: { after: 0, line: 260 },
        shading: { fill: CODE_BG, type: ShadingType.CLEAR },
        children: [new TextRun({ text: line || " ", font: "Consolas", size: 18, color: "1F2937" })],
    }));
}

function quote(text) {
    return new Paragraph({
        spacing: { before: 120, after: 120 },
        indent: { left: 360 },
        border: { left: { style: BorderStyle.SINGLE, size: 18, color: ACCENT_DARK, space: 10 } },
        children: [new TextRun({ text, italics: true, size: 22, color: MUTED, font: "Calibri" })],
    });
}

function tableCell(text, opts = {}) {
    return new TableCell({
        borders: cellBorders(),
        width: { size: opts.width, type: WidthType.DXA },
        shading: opts.header
            ? { fill: ACCENT_DARK, type: ShadingType.CLEAR }
            : (opts.alt ? { fill: ACCENT_LIGHT, type: ShadingType.CLEAR } : undefined),
        margins: { top: 80, bottom: 80, left: 120, right: 120 },
        children: [new Paragraph({
            alignment: opts.center ? AlignmentType.CENTER : AlignmentType.LEFT,
            children: [new TextRun({
                text,
                bold: opts.header || opts.bold,
                color: opts.header ? "FFFFFF" : "000000",
                size: 20,
                font: "Calibri",
            })],
        })],
    });
}

function makeTable(rows, colWidths) {
    const totalWidth = colWidths.reduce((a, b) => a + b, 0);
    const tableRows = rows.map((cells, rowIdx) =>
        new TableRow({
            children: cells.map((c, colIdx) =>
                tableCell(c, {
                    width: colWidths[colIdx],
                    header: rowIdx === 0,
                    alt: rowIdx > 0 && rowIdx % 2 === 0,
                    center: colIdx > 0,   // left-align first column
                })),
        })
    );
    return new Table({
        width: { size: totalWidth, type: WidthType.DXA },
        columnWidths: colWidths,
        rows: tableRows,
    });
}

// ─────────────────────────────────────────────────────────────────────────────
// Content blocks
// ─────────────────────────────────────────────────────────────────────────────

const titlePage = [
    new Paragraph({
        spacing: { before: 2400, after: 240 },
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "Relocation Recommender", bold: true, size: 56, color: ACCENT_DARK, font: "Calibri" })],
    }),
    new Paragraph({
        spacing: { after: 480 },
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "Персонализированная ML-система рекомендаций по переезду", size: 28, color: MUTED, font: "Calibri" })],
    }),
    new Paragraph({
        spacing: { after: 120 },
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "Документация проекта и ход разработки", italics: true, size: 24, color: MUTED, font: "Calibri" })],
    }),
    new Paragraph({
        spacing: { before: 3600 },
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "Технологии: Python · scikit-learn · HDBSCAN · UMAP · Plotly · Streamlit · LinUCB", size: 20, color: MUTED, font: "Calibri" })],
    }),
    new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "Источники данных: World Bank API · Numbeo · Henley Passport Index", size: 20, color: MUTED, font: "Calibri" })],
    }),
];

const intro = [
    H1("1. Введение и постановка задачи"),

    H2("1.1. Проблема"),
    P("Выбор страны для переезда — многомерная задача, в которой пользователь должен балансировать десятки факторов: стоимость жизни, безопасность, качество медицины, простоту визового режима, климат, языковой барьер и др. Существующие источники (Numbeo, Nomad List) дают лишь сырые рейтинги, не учитывая индивидуальные приоритеты пользователя."),
    P("Особую актуальность задача приобрела в контексте россиян: с 2022 года граждане РФ сталкиваются с уникальными ограничениями — санкциями, проблемами с банковским обслуживанием, закрытием части направлений. Система учитывает эти факторы как первоклассные признаки."),

    H2("1.2. Цель проекта"),
    P("Построить рабочий end-to-end ML-пайплайн, который:"),
    bullet("Собирает данные о странах из открытых источников (88 стран × 23 признака)"),
    bullet("Кластеризует страны по социально-экономическим характеристикам"),
    bullet("Принимает структурированный профиль пользователя (бюджет, приоритеты, языки, гражданство)"),
    bullet("Выдаёт топ-N стран с объяснением каждой рекомендации"),
    bullet("Учитывает российский контекст: безвизовый въезд, санкции, банкинг"),
    bullet("Имеет интерактивный веб-интерфейс с картой и сравнительными графиками"),
    bullet("Оптимизирует веса признаков через online-обучение (Contextual Bandit / LinUCB)"),

    H2("1.3. Ключевые метрики (итог)"),
    makeTable(
        [
            ["Метрика", "Значение", "Комментарий"],
            ["Hit Rate (все профили)", "96.6 %", "57 из 59 синтетических профилей"],
            ["Hit Rate (российские)", "100.0 %", "9 из 9 российских профилей"],
            ["NDCG@10 (все)", "0.7608", "Косинусная модель + российские бонусы"],
            ["NDCG@10 (чистые)", "0.7820", "Профили без шума"],
            ["NDCG@10 (российские)", "0.7368", "Профили с user_is_russian=True"],
            ["NDCG@10 LinUCB", "0.3491*", "На сложном тест-сете, +27.11 % vs base"],
            ["Coverage", "85.2 %", "75 из 88 стран попадают в рекомендации"],
            ["Diversity@10", "0.051", "Умеренная разнородность"],
            ["Стран в датасете", "88", "Из 238 исходных (фильтр >40% NA)"],
            ["Признаков", "23", "После MinMax-нормализации, вкл. 3 Russian-specific"],
        ],
        [3200, 2200, 4000]
    ),
    P("* Тест-сет в evaluate.py и rl_agent.py разный (59 vs 200 профилей, разные распределения); ниже разбираем подробнее.", { run: { italics: true, color: MUTED, size: 20 } }),
];

const architecture = [
    H1("2. Архитектура проекта"),

    H2("2.1. Слои системы"),
    P("Проект разделён на четыре слабосвязанных слоя — каждый можно использовать независимо:"),
    bullet([
        new TextRun({ text: "Data layer  ", bold: true, font: "Calibri", size: 22 }),
        new TextRun({ text: "— сбор и предобработка (data_collector.py → preprocessor.py)", size: 22, font: "Calibri" }),
    ]),
    bullet([
        new TextRun({ text: "ML layer  ", bold: true, font: "Calibri", size: 22 }),
        new TextRun({ text: "— кластеризация (clusterer.py), рекомендации (recommender.py)", size: 22, font: "Calibri" }),
    ]),
    bullet([
        new TextRun({ text: "Eval layer  ", bold: true, font: "Calibri", size: 22 }),
        new TextRun({ text: "— оценка качества (evaluate.py), online-обучение (rl_agent.py)", size: 22, font: "Calibri" }),
    ]),
    bullet([
        new TextRun({ text: "UI layer  ", bold: true, font: "Calibri", size: 22 }),
        new TextRun({ text: "— веб-приложение (app/streamlit_app.py)", size: 22, font: "Calibri" }),
    ]),

    H2("2.2. Поток данных"),
    ...code(
        "External APIs ──→ data/raw/*.csv ──→ data/processed/countries.csv\n" +
        "                                            │\n" +
        "        ┌───────────────────────────────────┼────────────────────────────┐\n" +
        "        ▼                                   ▼                            ▼\n" +
        "  clusterer.py                       recommender.py                   evaluate.py\n" +
        "  (KMeans + HDBSCAN + UMAP)          (UserProfile + cosine sim)       (NDCG@10)\n" +
        "        │                                   │                            │\n" +
        "        └────────────┬──────────────────────┘                            ▼\n" +
        "                     ▼                                              rl_agent.py\n" +
        "             streamlit_app.py                                       (LinUCB)\n" +
        "             (Plotly choropleth, radar, table)"
    ),

    H2("2.3. Структура папок"),
    ...code(
        "relocation_recommender/\n" +
        "├── data/\n" +
        "│   ├── raw/              worldbank.csv, numbeo.csv, visa_data.csv\n" +
        "│   └── processed/        countries.csv, metrics.json, rl_training.json\n" +
        "├── notebooks/\n" +
        "│   ├── 01_eda.ipynb\n" +
        "│   └── figures/          PNG-графики из EDA + clusters.html\n" +
        "├── src/\n" +
        "│   ├── data_collector.py\n" +
        "│   ├── preprocessor.py\n" +
        "│   ├── clusterer.py\n" +
        "│   ├── recommender.py\n" +
        "│   ├── explainer.py      (стаб, логика в recommender.explain)\n" +
        "│   ├── evaluate.py\n" +
        "│   └── rl_agent.py\n" +
        "├── app/\n" +
        "│   └── streamlit_app.py\n" +
        "├── requirements.txt\n" +
        "└── README.md"
    ),
];

const dataCollection = [
    H1("3. Сбор данных — data_collector.py"),

    H2("3.1. Три источника"),
    P("В файле реализованы три функции, по одной на источник. Все они сохраняют результат в data/raw/ и используются независимо."),

    H3("collect_worldbank()"),
    P("Скачивает 5 индикаторов через официальный World Bank API:"),
    makeTable(
        [
            ["Индикатор", "Код API", "Описание"],
            ["ВВП на душу населения", "NY.GDP.PCAP.CD", "USD в текущих ценах"],
            ["Безработица", "SL.UEM.TOTL.ZS", "% рабочей силы"],
            ["Продолжительность жизни", "SP.DYN.LE00.IN", "Лет при рождении"],
            ["Грамотность", "SE.ADT.LITR.ZS", "% взрослого населения"],
            ["Коэффициент Джини", "SI.POV.GINI", "Неравенство 0-100"],
        ],
        [3400, 2400, 3600]
    ),
    quote("Особенности реализации: пагинация (266 записей × 5 страниц для GDP), retry с экспоненциальным backoff (3 попытки), параметр mrv=1 для запроса самого свежего значения по каждой стране."),
    P("Важный баг, обнаруженный при первом запуске: World Bank API возвращает ISO3 в поле countryiso3code, а country.id содержит двухбуквенный alpha-2. Из-за этого первая версия фильтра len(iso3) != 3 отбрасывала все записи. Фикс: брать countryiso3code и одновременно проверять что country.id — валидный alpha-2 (отбрасывает агрегаты типа 1W, ZH, ZI)."),

    H3("collect_numbeo()"),
    P("Парсит две HTML-страницы Numbeo через BeautifulSoup:"),
    bullet("cost-of-living/rankings_by_country.jsp → 5 индексов цен"),
    bullet("quality-of-life/rankings_by_country.jsp → 4 индекса качества"),
    P("Скрейпинг устойчив к изменениям вёрстки: сначала ищется <table id=\"t2\">, затем fallback на <table class=\"stripe\">. Колонки определяются по частичному совпадению заголовка (без учёта регистра), поэтому переименование \"Health Care Index\" → \"Healthcare Index\" не сломает парсинг."),
    quote("Этический аспект: между запросами стоит пауза delay=2.0 секунды (вежливый скрапинг). User-Agent имитирует Chrome — без него Numbeo возвращает 403."),

    H3("collect_visa_data()"),
    P("В отличие от первых двух функций, эта НЕ ходит в сеть — данные захардкожены в виде списка из 65 словарей. Это сознательное решение:"),
    bullet("Visa-free count берётся из Henley Passport Index 2025 (нет публичного API)"),
    bullet("Residency difficulty — авторская субъективная оценка 1-5 (документации не существует)"),
    bullet("DNV / investor visa — статус актуален на 2025, меняется редко"),
    bullet("EU / Schengen — формально-юридические признаки"),
    P("Документ покрывает 65 наиболее популярных стран эмиграции, разбитых по регионам с комментариями."),

    H3("Три дополнительных колонки для россиян"),
    P("В visa_data добавлены три признака, специфичных для граждан РФ:"),
    makeTable(
        [
            ["Колонка", "Тип", "Описание"],
            ["ru_visa_free", "bool", "Безвизовый въезд / visa-on-arrival с паспортом РФ (по состоянию на 2025)"],
            ["ru_banking_access", "1–5", "Доступность банковского обслуживания: 5 = легко открыть счёт (ARM, GEO, SRB), 1 = практически невозможно (USA, CHE, страны Балтии)"],
            ["ru_sanctions_risk", "1–3", "Санкционный риск: 1 = нет ограничений (TUR, ARE, THA), 2 = частичные, 3 = полный пакет (EU, USA, GBR, AUS)"],
        ],
        [2800, 1200, 5400]
    ),
    P("Страны вне visa_data (не в списке 65) получают нейтральные дефолты в preprocessor.py: ru_visa_free=0, ru_banking_access=3, ru_sanctions_risk=1 — чтобы не попасть под фильтр >40% NA."),

    H2("3.2. Полученные объёмы"),
    makeTable(
        [
            ["Файл", "Стран", "Колонок"],
            ["worldbank.csv", "238", "6 (5 индикаторов + name)"],
            ["numbeo.csv", "155", "10 (9 индексов + name)"],
            ["visa_data.csv", "65", "10 (вкл. 3 Russian-specific)"],
        ],
        [3600, 2400, 3400]
    ),
];

const preprocessing = [
    H1("4. Препроцессинг — preprocessor.py"),

    H2("4.1. Задача и сложность"),
    P("Три источника используют разные ключи для идентификации стран:"),
    bullet("World Bank: ISO3 (USA, DEU, ...)"),
    bullet("Numbeo: текстовое название (\"United States\", \"South Korea\", ...)"),
    bullet("Visa data: текстовое название (\"UAE\", \"Czech Republic\", ...)"),
    P("Чтобы их объединить, нужно нормализовать все источники к единому ключу — ISO3."),

    H2("4.2. Алгоритм _to_iso3()"),
    P("Трёхуровневое разрешение с кэшированием:"),
    num("Словарь ручных переопределений (_NAME_OVERRIDES) для проблемных случаев: \"Hong Kong\" → HKG, \"South Korea\" → KOR, \"UAE\" → ARE, \"Czechia\" → CZE и др."),
    num("pycountry.countries.get(name=...) — точное совпадение по официальному названию"),
    num("pycountry.countries.search_fuzzy(...) — нечёткий поиск (ловит \"United States\" → \"United States of America\")"),
    P("Результаты кэшируются в _iso3_cache, чтобы не вызывать fuzzy-поиск повторно."),

    H2("4.3. Импутация пропусков"),
    P("После outer-merge остаётся много NaN. Простое заполнение глобальной медианой даёт смещение — для бедной африканской страны нет смысла подставлять медиану всех 87 стран. Поэтому используется двухуровневая схема:"),
    num("Назначение макрорегиона через _ISO3_TO_REGION (14 регионов: Northern Europe, Southeast Asia, ...)"),
    num("Импутация: NaN → медиана по региону → fallback на глобальную медиану (если регион полностью пустой)"),

    H2("4.4. Финальная статистика"),
    makeTable(
        [
            ["Шаг", "Значение"],
            ["Стран после outer join", "238"],
            ["Порог пропусков", "40 %"],
            ["Дропнуто", "150"],
            ["Осталось", "88"],
            ["Признаков", "23"],
            ["Пропусков после импутации", "0"],
            ["Колонок, прошедших MinMax-scale", "18"],
            ["Boolean-колонок (остались 0/1)", "5 (вкл. ru_visa_free)"],
        ],
        [5800, 3600]
    ),

    H2("4.5. Допущения"),
    quote("Порог 40% пропусков — компромисс. Меньше → теряем интересные страны (Iran, Bolivia). Больше → импутация слишком агрессивна. 40% даёт 88 стран — достаточно для кластеризации (~12 стран на кластер при K=7)."),
    P("Важная деталь: при добавлении трёх Russian-specific колонок страны, отсутствующие в visa_data (23 из 88), получали NaN по всем трём — это поднимало долю пропусков выше 40% и выбивало их из датасета. Решение: заполнять нейтральными дефолтами ещё до расчёта missing_rate (шаг 2b в build_dataset())."),
];

const eda = [
    H1("5. Разведочный анализ — notebooks/01_eda.ipynb"),

    H2("5.1. Содержание ноутбука"),
    P("Ноутбук выполняется в одну команду (jupyter nbconvert --execute) и генерирует 4 PNG-графика в notebooks/figures/."),

    H3("Корреляционная матрица"),
    P("Тепловая карта 20×20 показывает сильные корреляции:"),
    bullet("GDP per capita ↔ healthcare_index (r ≈ 0.78)"),
    bullet("Cost of living ↔ purchasing power (r ≈ 0.62)"),
    bullet("Safety ↔ pollution (отрицательная, r ≈ -0.55)"),
    bullet("EU member ↔ Schengen (почти 1:1)"),

    H3("Топ-10 по 8 ключевым индексам"),
    P("Сетка 2×4 barh-чартов. Для индексов \"чем ниже — тем лучше\" (cost, pollution, unemployment) сортировка инвертирована."),

    H3("Scatter: стоимость жизни vs безопасность"),
    P("Каждая точка — страна с подписью. Пунктирные линии по медианам делят плоскость на 4 квадранта: Cheap & Safe, Expensive & Safe, Cheap & Risky, Expensive & Risky. Помогает увидеть \"sweet spot\" — страны в верхнем левом квадранте."),

    H3("Scatter: GDP vs healthcare"),
    P("Точки окрашены по членству в ЕС, добавлена линия тренда (np.polyfit) и значение r в углу. Подтверждает гипотезу о сильной положительной корреляции."),
];

const clustering = [
    H1("6. Кластеризация — clusterer.py"),

    H2("6.1. Постановка"),
    P("Цель: сгруппировать 87 стран в осмысленные кластеры на основе всех 20 признаков, чтобы рекомендатель мог давать +0.1 бонус к score для стран из «правильного» кластера."),

    H2("6.2. KMeans с подбором K"),
    P("Функция run_kmeans(df, k_range=(4,12)) перебирает K от 4 до 12, для каждого считает:"),
    bullet("Inertia — внутрикластерную дисперсию (для метода локтя)"),
    bullet("Silhouette Score — насколько кластеры компактны и отделены"),
    P("Оптимальный K выбирается по максимуму силуэтной оценки. Локоть определяется как точка максимальной кривизны (вторая производная inertia)."),

    H3("Результат на реальных данных"),
    makeTable(
        [
            ["K", "Inertia", "Silhouette"],
            ["4", "929.5", "0.1794"],
            ["5", "877.0", "0.1864"],
            ["6", "767.1", "0.2388"],
            ["7", "691.1", "0.2599 ← best"],
            ["8", "625.5", "0.2207"],
            ["9", "563.8", "0.2275"],
            ["10-12", "→ 499.2", "снижается"],
        ],
        [2000, 3500, 3900]
    ),

    H2("6.3. HDBSCAN — альтернативная кластеризация"),
    P("В отличие от KMeans, HDBSCAN не требует задавать K и помечает выбросы как шум (-1). На наших данных нашёл 5 кластеров и 9 шумовых точек (Сингапур, ЮАР, Нигерия, Швейцария, Исландия — все статистически нетипичны)."),

    H2("6.4. UMAP-проекция в 2D"),
    P("UMAP с параметрами n_neighbors=12, min_dist=0.25 сохраняет глобальную структуру лучше t-SNE. Результат сохраняется как интерактивный HTML через Plotly (clusters.html) с hover-подписями и подписями страны на каждой точке."),

    H2("6.5. Именование кластеров — самое интересное"),
    P("Функция label_clusters() анализирует каждый кластер по двум осям:"),
    bullet("Центроид в [0,1]-пространстве признаков: gdp, cost, safety, eu_member, ..."),
    bullet("Доминирующий макрорегион: подсчёт стран по _ISO3_REGION"),
    P("Затем 12 правил с порогами выдают человеко-читаемое имя. Например:"),
    ...code(
        "if eu_frac >= 0.6 and gdp >= 0.55 and cost >= 0.55:\n" +
        "    return \"Богатая Европа (ЕС)\"\n" +
        "if e_asia_frac >= 0.35 and gulf_frac >= 0.30:\n" +
        "    return \"Богатая Азия и Залив\"\n" +
        "if n >= 15 and latam_frac >= 0.35:\n" +
        "    return \"Латинская Америка и развивающийся мир\""
    ),

    H3("Итоговые 7 кластеров"),
    makeTable(
        [
            ["#", "Название", "Стран", "Примеры"],
            ["0", "Доступная Европа (ЕС)", "21", "PRT, POL, HRV, BGR, MLT"],
            ["1", "Доступная Юго-Восточная Азия", "5", "THA, MYS, IDN, VNM, PHL"],
            ["2", "Англосфера — высокий уровень жизни", "9", "USA, CAN, AUS, NZL, SGP, GBR"],
            ["3", "Богатая Азия и Залив", "9", "JPN, KOR, ARE, QAT, SAU, TWN"],
            ["4", "Отдельная страна (ZAF)", "1", "ZAF — статистический выброс"],
            ["5", "Развивающийся мир", "30", "BRA, ARG, GEO, IND, MAR, MEX"],
            ["6", "Шенген — высокий уровень жизни", "12", "DEU, CHE, SWE, NOR, NLD, FRA"],
        ],
        [600, 3600, 800, 4400]
    ),
];

const recommender = [
    H1("7. Рекомендательная система — recommender.py"),

    H2("7.1. UserProfile (dataclass)"),
    P("Структурированный профиль пользователя:"),
    ...code(
        "@dataclass\n" +
        "class UserProfile:\n" +
        "    budget_usd:          float       # месячный бюджет\n" +
        "    safety_priority:     int         # 1-5\n" +
        "    climate_warm:        bool\n" +
        "    visa_easy_priority:  int         # 1-5\n" +
        "    healthcare_priority: int         # 1-5\n" +
        "    language:            list[str]   # знакомые языки\n" +
        "    # Russian-specific fields\n" +
        "    user_is_russian:     bool = False  # включает RU-бонусы/штрафы\n" +
        "    needs_banking_access:bool = False  # нужен счёт в иностранном банке\n" +
        "    has_eu_visa:         bool = False  # есть действующая шенгенская виза"
    ),
    P("Валидация диапазона 1-5 происходит в __post_init__, языки нормализуются к нижнему регистру. Три новых поля включаются исключительно при user_is_russian=True и не влияют на стандартный расчёт."),

    H2("7.2. profile_to_weights() — приоритеты → веса"),
    P("Приоритеты пользователя нелинейно отображаются в веса признаков для косинусного сходства:"),
    bullet("safety_priority=5 → safety_weight = 2.5 (доминирует)"),
    bullet("budget_usd=1500 → cost_weight ≈ 1.8 (дорогие страны сильно штрафуются)"),
    bullet("visa_easy_priority=3 → residency/DNV/eu_member умеренные"),
    bullet("user_is_russian=True → ru_sanctions_risk weight = 1.8 (ключевой фактор)"),
    bullet("needs_banking_access=True → ru_banking_access weight = 1.5"),
    P("Бюджет преобразуется в \"urgency\" по формуле: budget_urgency = 1 - sqrt(max_cost_index)."),

    H2("7.3. Пайплайн recommend()"),
    makeTable(
        [
            ["Шаг", "Логика"],
            ["1. Жёсткий фильтр", "cost_of_living_index ≤ budget_ceiling по 6-ступенчатой шкале"],
            ["2. Инверсия", "residency_difficulty, cost, pollution, ru_sanctions_risk → 1 - x (выше = лучше)"],
            ["3. Идеальный вектор", "Целевые значения из профиля: tight budget → ideal_cost=0.78, safety=5 → ideal_safety=1.0"],
            ["4. Косинусное сходство", "sim(W·ideal, W·country)"],
            ["5. Базовые бонусы", "+0.10 для лучшего кластера, +0.05 язык, +0.03 климат"],
            ["6. Российские бонусы", "+0.08 за безвиз РФ, −0.04 Шенген без визы, −0.06×risk санкции, +0.04×access банкинг"],
            ["7. Сортировка", "Top-N по итоговому score"],
        ],
        [2200, 7200]
    ),

    H2("7.4. Маппинг бюджет → cost ceiling"),
    P("Кусочно-постоянная функция _budget_to_max_cost():"),
    makeTable(
        [
            ["Бюджет $/мес", "Max cost index"],
            ["≤ 1 500", "0.28"],
            ["≤ 2 500", "0.42"],
            ["≤ 3 500", "0.57"],
            ["≤ 5 000", "0.72"],
            ["≤ 8 000", "0.88"],
            ["> 8 000", "1.00 (без фильтра)"],
        ],
        [4800, 4600]
    ),

    H2("7.5. explain() — обоснование рекомендации"),
    P("Для любой страны функция выдаёт развёрнутое объяснение:"),
    bullet("Декомпозиция score: косинус + бонус кластера + бонус климата + бонус языка"),
    bullet("Bar chart по каждому из 12 признаков с raw-значением (ASCII: █████░░░░░)"),
    bullet("Совпадение климата ✓ / нет"),
    bullet("Знакомые языки страны"),
    bullet("Автоматический список плюсов (топ-3 признака) и минусов (нижние-3)"),

    H2("7.6. Тест на 3 профилях"),
    P("Скрипт запускался на трёх синтетических пользователях:"),

    H3("Профиль 1: Цифровой кочевник"),
    P("$2k бюджет, тёплый климат, EN+ES, visa=5. Топ: MLT → PAN → CRI → MEX → GRC."),

    H3("Профиль 2: Семья с детьми"),
    P("$5k бюджет, safety=5, healthcare=5, EN+DE. Топ: AUT → DEU → CAN → AUS → GBR."),

    H3("Профиль 3: Пенсионер"),
    P("$3.5k бюджет, тёплый климат, safety=5, EN. Топ: MLT → ARE → PRT → CYP → EST."),

    quote("Поведение алгоритма \"осмысленно\": Мальта побеждает для номада и пенсионера одновременно (английский язык + ЕС + тёплый климат + доступная цена), но для семьи побеждает Австрия (безопасность и здравоохранение)."),
];

const ui = [
    H1("8. Веб-интерфейс — app/streamlit_app.py"),

    H2("8.1. Дизайн — Cyberpunk 2077"),
    P("Интерфейс выполнен в стиле Cyberpunk 2077: тёмный фон (#08080f), неоновые акценты жёлтый/cyan/пурпурный, строчные линии (scanline overlay). Шрифты:"),
    bullet("Exo 2 — заголовки: геометричный, sci-fi, но читабельный"),
    bullet("Inter — основной текст: современный, не режет глаза"),
    bullet("JetBrains Mono — блоки с кодом и цифрами"),
    P("CSS инжектируется через st.markdown(CYBERPUNK_CSS, unsafe_allow_html=True). Весь интерфейс переведён на русский язык."),

    H2("8.2. Левая панель"),

    H3("Блок «Профиль россиянина»"),
    bullet("Чекбокс «Я ИЗ РОССИИ» (по умолчанию включён) — активирует российские бонусы/штрафы"),
    bullet("Чекбокс «Нужен банковский счёт» — увеличивает вес ru_banking_access"),
    bullet("Чекбокс «Есть шенгенская / ЕС виза» — снимает штраф за Шенген-страны"),

    H3("Основные параметры"),
    bullet("Слайдер бюджета $500–10 000 (шаг $100)"),
    bullet("Безопасность 1–5, Тёплый климат 1–5 (≥3 → True), Лёгкость визы 1–5, Медицина 1–5"),
    bullet("Мультиселект знакомых языков (18 опций на русском; по умолчанию Русский + Английский при RU-режиме)"),
    bullet("Слайдер Top-N (5–30)"),

    H2("8.3. Основная область"),

    H3("Choropleth-карта мира"),
    P("Plotly choropleth с цветовой шкалой (красный → жёлтый). Диапазон color-axis обрезан по 5–95 перцентилям. Тёмная тема карты в стиле Cyberpunk: суша #1a1a28, океан #08080f, контуры cyan. Hover: страна, score, ключевые метрики. Клик → выбор страны."),

    H3("Таблица топ-N"),
    P("Колонки: ISO3, Страна (англ.), На русском, Оценка, Стоимость жизни, Безопасность, Сложность визы, ЕС, Климат. При включённом режиме россиянина добавляются колонки Безвиз РФ и Банкинг (++++ шкала). Строки кликабельны."),

    H3("Radar chart"),
    P("Plotly Scatterpolar по 8 осям: Безопасность, Здравоохранение, Доступность, Долголетие, Покупательная сила, Простота визы, ВВП, Экология. Цветовая схема Cyberpunk: жёлтая заливка, cyan-сетка."),

    H3("Детальный анализ"),
    P("Метрики оценки: Итог. оценка, Косинус, Бонус кластера, Климат/Язык. При RU-режиме — три дополнительные карточки: Безвиз РФ, Санкции, Банкинг РФ. Блок «Почему эта страна?» — развёрнутое объяснение через recommender.explain()."),

    H2("8.4. Информационные блоки"),
    bullet("«Объяснение параметра Сложность визы» — шкала 1–5 с примерами стран"),
    bullet("«Как работает режим Я из России» — описание трёх факторов: безвизовость, санкционный риск, банкинг"),

    H2("8.5. Развёртывание"),
    P("Приложение опубликовано на Streamlit Community Cloud:"),
    ...code("https://github.com/PacTu6ka/relocation_recommender\napp/streamlit_app.py → main branch"),
    P("Локальный запуск:"),
    ...code("streamlit run app/streamlit_app.py\n# → http://localhost:8501"),
];

const evaluation = [
    H1("9. Оффлайн-оценка — evaluate.py"),

    H2("9.1. Постановка"),
    P("Чтобы измерить качество рекомендаций без живых пользователей, генерируется 50 синтетических профилей, для каждого известна \"идеальная\" страна. Если рекомендатель правильно работает, эта страна должна попадать в топ-10."),

    H2("9.2. Генерация синтетических профилей"),
    P("Из 50 целевых стран (географически разнообразный список: Европа, Азия, Латинская Америка, ...) формируется профиль, который должен \"любить\" эту страну:"),
    bullet("budget_usd — минимальный, при котором страна проходит фильтр (по cost_of_living_index)"),
    bullet("safety_priority — round(safety_index × 5)"),
    bullet("healthcare_priority — round(healthcare_index × 5)"),
    bullet("visa_easy_priority — round((1 - residency_difficulty) × 5)"),
    bullet("climate_warm — страна в _WARM_CLIMATE?"),
    bullet("language — официальные / распространённые языки страны"),
    P("Половина (25) — \"чистые\" профили, вторая половина — с шумом ±1 на приоритеты (тест устойчивости)."),

    H2("9.3. Метрики"),

    H3("NDCG@10"),
    P("Discounted Cumulative Gain нормализованный. Для бинарной релевантности (страна либо идеальная, либо нет):"),
    ...code(
        "NDCG@10 = 1 / log2(rank + 1)   если идеал в топ-10\n" +
        "        = 0                     иначе"
    ),
    P("Среднее по 50 профилям."),

    H3("Coverage"),
    P("Доля стран из датасета, попадающих хотя бы в одну рекомендацию топ-10 (по всем 50 профилям). Показывает не \"схлопывается\" ли система на узкое подмножество."),

    H3("Diversity@10"),
    P("Среднее попарное косинусное расстояние между признаковыми векторами 10 рекомендованных стран. Низкое значение → \"копии одного типа\". Высокое → разнообразные предложения."),

    H2("9.4. Результаты"),
    makeTable(
        [
            ["Метрика", "Значение"],
            ["Hit Rate (все 59 профилей)", "96.6 % (57/59)"],
            ["Hit Rate (российские)", "100.0 % (9/9)"],
            ["NDCG@10 (все)", "0.7608"],
            ["NDCG@10 (чистые)", "0.7820"],
            ["NDCG@10 (шумные)", "0.7483"],
            ["NDCG@10 (российские)", "0.7368"],
            ["Coverage", "85.2 % (75/88)"],
            ["Diversity@10", "0.0508"],
        ],
        [5800, 3600]
    ),

    H3("Распределение по рангу"),
    P("Из 57 попаданий: 30 на 1 месте (53 %), 14 на 2 месте, 8 на 3 месте, 4 на 4 месте, 1 на 7 месте."),

    H3("Российские профили — топ результаты"),
    P("9 профилей с user_is_russian=True (целевые: GEO, ARM, SRB, ARE, THA, MNE, IDN, ARG, MEX):"),
    bullet("GEO, SRB, ARE, THA — rank 1 (идеальное совпадение)"),
    bullet("ARM — rank 2"),
    bullet("MNE, IDN, ARG, MEX — rank 3"),
    P("Все 9 попали в топ-10 (Hit Rate 100%). Diversity@10 у российских профилей выше (0.079 vs 0.035) — система рекомендует более разнообразный географический микс."),

    H3("Два промаха (из базовых профилей)"),
    bullet("POL (Польша) — вытесняется соседями (Чехия, Венгрия) с более выгодным профилем"),
    bullet("IND (Индия) — низкий бюджет $1500 + плохое здравоохранение + загрязнение уводят за топ-10"),

    quote("Интересный артефакт: российские профили показали более высокое Diversity@10 (0.079), чем базовые (0.035–0.050). Это связано с тем, что дружественных стран много и они географически разнообразны: Кавказ, Балканы, Ближний Восток, ЮВА, Латинская Америка."),
];

const rl = [
    H1("10. Online-обучение — rl_agent.py"),

    H2("10.1. Зачем"),
    P("profile_to_weights() — это статическая эвристика. Можем ли мы выучить, что для разных типов пользователей нужны разные \"тонкие\" корректировки весов? Это классическая задача contextual bandit."),

    H2("10.2. Постановка как Multi-Armed Bandit"),
    makeTable(
        [
            ["Элемент", "Реализация"],
            ["Контекст x", "9-мерный embedding профиля: [budget/10k, safety/5, climate, visa/5, health/5, has_english, has_latin_lang, has_continental_lang, bias]"],
            ["Действие a", "Одна из 8 \"рук\" — шаблонов корректировки весов: baseline, boost_safety, boost_health, boost_cost, boost_visa, boost_eu, damp_gdp, boost_clean"],
            ["Награда r", "NDCG@10 относительно известной идеальной страны (как в evaluate.py)"],
        ],
        [2800, 6600]
    ),

    H2("10.3. LinUCB — алгоритм"),
    P("Disjoint LinUCB: для каждой руки a поддерживаются:"),
    bullet("A_a ∈ ℝ^(d×d) — precision-матрица (init: I)"),
    bullet("b_a ∈ ℝ^d — взвешенная сумма наград × контекстов (init: 0)"),

    H3("Action selection (UCB)"),
    ...code(
        "θ_a = A_a⁻¹ b_a                                      # ridge regression\n" +
        "UCB(a) = θ_a · x + α √(x · A_a⁻¹ · x)               # exploration bonus\n" +
        "â = argmax_a UCB(a)"
    ),

    H3("Update"),
    ...code(
        "A_a ← A_a + x xᵀ\n" +
        "b_a ← b_a + r · x"
    ),
    P("Где α=0.6 — коэффициент исследования. Реализация занимает ровно 32 строки чистого кода."),

    H2("10.4. Обучение"),
    P("1000 итераций случайных синтетических пользователей. На каждом шаге:"),
    num("Семплируем (профиль, идеал) из 60 целевых стран"),
    num("Считаем контекст x = _context_vector(profile)"),
    num("base_weights = profile_to_weights(profile)"),
    num("Агент выбирает руку → новые веса = base × шаблон_руки"),
    num("Запускаем рекомендатель с этими весами, получаем top-10"),
    num("r = NDCG@10 vs известный идеал → обновляем агента"),

    H2("10.5. Результаты"),
    makeTable(
        [
            ["", "Baseline", "LinUCB"],
            ["NDCG@10 (200 тестов)", "0.2746", "0.3491"],
            ["Δ улучшение", "—", "+27.11 %"],
            ["Wins (agent > base)", "—", "70 (35 %)"],
            ["Losses (agent < base)", "—", "25 (12 %)"],
            ["Ties", "—", "105 (52 %)"],
        ],
        [3600, 2900, 2900]
    ),
    P("Распределение рук на тесте: boost_cost (28.5%), boost_safety (21%), boost_visa (18%), boost_eu (13%), baseline (10.5%). Боты практически не используют boost_health и не используют boost_clean — узкие ниши."),

    quote("Важный нюанс: тестовый сет в rl_agent.py труднее, чем в evaluate.py. Случайные профили генерируются с +20% jitter бюджета и ±1 шумом на каждом приоритете. Поэтому baseline NDCG=0.27 ниже, чем 0.79 в evaluate.py — это просто другая нагрузка."),
];

const assumptions = [
    H1("11. Допущения и ограничения"),

    H2("11.1. Данные"),
    bullet("Статичный snapshot — данные за 2024-2025. Реальный продукт нуждался бы в апдейте раз в квартал."),
    bullet("Numbeo — crowd-sourced, точность зависит от количества репортов. Для редко посещаемых стран индексы могут быть смещены."),
    bullet("Visa difficulty — субъективная оценка автора. Реальный продукт нуждался бы в опросе экспертов / парсинге официальных правил."),
    bullet("ВВП на душу как прокси для \"уровня жизни\" — спорно для стран с высоким Джини (Бразилия, ЮАР)."),

    H2("11.2. Модель"),
    bullet("Косинусное сходство предполагает, что все измерения сравнимы по \"единицам\". MinMax-scaling это обеспечивает, но \"safety на 0.8\" не эквивалентно \"healthcare на 0.8\" в реальной полезности."),
    bullet("Бонусы +0.10/+0.05/+0.03 подобраны эвристически. Их можно было бы откалибровать через A/B-тест на живых пользователях."),
    bullet("Жёсткий фильтр по бюджету — пороговая функция. В реальности бюджет это распределение с дисперсией, а не точка отсечения."),

    H2("11.3. Оценка"),
    bullet("Синтетические профили — circular evaluation. Идеал в evaluate.py определяется из тех же признаков, что использует рекомендатель. Это завышает оценку. Честный benchmark нуждался бы в живых пользователях с независимо записанным \"идеалом\"."),
    bullet("NDCG@10 при бинарной релевантности игнорирует, что страны 2 и 3 могут быть почти идеально подходящими."),

    H2("11.4. Покрытие"),
    bullet("87 стран — это после фильтрации >40% NA. Например, Куба, КНДР, Иран — отсутствуют из-за Numbeo. Для них рекомендация невозможна."),
    bullet("Языковая карта _LANG_COUNTRIES упрощена. Швейцария говорит на 4 языках, в нашей модели — только французский+немецкий."),

    H2("11.5. Этическое"),
    bullet("\"Идеальная страна\" — социально нагруженный концепт. Алгоритм может неявно усиливать миграцию из бедных стран в богатые."),
    bullet("Climate features не учитывают климат-изменения — Греция через 20 лет может стать менее привлекательной."),
];

const russian_adaptation = [
    H1("12. Адаптация для россиян"),

    H2("12.1. Контекст"),
    P("С 2022 года граждане РФ столкнулись с уникальными ограничениями при эмиграции, не отражёнными в стандартных международных рейтингах. Стандартный recommender не учитывал, что для человека с паспортом РФ разница между Грузией и Германией — не только в стоимости жизни, но и в возможности въехать, открыть счёт, получить ВНЖ без санкционных последствий."),

    H2("12.2. Три новых признака"),
    makeTable(
        [
            ["Признак", "Диапазон", "Вес в модели", "Примеры (лучшие)"],
            ["ru_visa_free", "0 / 1", "+0.08 бонус", "GEO, ARM, SRB, TUR, ARE, THA, LatAm"],
            ["ru_banking_access", "1–5 → [0,1]", "+0.04 × val (если нужен счёт)", "ARM (5), GEO (5), SRB (5), TUR (4)"],
            ["ru_sanctions_risk", "1–3 → [0,1]", "−0.06 × risk (инвертирован)", "GEO, SRB, THA = 1; EU, USA, GBR = 3"],
        ],
        [2600, 1600, 2800, 2400]
    ),

    H2("12.3. Логика бонусов в recommend()"),
    ...code(
        "if profile.user_is_russian:\n" +
        "    # +0.08 за безвизовый въезд\n" +
        "    ru_visa_bonus = where(ru_visa_free > 0.5, +0.08, 0)\n" +
        "\n" +
        "    # -0.04 за Шенген без визы (снимается при has_eu_visa=True)\n" +
        "    if not profile.has_eu_visa:\n" +
        "        ru_visa_bonus += where(schengen & ~ru_visa_free, -0.04, 0)\n" +
        "\n" +
        "    # санкционный штраф: -0.06 × risk (0–1 после MinMax)\n" +
        "    ru_sanctions_penalty = -0.06 × ru_sanctions_risk\n" +
        "\n" +
        "    # банкинг: +0.04 × access (если needs_banking_access)\n" +
        "    if profile.needs_banking_access:\n" +
        "        ru_banking_bonus = +0.04 × ru_banking_access"
    ),

    H2("12.4. Результат на российских тест-профилях"),
    P("9 профилей с целевыми странами (популярные направления эмиграции из РФ):"),
    makeTable(
        [
            ["Целевая страна", "Ранг", "NDCG@10"],
            ["Грузия (GEO)", "1", "1.0000"],
            ["Сербия (SRB)", "1", "1.0000"],
            ["ОАЭ (ARE)", "1", "1.0000"],
            ["Таиланд (THA)", "1", "1.0000"],
            ["Армения (ARM)", "2", "0.6309"],
            ["Черногория (MNE)", "3", "0.5000"],
            ["Индонезия (IDN)", "3", "0.5000"],
            ["Аргентина (ARG)", "3", "0.5000"],
            ["Мексика (MEX)", "3", "0.5000"],
        ],
        [3400, 2000, 4000]
    ),
    quote("100% Hit Rate на российских профилях означает, что все 9 дружественных стран попали в рекомендации. Страны с высоким санкционным риском (EU, USA, GBR) опустились, не исчезнув полностью — пользователь сам принимает финальное решение."),
];

const conclusion = [
    H1("13. Итоги и дальнейшие улучшения"),

    H2("13.1. Что получилось"),
    bullet([
        new TextRun({ text: "End-to-end ML-пайплайн ", bold: true, font: "Calibri", size: 22 }),
        new TextRun({ text: "от сбора сырых данных до интерактивного веб-интерфейса", size: 22, font: "Calibri" }),
    ]),
    bullet([
        new TextRun({ text: "7 осмысленных кластеров ", bold: true, font: "Calibri", size: 22 }),
        new TextRun({ text: "(silhouette 0.26), независимо подтверждённых HDBSCAN", size: 22, font: "Calibri" }),
    ]),
    bullet([
        new TextRun({ text: "96 % hit-rate ", bold: true, font: "Calibri", size: 22 }),
        new TextRun({ text: "на синтетических профилях, 60 % попаданий сразу на 1 место", size: 22, font: "Calibri" }),
    ]),
    bullet([
        new TextRun({ text: "+27 % NDCG@10 ", bold: true, font: "Calibri", size: 22 }),
        new TextRun({ text: "после обучения LinUCB на 1000 пользователях", size: 22, font: "Calibri" }),
    ]),
    bullet([
        new TextRun({ text: "Объясняемость ", bold: true, font: "Calibri", size: 22 }),
        new TextRun({ text: "— каждая рекомендация имеет полную декомпозицию по 12 признакам", size: 22, font: "Calibri" }),
    ]),

    H2("13.2. Что можно улучшить"),

    H3("Данные"),
    bullet("Подключить Eurostat для дополнительных индикаторов (зарплаты, ставки аренды)"),
    bullet("Использовать official visa policy databases (IATA Travel Centre) вместо ручного словаря"),
    bullet("Климат: вместо bool — реальные данные NOAA / OpenWeatherMap с временами года"),

    H3("Модель"),
    bullet("Learning-to-rank вместо косинуса (XGBoost-Ranker, LambdaMART)"),
    bullet("Replace LinUCB на Neural Bandit (DeepFM + UCB) для нелинейных контекстов"),
    bullet("Onboarding-flow с активным обучением: показать 5 случайных стран, попросить ранжировать, выучить веса"),

    H3("UI/UX"),
    bullet("Сохранение профилей в URL (shareable links)"),
    bullet("Сравнение нескольких стран рядом (radar overlay)"),
    bullet("Симуляция \"что-если\": изменить один приоритет → как сдвинется рейтинг"),

    H3("Production"),
    bullet("Cron-job для апдейта данных раз в месяц"),
    bullet("FastAPI-эндпоинт /recommend для интеграции с мобильным приложением"),
    bullet("A/B-тестирование версий рекомендателя (baseline vs LinUCB vs Neural)"),
    bullet("Метрики в продакшене: CTR на детальный экран страны, conversion на \"добавить в избранное\""),

    H2("13.3. Заключение"),
    P("Проект демонстрирует, что для задачи рекомендаций даже относительно простые методы (косинусное сходство + регионально-средняя импутация + LinUCB) могут давать осмысленные и интерпретируемые результаты, если правильно подобраны данные и метрики.", { justify: true }),
    P("Главный урок: качество рекомендаций определяется не сложностью модели, а качеством признаков и правильной формулировкой задачи. До перехода к нейронным сетям и трансформерам стоит инвестировать в:", { justify: true }),
    num("Хорошие источники данных"),
    num("Осмысленную нормализацию и импутацию"),
    num("Интерпретируемую функцию reward для оценки"),
    num("Объясняемость каждой рекомендации"),

    new Paragraph({
        spacing: { before: 480, after: 240 },
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "— Конец документа —", italics: true, size: 20, color: MUTED, font: "Calibri" })],
    }),
];

// ─────────────────────────────────────────────────────────────────────────────
// Document assembly
// ─────────────────────────────────────────────────────────────────────────────

const doc = new Document({
    creator: "Relocation Recommender Project",
    title: "Relocation Recommender — Project Documentation",
    styles: {
        default: { document: { run: { font: "Calibri", size: 22 } } },
        paragraphStyles: [
            { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
              run: { size: 36, bold: true, color: ACCENT_DARK, font: "Calibri" },
              paragraph: { spacing: { before: 360, after: 240 }, outlineLevel: 0 } },
            { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
              run: { size: 28, bold: true, color: ACCENT_DARK, font: "Calibri" },
              paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 1 } },
            { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
              run: { size: 24, bold: true, color: "2F5496", font: "Calibri" },
              paragraph: { spacing: { before: 180, after: 80 }, outlineLevel: 2 } },
        ],
    },
    numbering: {
        config: [
            { reference: "bullets",
              levels: [
                  { level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
                    style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
                  { level: 1, format: LevelFormat.BULLET, text: "◦", alignment: AlignmentType.LEFT,
                    style: { paragraph: { indent: { left: 1440, hanging: 360 } } } },
              ]},
            { reference: "numbers",
              levels: [
                  { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
                    style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
              ]},
        ],
    },
    sections: [{
        properties: {
            page: {
                size: { width: 11906, height: 16838 },   // A4
                margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
            },
        },
        headers: {
            default: new Header({
                children: [new Paragraph({
                    alignment: AlignmentType.RIGHT,
                    children: [new TextRun({
                        text: "Relocation Recommender — Documentation",
                        size: 18, color: MUTED, italics: true, font: "Calibri",
                    })],
                })],
            }),
        },
        footers: {
            default: new Footer({
                children: [new Paragraph({
                    alignment: AlignmentType.CENTER,
                    children: [
                        new TextRun({ text: "Стр. ", size: 18, color: MUTED, font: "Calibri" }),
                        new TextRun({ children: [PageNumber.CURRENT], size: 18, color: MUTED, font: "Calibri" }),
                        new TextRun({ text: " из ", size: 18, color: MUTED, font: "Calibri" }),
                        new TextRun({ children: [PageNumber.TOTAL_PAGES], size: 18, color: MUTED, font: "Calibri" }),
                    ],
                })],
            }),
        },
        children: [
            ...titlePage,
            ...intro,
            ...architecture,
            ...dataCollection,
            ...preprocessing,
            ...eda,
            ...clustering,
            ...recommender,
            ...ui,
            ...evaluation,
            ...rl,
            ...assumptions,
            ...russian_adaptation,
            ...conclusion,
        ],
    }],
});

Packer.toBuffer(doc).then(buffer => {
    const outPath = path.join(__dirname, "Relocation_Recommender_Documentation.docx");
    fs.writeFileSync(outPath, buffer);
    console.log("✓ Saved to:", outPath);
    console.log("  Size:", (buffer.length / 1024).toFixed(1), "KB");
});
