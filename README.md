# Система оцінювання ризику ЦД2

Автор: Самолюк Т.Г., 251з

Тема роботи: Визначення ризиків захворювань на основі можливих факторів впливу

Система оцінювання ризику цукрового діабету 2 типу на даних NHANES: описова статистика, опитувальник FINDRISC, три ML-моделі та інтерфейс на Streamlit.

## Швидкий старт

У проєкті вже є оброблені дані (`data/processed/`) і навчені моделі (`models/`), тому застосунок запускається без перерахунку:

```bash
uv sync                                    # встановити залежності
uv run streamlit run app/Головна.py        # запустити застосунок
```

Застосунок відкривається на `http://localhost:8501`. Команди однакові для Windows, macOS і Linux; `make` не потрібен.

## Повна перебудова

Потрібна, якщо даних і моделей немає або їх треба оновити:

```bash
uv sync
uv run python scripts/download_data.py     # сирі XPT з CDC → data/raw/ (~46 МБ, не в git)
uv run python scripts/build_cohort.py      # data/processed/{cohort.parquet, flow.json, variables.csv}
uv run python scripts/train_models.py      # models/*.joblib, metrics.json, test_predictions.parquet (seed 42)
uv run python scripts/build_results.py     # data/processed/results.xlsx + results_numbers.json
uv run streamlit run app/Головна.py
```

## Структура

```
diabetes-risk/
├── app/                  # Streamlit: тонкий шар, без бізнес-логіки
│   ├── Головна.py
│   ├── _common.py
│   └── pages/
├── src/risk/             # уся логіка (чисті функції)
│   ├── config.py
│   ├── errors.py         # винятки предметної області
│   ├── data/             # завантаження, гармонізація, аналітична вибірка, ваги
│   ├── stats/            # описова статистика, гіпотези, OR, кореляції
│   ├── models/           # конвеєри, навчання, оцінка, DeLong, калібрування, SHAP
│   ├── findrisc/         # підрахунок балів FINDRISC
│   ├── external/         # порівняння з Україною
│   ├── viz/              # графіки та схеми matplotlib
│   ├── report/           # results.py — єдине джерело таблиць/рисунків/чисел; excel.py; glossary.py
│   ├── person.py         # оцінка ризику для однієї людини (форма, сценарії)
│   └── interpret.py      # автотексти-пояснення
├── scripts/              # завантаження даних, вибірка, навчання, таблиці, схеми
├── models/               # навчені моделі (*.joblib)
├── data/{processed,external}/
├── tests/{unit,integration,app}/
├── .streamlit/           # налаштування Streamlit
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── pyproject.toml
├── requirements.txt
└── uv.lock
```

## Блоки дослідження → сторінки застосунку

| Блок схеми | Сторінка |
|:--|:--|
| 1. Збір та завантаження даних; 2. Попередня обробка | 1 «Дані і вибірка» |
| 3. Статистичне дослідження вибірки | 2 «Статистика» |
| 4. Аналіз факторів ризику | 3 «Фактори ризику» |
| 5. Математичне моделювання; 7. Перевірка та оцінювання моделі | 4 «Моделі» |
| 6. Порівняння моделей-аналогів | 5 «FINDRISC» |
| 8. Оцінювання індивідуального ризику | 7 «Оцінка ризику» |
| 9. Візуалізація та формування звіту | усі сторінки; 8 «Експорт» |

Сторінка 6 «Порівняння з Україною» належить до складової «Дослідження статистичних вибірок / підтвердження існуючими даними».

## Команди make

Це скорочення для систем, де встановлено `make`. Відповідники `uv` вище наведено для запуску застосунку, перебудови даних і моделей та тестів; решта цілей (`lint`, `fmt`, `requirements`, `all`) доступна лише через `make`.

| Команда | Що робить |
|---|---|
| `make sync` | Синхронізує середовище з `uv.lock` |
| `make data` | Завантажує сирі дані NHANES з CDC (~46 МБ) |
| `make cohort` | Будує аналітичну вибірку |
| `make train` | Навчає моделі й зберігає їх у `models/` |
| `make lint` | Перевіряє ruff (check + format --check) |
| `make fmt` | Форматує код і виправляє ruff |
| `make test` | Швидкі тести з покриттям `src/risk` ≥ 95 % |
| `make test-all` | Усі тести, включно з `slow` |
| `make app` | Запускає Streamlit-застосунок |
| `make results` | Збирає таблиці й числа (`data/processed/results.xlsx`, `results_numbers.json`) |
| `make all` | Повний конвеєр від даних до таблиць |
| `make requirements` | Генерує `requirements.txt` з `uv.lock` |

## Дані та використання

Дані — National Health and Nutrition Examination Survey (NHANES), відкриті дані CDC/NCHS: https://wwwn.cdc.gov/nchs/nhanes/. Використано цикли 2015–2016 і 2017–березень 2020. Сирі файли не зберігаються в репозиторії (`data/raw/` в `.gitignore`); їх завантажує `make data`.

Система — дослідницький інструмент, а не медичний виріб. Вона не ставить діагноз і не замінює консультацію лікаря чи лабораторний аналіз.

## Тести

Швидкі тести (без маркера `slow`) з перевіркою покриття `src/risk` не менше 95 %:

```bash
uv run python scripts/build_results.py     # лише якщо немає data/processed/results.xlsx
uv run pytest -m "not slow" --cov=src/risk --cov-report=term-missing --cov-fail-under=95
```

Файла `data/processed/results.xlsx` немає в git, тому перший рядок його створює (≈ 5 с). Для цього потрібні лише `data/processed/{cohort.parquet, flow.json, variables.csv}` і `models/`, сирі дані не потрібні.

Усі тести, включно з `slow`:

```bash
uv run pytest -m "" --cov=src/risk --cov-report=term-missing
```

Тестам `slow` потрібні `data/processed/cohort.parquet` і `models/`. Один із них пропускається, якщо немає `data/raw/GHB_I.xpt`; його створює `uv run python scripts/download_data.py`.

## Деплой

Розміщення: власний сервер, контейнер Docker за зворотним проксі з HTTPS.

Образ збирається з `Dockerfile` у корені. Застосунок слухає порт 8501. Перевірка стану: `/_stcore/health`.

```bash
docker compose up -d --build     # http://localhost:8501
docker compose logs -f app
```

Образ: `python:3.12-slim`, multi-stage, залежності з `uv.lock`; у контейнер копіюються `src/`, `app/`, `data/processed/`, `data/external/`, `models/`. Контейнер працює від користувача `app`.

---

Результат є дослідницькою оцінкою моделі, а не медичним діагнозом.
