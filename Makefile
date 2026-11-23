# Makefile: команди проєкту (порядок повного конвеєра — ціль all)

.PHONY: sync data cohort train lint fmt test test-all app results requirements all

# Синхронізувати віртуальне середовище з uv.lock
sync:
	uv sync

# Завантажити сирі дані NHANES з CDC
data:
	uv run python scripts/download_data.py

# Побудувати оброблену когорту
cohort:
	uv run python scripts/build_cohort.py

# Навчити ML-моделі й зберегти їх у models/
train:
	uv run python scripts/train_models.py

# Перевірити стиль коду (ruff check + format --check)
lint:
	uv run ruff check . && uv run ruff format --check .

# Автоформатування та автовиправлення ruff
fmt:
	uv run ruff format . && uv run ruff check --fix .

# Швидкі тести (без slow) з перевіркою покриття src/risk не менше 95 %;
# спершу гарантує наявність data/processed/results.xlsx (його не буває в чистому клоні)
test: data/processed/results.xlsx
	uv run pytest -m "not slow" --cov=src/risk --cov-report=term-missing --cov-fail-under=95

# Усі тести, включно з slow (потрібні дані)
test-all:
	uv run pytest -m "" --cov=src/risk --cov-report=term-missing

# Запустити Streamlit-застосунок
app:
	uv run streamlit run app/Головна.py

# Таблиці й числа для документів: data/processed/results.xlsx, results_numbers.json
results:
	uv run python scripts/build_results.py

# Файлова ціль для `test`: будує results.xlsx лише якщо його немає (≈ 5 с, без сирих даних)
data/processed/results.xlsx:
	uv run python scripts/build_results.py

# Згенерувати requirements.txt з uv.lock
requirements:
	uv export --no-dev --no-hashes --format requirements-txt -o requirements.txt

# Повний конвеєр: дані, вибірка, моделі, тести, таблиці
all: data cohort train test results
