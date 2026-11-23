"""Пайплайни моделей, сітки гіперпараметрів, CV-спліттер і запобіжник заборонених ознак.

Препроцесинг (імпутація, масштабування) — усередині Pipeline, тож fit відбувається лише на
тренувальних фолдах і не бачить валідаційних/тестових рядків.
"""

from collections.abc import Callable, Iterable

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from risk.config import CV_FOLDS, FEATURES, FORBIDDEN_FEATURE_PATTERNS, MONOTONIC_FEATURES, SEED
from risk.errors import ForbiddenFeatureError


def make_logit() -> Pipeline:
    """Медіанна імпутація → стандартизація → логістична регресія."""
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=SEED)),
        ]
    )


def monotonic_cst() -> list[int]:
    """Вектор монотонних обмежень у порядку FEATURES (0 — без обмеження)."""
    return [MONOTONIC_FEATURES.get(f, 0) for f in FEATURES]


def make_rf() -> Pipeline:
    """Медіанна імпутація → Random Forest з монотонними обмеженнями (RF не приймає NaN)."""
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300, random_state=SEED, n_jobs=-1, monotonic_cst=monotonic_cst()
                ),
            ),
        ]
    )


def make_hgb() -> Pipeline:
    """HistGradientBoosting з монотонними обмеженнями, без препроцесингу: NaN — нативно.

    early_stopping=False: за замовчуванням ("auto") воно вмикається лише при n > 10 000, тобто
    на CV-фолдах (~8.4 тис.) його немає, а на refit (~10.5 тис.) є — CV оцінював би іншу модель.
    """
    return Pipeline(
        [
            (
                "model",
                HistGradientBoostingClassifier(
                    random_state=SEED,
                    max_iter=200,
                    early_stopping=False,
                    monotonic_cst=monotonic_cst(),
                ),
            )
        ]
    )


MAKERS: dict[str, Callable[[], Pipeline]] = {"logit": make_logit, "rf": make_rf, "hgb": make_hgb}

# Невелика сітка: перевірка кількох конфігурацій, а не повний підбір; "model__" — крок пайплайна
PARAM_GRIDS: dict[str, dict[str, list]] = {
    "logit": {"model__C": [0.1, 1, 10]},
    "rf": {"model__max_depth": [None, 8], "model__min_samples_leaf": [5, 20]},
    "hgb": {"model__learning_rate": [0.05, 0.1], "model__max_leaf_nodes": [15, 31]},
}


def cv_splitter() -> StratifiedKFold:
    """Stratified 5-fold з перемішуванням: однакові фолди для GridSearchCV і OOF."""
    return StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)


def assert_no_forbidden_features(columns: Iterable[str]) -> None:
    """ForbiddenFeatureError, якщо назва колонки містить глікемічний маркер або ціль."""
    patterns = [p.lower() for p in FORBIDDEN_FEATURE_PATTERNS]
    bad = [c for c in columns if any(p in str(c).lower() for p in patterns)]
    if bad:
        raise ForbiddenFeatureError(
            f"Заборонені ознаки в матриці (складові цілі — витік): {', '.join(map(str, bad))}"
        )
