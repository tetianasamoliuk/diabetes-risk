"""Доменні винятки проєкту."""


class RiskError(Exception):
    """Базовий виняток проєкту."""


class DownloadError(RiskError):
    """Файл NHANES не завантажився або не є XPT."""


class DataValidationError(RiskError):
    """Дані не пройшли валідацію (структура, типи, діапазони)."""


class ForbiddenFeatureError(RiskError):
    """У матрицю ознак потрапила заборонена колонка (глікемічний маркер або ціль)."""


class ModelNotTrainedError(RiskError):
    """Збережених моделей немає — потрібно виконати scripts/train_models.py."""
