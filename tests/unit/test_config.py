"""Запобіжник методології: у списку ознак немає глікемічних маркерів і цілі."""

import re

from risk import config


def test_features_have_no_forbidden_patterns() -> None:
    for feature in config.FEATURES:
        for pattern in config.FORBIDDEN_FEATURE_PATTERNS:
            assert pattern.lower() not in feature.lower(), f"{feature} містить {pattern}"


def test_every_feature_has_ukrainian_label() -> None:
    assert set(config.FEATURES) <= set(config.FEATURE_LABELS_UK)


def test_factor_lists_cover_features() -> None:
    assert set(config.CONTINUOUS_FACTORS) | set(config.BINARY_FACTORS) == set(config.FEATURES)


def test_author_format() -> None:
    pattern = r"^[А-ЯІЇЄҐ][а-яіїєґ'’-]+ [А-ЯІЇЄҐ]\.[А-ЯІЇЄҐ]\., \S+$"
    assert re.fullmatch(pattern, config.AUTHOR)
    assert not re.search(r"\d{4}", config.AUTHOR)
    assert not re.search(r"університет|ун-т", config.AUTHOR, re.IGNORECASE)


def test_topic_is_plain_title() -> None:
    assert config.TOPIC.strip()
    assert not config.TOPIC.endswith(".")
