"""Спільне для тестів застосунку: синтетичний файл користувача."""

import numpy as np
import pandas as pd


def user_csv(n: int = 200, **override: object) -> bytes:
    """CSV користувача: n рядків (вік, стать, ІМТ, талія, t2d; seed 0), override замінює колонки."""
    rng = np.random.default_rng(0)
    raw = pd.DataFrame(
        {
            "age": rng.integers(20, 80, n),
            "male": rng.integers(0, 2, n),
            "bmi": rng.normal(28, 4, n).clip(15, 60),
            "waist": rng.normal(95, 10, n).clip(60, 150),
            "t2d": rng.integers(0, 2, n),
        }
    )
    return raw.assign(**override).to_csv(index=False).encode()
