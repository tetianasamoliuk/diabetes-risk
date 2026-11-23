"""Таблиці та тексти з models/metrics.json (для сторінки «Моделі» і для експорту)."""

from collections.abc import Callable

import numpy as np
import pandas as pd

from risk import interpret
from risk.config import ALPHA, MODEL_NAMES_UK
from risk.errors import DataValidationError
from risk.findrisc.score import VARIANT_TITLES
from risk.models.delong import auc_ci, delong_test
from risk.stats.hypothesis import holm

NAN = float("nan")

# Назви гіперпараметрів у таблиці 4.2 (ключі — імена scikit-learn без префікса model__)
PARAM_NAMES_UK = {
    "C": "обернена сила регуляризації C",
    "max_depth": "глибина дерев",
    "min_samples_leaf": "мінімум людей у листку",
    "learning_rate": "швидкість навчання",
    "max_leaf_nodes": "листків у дереві",
}


def _uk(key: str) -> str:
    return MODEL_NAMES_UK.get(key, key)


def _param_value(value: object) -> str:
    return "без обмеження" if value is None else str(value)


def _param_name(key: str) -> str:
    name = key.removeprefix("model__")
    return PARAM_NAMES_UK.get(name, name)


def _params(d: dict) -> str:
    """{'model__max_depth': None, …} → «глибина дерев: без обмеження; …»."""
    parts = [f"{_param_name(k)}: {_param_value(v)}" for k, v in d.items()]
    return "; ".join(parts) or "—"


def _metrics_row(key: str, metrics: dict) -> dict:
    t, cv = metrics["test"][key], metrics["cv"][key]
    return {
        "AUC крос-валідації (середнє ± SD)": f"{cv['cv_auc_mean']:.3f} ± {cv['cv_auc_sd']:.3f}",
        "AUC (тест)": round(t["auc"], 3),
        "95% ДІ AUC (тест)": f"{t['auc_ci_low']:.3f}–{t['auc_ci_high']:.3f}",
        "Частка правильних (accuracy)": round(t["accuracy"], 3),
        "Точність позитивних (precision)": round(t["precision"], 3),
        "Чутливість (recall)": round(t["recall"], 3),
        "Специфічність": round(t["specificity"], 3),
        "F1": round(t["f1"], 3),
        "Оцінка Браєра": round(t["brier"], 3),
        "Поріг (індекс Юдена)": round(t["threshold"], 3),
        "Обрані параметри": _params(metrics["best_params"][key]),
    }


def metrics_summary_table(metrics: dict) -> pd.DataFrame:
    """Рядок на модель: CV, test-метрики, поріг, параметри."""
    keys = list(metrics["test"])
    return pd.DataFrame([_metrics_row(k, metrics) for k in keys], index=[_uk(k) for k in keys])


def pct_ci(low: float, high: float) -> str:
    """Межі ДІ часток 0–1 у відсотках: «75.2–82.4»."""
    return f"{low * 100:.1f}–{high * 100:.1f}"


def _ci_row(t: dict) -> dict:
    return {
        "Чутливість, %": round(t["recall"] * 100, 1),
        "Чутливість: 95% ДІ, %": pct_ci(t["se_ci_low"], t["se_ci_high"]),
        "Специфічність, %": round(t["specificity"] * 100, 1),
        "Специфічність: 95% ДІ, %": pct_ci(t["sp_ci_low"], t["sp_ci_high"]),
        "Точність позитивних (PPV), %": round(t["precision"] * 100, 1),
        "PPV: 95% ДІ, %": pct_ci(t["ppv_ci_low"], t["ppv_ci_high"]),
        "Нахил калібрування": round(t["calib_slope"], 3),
        "Зсув калібрування (log-odds)": round(t["calib_intercept"], 3),
    }


def metrics_ci_table(metrics: dict) -> pd.DataFrame:
    """Рядок на модель: Se, Sp, PPV з 95% ДІ Вілсона; нахил і зсув калібрування."""
    rows = {_uk(k): _ci_row(t) for k, t in metrics["test"].items()}
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("Модель").reset_index()


def _verdict(p: float) -> str:
    return "значуща" if p < ALPHA else "незначуща"


def delong_holm(metrics: dict) -> list[float]:
    """Скориговані за Холмом p попарних DeLong-порівнянь (порядок — як у metrics["delong"])."""
    return holm([r["p"] for r in metrics["delong"]])


def findrisc_holm(metrics: dict) -> list[float]:
    """Скориговані за Холмом p порівнянь «модель проти FINDRISC» (окрема сім'я гіпотез)."""
    return holm([r["p"] for r in metrics["delong_vs_findrisc"]])


def delong_table(metrics: dict) -> pd.DataFrame:
    """Попарні порівняння AUC (тест DeLong); висновок — за p з поправкою Холма."""
    rows = [
        {
            "Модель A": _uk(r["a"]),
            "Модель B": _uk(r["b"]),
            "AUC моделі A": round(r["auc_a"], 3),
            "AUC моделі B": round(r["auc_b"], 3),
            "Різниця AUC (A − B)": round(r["diff"], 3),
            "95% ДІ різниці": f"{r['ci_low']:.3f}…{r['ci_high']:.3f}",
            "z-статистика": round(r["z"], 2),
            "p": interpret.fmt_p(r["p"]),
            "p (Холм)": interpret.fmt_p(p_holm),
            "Висновок": _verdict(p_holm),
        }
        for r, p_holm in zip(metrics["delong"], delong_holm(metrics), strict=True)
    ]
    return pd.DataFrame(rows)


def findrisc_auc_table(metrics: dict) -> pd.DataFrame:
    """AUC варіантів FINDRISC a/b/c на тесті."""
    rows = [
        {
            "Варіант": k,
            "Опис": VARIANT_TITLES[k],
            "AUC": round(v["auc"], 3),
            "95% ДІ": f"{v['ci_low']:.3f}–{v['ci_high']:.3f}",
            "N": v["n"],
            "Позитивних": v["n_pos"],
            "Придатний для порівняння": "так" if v["valid"] else "ні",
        }
        for k, v in metrics["findrisc"].items()
    ]
    return pd.DataFrame(rows)


def findrisc_vs_model_table(metrics: dict) -> pd.DataFrame:
    """Основна модель проти варіантів FINDRISC (тест DeLong); висновок — за p (Холм)."""
    rows = [
        {
            "Модель": _uk(r["model"]),
            "Варіант FINDRISC": r["variant"],
            "N": r["n"],
            "AUC моделі": round(r["auc_a"], 3),
            "AUC FINDRISC": round(r["auc_b"], 3),
            "Різниця AUC": round(r["diff"], 3),
            "95% ДІ різниці": f"{r['ci_low']:.3f}…{r['ci_high']:.3f}",
            "z-статистика": round(r["z"], 2),
            "p": interpret.fmt_p(r["p"]),
            "p (Холм)": interpret.fmt_p(p_holm),
            "Висновок": _verdict(p_holm),
        }
        for r, p_holm in zip(metrics["delong_vs_findrisc"], findrisc_holm(metrics), strict=True)
    ]
    return pd.DataFrame(rows)


def calibration_frames(metrics: dict) -> dict[str, pd.DataFrame]:
    """Калібрувальні таблиці по моделях (для figures.calibration_curves)."""
    return {k: pd.DataFrame(v) for k, v in metrics["calibration"].items()}


def confusion_counts(metrics: dict, name: str) -> tuple[int, int, int, int]:
    """(tn, fp, fn, tp) на тесті."""
    t = metrics["test"][name]
    return t["tn"], t["fp"], t["fn"], t["tp"]


def _best_pairs(metrics: dict, best: str) -> list[tuple[dict, float]]:
    """DeLong-порівняння основної моделі з іншими разом із p (Холм)."""
    pairs = zip(metrics["delong"], delong_holm(metrics), strict=True)
    return [(r, p_holm) for r, p_holm in pairs if best in (r["a"], r["b"])]


def _other(r: dict, best: str) -> str:
    return r["b"] if r["a"] == best else r["a"]


def _pair_text(metrics: dict, best: str, other: str) -> str:
    for r, p_holm in _best_pairs(metrics, best):
        if _other(r, best) == other:
            a, b = ("a", "b") if r["a"] == best else ("b", "a")
            return interpret.delong_text(
                _uk(r[a]), r[f"auc_{a}"], _uk(r[b]), r[f"auc_{b}"], r["p"], p_holm=p_holm
            )
    raise DataValidationError(f"Немає DeLong-порівняння {best} / {other}")


def _holm_summary(metrics: dict, best: str) -> str:
    """Підсумок після поправки Холма: з якими моделями AUC основної відрізняється значуще."""
    head = f"З поправкою Холма на кількість порівнянь ({len(metrics['delong'])})"
    sig = [f"«{_uk(_other(r, best))}»" for r, p in _best_pairs(metrics, best) if p < ALPHA]
    if not sig:
        return (
            f"{head} основна модель «{_uk(best)}» статистично не відрізняється за AUC "
            "від жодної з інших моделей."
        )
    return (
        f"{head} різниця AUC основної моделі «{_uk(best)}» статистично значуща порівняно з: "
        f"{', '.join(sig)}."
    )


def _rule_text(metrics: dict, best: str) -> str:
    """Речення про вибір за CV; якщо тестовий AUC в іншої вищий, це пояснюється."""
    text = (
        f"Модель «{_uk(best)}» обрано за середнім AUC крос-валідації на навчальній "
        "вибірці, а не за тестом, щоб тестові дані не впливали на вибір."
    )
    auc = {k: v["auc"] for k, v in metrics["test"].items()}
    if max(auc, key=lambda k: auc[k]) != best:
        text += " Вищий тестовий AUC в іншої моделі цього вибору не змінює."
    return text


def best_model_conclusion(metrics: dict) -> str:
    """Підсумок за Холмом, основна модель (best_model) проти кожної іншої, примітка про вибір."""
    best = metrics["best_model"]
    parts = [_holm_summary(metrics, best)]
    parts += [_pair_text(metrics, best, k) for k in metrics["test"] if k != best]
    if "best_model_rule" in metrics:
        parts.append(_rule_text(metrics, best))
    return "\n\n".join(parts)


# ---------- підгрупи: діагностований і недіагностований ЦД2 (тестова вибірка) ----------

SUBGROUPS = {"undiag": "Недіагностований ЦД2", "diag": "Діагностований ЦД2"}
SUBGROUP_SCORES = ("logit", "rf", "hgb", "findrisc_a")
COMPARED_WITH_BEST = ("logit", "findrisc_a")  # з чим порівнюємо основну модель у підгрупах
SCORE_NAMES = {**MODEL_NAMES_UK, "findrisc_a": "FINDRISC (a)"}
SUBGROUP_CAPTION = (
    "AUC моделей і FINDRISC (a) окремо для недіагностованого і діагностованого ЦД2 "
    "(тестова вибірка)"
)
_NO_PAIR = {"n_pair": NAN, "auc_best_pair": NAN, "diff": NAN, "p": NAN}


def _col(key: str) -> str:
    """Колонка test_predictions з оцінкою: p_<модель> або findrisc_a."""
    return key if key.startswith("findrisc") else f"p_{key}"


def subgroup(preds: pd.DataFrame, key: str) -> pd.DataFrame:
    """Особи без ЦД2 і хворі з уже встановленим діагнозом (diag) або без нього (undiag)."""
    sick, diagnosed = preds["t2d"] == 1, preds["diq010"] == 1
    return preds[(sick & (diagnosed if key == "diag" else ~diagnosed)) | ~sick]


def _auc_cells(d: pd.DataFrame, col: str) -> dict:
    d = d[d[col].notna()]
    auc, lo, hi = auc_ci(d["t2d"].to_numpy(), d[col].to_numpy())
    return {"n": len(d), "n_pos": int(d["t2d"].sum()), "auc": auc, "ci_low": lo, "ci_high": hi}


def _vs_best(d: pd.DataFrame, best: str, key: str) -> dict:
    """DeLong «основна модель проти key» на людях, у яких є обидві оцінки."""
    if key == best or key not in COMPARED_WITH_BEST:
        return _NO_PAIR
    pair = d.dropna(subset=[_col(best), _col(key)])
    r = delong_test(pair["t2d"].to_numpy(), pair[_col(best)].to_numpy(), pair[_col(key)].to_numpy())
    return {"n_pair": len(pair), "auc_best_pair": r.auc_a, "diff": r.diff, "p": r.p}


def subgroup_auc(preds: pd.DataFrame, best: str) -> pd.DataFrame:
    """Рядок на (підгрупу, оцінку): N, позитивні, AUC з 95% ДІ і DeLong з основною моделлю.

    p_holm — поправка Холма на всі порівняння таблиці (окрема сім'я гіпотез).
    """
    rows = []
    for g in SUBGROUPS:
        d = subgroup(preds, g)
        rows += [
            {"group": g, "score": k, **_auc_cells(d, _col(k)), **_vs_best(d, best, k)}
            for k in SUBGROUP_SCORES
        ]
    out = pd.DataFrame(rows)
    return out.assign(p_holm=holm(out["p"]))


def _cell(x: float, fmt: Callable[[float], str]) -> str:
    return "—" if np.isnan(x) else fmt(x)


def _signed(x: float) -> str:
    return f"{x:+.3f}".replace("-", interpret.MINUS)


def subgroup_view(raw: pd.DataFrame, best: str) -> pd.DataFrame:
    """Таблиця підгруп українською; «—» там, де порівняння з основною моделлю немає."""
    names = {k: SCORE_NAMES[k] + (" (основна)" if k == best else "") for k in SUBGROUP_SCORES}
    ci = zip(raw["ci_low"], raw["ci_high"], strict=True)
    return pd.DataFrame({
        "Підгрупа": raw["group"].map(lambda g: f"{SUBGROUPS[g]} проти осіб без ЦД2"),
        "Оцінка": raw["score"].map(names),
        "N": raw["n"],
        "З ЦД2": raw["n_pos"],
        "AUC": raw["auc"].round(3),
        "95% ДІ": [f"{lo:.3f}–{hi:.3f}" for lo, hi in ci],
        "Різниця AUC (основна − ця)": raw["diff"].map(lambda x: _cell(x, _signed)),
        "p (ДеЛонга)": raw["p"].map(lambda x: _cell(x, interpret.fmt_p)),
        "p (Холм)": raw["p_holm"].map(lambda x: _cell(x, interpret.fmt_p)),
    })  # fmt: skip


def subgroup_numbers(raw: pd.DataFrame) -> dict[str, float | int]:
    """Ключі для документів; k = <підгрупа>_<оцінка>, напр. undiag_findrisc_a.

    sub_k_auc, sub_k_auc_lo, sub_k_auc_hi, sub_k_n, sub_k_pos — для кожного рядка;
    для порівнянь з основною моделлю ще sub_k_auc_diff (основна − ця), sub_k_n_pair,
    p_sub_k (ДеЛонга) і p_holm_sub_k (Холм на всі порівняння таблиці).
    """
    out: dict[str, float | int] = {}
    for r in raw.itertuples():
        k = f"{r.group}_{r.score}"
        out |= {f"sub_{k}_auc": round(r.auc, 3), f"sub_{k}_auc_lo": round(r.ci_low, 3),
                f"sub_{k}_auc_hi": round(r.ci_high, 3), f"sub_{k}_n": int(r.n),
                f"sub_{k}_pos": int(r.n_pos)}  # fmt: skip
        if not np.isnan(r.p):
            out |= {f"sub_{k}_auc_diff": round(r.diff, 3), f"sub_{k}_n_pair": int(r.n_pair),
                    f"p_sub_{k}": float(r.p), f"p_holm_sub_{k}": float(r.p_holm)}  # fmt: skip
    return out


def subgroup_comparisons(raw: pd.DataFrame, best: str) -> list[str]:
    """Речення DeLong для кожного порівняння таблиці підгруп (AUC — на спільних людях)."""
    return [
        f"{SUBGROUPS[r.group]} — "
        + interpret.delong_text(
            SCORE_NAMES[best], r.auc_best_pair, SCORE_NAMES[r.score], r.auc, r.p, p_holm=r.p_holm
        )
        for r in raw.dropna(subset=["p"]).itertuples()
    ]


def _gap_sentence(raw: pd.DataFrame, key: str) -> str:
    rows = raw[raw["score"] == key].set_index("group")["p_holm"]
    return (
        f"Порівняно з «{SCORE_NAMES[key]}» різниця AUC серед діагностованих "
        f"{_verdict(rows['diag'])}, серед недіагностованих — {_verdict(rows['undiag'])}."
    )


def _undiag_verdict(proved: bool, n_pos: int) -> str:
    if proved:
        return (
            "Перевага основної моделі зберігається і для недіагностованого діабету, але випадків "
            f"лише {n_pos}, тому оцінка нестабільна."
        )
    return (
        "Тобто перевагу основної моделі показано на змішаній вибірці, де переважають уже "
        "діагностовані; для виявлення недіагностованого діабету вона не доведена. Випадків "
        f"недіагностованого діабету лише {n_pos}, тому ця оцінка нестабільна: довірчі "
        "інтервали AUC широкі."
    )


def _positives(raw: pd.DataFrame, best: str) -> pd.Series:
    return raw[raw["score"] == best].set_index("group")["n_pos"]


def pct_diagnosed(raw: pd.DataFrame, best: str) -> float:
    """Частка (%) уже діагностованих серед людей з ЦД2 у тестовій вибірці."""
    pos = _positives(raw, best)
    return float(100 * pos["diag"] / pos.sum())


def subgroup_conclusion(raw: pd.DataFrame, best: str) -> str:
    """Висновок аналізу підгруп з чисел таблиці (значущість — за p з поправкою Холма)."""
    undiag = raw[(raw["group"] == "undiag") & raw["p"].notna()]
    proved = bool(((undiag["p_holm"] < ALPHA) & (undiag["diff"] > 0)).all())
    parts = [f"Серед людей з ЦД2 у тестовій вибірці {pct_diagnosed(raw, best):.0f} % "
             "уже мають діагноз (з анкети NHANES)."]  # fmt: skip
    parts += [_gap_sentence(raw, k) for k in COMPARED_WITH_BEST if k != best]
    parts.append(_undiag_verdict(proved, int(_positives(raw, best)["undiag"])))
    return " ".join(parts)
