"""Автопояснення результатів простою мовою (українською).

Усі функції чисті: повертають рядок, числа беруться лише з аргументів.
Частки (чутливість, специфічність, ймовірності) передаються як 0–1;
поширеність (prevalence_text) — у відсотках.
"""

import math
from bisect import bisect_right
from collections.abc import Mapping

import pandas as pd

from risk.config import ALPHA, FEATURE_LABELS_UK
from risk.stats.hypothesis import MIN_NORMALITY_N

# Словесна шкала AUC за Hosmer, Lemeshow і Sturdivant (2013); межа належить верхній категорії
AUC_BOUNDS = [0.7, 0.8, 0.9]
AUC_GRADES = ["слабка", "прийнятна", "відмінна", "видатна"]
BRIER_COIN = 0.25  # Brier для моделі «підкидання монети»
CALIBRATION_OK_PP = 2.0  # допустима різниця середнього прогнозу й частки хворих, п.п.
CALIBRATION_SLOPE_TOL = 0.1  # нахил калібрування в межах 1 ± 0.1 вважаємо близьким до ідеалу
SHAP_TOP = 3  # скільки чинників показувати в кожному напрямку
MINUS = "−"
VIF_LIMIT = 5.0  # понад це ознаки вважаємо колінеарними

# Як називати ознаку в тексті, коли відоме її значення: (є, немає)
BINARY_PHRASES: dict[str, tuple[str, str]] = {
    "male": ("чоловіча стать", "жіноча стать"),
    "fam_dm": ("діабет у близьких родичів", "відсутність діабету в близьких родичів"),
    "bp_med": ("прийом ліків від тиску", "відсутність прийому ліків від тиску"),
    "cycle_2017": ("обстеження в періоді 2017–2020", "обстеження в періоді 2015–2016"),
}
# Підписи бінарних ознак на рисунку (значення «так» / «ні» стоїть окремо)
BINARY_CAPTIONS: dict[str, str] = {
    "male": "Стать: чоловіча",
    "fam_dm": "Діабет у близьких родичів",
    "bp_med": "Приймає ліки від тиску",
    "cycle_2017": "Період обстеження: 2017–2020",
}
# Неперервні ознаки: (назва в тексті, одиниця, знаків після коми)
CONTINUOUS_PHRASES: dict[str, tuple[str, str, int]] = {
    "age": ("вік", "років", 0),
    "bmi": ("ІМТ", "кг/м²", 1),
    "waist": ("окружність талії", "см", 0),
    "sbp": ("систолічний тиск", "мм рт. ст.", 0),
    "dbp": ("діастолічний тиск", "мм рт. ст.", 0),
    "tc": ("загальний холестерин", "мг/дл", 0),
    "hdl": ("ЛПВЩ-холестерин", "мг/дл", 0),
    "pa_h_week": ("фізична активність", "год/тиждень", 1),
}
# Короткі назви ознак у формулі логістичної регресії
FORMULA_NAMES_UK: dict[str, str] = {
    "age": "вік",
    "male": "стать",
    "bmi": "ІМТ",
    "waist": "талія",
    "sbp": "САТ",
    "dbp": "ДАТ",
    "tc": "ХС",
    "hdl": "ЛПВЩ",
    "pa_h_week": "активність",
    "fam_dm": "спадковість",
    "bp_med": "ліки від тиску",
    "cycle_2017": "період 2017–2020",
}
FORMULA_TERMS_PER_LINE = 4
LOGIT_LEGEND = (
    "p̂ — ймовірність діабету, z — лінійний предиктор, числа перед ознаками — коефіцієнти β "
    "з таблиці нижче. Ознаки: вік (роки), стать (1 — чоловіча), ІМТ (кг/м²), талія (см), "
    "САТ і ДАТ (мм рт. ст.), ХС — загальний холестерин і ЛПВЩ (мг/дл), активність "
    "(год/тиждень), спадковість (1 — діабет у близьких родичів), ліки від тиску (1 — приймає), "
    "період 2017–2020 (1 — так)."
)
LOGIT_SAME_FORM_NOTE = (
    "Прогнозна модель має ту саму форму, але інші коефіцієнти: вони отримані з регуляризацією "
    "на стандартизованих ознаках."
)
SHAP_AXES_NOTE = (
    "f(x) — прогноз моделі для цієї людини у log-odds (логарифм шансів), E[f(X)] — середній "
    "прогноз на навчальній вибірці."
)

# Як обирається тест для числових ознак (правило — в risk.stats.hypothesis)
TEST_CHOICE_NOTE = (
    "Тест для числової ознаки обирається за правилом. Спершу в кожній групі перевіряється "
    "нормальність розподілу (критерій Д'Агостіно–Пірсона): розподіл вважається нормальним, "
    f"якщо p ≥ {ALPHA} і в групі щонайменше {MIN_NORMALITY_N} людей. Якщо нормальні обидві "
    "групи, застосовується t-критерій Велча, інакше критерій Манна–Вітні. Частки порівнюються "
    "критерієм χ²."
)

# Назви тестів українською: ключ — підрядок у нормалізованій назві.
TEST_NAMES_UK = {
    "welch": "t-критерій Велча",
    "mann": "критерій Манна–Вітні",
    "chi": "критерій χ²",
}


def fmt_p(p: float) -> str:
    """p-value: «< 0.001» або три знаки; біля порога 0.05 — чотири (щоб 0.0499 не став «0.050»)."""
    if p < 0.001:
        return "< 0.001"
    return f"{p:.4f}" if 0.045 <= p < 0.055 else f"{p:.3f}"


def plural_uk(n: int, forms: tuple[str, str, str]) -> str:
    """Форма слова за числом: (1, 2–4, 5+), напр. «помилка», «помилки», «помилок»."""
    if n % 10 == 1 and n % 100 != 11:
        return forms[0]
    return forms[1] if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14 else forms[2]


def fmt_pct(x: float) -> str:
    """Відсотки з одним знаком: «2.0 %»."""
    return f"{x:.1f} %"


def fmt_pp(x: float) -> str:
    """Зміна у відсоткових пунктах зі знаком: «+0.3 п.п.», «−0.2 п.п.», «0.0 п.п.»."""
    r = round(x, 1)
    if r == 0:
        return "0.0 п.п."
    return f"{r:+.1f}".replace("-", MINUS) + " п.п."


def fmt_cell(x: float | None, digits: int = 2, cap: float | None = None) -> str:
    """Число для таблиці: «—» замість nan/inf, «> cap» для надвеликих значень."""
    if x is None or not math.isfinite(x):
        return "—"
    if cap is not None and x > cap:
        return f"> {cap:g}"
    return f"{x:.{digits}f}"


def fmt_num(x: float, digits: int = 2) -> str:
    """Число з фіксованою кількістю знаків."""
    return f"{x:.{digits}f}"


def _pct(x: float, digits: int = 1) -> str:
    """Частка 0–1 у відсотках (без знака %)."""
    return fmt_num(x * 100, digits)


def p_short(p: float) -> str:
    """«p < 0.001» або «p = 0.372» — без подвійного знака."""
    f = fmt_p(p)
    return f"p {f}" if f.startswith("<") else f"p = {f}"


def p_value_text(p: float, alpha: float = 0.05) -> str:
    """Висновок за p-value відносно рівня значущості alpha."""
    if p < alpha:
        return f"Різниця статистично значуща (p = {fmt_p(p)}), нульову гіпотезу відхиляємо."
    return f"Різниця не є статистично значущою (p = {fmt_p(p)}), нульову гіпотезу не відхиляємо."


def or_text(
    label: str,
    odds_ratio: float,
    ci_low: float,
    ci_high: float,
    reference: str | None = None,
    scale_label: str | None = None,
) -> str:
    """Пояснення OR з ДІ. scale_label (наприклад «10 мг/дл») — для неперервних факторів."""
    ci = f"(95% ДІ {fmt_num(ci_low)}–{fmt_num(ci_high)})"
    val = fmt_num(odds_ratio)
    if scale_label:
        head = f"Кожні {scale_label} ({label}) змінюють шанси мати ЦД2 у {val} раза {ci}."
    else:
        ref = reference or "у решти"
        who = f"У людей з {label}"
        if val == fmt_num(1):
            head = f"{who} шанси мати ЦД2 не відрізняються від шансів {ref} {ci}."
        elif odds_ratio > 1:
            head = f"{who} шанси мати ЦД2 у {val} раза вищі {ci}, ніж {ref}."
        else:
            head = f"{who} шанси мати ЦД2 становлять {val} від шансів {ref} {ci}."
    if ci_low <= 1 <= ci_high:
        return f"{head} Інтервал містить 1, тому зв'язок статистично не значущий."
    return f"{head} Інтервал не містить 1, тому зв'язок статистично значущий."


def auc_text(auc: float, ci_low: float | None = None, ci_high: float | None = None) -> str:
    """Пояснення AUC за шкалою якості."""
    grade = AUC_GRADES[bisect_right(AUC_BOUNDS, auc)]
    ci = ""
    if ci_low is not None and ci_high is not None:
        ci = f" (95% ДІ {fmt_num(ci_low)}–{fmt_num(ci_high)})"
    return (
        f"AUC = {fmt_num(auc)}{ci} — це ймовірність, що випадковий хворий отримає "
        f"вищий бал ризику, ніж випадковий здоровий. "
        f"Висновок: розрізнювальна здатність — {grade}."
    )


def _mean_calibration(mean_pred: float, mean_obs: float) -> tuple[bool, str]:
    """(чи близький середній прогноз до частки хворих, речення про це)."""
    diff = (mean_pred - mean_obs) * 100
    pred, obs = _pct(mean_pred), _pct(mean_obs)
    if abs(diff) < CALIBRATION_OK_PP:
        return True, f"Середній прогноз {pred}% близький до частки хворих {obs}%."
    verb = "завищує" if diff > 0 else "занижує"
    return False, (
        f"Модель {verb} ризик у середньому на {fmt_num(abs(diff), 1)} п.п. "
        f"(прогноз {pred}%, фактично {obs}%)."
    )


def _slope_calibration(slope: float) -> tuple[bool, str]:
    """(чи нахил калібрування близький до 1, речення про розкид прогнозів)."""
    head = f"Нахил калібрування {fmt_num(slope)}"
    if abs(slope - 1) < CALIBRATION_SLOPE_TOL:
        return True, f"{head} близький до 1: розкид прогнозів відповідає дійсності."
    if slope < 1:
        return False, (
            f"{head} менший за 1: прогнози надто крайні — високі ризики модель завищує, "
            "а низькі занижує."
        )
    return False, (
        f"{head} більший за 1: прогнози надто стиснуті до середнього — високі ризики модель "
        "занижує, а низькі завищує."
    )


def calibration_text(brier: float, mean_pred: float, mean_obs: float, slope: float) -> str:
    """Пояснення калібрування: Brier, зсув середнього прогнозу і нахил калібрування.

    «Добре» — лише коли і середній прогноз, і нахил близькі до ідеалу.
    """
    head = (
        f"Оцінка Браєра = {fmt_num(brier)}: це середня квадратична помилка ймовірностей, "
        f"менше — краще, {fmt_num(BRIER_COIN)} — як підкидання монети."
    )
    mean_ok, mean = _mean_calibration(mean_pred, mean_obs)
    slope_ok, spread = _slope_calibration(slope)
    if mean_ok and slope_ok:
        return f"{head} Калібрування добре. {mean} {spread}"
    return (
        f"{head} {mean} {spread} Тому ймовірності цієї моделі варто читати як орієнтир, "
        "а не як точні відсотки."
    )


def delong_text(
    name_a: str,
    auc_a: float,
    name_b: str,
    auc_b: float,
    p: float,
    alpha: float = 0.05,
    p_holm: float | None = None,
) -> str:
    """Порівняння двох AUC за тестом ДеЛонга; name_a — обрана (опорна) модель.

    p_holm — p з поправкою Холма на кількість порівнянь; якщо задано, висновок робиться за ним.
    """
    diff = f"{auc_a - auc_b:+.3f}".replace("-", MINUS)
    head = f"{name_a}: AUC {fmt_num(auc_a, 3)}; {name_b}: AUC {fmt_num(auc_b, 3)}. "
    if p_holm is None:
        return f"{head}Різниця {diff} {_is(p, alpha)} статистично значущою ({p_short(p)})."
    text = (
        f"{head}Різниця {diff} {_is(p_holm, alpha)} статистично значущою з поправкою Холма "
        f"на кількість порівнянь (сире {p_short(p)}, з поправкою {p_short(p_holm)})."
    )
    if p < alpha <= p_holm:
        text += " Без поправки вона виглядала б значущою."
    return text


def _is(p: float, alpha: float) -> str:
    return "є" if p < alpha else "не є"


def feature_phrase(name: str, value: float | None) -> str | None:
    """Ознака зі значенням для тексту: «окружність талії 90 см», «відсутність діабету в родичів»."""
    if value is None or value != value:
        return None
    if name in BINARY_PHRASES:
        return BINARY_PHRASES[name][0 if value else 1]
    if name in CONTINUOUS_PHRASES:
        label, unit, digits = CONTINUOUS_PHRASES[name]
        return f"{label} {value:.{digits}f} {unit}"
    return None


def feature_display(name: str, value: float) -> tuple[str, str]:
    """(значення, підпис) ознаки для рисунка: («так», «Діабет у родичів»), («26.0», «ІМТ»)."""
    if name in BINARY_CAPTIONS:
        return ("так" if value else "ні"), BINARY_CAPTIONS[name]
    digits = CONTINUOUS_PHRASES.get(name, ("", "", 0))[2]
    return f"{value:.{digits}f}", FEATURE_LABELS_UK.get(name, name)


def _shap_part(
    items: list[tuple[str, float]],
    labels: dict[str, str],
    sign: str,
    values: Mapping[str, float] | None = None,
) -> str:
    """Список «назва (±значення)» для SHAP."""
    parts = []
    for name, v in items:
        text = feature_phrase(name, values.get(name) if values else None)
        parts.append(f"{text or labels.get(name, name)} ({sign}{fmt_num(abs(v))})")
    return ", ".join(parts)


def shap_text(
    contributions: list[tuple[str, float]],
    labels: dict[str, str],
    values: Mapping[str, float] | None = None,
) -> str:
    """Пояснення внесків ознак (SHAP, log-odds) у прогноз для однієї людини.

    values — значення ознак людини: з ними чинники пишуться з конкретним значенням.
    """
    if not contributions:
        return "Внесків окремих чинників немає, тому пояснити цей прогноз нічим."
    ordered = sorted(contributions, key=lambda c: c[1], reverse=True)
    up = [c for c in ordered if c[1] > 0][:SHAP_TOP]
    down = [c for c in reversed(ordered) if c[1] < 0][:SHAP_TOP]
    parts = []
    if up:
        parts.append(f"Найбільше підвищили оцінку моделі: {_shap_part(up, labels, '+', values)}")
    if down:
        parts.append(f"Найбільше знизили: {_shap_part(down, labels, MINUS, values)}")
    body = ". ".join(parts) if parts else "Жоден чинник не змінив оцінку моделі"
    return (
        f"{body}. Значення подано в log-odds: знак «+» збільшує оцінку моделі, "
        f"знак «{MINUS}» зменшує. Це опис роботи моделі, а не причини діабету."
    )


def _formula_term(name: str, coef: float) -> str:
    sign = "-" if coef < 0 else "+"
    return f"{sign} {abs(coef):.3f}\\,\\text{{{FORMULA_NAMES_UK.get(name, name)}}}"


def logit_formula_aligned(coefs: Mapping[str, float]) -> str:
    """LaTeX формули логістичної регресії в кілька рядків з короткими українськими назвами."""
    terms = [_formula_term(n, c) for n, c in coefs.items() if n != "const"]
    step = FORMULA_TERMS_PER_LINE
    lines = [f"{coefs['const']:.3f} " + " ".join(terms[:step])]
    lines += [" ".join(terms[i : i + step]) for i in range(step, len(terms), step)]
    body = " \\\\\n&\\quad ".join(lines)
    return (
        "\\begin{aligned}\n\\hat p &= \\frac{1}{1+e^{-z}} \\\\\n"
        f"z &= {body}\n\\end{{aligned}}"
    )


def prevalence_text(
    weighted_pct: float, ci_low_pct: float, ci_high_pct: float, unweighted_pct: float
) -> str:
    """Поширеність: зважена оцінка проти незваженої. Аргументи у відсотках."""
    return (
        f"З урахуванням ваг опитування поширеність становить {weighted_pct:.1f}% "
        f"(95% ДІ {ci_low_pct:.1f}–{ci_high_pct:.1f}%). "
        f"Незважена частка у вибірці {unweighted_pct:.1f}%: NHANES навмисно надмірно представляє "
        "окремі групи, тому зважена оцінка ближча до населення США."
    )


def test_name_uk(test_name: str) -> str:
    """Українська назва тесту; невідому повертаємо як є."""
    key = test_name.lower()
    return next((uk for k, uk in TEST_NAMES_UK.items() if k in key), test_name)


def test_text(
    variable_label: str,
    test_name: str,
    statistic: float,
    p: float,
    p_holm: float,
    alpha: float = 0.05,
) -> str:
    """Результат порівняння груп: сире p, p після поправки Холма, висновок."""
    return (
        f"{variable_label}: {test_name_uk(test_name)}, статистика = {fmt_num(statistic)}, "
        f"сире p = {fmt_p(p)}, після поправки Холма p = {fmt_p(p_holm)}. "
        f"{p_value_text(p_holm, alpha)}"
    )


def threshold_text(threshold: float, sensitivity: float, specificity: float) -> str:
    """Чому поріг не 0.5 і що означають чутливість та специфічність."""
    return (
        f"Поріг класифікації {_pct(threshold)} % замість 50 %, бо хворих менше, ніж здорових; "
        "поріг за індексом Юдена максимізує суму чутливості й специфічності. "
        f"Чутливість {_pct(sensitivity)}% — стільки хворих модель виявляє, "
        f"специфічність {_pct(specificity)}% — стільки здорових вона правильно не лякає."
    )


def vif_text(vif_table: pd.DataFrame) -> str:
    """Висновок про колінеарність: таблиця з колонкою «vif», індекс — коди ознак (як у моделі)."""
    vif = vif_table["vif"].drop(index="const", errors="ignore").dropna()
    high = vif[vif > VIF_LIMIT].sort_values(ascending=False)
    if high.empty:
        return f"VIF усіх ознак нижче {VIF_LIMIT:.0f}: колінеарність незначна."
    names = " і ".join(
        f"«{FEATURE_LABELS_UK.get(str(i), str(i)).split(',')[0]}»" for i in high.index
    )
    return (
        f"VIF для {names} понад {VIF_LIMIT:.0f} (до {fmt_num(float(high.iloc[0]), 1)}): "
        "ознаки колінеарні, тому їхні OR у багатофакторній моделі треба тлумачити обережно."
    )
