"""Конфігурація проєкту: шляхи, seed, списки ознак, пороги.

Єдине місце, де визначені константи методології.
"""

from pathlib import Path

from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"
MODELS_DIR = ROOT / "models"

SEED = 42
ALPHA = 0.05  # рівень значущості для всіх тестів
Z95 = float(norm.ppf(0.975))  # квантиль нормального розподілу для 95 % довірчих інтервалів
# Параметри валідації моделей: єдине джерело для коду і текстів сторінок
TEST_SIZE = 0.2  # частка тестової вибірки
TEST_PCT = round(TEST_SIZE * 100)  # те саме у відсотках для текстів
TRAIN_PCT = 100 - TEST_PCT
CV_FOLDS = 5  # стратифікованих фолдів крос-валідації
CALIBRATION_BINS = 10  # груп калібрувальної кривої

# Цикли NHANES: рік у URL, схема імені файлу, MEC-вага і коефіцієнт для об'єднання циклів
# (частка тривалості циклу: 2 і 3.2 роки з 5.2, за методикою CDC)
CYCLES: dict[str, dict] = {
    "2015-2016": {
        "year": 2015,
        "suffix": "_I",
        "prefix": "",
        "weight": "WTMEC2YR",
        "weight_k": 2 / 5.2,
    },
    "2017-2020": {
        "year": 2017,
        "suffix": "",
        "prefix": "P_",
        "weight": "WTMECPRP",
        "weight_k": 3.2 / 5.2,
    },
}
# Модулі NHANES; у циклі 2017-2020 замість BPX є лише осцилометричний P_BPXO
MODULES = ["DEMO", "BMX", "BPX", "GHB", "DIQ", "MCQ", "PAQ", "BPQ", "TCHOL", "HDL"]

# Колонки тиску по циклах: у 2015–2016 аускультативні виміри, у 2017–2020 осцилометричні
SBP_COLS = {
    "2015-2016": [f"BPXSY{i}" for i in range(1, 5)],
    "2017-2020": [f"BPXOSY{i}" for i in range(1, 4)],
}
DBP_COLS = {
    "2015-2016": [f"BPXDI{i}" for i in range(1, 5)],
    "2017-2020": [f"BPXODI{i}" for i in range(1, 4)],
}

# GPAQ: (чи займався 1/2, днів/тиждень, хвилин/день).
# Домени: робота інтенсивна/помірна, транспорт, дозвілля інтенсивне/помірне
PA_DOMAINS = [
    ("PAQ605", "PAQ610", "PAD615"),
    ("PAQ620", "PAQ625", "PAD630"),
    ("PAQ635", "PAQ640", "PAD645"),
    ("PAQ650", "PAQ655", "PAD660"),
    ("PAQ665", "PAQ670", "PAD675"),
]

HBA1C_THRESHOLD = 6.5
ADULT_AGE = 20
ACTIVE_MIN_PER_WEEK = 210  # ≈ 30 хв/день для пункту FINDRISC «активність»
# Межі окружності талії, см: (нижня, верхня) для чоловіків (True) і жінок (False);
# верхня межа — також поріг абдомінального ожиріння в клінічних OR
WAIST_CUTS_CM: dict[bool, tuple[int, int]] = {True: (94, 102), False: (80, 88)}

# Ознаки моделей. Глікемічні маркери заборонені, бо вони є складовими цілі.
FEATURES = [
    "age",
    "male",
    "bmi",
    "waist",
    "sbp",
    "dbp",
    "tc",
    "hdl",
    "pa_h_week",
    "fam_dm",
    "bp_med",
    "cycle_2017",
]
# Монотонні обмеження ансамблів дерев (+1 — ризик не спадає з ознакою, −1 — не зростає).
# Лише для ознак сценаріїв «що якщо»: менші ІМТ чи талія або більша активність не можуть
# підвищити оцінку; решта ознак без обмежень. Логістична регресія монотонна сама.
MONOTONIC_FEATURES: dict[str, int] = {"bmi": 1, "waist": 1, "pa_h_week": -1}
FORBIDDEN_FEATURE_PATTERNS = (
    "LBXGH",
    "LBXGLU",
    "LBXIN",  # інсулін (NHANES-код)
    "DIQ",
    "hba1c",
    "glu",
    "diq010",
    "high_glu_hist",
    "t2d",
    "t1d",
    "insulin",
    "INS",
)

# Клінічний крок, на який рахується OR для неперервних факторів
OR_SCALE: dict[str, float] = {
    "age": 10,
    "bmi": 5,
    "waist": 10,
    "sbp": 10,
    "dbp": 10,
    "tc": 10,
    "hdl": 10,
    "pa_h_week": 1,
}
OR_SCALE_LABELS_UK: dict[str, str] = {
    "age": "на 10 років",
    "bmi": "на 5 кг/м²",
    "waist": "на 10 см",
    "sbp": "на 10 мм рт. ст.",
    "dbp": "на 10 мм рт. ст.",
    "tc": "на 10 мг/дл",
    "hdl": "на 10 мг/дл",
    "pa_h_week": "на 1 год/тиждень",
}

CONTINUOUS_FACTORS = ["age", "bmi", "waist", "sbp", "dbp", "tc", "hdl", "pa_h_week"]
BINARY_FACTORS = ["male", "fam_dm", "bp_med", "cycle_2017"]

FEATURE_LABELS_UK: dict[str, str] = {
    "age": "Вік, років",
    "male": "Стать (чоловіча)",
    "bmi": "Індекс маси тіла, кг/м²",
    "waist": "Окружність талії, см",
    "sbp": "Систолічний АТ, мм рт. ст.",
    "dbp": "Діастолічний АТ, мм рт. ст.",
    "tc": "Загальний холестерин, мг/дл",
    "hdl": "ЛПВЩ-холестерин, мг/дл",
    "pa_h_week": "Фізична активність, год/тиждень",
    "pa_min_week": "Фізична активність, хв/тиждень",
    "fam_dm": "Діабет у близьких родичів",
    "bp_med": "Приймає ліки від тиску",
    "cycle_2017": "Цикл 2017–2020",
    "hba1c": "HbA1c, %",
    "t2d": "Цукровий діабет 2 типу",
    "t2d_undiag": "Недіагностований діабет (HbA1c ≥ 6.5)",
    "high_glu_hist": "Підвищена глюкоза в анамнезі",
    "chol_med": "Приймає ліки від холестерину",
    "t1d_proxy": "Ймовірний ЦД1 серед діагностованих (діагноз до 30 років та інсулін)",
}

# Код NHANES та одиниці для таблиці змінних (сторінка «Дані і вибірка»)
NHANES_CODES: dict[str, tuple[str, str]] = {
    "age": ("RIDAGEYR", "років"),
    "male": ("RIAGENDR", "1 = чоловік"),
    "bmi": ("BMXBMI", "кг/м²"),
    "waist": ("BMXWAIST", "см"),
    "sbp": ("BPXSY1–4 / BPXOSY1–3", "мм рт. ст."),
    "dbp": ("BPXDI1–4 / BPXODI1–3", "мм рт. ст."),
    "tc": ("LBXTC", "мг/дл"),
    "hdl": ("LBDHDD", "мг/дл"),
    "pa_min_week": ("PAQ605–PAD675 (GPAQ)", "хв/тиждень"),
    "fam_dm": ("MCQ300C", "1 = так"),
    "bp_med": ("BPQ050A (BPQ020, BPQ040A)", "1 = так"),
    "high_glu_hist": ("DIQ160, DIQ010 = 3", "1 = так"),
    "hba1c": ("LBXGH", "%"),
    "t2d": ("DIQ010 = 1 або LBXGH ≥ 6.5", "1 = так"),
    "chol_med": ("BPQ100D (BPQ080, BPQ090D)", "1 = так"),
    "t1d_proxy": ("DIQ010, DID040, DIQ050", "1 = так"),
}

MODEL_NAMES_UK = {"logit": "Логістична регресія", "rf": "Random Forest", "hgb": "Gradient Boosting"}

DISCLAIMER = "Результат є дослідницькою оцінкою моделі, а не медичним діагнозом."

# Рядок автора: «Прізвище І.Б., група»; єдине джерело для застосунку й документів
AUTHOR = "Самолюк Т.Г., 251з"
# Затверджена назва теми роботи
TOPIC = "Визначення ризиків захворювань на основі можливих факторів впливу"
