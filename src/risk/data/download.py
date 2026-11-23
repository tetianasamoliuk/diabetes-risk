"""Завантаження XPT-файлів NHANES з CDC (I/O, мережа)."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from risk.config import CYCLES, MODULES, RAW_DIR
from risk.errors import DownloadError

BASE_URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{name}.xpt"
XPT_MAGIC = b"HEADER RECORD"
TIMEOUT_S = 120
WORKERS = 8


def file_name(module: str, cycle: str) -> str:
    """Ім'я файлу модуля в циклі; у 2017-2020 тиск лише осцилометричний (P_BPXO)."""
    cfg = CYCLES[cycle]
    if module == "BPX" and cfg["prefix"]:
        return f"{cfg['prefix']}BPXO"
    return f"{cfg['prefix']}{module}{cfg['suffix']}"


def download_file(year: int, name: str, dest: Path) -> Path:
    """Качає файл у dest; CDC на хибний URL віддає HTML (код 200) — перевіряємо заголовок."""
    url = BASE_URL.format(year=year, name=name)
    try:
        with urlopen(url, timeout=TIMEOUT_S) as resp:
            data = resp.read()
    except (URLError, TimeoutError) as exc:
        raise DownloadError(f"{name}: помилка мережі ({exc})") from exc
    if not data.startswith(XPT_MAGIC):
        raise DownloadError(f"{name}: відповідь не є XPT ({data[:60]!r})")
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / f"{name}.xpt"
    path.write_bytes(data)
    return path


def _fetch_missing(job: tuple[int, str], dest: Path) -> Path:
    year, name = job
    path = dest / f"{name}.xpt"
    return path if path.exists() else download_file(year, name, dest)


def download_all(dest: Path = RAW_DIR) -> list[Path]:
    """Качає всі модулі обох циклів; наявні файли пропускає."""
    jobs = [
        (cfg["year"], file_name(module, cycle))
        for cycle, cfg in CYCLES.items()
        for module in MODULES
    ]
    with ThreadPoolExecutor(WORKERS) as pool:
        return list(pool.map(lambda job: _fetch_missing(job, dest), jobs))
