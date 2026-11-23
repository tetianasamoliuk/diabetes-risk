"""Завантаження XPT-файлів NHANES: імена файлів, перевірка вмісту, пропуск наявних."""

from pathlib import Path
from urllib.error import URLError

import pytest

from risk.data import download
from risk.data.download import XPT_MAGIC, download_file, file_name
from risk.errors import DownloadError


class FakeResp:
    """Мінімальна заміна відповіді `urlopen`."""

    def __init__(self, data: bytes) -> None:
        """Запам'ятовує тіло відповіді."""
        self.data = data

    def __enter__(self) -> "FakeResp":
        """Вхід у контекст."""
        return self

    def __exit__(self, *args: object) -> None:
        """Вихід із контексту."""
        return None

    def read(self) -> bytes:
        """Тіло відповіді."""
        return self.data


@pytest.mark.parametrize(
    ("module", "cycle", "expected"),
    [
        ("DEMO", "2015-2016", "DEMO_I"),
        ("BPX", "2015-2016", "BPX_I"),
        ("DEMO", "2017-2020", "P_DEMO"),
        ("BPX", "2017-2020", "P_BPXO"),
        ("TCHOL", "2017-2020", "P_TCHOL"),
    ],
)
def test_file_name(module: str, cycle: str, expected: str) -> None:
    assert file_name(module, cycle) == expected


def test_html_response_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(download, "urlopen", lambda *a, **k: FakeResp(b"<html>404</html>"))
    with pytest.raises(DownloadError):
        download_file(2015, "X", tmp_path)
    assert not list(tmp_path.iterdir())


def test_xpt_written(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    payload = XPT_MAGIC + b" rest"
    monkeypatch.setattr(download, "urlopen", lambda *a, **k: FakeResp(payload))
    path = download_file(2015, "DEMO_I", tmp_path)
    assert path == tmp_path / "DEMO_I.xpt"
    assert path.read_bytes() == payload


def test_download_all_skips_existing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[str] = []

    def fake(url: str, **kw: object) -> FakeResp:
        calls.append(url)
        return FakeResp(XPT_MAGIC)

    monkeypatch.setattr(download, "urlopen", fake)
    (tmp_path / "DEMO_I.xpt").write_bytes(XPT_MAGIC)
    paths = download.download_all(tmp_path)
    assert len(paths) == 20
    assert len(calls) == 19
    assert all("DEMO_I" not in c for c in calls)


def test_download_file_network_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def boom(url: str, **kw: object) -> FakeResp:
        raise URLError("no network")

    monkeypatch.setattr(download, "urlopen", boom)
    with pytest.raises(DownloadError, match="помилка мережі"):
        download_file(2015, "DEMO_I", tmp_path)
