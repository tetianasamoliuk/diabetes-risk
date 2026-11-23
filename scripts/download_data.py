"""Завантажує сирі XPT-файли NHANES у data/raw/ (наявні пропускає)."""

import argparse

from risk.data.download import download_all


def main() -> None:
    """Точка входу CLI."""
    argparse.ArgumentParser(description=__doc__).parse_args()
    for path in download_all():
        print(path)


if __name__ == "__main__":
    main()
