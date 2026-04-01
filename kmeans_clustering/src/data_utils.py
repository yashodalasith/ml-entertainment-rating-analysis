from pathlib import Path

import pandas as pd


def locate_file(file_name: str, start_dir: Path | None = None) -> Path:
    """Locate a dataset file in common project locations."""
    cwd = start_dir or Path.cwd()
    search_roots = [
        cwd / "data",
        cwd,
        cwd.parent / "data",
        cwd.parent,
    ]

    for root in search_roots:
        candidate = root / file_name
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Could not find {file_name}. Put it in kmeans_clustering/data or project-root/data."
    )


def load_raw_tables(start_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """Load animes, profiles and reviews tables from CSV files."""
    animes_path = locate_file("animes.csv", start_dir=start_dir)
    profiles_path = locate_file("profiles.csv", start_dir=start_dir)
    reviews_path = locate_file("reviews.csv", start_dir=start_dir)

    return {
        "animes": pd.read_csv(animes_path),
        "profiles": pd.read_csv(profiles_path),
        "reviews": pd.read_csv(reviews_path),
        "paths": {
            "animes": animes_path,
            "profiles": profiles_path,
            "reviews": reviews_path,
        },
    }
