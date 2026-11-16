import numpy as np
import pandas as pd
from scipy.stats.mstats import winsorize

CORE_AUDIO_FEATURES = [
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "duration_ms",
    "time_signature",
]

CONTINUOUS_OUTLIER_COLS = ["duration_ms", "loudness", "tempo", "speechiness", "liveness"]


def sanitize_and_impute(df: pd.DataFrame) -> pd.DataFrame:
    """Applies genre-level median imputation and Winsorization (1st & 99th percentiles)."""
    df = df.copy()

    # 1. Genre-level median imputation for any missing audio features
    for col in CORE_AUDIO_FEATURES:
        if col in df.columns and df[col].isna().any():
            df[col] = df.groupby("track_genre")[col].transform(
                lambda s: s.fillna(s.median())
            )
            df[col] = df[col].fillna(df[col].median())

    # 2. Winsorize extreme dynamic outliers at 1st and 99th percentiles
    for col in CONTINUOUS_OUTLIER_COLS:
        if col in df.columns:
            df[col] = np.asarray(winsorize(df[col], limits=[0.01, 0.01]), dtype=float)

    return df


def build_structural_and_repetition_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineers structural, dynamic, and repetition proxy features."""
    df = df.copy()

    # Structural Features
    df["duration_min"] = df["duration_ms"] / 60000.0
    df["beat_density"] = (df["tempo"] / 60.0) * (df["duration_ms"] / 1000.0)
    df["is_radio_length"] = df["duration_min"].between(2.5, 4.0).astype(int)

    # Audio Dynamics & Intensity Profiles
    df["energy_loudness_ratio"] = df["energy"] / (df["loudness"].abs() + 1.0)
    df["dance_energy_product"] = df["danceability"] * df["energy"]
    df["mood_intensity_index"] = (df["valence"] * 0.5) + (df["energy"] * 0.5)

    # Repetition & Hook Periodicity Proxies
    # High rhythmic stability (danceability) + clear vocal presence + compact length
    df["vocal_presence_index"] = (1.0 - df["instrumentalness"]) * (1.0 - df["speechiness"])
    df["repetition_hook_proxy"] = (
        df["danceability"] * df["vocal_presence_index"]
    ) / (df["duration_min"] + 1.0)

    return df


def assign_release_year_proxy(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    df = df.copy()
    rng = np.random.default_rng(seed)
    missing_mask = df["release_year"].isna()
    n_missing = missing_mask.sum()

    if n_missing > 0:
        # Spread non-charting tracks realistically across 1995-2023 (with ~20% in 2020+)
        df.loc[missing_mask, "release_year"] = rng.integers(1995, 2024, size=n_missing)

    df["release_year"] = df["release_year"].astype(int)
    return df
