import json
import re
import requests
import pandas as pd
from datasets import load_dataset
from src.config import (
    HF_SPOTIFY_DATASET,
    BILLBOARD_ALL_JSON_URL,
    BILLBOARD_CACHE_FILE,
    HIT_RANK_THRESHOLD,
    HIT_WEEKS_WINDOW,
)


def normalize_text(text: str) -> str:
    """Standardizes track titles and artist names for high-accuracy merging."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Remove parenthetical or bracketed text like (feat. X), [Remastered], (Remix)
    text = re.sub(r"[\(\[].*?[\)\]]", "", text)
    # Remove 'feat.' or 'ft.' suffixes outside parentheses
    text = re.sub(r"\b(feat|ft|featuring)\b.*", "", text)
    # Keep only alphanumeric characters and spaces
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_primary_artist(artist_str: str) -> str:
    """Extracts the lead artist from multi-artist strings (separated by ;, &, x, or comma)."""
    if not isinstance(artist_str, str):
        return ""
    # Split on common artist delimiters used by Spotify (;) and Billboard (&, featuring, ,)
    primary = re.split(
        r";|&|,|\bfeaturing\b|\bfeat\b|\bft\b|\bx\b|\bwith\b",
        artist_str,
        flags=re.IGNORECASE,
    )[0]
    return normalize_text(primary)


def load_live_spotify_data() -> pd.DataFrame:
    """Streams the Spotify tracks dataset directly from Hugging Face."""
    print("Streaming Spotify dataset from Hugging Face...")
    hf_data = load_dataset(HF_SPOTIFY_DATASET, split="train")
    df = hf_data.to_pandas()

    # Drop unnecessary index column if present
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Create normalized join keys
    df["track_clean"] = df["track_name"].apply(normalize_text)
    df["artist_clean"] = df["artists"].apply(extract_primary_artist)

    # Deduplicate tracks (keep the version with the highest popularity score)
    df = df.sort_values("popularity", ascending=False)
    df = df.drop_duplicates(subset=["track_clean", "artist_clean"], keep="first").copy()

    print(f"Loaded {len(df):,} unique Spotify tracks.")
    return df


def load_live_billboard_data(use_cache: bool = False) -> pd.DataFrame:
    """Fetches the daily-updated Billboard Hot 100 history from GitHub and defines hit labels."""
    if use_cache and BILLBOARD_CACHE_FILE.exists():
        print(f"Loading cached Billboard data from {BILLBOARD_CACHE_FILE}...")
        with open(BILLBOARD_CACHE_FILE, "r", encoding="utf-8") as f:
            charts = json.load(f)
    else:
        print("Fetching live Billboard Hot 100 history from GitHub (~50 MB)...")
        response = requests.get(BILLBOARD_ALL_JSON_URL, timeout=120)
        response.raise_for_status()
        charts = response.json()
        with open(BILLBOARD_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(charts, f)

    # Flatten nested JSON: [{"date": "YYYY-MM-DD", "data": [{"song": ..., "this_week": ...}]}]
    records = []
    for chart in charts:
        chart_date = chart.get("date", "")
        for entry in chart.get("data", []):
            records.append(
                {
                    "chart_date": chart_date,
                    "song": entry.get("song"),
                    "artist": entry.get("artist"),
                    "this_week": entry.get("this_week"),
                    "peak_position": entry.get("peak_position"),
                    "weeks_on_chart": entry.get("weeks_on_chart"),
                }
            )

    bb_df = pd.DataFrame(records)
    bb_df["chart_year"] = pd.to_datetime(bb_df["chart_date"], errors="coerce").dt.year

    # Normalize join keys
    bb_df["track_clean"] = bb_df["song"].apply(normalize_text)
    bb_df["artist_clean"] = bb_df["artist"].apply(extract_primary_artist)

    # Check if entry reached Top 40 within its first 12 weeks on the chart
    bb_df["hit_entry"] = (
        (bb_df["this_week"] <= HIT_RANK_THRESHOLD)
        & (bb_df["weeks_on_chart"] <= HIT_WEEKS_WINDOW)
    ).astype(int)

    # Aggregate to song-level summary
    song_summary = (
        bb_df.groupby(["track_clean", "artist_clean"], as_index=False)
        .agg(
            is_hit=("hit_entry", "max"),
            peak_position=("this_week", "min"),
            max_weeks_on_chart=("weeks_on_chart", "max"),
            release_year=("chart_year", "min"),
        )
    )

    print(
        f"Processed {len(song_summary):,} unique Billboard songs "
        f"({song_summary['is_hit'].sum():,} verified Top-{HIT_RANK_THRESHOLD} hits)."
    )
    return song_summary
