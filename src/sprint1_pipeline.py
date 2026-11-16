import pandas as pd
from src.config import OUTPUT_PARQUET, OUTPUT_CSV
from src.data_ingestion import load_live_spotify_data, load_live_billboard_data
from src.feature_engineering import (
    sanitize_and_impute,
    build_structural_and_repetition_features,
    assign_release_year_proxy,
)


def run_sprint1_pipeline(use_cached_billboard: bool = False) -> pd.DataFrame:
    # Step 1: Ingest Live Datasets
    spotify_df = load_live_spotify_data()
    billboard_df = load_live_billboard_data(use_cache=use_cached_billboard)

    # Step 2: Join Datasets on Normalized Track + Primary Artist
    print("Merging Spotify features with Billboard ground-truth labels...")
    master_df = pd.merge(
        spotify_df,
        billboard_df,
        on=["track_clean", "artist_clean"],
        how="left",
    )

    # Fill non-charting tracks with 0 for 'is_hit'
    master_df["is_hit"] = master_df["is_hit"].fillna(0).astype(int)
    master_df["charted_on_billboard"] = master_df["peak_position"].notna().astype(int)
    master_df["peak_position"] = master_df["peak_position"].fillna(0).astype(int)
    master_df["max_weeks_on_chart"] = master_df["max_weeks_on_chart"].fillna(0).astype(int)

    # Step 3: Clean Data and Engineer Features
    print("Running quality assurance and feature engineering...")
    master_df = sanitize_and_impute(master_df)
    master_df = build_structural_and_repetition_features(master_df)
    master_df = assign_release_year_proxy(master_df)

    # Step 4: Organize Final Columns
    final_columns = [
        # Metadata
        "track_id",
        "track_name",
        "artists",
        "album_name",
        "track_genre",
        "release_year",
        "explicit",
        "popularity",
        # Core Audio Features
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
        # Engineered Structural & Repetition Features
        "duration_min",
        "beat_density",
        "is_radio_length",
        "energy_loudness_ratio",
        "dance_energy_product",
        "mood_intensity_index",
        "vocal_presence_index",
        "repetition_hook_proxy",
        # Target & Chart Metrics
        "charted_on_billboard",
        "peak_position",
        "max_weeks_on_chart",
        "is_hit",
    ]

    available_cols = [c for c in final_columns if c in master_df.columns]
    final_df = master_df[available_cols].copy()

    # Step 5: Save Outputs
    final_df.to_parquet(OUTPUT_PARQUET, index=False)
    final_df.to_csv(OUTPUT_CSV, index=False)

    print("\n" + "=" * 50)
    print("SPRINT 1 COMPLETE: MASTER FEATURE TABLE SUMMARY")
    print("=" * 50)
    print(f"Total Validated Tracks : {len(final_df):,}")
    print(f"Total Features         : {len(final_df.columns)}")
    print(f"Billboard Charted      : {final_df['charted_on_billboard'].sum():,}")
    print(f"Top 40 Hits (is_hit=1) : {final_df['is_hit'].sum():,} ({final_df['is_hit'].mean() * 100:.2f}%)")
    print(f"Saved Parquet to       : {OUTPUT_PARQUET}")
    print(f"Saved CSV to           : {OUTPUT_CSV}")
    print("=" * 50)

    return final_df


if __name__ == "__main__":
    # Set use_cached_billboard=True after your first run to avoid re-downloading 50MB every time
    run_sprint1_pipeline(use_cached_billboard=False)
