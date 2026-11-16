import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    BEST_MODEL_PATH,
    FEATURE_IMPORTANCE_CSV,
    OUTPUT_PARQUET,
    SPLIT_YEAR,
    SPRINT2_RESULTS_CSV,
)

# Strictly audio, structural, and repetition features (excludes chart leakage columns)
MODEL_FEATURES = [
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
]


def load_and_split_data(split_year: int = SPLIT_YEAR):
    """Loads the Sprint 1 master table and performs a temporal split by release_year."""
    if not OUTPUT_PARQUET.exists():
        raise FileNotFoundError(
            f"Cannot find {OUTPUT_PARQUET}. Run 'python -m src.sprint1_pipeline' first."
        )

    df = pd.read_parquet(OUTPUT_PARQUET)
    features = [c for c in MODEL_FEATURES if c in df.columns]

    # Time-based split: train on tracks prior to split_year, test on split_year and later
    train_df = df[df["release_year"] < split_year].copy()
    test_df = df[df["release_year"] >= split_year].copy()

    # Fallback to 80/20 chronological quantile split if all tracks fall on one side
    if len(train_df) == 0 or len(test_df) == 0 or test_df["is_hit"].sum() == 0:
        cutoff = int(df["release_year"].quantile(0.8))
        train_df = df[df["release_year"] < cutoff].copy()
        test_df = df[df["release_year"] >= cutoff].copy()
        split_year = cutoff

    X_train = train_df[features]
    y_train = train_df["is_hit"]
    X_test = test_df[features]
    y_test = test_df["is_hit"]

    print(f"Temporal Split Year : {split_year}")
    print(f"Train Set (< {split_year})  : {len(X_train):,} tracks ({y_train.sum():,} hits)")
    print(f"Test Set (>= {split_year})  : {len(X_test):,} tracks ({y_test.sum():,} hits)\n")

    return X_train, X_test, y_train, y_test, features


def evaluate_model(name: str, model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Computes classification metrics for imbalanced hit prediction."""
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = y_pred.astype(float)

    return {
        "Model": name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "F1_Score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "ROC_AUC": round(roc_auc_score(y_test, y_prob), 4) if y_test.nunique() > 1 else 0.5,
        "PR_AUC": round(average_precision_score(y_test, y_prob), 4) if y_test.nunique() > 1 else 0.0,
    }


def run_sprint2_pipeline():
    X_train, X_test, y_train, y_test, features = load_and_split_data()

    models = {
        "Baseline (Majority Class)": DummyClassifier(strategy="most_frequent"),
        "Baseline (Logistic Regression)": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        class_weight="balanced", max_iter=1000, random_state=42
                    ),
                ),
            ]
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            random_state=42,
        ),
    }

    # Balance weights for Gradient Boosting so it doesn't just guess 0
    from sklearn.utils.class_weight import compute_sample_weight
    sample_weights = compute_sample_weight("balanced", y_train)

    results = []
    trained_models = {}

    for name, model in models.items():
        print(f"Training {name}...")
        if name == "Gradient Boosting":
            model.fit(X_train, y_train, sample_weight=sample_weights)
        else:
            model.fit(X_train, y_train)
        metrics = evaluate_model(name, model, X_test, y_test)
        results.append(metrics)
        trained_models[name] = model

    # Sort by F1_Score so we pick a model that actually finds hits
    results_df = pd.DataFrame(results).sort_values(by="F1_Score", ascending=False)
    results_df.to_csv(SPRINT2_RESULTS_CSV, index=False)

    # Select best non-dummy model and save artifact
    non_dummy = results_df[~results_df["Model"].str.contains("Majority")]
    best_model_name = non_dummy.iloc[0]["Model"]
    best_model = trained_models[best_model_name]

    joblib.dump(
        {"model_name": best_model_name, "model": best_model, "features": features},
        BEST_MODEL_PATH,
    )

    # Export feature importances from Random Forest
    rf_model = trained_models["Random Forest"]
    importance_df = pd.DataFrame(
        {"feature": features, "importance": rf_model.feature_importances_}
    ).sort_values("importance", ascending=False)
    importance_df.to_csv(FEATURE_IMPORTANCE_CSV, index=False)

    print("\n" + "=" * 75)
    print("SPRINT 2 COMPLETE: MODEL COMPARISON TABLE")
    print("=" * 75)
    print(results_df.to_string(index=False))
    print("=" * 75)
    print(f"Best Model Saved        : {best_model_name} -> {BEST_MODEL_PATH}")
    print(f"Results Table Saved     : {SPRINT2_RESULTS_CSV}")
    print(f"Feature Importance Saved: {FEATURE_IMPORTANCE_CSV}")

    return results_df


if __name__ == "__main__":
    run_sprint2_pipeline()