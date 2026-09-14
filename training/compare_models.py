import json
import shutil
from pathlib import Path

import joblib
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

TEST_FILE = DATA_DIR / "test.csv"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

def calculate_sklearn_metrics(model_file, vectorizer_file):
    """
    Evaluate a saved sklearn model directly on test.csv.
    """

    model = joblib.load(model_file)
    vectorizer = joblib.load(vectorizer_file)

    df = pd.read_csv(TEST_FILE)

    required_columns = {"text", "disease"}

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns in {TEST_FILE}: {missing}"
        )

    df["text"] = df["text"].fillna("").astype(str)
    df["disease"] = df["disease"].fillna("").astype(str)

    X = vectorizer.transform(df["text"])
    y_true = df["disease"].values

    predictions = model.predict(X)

    accuracy = (predictions == y_true).mean()

    # Weighted F1 and Macro F1 without requiring sklearn metrics
    from sklearn.metrics import f1_score

    weighted_f1 = f1_score(
        y_true,
        predictions,
        average="weighted",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_true,
        predictions,
        average="macro",
        zero_division=0
    )

    return {
        "model": "Improved Logistic Regression",
        "model_family": "sklearn",
        "test_accuracy": float(accuracy),
        "test_weighted_f1": float(weighted_f1),
        "test_macro_f1": float(macro_f1),
    }


# ============================================================
# LOAD BIOBERT METRICS
# ============================================================

def load_biobert_metrics():

    file = MODELS_DIR / "biobert_metrics.json"

    if not file.exists():
        return None

    with open(file, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    return {
        "model": "BioBERT",
        "model_family": "transformer",
        "validation_accuracy": metrics.get(
            "validation_accuracy"
        ),
        "validation_weighted_f1": metrics.get(
            "validation_weighted_f1"
        ),
        "validation_macro_f1": metrics.get(
            "validation_macro_f1"
        ),
        "validation_top_3_accuracy": metrics.get(
            "validation_top_3_accuracy"
        ),
        "validation_top_5_accuracy": metrics.get(
            "validation_top_5_accuracy"
        ),
        "test_accuracy": metrics.get(
            "test_accuracy"
        ),
        "test_weighted_f1": metrics.get(
            "test_weighted_f1"
        ),
        "test_macro_f1": metrics.get(
            "test_macro_f1"
        ),
        "test_top_3_accuracy": metrics.get(
            "test_top_3_accuracy"
        ),
        "test_top_5_accuracy": metrics.get(
            "test_top_5_accuracy"
        ),
    }


# ============================================================
# LOAD LSTM METRICS
# ============================================================

def load_lstm_metrics():

    file = MODELS_DIR / "lstm_metrics.json"

    if not file.exists():
        return None

    with open(file, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    return {
        "model": "Bidirectional LSTM",
        "model_family": "lstm",
        "validation_accuracy": metrics.get(
            "validation_accuracy"
        ),
        "validation_weighted_f1": metrics.get(
            "validation_weighted_f1"
        ),
        "validation_macro_f1": metrics.get(
            "validation_macro_f1"
        ),
        "validation_top_3_accuracy": metrics.get(
            "validation_top_3_accuracy"
        ),
        "validation_top_5_accuracy": metrics.get(
            "validation_top_5_accuracy"
        ),
        "test_accuracy": metrics.get(
            "test_accuracy"
        ),
        "test_weighted_f1": metrics.get(
            "test_weighted_f1"
        ),
        "test_macro_f1": metrics.get(
            "test_macro_f1"
        ),
        "test_top_3_accuracy": metrics.get(
            "test_top_3_accuracy"
        ),
        "test_top_5_accuracy": metrics.get(
            "test_top_5_accuracy"
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)

    results = []

    # --------------------------------------------------------
    # IMPROVED LOGISTIC REGRESSION
    # --------------------------------------------------------

    improved_model = MODELS_DIR / "improved_model.joblib"
    improved_vectorizer = (
        MODELS_DIR / "improved_tfidf_vectorizer.joblib"
    )

    if (
        improved_model.exists()
        and improved_vectorizer.exists()
    ):

        print()
        print("Evaluating Improved Logistic Regression...")

        improved_metrics = calculate_sklearn_metrics(
            improved_model,
            improved_vectorizer
        )

        # Known validation metrics from previous training
        improved_metrics[
            "validation_accuracy"
        ] = 0.8686

        improved_metrics[
            "validation_weighted_f1"
        ] = 0.8467

        improved_metrics[
            "validation_macro_f1"
        ] = 0.8375

        results.append(improved_metrics)

        print(
            f"Test Accuracy: "
            f"{improved_metrics['test_accuracy']:.4f}"
        )

        print(
            f"Test Weighted F1: "
            f"{improved_metrics['test_weighted_f1']:.4f}"
        )

        print(
            f"Test Macro F1: "
            f"{improved_metrics['test_macro_f1']:.4f}"
        )

    else:

        print()
        print(
            "Improved model files were not found."
        )


    # --------------------------------------------------------
    # BIOBERT
    # --------------------------------------------------------

    biobert = load_biobert_metrics()

    if biobert is not None:

        results.append(biobert)

    else:

        print(
            "Skipping missing BioBERT metrics file."
        )


    # --------------------------------------------------------
    # LSTM
    # --------------------------------------------------------

    lstm = load_lstm_metrics()

    if lstm is not None:

        results.append(lstm)

    else:

        print(
            "Skipping missing LSTM metrics file."
        )


    # --------------------------------------------------------
    # CHECK RESULTS
    # --------------------------------------------------------

    if not results:

        raise RuntimeError(
            "No model results were found."
        )


    # --------------------------------------------------------
    # DATAFRAME
    # --------------------------------------------------------

    df = pd.DataFrame(results)

    # Ensure expected columns exist
    expected_columns = [
        "model",
        "model_family",
        "validation_accuracy",
        "validation_weighted_f1",
        "validation_macro_f1",
        "validation_top_5_accuracy",
        "test_accuracy",
        "test_weighted_f1",
        "test_macro_f1",
        "test_top_5_accuracy",
    ]

    for column in expected_columns:

        if column not in df.columns:

            df[column] = None


    # --------------------------------------------------------
    # SELECTION SCORE
    # --------------------------------------------------------

    def selection_score(row):

        macro = row["validation_macro_f1"]

        top5 = row["validation_top_5_accuracy"]

        if pd.notna(macro) and pd.notna(top5):

            return (
                0.5 * float(macro)
                + 0.5 * float(top5)
            )

        if pd.notna(macro):

            return float(macro)

        if pd.notna(row["test_macro_f1"]):

            return float(row["test_macro_f1"])

        return 0.0


    df["selection_score"] = df.apply(
        selection_score,
        axis=1
    )


    # ========================================================
    # DISPLAY
    # ========================================================

    print()
    print("=" * 80)
    print("MODEL RESULTS")
    print("=" * 80)

    display_columns = [
        "model",
        "validation_accuracy",
        "validation_macro_f1",
        "validation_top_5_accuracy",
        "test_accuracy",
        "test_weighted_f1",
        "test_macro_f1",
        "test_top_5_accuracy",
        "selection_score",
    ]

    print(
        df[display_columns].to_string(
            index=False
        )
    )


    # ========================================================
    # BEST MODEL
    # ========================================================

    best_index = df["selection_score"].idxmax()

    best = df.loc[best_index]

    best_model_name = best["model"]
    best_model_family = best["model_family"]
    best_score = float(best["selection_score"])


    print()
    print("=" * 80)
    print(f"BEST MODEL: {best_model_name}")
    print(f"Model family: {best_model_family}")
    print(f"Selection score: {best_score:.4f}")
    print("=" * 80)


    # ========================================================
    # ACTIVATE BEST MODEL
    # ========================================================

    active_model = None

    if best_model_family == "sklearn":

        source_model = (
            MODELS_DIR /
            "improved_model.joblib"
        )

        source_vectorizer = (
            MODELS_DIR /
            "improved_tfidf_vectorizer.joblib"
        )

        active_model_path = (
            MODELS_DIR /
            "active_model.joblib"
        )

        active_vectorizer_path = (
            MODELS_DIR /
            "active_vectorizer.joblib"
        )

        shutil.copy2(
            source_model,
            active_model_path
        )

        shutil.copy2(
            source_vectorizer,
            active_vectorizer_path
        )

        active_model = str(active_model_path)


    elif best_model_family == "transformer":

        source_model = (
            MODELS_DIR /
            "biobert_model.pt"
        )

        active_model_path = (
            MODELS_DIR /
            "active_biobert_model.pt"
        )

        if source_model.exists():

            shutil.copy2(
                source_model,
                active_model_path
            )

            active_model = str(
                active_model_path
            )


    elif best_model_family == "lstm":

        source_model = (
            MODELS_DIR /
            "lstm_model.pt"
        )

        active_model_path = (
            MODELS_DIR /
            "active_lstm_model.pt"
        )

        if source_model.exists():

            shutil.copy2(
                source_model,
                active_model_path
            )

            active_model = str(
                active_model_path
            )


    # ========================================================
    # METADATA
    # ========================================================

    metadata = {
        "best_model": best_model_name,
        "model_family": best_model_family,
        "selection_score": best_score,
        "active_model": active_model,
        "test_accuracy": (
            None
            if pd.isna(best["test_accuracy"])
            else float(best["test_accuracy"])
        ),
        "test_weighted_f1": (
            None
            if pd.isna(best["test_weighted_f1"])
            else float(best["test_weighted_f1"])
        ),
        "test_macro_f1": (
            None
            if pd.isna(best["test_macro_f1"])
            else float(best["test_macro_f1"])
        ),
    }


    metadata_file = (
        MODELS_DIR /
        "best_model_metadata.json"
    )

    with open(
        metadata_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4,
            ensure_ascii=False
        )


    # ========================================================
    # SAVE COMPARISON FILES
    # ========================================================

    comparison_csv = (
        RESULTS_DIR /
        "model_comparison.csv"
    )

    comparison_json = (
        RESULTS_DIR /
        "model_comparison.json"
    )

    df.to_csv(
        comparison_csv,
        index=False
    )

    with open(
        comparison_json,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=4,
            ensure_ascii=False
        )


    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print()
    print("=" * 80)
    print("ACTIVE MODEL")
    print("=" * 80)

    print(
        f"Best model: {best_model_name}"
    )

    print(
        f"Family: {best_model_family}"
    )

    print(
        f"Test Accuracy: "
        f"{float(best['test_accuracy']):.4f}"
    )

    print()
    print("Active model metadata:")
    print(metadata_file)

    print()
    print("Comparison CSV:")
    print(comparison_csv)

    print()
    print("Comparison JSON:")
    print(comparison_json)

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()