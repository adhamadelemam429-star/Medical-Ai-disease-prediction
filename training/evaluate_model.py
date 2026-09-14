from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

TEST_FILE = DATA_DIR / "test.csv"

MODEL_FILE = MODELS_DIR / "baseline_model.joblib"
VECTORIZER_FILE = MODELS_DIR / "tfidf_vectorizer.joblib"

RESULT_FILE = RESULTS_DIR / "final_evaluation.json"


def main():

    print("=" * 60)
    print("FINAL MODEL EVALUATION")
    print("=" * 60)

    if not MODEL_FILE.exists():

        raise FileNotFoundError(
            f"Model not found: {MODEL_FILE}"
        )

    if not VECTORIZER_FILE.exists():

        raise FileNotFoundError(
            f"Vectorizer not found: "
            f"{VECTORIZER_FILE}"
        )

    if not TEST_FILE.exists():

        raise FileNotFoundError(
            f"Test dataset not found: "
            f"{TEST_FILE}"
        )

    model = joblib.load(
        MODEL_FILE
    )

    vectorizer = joblib.load(
        VECTORIZER_FILE
    )

    test_df = pd.read_csv(
        TEST_FILE
    )

    X_test = vectorizer.transform(
        test_df["text"]
    )

    y_test = test_df["disease"]

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    weighted_f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0
    )

    print()
    print(
        f"Test Accuracy: "
        f"{accuracy:.4f}"
    )

    print(
        f"Test Weighted F1: "
        f"{weighted_f1:.4f}"
    )

    results = {
        "model":
            "TF-IDF + Logistic Regression",

        "test_samples":
            len(test_df),

        "test_accuracy":
            float(accuracy),

        "test_weighted_f1":
            float(weighted_f1),

        "classification_report":
            report
    }

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        RESULT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            ensure_ascii=False,
            indent=4
        )

    print()
    print(
        f"Evaluation saved to:"
    )

    print(
        RESULT_FILE
    )

    print()
    print("=" * 60)
    print("EVALUATION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":

    main()