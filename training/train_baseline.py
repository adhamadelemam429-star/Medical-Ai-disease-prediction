from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

TRAIN_FILE = DATA_DIR / "train.csv"
VALIDATION_FILE = DATA_DIR / "validation.csv"
TEST_FILE = DATA_DIR / "test.csv"

MODEL_FILE = MODELS_DIR / "baseline_model.joblib"
VECTORIZER_FILE = MODELS_DIR / "tfidf_vectorizer.joblib"
RESULT_FILE = RESULTS_DIR / "baseline_results.json"


def load_data():

    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    train_df = pd.read_csv(
        TRAIN_FILE
    )

    validation_df = pd.read_csv(
        VALIDATION_FILE
    )

    test_df = pd.read_csv(
        TEST_FILE
    )

    print(
        f"Training samples: {len(train_df)}"
    )

    print(
        f"Validation samples: {len(validation_df)}"
    )

    print(
        f"Test samples: {len(test_df)}"
    )

    return (
        train_df,
        validation_df,
        test_df
    )


def train_model(
    train_df
):

    print()
    print("=" * 60)
    print("TRAINING BASELINE MODEL")
    print("=" * 60)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        sublinear_tf=True
    )

    X_train = vectorizer.fit_transform(
        train_df["text"]
    )

    y_train = train_df["disease"]

    print(
        f"Training matrix shape: "
        f"{X_train.shape}"
    )

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    print(
        "Baseline model trained successfully."
    )

    return (
        vectorizer,
        model
    )


def evaluate_model(
    vectorizer,
    model,
    validation_df,
    test_df
):

    print()
    print("=" * 60)
    print("EVALUATING MODEL")
    print("=" * 60)

    X_validation = vectorizer.transform(
        validation_df["text"]
    )

    y_validation = validation_df["disease"]

    validation_predictions = model.predict(
        X_validation
    )

    validation_accuracy = accuracy_score(
        y_validation,
        validation_predictions
    )

    validation_f1 = f1_score(
        y_validation,
        validation_predictions,
        average="weighted",
        zero_division=0
    )

    print()
    print("VALIDATION RESULTS")
    print(
        f"Accuracy: "
        f"{validation_accuracy:.4f}"
    )

    print(
        f"Weighted F1: "
        f"{validation_f1:.4f}"
    )

    X_test = vectorizer.transform(
        test_df["text"]
    )

    y_test = test_df["disease"]

    test_predictions = model.predict(
        X_test
    )

    test_accuracy = accuracy_score(
        y_test,
        test_predictions
    )

    test_f1 = f1_score(
        y_test,
        test_predictions,
        average="weighted",
        zero_division=0
    )

    print()
    print("TEST RESULTS")

    print(
        f"Accuracy: "
        f"{test_accuracy:.4f}"
    )

    print(
        f"Weighted F1: "
        f"{test_f1:.4f}"
    )

    report = classification_report(
        y_test,
        test_predictions,
        output_dict=True,
        zero_division=0
    )

    results = {
        "model": "TF-IDF + Logistic Regression",
        "training_samples": len(
            validation_df
        ),
        "validation_samples": len(
            validation_df
        ),
        "test_samples": len(
            test_df
        ),
        "validation_accuracy":
            float(validation_accuracy),
        "validation_weighted_f1":
            float(validation_f1),
        "test_accuracy":
            float(test_accuracy),
        "test_weighted_f1":
            float(test_f1),
        "classification_report":
            report
    }

    return results


def save_model(
    vectorizer,
    model,
    results
):

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        vectorizer,
        VECTORIZER_FILE
    )

    joblib.dump(
        model,
        MODEL_FILE
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
    print("=" * 60)
    print("MODEL SAVED")
    print("=" * 60)

    print(
        f"Vectorizer: "
        f"{VECTORIZER_FILE}"
    )

    print(
        f"Model: "
        f"{MODEL_FILE}"
    )

    print(
        f"Results: "
        f"{RESULT_FILE}"
    )


def main():

    (
        train_df,
        validation_df,
        test_df
    ) = load_data()

    (
        vectorizer,
        model
    ) = train_model(
        train_df
    )

    results = evaluate_model(
        vectorizer,
        model,
        validation_df,
        test_df
    )

    save_model(
        vectorizer,
        model,
        results
    )

    print()
    print("=" * 60)
    print("BASELINE TRAINING COMPLETED")
    print("=" * 60)


if __name__ == "__main__":

    main()