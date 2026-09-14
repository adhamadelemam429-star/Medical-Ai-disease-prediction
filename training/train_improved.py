from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

TRAIN_FILE = DATA_DIR / "train.csv"
VALIDATION_FILE = DATA_DIR / "validation.csv"
TEST_FILE = DATA_DIR / "test.csv"

MODEL_FILE = MODELS_DIR / "improved_model.joblib"
VECTORIZER_FILE = MODELS_DIR / "improved_tfidf_vectorizer.joblib"
RESULT_FILE = RESULTS_DIR / "improved_results.json"

RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    train_df = pd.read_csv(TRAIN_FILE)
    validation_df = pd.read_csv(VALIDATION_FILE)
    test_df = pd.read_csv(TEST_FILE)

    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(validation_df)}")
    print(f"Test samples: {len(test_df)}")

    return train_df, validation_df, test_df


# ============================================================
# TF-IDF
# ============================================================

def build_vectorizer():

    return TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.98,
        sublinear_tf=True,
        max_features=30000
    )


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(train_df):

    print()
    print("=" * 60)
    print("TRAINING IMPROVED MODEL")
    print("=" * 60)

    vectorizer = build_vectorizer()

    X_train = vectorizer.fit_transform(
        train_df["text"]
    )

    y_train = train_df["disease"]

    print(
        f"Training matrix shape: {X_train.shape}"
    )

    print()
    print("Creating Logistic Regression classifier...")

    base_model = LogisticRegression(
        C=1.5,
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear",
        random_state=RANDOM_STATE
    )

    print(
        "Creating One-vs-Rest classifier..."
    )

    model = OneVsRestClassifier(
        base_model,
        n_jobs=1
    )

    print()
    print(
        "Training model. Please wait..."
    )

    model.fit(
        X_train,
        y_train
    )

    print()
    print(
        "Improved model trained successfully."
    )

    return vectorizer, model


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    vectorizer,
    model,
    dataframe,
    dataset_name
):

    X = vectorizer.transform(
        dataframe["text"]
    )

    y = dataframe["disease"]

    predictions = model.predict(
        X
    )

    accuracy = accuracy_score(
        y,
        predictions
    )

    weighted_f1 = f1_score(
        y,
        predictions,
        average="weighted",
        zero_division=0
    )

    macro_f1 = f1_score(
        y,
        predictions,
        average="macro",
        zero_division=0
    )

    print()
    print("=" * 60)
    print(f"{dataset_name} RESULTS")
    print("=" * 60)

    print(
        f"Accuracy: {accuracy:.4f}"
    )

    print(
        f"Weighted F1: {weighted_f1:.4f}"
    )

    print(
        f"Macro F1: {macro_f1:.4f}"
    )

    report = classification_report(
        y,
        predictions,
        output_dict=True,
        zero_division=0
    )

    return {
        "accuracy": float(accuracy),
        "weighted_f1": float(weighted_f1),
        "macro_f1": float(macro_f1),
        "classification_report": report
    }


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
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
    print("IMPROVED MODEL SAVED")
    print("=" * 60)

    print(
        f"Model: {MODEL_FILE}"
    )

    print(
        f"Vectorizer: {VECTORIZER_FILE}"
    )

    print(
        f"Results: {RESULT_FILE}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    train_df, validation_df, test_df = load_data()

    vectorizer, model = train_model(
        train_df
    )

    validation_results = evaluate(
        vectorizer,
        model,
        validation_df,
        "VALIDATION"
    )

    test_results = evaluate(
        vectorizer,
        model,
        test_df,
        "TEST"
    )

    results = {

        "model":
            "Improved TF-IDF + OneVsRest Logistic Regression",

        "configuration": {

            "ngram_range":
                "(1, 2)",

            "max_features":
                30000,

            "C":
                1.5,

            "class_weight":
                "balanced",

            "solver":
                "liblinear",

            "wrapper":
                "OneVsRestClassifier",

            "n_jobs":
                1,

            "max_iter":
                1000
        },

        "training_samples":
            len(train_df),

        "validation_samples":
            len(validation_df),

        "test_samples":
            len(test_df),

        "validation":
            validation_results,

        "test":
            test_results
    }

    save_results(
        vectorizer,
        model,
        results
    )

    print()
    print("=" * 60)
    print("IMPROVED TRAINING COMPLETED")
    print("=" * 60)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()