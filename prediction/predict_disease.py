import json
from pathlib import Path

import joblib
import numpy as np
import torch

from database.operations import get_condition


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODELS_DIR = PROJECT_ROOT / "models"

METADATA_FILE = (
    MODELS_DIR
    / "best_model_metadata.json"
)


# ============================================================
# LOAD ACTIVE MODEL METADATA
# ============================================================

def _load_metadata():

    if not METADATA_FILE.exists():

        raise FileNotFoundError(
            "No active model metadata found.\n"
            "Run:\n"
            "python training\\compare_models.py"
        )

    with open(
        METADATA_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# ============================================================
# LOAD SKLEARN MODEL
# ============================================================

def _load_sklearn():

    model_file = (
        MODELS_DIR
        / "active_model.joblib"
    )

    vectorizer_file = (
        MODELS_DIR
        / "active_vectorizer.joblib"
    )

    if not model_file.exists():

        raise FileNotFoundError(
            f"Active sklearn model not found:\n"
            f"{model_file}"
        )

    if not vectorizer_file.exists():

        raise FileNotFoundError(
            f"Active vectorizer not found:\n"
            f"{vectorizer_file}"
        )

    model = joblib.load(
        model_file
    )

    vectorizer = joblib.load(
        vectorizer_file
    )

    return model, vectorizer


# ============================================================
# LOAD LSTM
# ============================================================

def _load_lstm():

    from training.train_lstm import (
        LSTMClassifier,
        encode_text
    )

    labels_file = (
        MODELS_DIR
        / "lstm_labels.json"
    )

    vocabulary_file = (
        MODELS_DIR
        / "lstm_vocab.json"
    )

    model_file = (
        MODELS_DIR
        / "active_lstm_model.pt"
    )

    if not labels_file.exists():

        raise FileNotFoundError(
            f"LSTM labels not found:\n"
            f"{labels_file}"
        )

    if not vocabulary_file.exists():

        raise FileNotFoundError(
            f"LSTM vocabulary not found:\n"
            f"{vocabulary_file}"
        )

    if not model_file.exists():

        raise FileNotFoundError(
            f"Active LSTM model not found:\n"
            f"{model_file}"
        )

    with open(
        labels_file,
        "r",
        encoding="utf-8"
    ) as file:

        labels = json.load(file)

    with open(
        vocabulary_file,
        "r",
        encoding="utf-8"
    ) as file:

        vocabulary = json.load(file)

    model = LSTMClassifier(
        len(vocabulary),
        len(labels)
    )

    model.load_state_dict(
        torch.load(
            model_file,
            map_location="cpu"
        )
    )

    model.eval()

    return (
        model,
        labels,
        vocabulary,
        encode_text
    )


# ============================================================
# LOAD BIOBERT
# ============================================================

def _load_biobert():

    from transformers import (
        AutoTokenizer
    )

    from training.train_biobert import (
        BioBERTClassifier,
        MAX_LENGTH
    )

    labels_file = (
        MODELS_DIR
        / "biobert_labels.json"
    )

    tokenizer_dir = (
        MODELS_DIR
        / "biobert_tokenizer"
    )

    model_file = (
        MODELS_DIR
        / "active_biobert_model.pt"
    )

    if not labels_file.exists():

        raise FileNotFoundError(
            f"BioBERT labels not found:\n"
            f"{labels_file}"
        )

    if not tokenizer_dir.exists():

        raise FileNotFoundError(
            f"BioBERT tokenizer not found:\n"
            f"{tokenizer_dir}"
        )

    if not model_file.exists():

        raise FileNotFoundError(
            f"Active BioBERT model not found:\n"
            f"{model_file}\n\n"
            f"Run compare_models.py first."
        )

    with open(
        labels_file,
        "r",
        encoding="utf-8"
    ) as file:

        labels = json.load(file)

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            tokenizer_dir
        )
    )

    model = BioBERTClassifier(
        len(labels)
    )

    model.load_state_dict(
        torch.load(
            model_file,
            map_location="cpu"
        )
    )

    model.eval()

    return (
        model,
        tokenizer,
        labels,
        MAX_LENGTH
    )


# ============================================================
# CONDITION INFORMATION
# ============================================================

def _get_condition_information(
    disease
):

    try:

        condition = get_condition(
            disease
        )

    except Exception:

        condition = None

    if not condition:

        return {

            "warnings": "",

            "recommendations": "",

            "emergency": False
        }

    return {

        "warnings":
            condition.get(
                "warnings",
                ""
            ),

        "recommendations":
            condition.get(
                "recommendations",
                ""
            ),

        "emergency":
            bool(
                condition.get(
                    "emergency_heuristic",
                    False
                )
            )
    }


# ============================================================
# MODEL NAME
# ============================================================

def _get_model_name(
    metadata,
    model_family
):

    model_name = metadata.get(
        "model_name"
    )

    if model_name:
        return model_name

    if model_family == "sklearn":
        return "Improved Logistic Regression"

    if model_family == "lstm":
        return "Bidirectional LSTM"

    if model_family == "transformer":
        return "BioBERT"

    return "Unknown"


# ============================================================
# PREDICT DISEASE
# ============================================================

def predict_disease(
    text,
    top_k=5,
    age=None,
    gender=None
):

    # ========================================================
    # VALIDATE INPUT
    # ========================================================

    if not isinstance(
        text,
        str
    ):

        raise ValueError(
            "Symptoms text must be a string."
        )

    text = text.strip()

    if not text:

        raise ValueError(
            "Symptoms text cannot be empty."
        )

    if not isinstance(
        top_k,
        int
    ):

        raise ValueError(
            "top_k must be an integer."
        )

    if top_k < 1:

        raise ValueError(
            "top_k must be at least 1."
        )

    # ========================================================
    # LOAD METADATA
    # ========================================================

    metadata = _load_metadata()

    model_family = metadata.get(
        "model_family"
    )

    # ========================================================
    # SKLEARN
    # ========================================================

    if model_family == "sklearn":

        model, vectorizer = (
            _load_sklearn()
        )

        transformed_text = (
            vectorizer.transform(
                [text]
            )
        )

        probabilities = (
            model.predict_proba(
                transformed_text
            )[0]
        )

        classes = np.asarray(
            model.classes_
        )

    # ========================================================
    # LSTM
    # ========================================================

    elif model_family == "lstm":

        (
            model,
            labels,
            vocabulary,
            encode_text
        ) = _load_lstm()

        encoded_text, _ = (
            encode_text(
                text,
                vocabulary
            )
        )

        input_tensor = torch.tensor(
            [encoded_text],
            dtype=torch.long
        )

        with torch.no_grad():

            logits = model(
                input_tensor
            )

            probabilities = (
                torch.softmax(
                    logits,
                    dim=1
                )
                .cpu()
                .numpy()[0]
            )

        classes = np.asarray(
            labels
        )

    # ========================================================
    # BIOBERT
    # ========================================================

    elif model_family == "transformer":

        (
            model,
            tokenizer,
            labels,
            max_length
        ) = _load_biobert()

        encoded = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )

        with torch.no_grad():

            logits = model(
                **encoded
            )

            probabilities = (
                torch.softmax(
                    logits,
                    dim=1
                )
                .cpu()
                .numpy()[0]
            )

        classes = np.asarray(
            labels
        )

    else:

        raise ValueError(
            f"Unsupported model family: "
            f"{model_family}"
        )

    # ========================================================
    # RANK PREDICTIONS
    # ========================================================

    number_of_predictions = min(
        top_k,
        len(classes)
    )

    ranked_indices = np.argsort(
        probabilities
    )[::-1][
        :number_of_predictions
    ]

    predictions = []

    for index in ranked_indices:

        disease = str(
            classes[index]
        )

        information = (
            _get_condition_information(
                disease
            )
        )

        prediction = {

            "rank":
                len(predictions) + 1,

            "disease":
                disease,

            "confidence":
                round(
                    float(
                        probabilities[index]
                    ),
                    6
                ),

            "warnings":
                information[
                    "warnings"
                ],

            "recommendations":
                information[
                    "recommendations"
                ],

            "emergency":
                information[
                    "emergency"
                ]
        }

        predictions.append(
            prediction
        )

    # ========================================================
    # FINAL RESPONSE
    # ========================================================

    return {

        "model":
            _get_model_name(
                metadata,
                model_family
            ),

        "model_family":
            model_family,

        "symptoms":
            text,

        "metadata": {

            "age":
                age,

            "gender":
                gender,

            "used_by_model":
                False,

            "note":
                "Age and gender are accepted by the API, "
                "but they are not used by the trained model "
                "because the training dataset contains "
                "text, disease, and section columns only."
        },

        "predictions":
            predictions,

        "disclaimer":
            "This system provides model-based candidate "
            "conditions from the training data and is not "
            "a medical diagnosis. Seek professional medical "
            "care for personal medical decisions or emergencies."
    }


# ============================================================
# COMMAND LINE TEST
# ============================================================

def main():

    print()
    print("=" * 70)
    print("MEDICAL AI DISEASE PREDICTION")
    print("=" * 70)

    symptoms = input(
        "Enter patient symptoms: "
    ).strip()

    if not symptoms:

        print(
            "No symptoms entered."
        )

        return

    try:

        result = predict_disease(
            symptoms,
            top_k=5
        )

        print()
        print("=" * 70)
        print("MODEL")
        print("=" * 70)

        print(
            result["model"]
        )

        print()
        print("=" * 70)
        print("PREDICTIONS")
        print("=" * 70)

        for prediction in (
            result["predictions"]
        ):

            print(
                f"\n"
                f"{prediction['rank']}. "
                f"{prediction['disease']}"
            )

            print(
                f"Confidence: "
                f"{prediction['confidence']:.2%}"
            )

            if prediction[
                "warnings"
            ]:

                print(
                    "Warnings: "
                    f"{prediction['warnings']}"
                )

            if prediction[
                "recommendations"
            ]:

                print(
                    "Recommendations: "
                    f"{prediction['recommendations']}"
                )

            if prediction[
                "emergency"
            ]:

                print(
                    "WARNING: "
                    "Emergency indicators found."
                )

        print()
        print("=" * 70)

    except Exception as error:

        print()
        print(
            "Prediction error:"
        )

        print(
            error
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()