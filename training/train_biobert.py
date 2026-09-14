import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel


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


# ============================================================
# BIOBERT CONFIGURATION
# ============================================================

MODEL_NAME = "dmis-lab/biobert-base-cased-v1.2"

MAX_LENGTH = 192

BATCH_SIZE = 4

EPOCHS = 3

LEARNING_RATE = 2e-5

WEIGHT_DECAY = 0.01

GRADIENT_ACCUMULATION = 4

SEED = 42


# ============================================================
# OUTPUT FILES
# ============================================================

MODEL_FILE = MODELS_DIR / "biobert_model.pt"

LABELS_FILE = MODELS_DIR / "biobert_labels.json"

CONFIG_FILE = MODELS_DIR / "biobert_config.json"

TOKENIZER_DIR = MODELS_DIR / "biobert_tokenizer"

METRICS_FILE = MODELS_DIR / "biobert_metrics.json"

RESULT_FILE = RESULTS_DIR / "biobert_results.json"


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# RANDOM SEED
# ============================================================

def set_seed(seed):
    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# LOAD CSV
# ============================================================

def load_dataframe(path):
    print()
    print(f"Loading dataset:")
    print(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found:\n{path}"
        )

    df = pd.read_csv(path)

    print(
        f"Columns found: {list(df.columns)}"
    )

    print(
        f"Rows found: {len(df)}"
    )

    # --------------------------------------------------------
    # IMPORTANT
    # Actual project CSV columns are:
    #
    # text
    # disease
    # section
    #
    # NOT:
    # text
    # label
    # --------------------------------------------------------

    required_columns = {
        "text",
        "disease"
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:

        raise ValueError(
            f"\nMissing required columns in:\n"
            f"{path}\n\n"
            f"Missing columns: "
            f"{sorted(missing_columns)}\n\n"
            f"Expected columns:\n"
            f"['text', 'disease', 'section']"
        )

    # --------------------------------------------------------
    # Clean text
    # --------------------------------------------------------

    df["text"] = (
        df["text"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # Clean disease
    # --------------------------------------------------------

    df["disease"] = (
        df["disease"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # Remove empty records
    # --------------------------------------------------------

    df = df[
        (df["text"] != "")
        &
        (df["disease"] != "")
    ].copy()

    df = df.reset_index(drop=True)

    print(
        f"Rows after cleaning: {len(df)}"
    )

    return df


# ============================================================
# DATASET
# ============================================================

class MedicalDataset(Dataset):

    def __init__(
        self,
        dataframe,
        tokenizer,
        label_encoder
    ):

        self.texts = (
            dataframe["text"]
            .tolist()
        )

        self.labels = (
            label_encoder
            .transform(
                dataframe["disease"]
            )
            .astype(np.int64)
        )

        self.tokenizer = tokenizer

    def __len__(self):

        return len(self.labels)

    def __getitem__(self, index):

        text = self.texts[index]

        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        item = {
            key: value.squeeze(0)
            for key, value
            in encoded.items()
        }

        item["labels"] = torch.tensor(
            self.labels[index],
            dtype=torch.long
        )

        return item


# ============================================================
# BIOBERT CLASSIFIER
# ============================================================

class BioBERTClassifier(nn.Module):

    def __init__(
        self,
        num_classes
    ):

        super().__init__()

        print()
        print(
            "Loading BioBERT backbone..."
        )

        self.bert = AutoModel.from_pretrained(
            MODEL_NAME
        )

        hidden_size = (
            self.bert.config.hidden_size
        )

        self.dropout = nn.Dropout(
            0.3
        )

        self.classifier = nn.Linear(
            hidden_size,
            num_classes
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids=None
    ):

        bert_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }

        if token_type_ids is not None:

            bert_inputs[
                "token_type_ids"
            ] = token_type_ids

        outputs = self.bert(
            **bert_inputs
        )

        # ----------------------------------------------------
        # Use pooler output if available.
        # Otherwise use CLS token.
        # ----------------------------------------------------

        if getattr(
            outputs,
            "pooler_output",
            None
        ) is not None:

            pooled_output = (
                outputs.pooler_output
            )

        else:

            pooled_output = (
                outputs
                .last_hidden_state[:, 0, :]
            )

        pooled_output = (
            self.dropout(
                pooled_output
            )
        )

        logits = self.classifier(
            pooled_output
        )

        return logits


# ============================================================
# TOP-K ACCURACY
# ============================================================

def calculate_top_k_accuracy(
    probabilities,
    labels,
    k
):

    if probabilities.shape[1] == 0:
        return 0.0

    k = min(
        k,
        probabilities.shape[1]
    )

    top_k_indices = np.argsort(
        probabilities,
        axis=1
    )[:, -k:]

    correct = 0

    for true_label, predicted_labels in zip(
        labels,
        top_k_indices
    ):

        if true_label in predicted_labels:

            correct += 1

    return (
        correct
        /
        len(labels)
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    model,
    dataloader,
    num_classes
):

    model.eval()

    all_logits = []

    all_labels = []

    total_loss = 0.0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():

        for batch in dataloader:

            labels = batch.pop(
                "labels"
            )

            batch = {
                key: value.to(DEVICE)
                for key, value
                in batch.items()
            }

            labels_device = labels.to(
                DEVICE
            )

            logits = model(
                **batch
            )

            loss = criterion(
                logits,
                labels_device
            )

            total_loss += (
                loss.item()
                *
                len(labels)
            )

            all_logits.append(
                logits.cpu()
            )

            all_labels.append(
                labels
            )

    if not all_labels:

        return {
            "loss": 0.0,
            "accuracy": 0.0,
            "weighted_f1": 0.0,
            "macro_f1": 0.0,
            "top_3_accuracy": 0.0,
            "top_5_accuracy": 0.0
        }

    logits = torch.cat(
        all_logits
    ).numpy()

    labels = torch.cat(
        all_labels
    ).numpy()

    predictions = np.argmax(
        logits,
        axis=1
    )

    probabilities = (
        torch.softmax(
            torch.from_numpy(logits),
            dim=1
        )
        .numpy()
    )

    accuracy = accuracy_score(
        labels,
        predictions
    )

    weighted_f1 = f1_score(
        labels,
        predictions,
        average="weighted",
        zero_division=0
    )

    macro_f1 = f1_score(
        labels,
        predictions,
        average="macro",
        zero_division=0
    )

    top_3_accuracy = (
        calculate_top_k_accuracy(
            probabilities,
            labels,
            3
        )
    )

    top_5_accuracy = (
        calculate_top_k_accuracy(
            probabilities,
            labels,
            5
        )
    )

    average_loss = (
        total_loss
        /
        len(labels)
    )

    return {

        "loss": float(
            average_loss
        ),

        "accuracy": float(
            accuracy
        ),

        "weighted_f1": float(
            weighted_f1
        ),

        "macro_f1": float(
            macro_f1
        ),

        "top_3_accuracy": float(
            top_3_accuracy
        ),

        "top_5_accuracy": float(
            top_5_accuracy
        )
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("BIOBERT MEDICAL DISEASE CLASSIFICATION")
    print("=" * 70)

    print()
    print(
        f"BioBERT model: {MODEL_NAME}"
    )

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Max sequence length: {MAX_LENGTH}"
    )

    print(
        f"Batch size: {BATCH_SIZE}"
    )

    print(
        f"Epochs: {EPOCHS}"
    )

    print(
        f"Learning rate: {LEARNING_RATE}"
    )

    print("=" * 70)

    # ========================================================
    # SEED
    # ========================================================

    set_seed(SEED)

    # ========================================================
    # CREATE DIRECTORIES
    # ========================================================

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # LOAD DATA
    # ========================================================

    train_df = load_dataframe(
        TRAIN_FILE
    )

    validation_df = load_dataframe(
        VALIDATION_FILE
    )

    test_df = load_dataframe(
        TEST_FILE
    )

    print()
    print("=" * 70)
    print("DATASET INFORMATION")
    print("=" * 70)

    print(
        f"Training samples: "
        f"{len(train_df)}"
    )

    print(
        f"Validation samples: "
        f"{len(validation_df)}"
    )

    print(
        f"Test samples: "
        f"{len(test_df)}"
    )

    # ========================================================
    # LABEL ENCODER
    # ========================================================

    print()
    print("=" * 70)
    print("LABEL ENCODING")
    print("=" * 70)

    # --------------------------------------------------------
    # Get diseases from training data
    # --------------------------------------------------------

    training_diseases = sorted(
        train_df["disease"]
        .unique()
        .tolist()
    )

    validation_diseases = set(
        validation_df["disease"]
        .unique()
        .tolist()
    )

    test_diseases = set(
        test_df["disease"]
        .unique()
        .tolist()
    )

    training_disease_set = set(
        training_diseases
    )

    validation_missing = (
        validation_diseases
        -
        training_disease_set
    )

    test_missing = (
        test_diseases
        -
        training_disease_set
    )

    if validation_missing:

        raise ValueError(
            "Validation contains diseases "
            "that do not exist in training:\n"
            f"{sorted(validation_missing)}"
        )

    if test_missing:

        raise ValueError(
            "Test contains diseases "
            "that do not exist in training:\n"
            f"{sorted(test_missing)}"
        )

    # ========================================================
    # FIT LABEL ENCODER
    # ========================================================

    label_encoder = LabelEncoder()

    label_encoder.fit(
        training_diseases
    )

    num_classes = len(
        label_encoder.classes_
    )

    print(
        f"Number of diseases: "
        f"{num_classes}"
    )

    # ========================================================
    # SAVE LABELS
    # ========================================================

    with open(
        LABELS_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            label_encoder
            .classes_
            .tolist(),
            file,
            ensure_ascii=False,
            indent=2
        )

    print(
        f"Labels saved to:"
    )

    print(
        LABELS_FILE
    )

    # ========================================================
    # LOAD TOKENIZER
    # ========================================================

    print()
    print("=" * 70)
    print("LOADING BIOBERT TOKENIZER")
    print("=" * 70)

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            MODEL_NAME
        )
    )

    TOKENIZER_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    tokenizer.save_pretrained(
        TOKENIZER_DIR
    )

    print(
        f"Tokenizer saved to:"
    )

    print(
        TOKENIZER_DIR
    )

    # ========================================================
    # CREATE DATASETS
    # ========================================================

    print()
    print("=" * 70)
    print("CREATING DATASETS")
    print("=" * 70)

    train_dataset = MedicalDataset(
        train_df,
        tokenizer,
        label_encoder
    )

    validation_dataset = MedicalDataset(
        validation_df,
        tokenizer,
        label_encoder
    )

    test_dataset = MedicalDataset(
        test_df,
        tokenizer,
        label_encoder
    )

    # ========================================================
    # CREATE DATALOADERS
    # ========================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # ========================================================
    # CREATE MODEL
    # ========================================================

    print()
    print("=" * 70)
    print("CREATING BIOBERT CLASSIFIER")
    print("=" * 70)

    model = BioBERTClassifier(
        num_classes
    )

    model = model.to(
        DEVICE
    )

    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # ========================================================
    # LOSS FUNCTION
    # ========================================================

    criterion = nn.CrossEntropyLoss()

    # ========================================================
    # BEST MODEL TRACKING
    # ========================================================

    best_score = -1.0

    best_validation_metrics = None

    # ========================================================
    # TRAINING
    # ========================================================

    print()
    print("=" * 70)
    print("STARTING BIOBERT TRAINING")
    print("=" * 70)

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        model.train()

        running_loss = 0.0

        optimizer.zero_grad(
            set_to_none=True
        )

        for step, batch in enumerate(
            train_loader,
            start=1
        ):

            labels = batch.pop(
                "labels"
            )

            batch = {
                key: value.to(DEVICE)
                for key, value
                in batch.items()
            }

            labels = labels.to(
                DEVICE
            )

            # ------------------------------------------------
            # Forward pass
            # ------------------------------------------------

            logits = model(
                **batch
            )

            # ------------------------------------------------
            # Loss
            # ------------------------------------------------

            loss = criterion(
                logits,
                labels
            )

            # ------------------------------------------------
            # Gradient accumulation
            # ------------------------------------------------

            scaled_loss = (
                loss
                /
                GRADIENT_ACCUMULATION
            )

            scaled_loss.backward()

            # ------------------------------------------------
            # Optimizer step
            # ------------------------------------------------

            if (
                step
                %
                GRADIENT_ACCUMULATION
                == 0
                or
                step
                ==
                len(train_loader)
            ):

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0
                )

                optimizer.step()

                optimizer.zero_grad(
                    set_to_none=True
                )

            running_loss += (
                loss.item()
            )

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            if step % 50 == 0:

                print(
                    f"Epoch "
                    f"{epoch}/{EPOCHS} "
                    f"| Step "
                    f"{step}/{len(train_loader)} "
                    f"| Loss "
                    f"{loss.item():.4f}"
                )

        # ====================================================
        # TRAINING LOSS
        # ====================================================

        epoch_loss = (
            running_loss
            /
            len(train_loader)
        )

        # ====================================================
        # VALIDATION
        # ====================================================

        validation_metrics = evaluate(
            model,
            validation_loader,
            num_classes
        )

        # ====================================================
        # MODEL SELECTION SCORE
        # ====================================================

        selection_score = (
            0.5
            *
            validation_metrics[
                "macro_f1"
            ]
            +
            0.5
            *
            validation_metrics[
                "top_5_accuracy"
            ]
        )

        # ====================================================
        # PRINT EPOCH RESULTS
        # ====================================================

        print()
        print("-" * 70)

        print(
            f"Epoch {epoch}/{EPOCHS}"
        )

        print(
            f"Training Loss: "
            f"{epoch_loss:.4f}"
        )

        print(
            f"Validation Accuracy: "
            f"{validation_metrics['accuracy']:.4f}"
        )

        print(
            f"Validation Weighted F1: "
            f"{validation_metrics['weighted_f1']:.4f}"
        )

        print(
            f"Validation Macro F1: "
            f"{validation_metrics['macro_f1']:.4f}"
        )

        print(
            f"Validation Top-3 Accuracy: "
            f"{validation_metrics['top_3_accuracy']:.4f}"
        )

        print(
            f"Validation Top-5 Accuracy: "
            f"{validation_metrics['top_5_accuracy']:.4f}"
        )

        print(
            f"Selection Score: "
            f"{selection_score:.4f}"
        )

        print("-" * 70)

        # ====================================================
        # SAVE BEST MODEL
        # ====================================================

        if selection_score > best_score:

            best_score = (
                selection_score
            )

            best_validation_metrics = (
                validation_metrics.copy()
            )

            torch.save(
                model.state_dict(),
                MODEL_FILE
            )

            print(
                "BEST MODEL UPDATED"
            )

            print(
                f"Saved to:"
            )

            print(
                MODEL_FILE
            )

    # ========================================================
    # CHECK BEST MODEL
    # ========================================================

    if not MODEL_FILE.exists():

        raise RuntimeError(
            "BioBERT model was not saved."
        )

    # ========================================================
    # LOAD BEST MODEL
    # ========================================================

    print()
    print("=" * 70)
    print("LOADING BEST BIOBERT MODEL")
    print("=" * 70)

    model.load_state_dict(
        torch.load(
            MODEL_FILE,
            map_location=DEVICE
        )
    )

    model.eval()

    # ========================================================
    # FINAL VALIDATION
    # ========================================================

    print()
    print("=" * 70)
    print("FINAL VALIDATION")
    print("=" * 70)

    final_validation = evaluate(
        model,
        validation_loader,
        num_classes
    )

    print(
        f"Validation Accuracy: "
        f"{final_validation['accuracy']:.4f}"
    )

    print(
        f"Validation Weighted F1: "
        f"{final_validation['weighted_f1']:.4f}"
    )

    print(
        f"Validation Macro F1: "
        f"{final_validation['macro_f1']:.4f}"
    )

    print(
        f"Validation Top-3 Accuracy: "
        f"{final_validation['top_3_accuracy']:.4f}"
    )

    print(
        f"Validation Top-5 Accuracy: "
        f"{final_validation['top_5_accuracy']:.4f}"
    )

    # ========================================================
    # FINAL TEST
    # ========================================================

    print()
    print("=" * 70)
    print("FINAL TEST")
    print("=" * 70)

    final_test = evaluate(
        model,
        test_loader,
        num_classes
    )

    print(
        f"Test Accuracy: "
        f"{final_test['accuracy']:.4f}"
    )

    print(
        f"Test Weighted F1: "
        f"{final_test['weighted_f1']:.4f}"
    )

    print(
        f"Test Macro F1: "
        f"{final_test['macro_f1']:.4f}"
    )

    print(
        f"Test Top-3 Accuracy: "
        f"{final_test['top_3_accuracy']:.4f}"
    )

    print(
        f"Test Top-5 Accuracy: "
        f"{final_test['top_5_accuracy']:.4f}"
    )

    # ========================================================
    # SAVE CONFIG
    # ========================================================

    config = {

        "model_name":
            MODEL_NAME,

        "max_length":
            MAX_LENGTH,

        "batch_size":
            BATCH_SIZE,

        "epochs":
            EPOCHS,

        "learning_rate":
            LEARNING_RATE,

        "weight_decay":
            WEIGHT_DECAY,

        "gradient_accumulation":
            GRADIENT_ACCUMULATION,

        "num_classes":
            num_classes,

        "hidden_size":
            int(
                model
                .bert
                .config
                .hidden_size
            ),

        "dropout":
            0.3,

        "device":
            str(DEVICE)
    }

    with open(
        CONFIG_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            config,
            file,
            ensure_ascii=False,
            indent=2
        )

    # ========================================================
    # SAVE METRICS
    # ========================================================

    metrics = {

        "model":
            "BioBERT",

        "model_family":
            "transformer",

        "model_name":
            MODEL_NAME,

        "training_samples":
            len(train_df),

        "validation_samples":
            len(validation_df),

        "test_samples":
            len(test_df),

        "num_classes":
            num_classes,

        "validation":
            final_validation,

        "test":
            final_test,

        "best_validation_score":
            float(best_score),

        "best_validation_metrics":
            best_validation_metrics,

        "configuration":
            config
    }

    with open(
        METRICS_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metrics,
            file,
            ensure_ascii=False,
            indent=2
        )

    # ========================================================
    # SAVE RESULT FILE
    # ========================================================

    with open(
        RESULT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metrics,
            file,
            ensure_ascii=False,
            indent=2
        )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print()
    print("=" * 70)
    print("BIOBERT TRAINING COMPLETED")
    print("=" * 70)

    print()
    print(
        f"Model:"
    )

    print(
        MODEL_NAME
    )

    print()
    print(
        f"Test Accuracy: "
        f"{final_test['accuracy']:.4f}"
    )

    print(
        f"Test Weighted F1: "
        f"{final_test['weighted_f1']:.4f}"
    )

    print(
        f"Test Macro F1: "
        f"{final_test['macro_f1']:.4f}"
    )

    print(
        f"Test Top-5 Accuracy: "
        f"{final_test['top_5_accuracy']:.4f}"
    )

    print()
    print(
        "Files created:"
    )

    print(
        MODEL_FILE
    )

    print(
        LABELS_FILE
    )

    print(
        CONFIG_FILE
    )

    print(
        TOKENIZER_DIR
    )

    print(
        METRICS_FILE
    )

    print(
        RESULT_FILE
    )

    print()
    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()