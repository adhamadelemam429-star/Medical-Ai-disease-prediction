import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_FILE = PROJECT_ROOT / "data" / "train.csv"
VALIDATION_FILE = PROJECT_ROOT / "data" / "validation.csv"
TEST_FILE = PROJECT_ROOT / "data" / "test.csv"

MODEL_DIR = PROJECT_ROOT / "models"

MODEL_FILE = MODEL_DIR / "lstm_model.pt"
LABEL_FILE = MODEL_DIR / "lstm_labels.json"
METRICS_FILE = MODEL_DIR / "lstm_metrics.json"


# =========================================================
# REPRODUCIBILITY
# =========================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# =========================================================
# CONFIGURATION
# =========================================================

MAX_VOCAB = 20000
MAX_LENGTH = 250

EMBEDDING_DIM = 256
HIDDEN_DIM = 256

NUM_LAYERS = 2

DROPOUT = 0.35

BATCH_SIZE = 32

EPOCHS = 20

LEARNING_RATE = 0.0005

WEIGHT_DECAY = 1e-4

PATIENCE = 5

GRADIENT_CLIP = 1.0


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# =========================================================
# TOKENIZATION
# =========================================================

def tokenize(text):

    text = str(text).lower()

    tokens = text.split()

    return tokens


# =========================================================
# VOCABULARY
# =========================================================

def build_vocabulary(texts):

    frequency = {}

    for text in texts:

        for token in tokenize(text):

            frequency[token] = (
                frequency.get(token, 0) + 1
            )

    sorted_words = sorted(
        frequency.items(),
        key=lambda item: item[1],
        reverse=True
    )

    vocabulary = {
        "<PAD>": 0,
        "<UNK>": 1
    }

    for word, _ in sorted_words:

        if len(vocabulary) >= MAX_VOCAB:
            break

        if word not in vocabulary:

            vocabulary[word] = len(vocabulary)

    return vocabulary


# =========================================================
# ENCODE TEXT
# =========================================================

def encode_text(text, vocabulary):

    tokens = tokenize(text)

    ids = [
        vocabulary.get(
            token,
            vocabulary["<UNK>"]
        )
        for token in tokens
    ]

    ids = ids[:MAX_LENGTH]

    if len(ids) == 0:

        ids = [vocabulary["<UNK>"]]

    length = len(ids)

    if length < MAX_LENGTH:

        ids += [
            vocabulary["<PAD>"]
        ] * (
            MAX_LENGTH - length
        )

    return ids, length


# =========================================================
# DATASET
# =========================================================

class MedicalDataset(Dataset):

    def __init__(
        self,
        dataframe,
        vocabulary,
        label_encoder
    ):

        self.texts = []

        self.lengths = []

        for text in dataframe["text"]:

            ids, length = encode_text(
                text,
                vocabulary
            )

            self.texts.append(ids)

            self.lengths.append(length)

        self.labels = label_encoder.transform(
            dataframe["disease"]
        )

    def __len__(self):

        return len(self.labels)

    def __getitem__(self, index):

        return (
            torch.tensor(
                self.texts[index],
                dtype=torch.long
            ),
            torch.tensor(
                self.lengths[index],
                dtype=torch.long
            ),
            torch.tensor(
                self.labels[index],
                dtype=torch.long
            )
        )


# =========================================================
# ATTENTION
# =========================================================

class AttentionLayer(nn.Module):

    def __init__(self, hidden_dim):

        super().__init__()

        self.attention = nn.Linear(
            hidden_dim,
            1
        )

    def forward(
        self,
        outputs,
        mask
    ):

        scores = self.attention(
            outputs
        ).squeeze(-1)

        scores = scores.masked_fill(
            ~mask,
            -1e9
        )

        weights = torch.softmax(
            scores,
            dim=1
        )

        context = torch.bmm(
            weights.unsqueeze(1),
            outputs
        ).squeeze(1)

        return context


# =========================================================
# BIDIRECTIONAL LSTM
# =========================================================

class LSTMClassifier(nn.Module):

    def __init__(
        self,
        vocab_size,
        num_classes
    ):

        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            EMBEDDING_DIM,
            padding_idx=0
        )

        self.embedding_dropout = nn.Dropout(
            0.15
        )

        self.lstm = nn.LSTM(
            input_size=EMBEDDING_DIM,
            hidden_size=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=DROPOUT if NUM_LAYERS > 1 else 0.0
        )

        self.attention = AttentionLayer(
            HIDDEN_DIM * 2
        )

        self.dropout = nn.Dropout(
            DROPOUT
        )

        self.classifier = nn.Sequential(

            nn.Linear(
                HIDDEN_DIM * 4,
                HIDDEN_DIM * 2
            ),

            nn.ReLU(),

            nn.Dropout(
                DROPOUT
            ),

            nn.Linear(
                HIDDEN_DIM * 2,
                num_classes
            )
        )

    def forward(
        self,
        x,
        lengths
    ):

        embedded = self.embedding(
            x
        )

        embedded = self.embedding_dropout(
            embedded
        )

        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        packed_output, (
            hidden,
            cell
        ) = self.lstm(
            packed
        )

        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=x.size(1)
        )

        forward_hidden = hidden[-2]

        backward_hidden = hidden[-1]

        final_hidden = torch.cat(
            (
                forward_hidden,
                backward_hidden
            ),
            dim=1
        )

        positions = torch.arange(
            x.size(1),
            device=x.device
        ).unsqueeze(0)

        mask = positions < lengths.to(
            x.device
        ).unsqueeze(1)

        attention_context = self.attention(
            outputs,
            mask
        )

        combined = torch.cat(
            (
                final_hidden,
                attention_context
            ),
            dim=1
        )

        combined = self.dropout(
            combined
        )

        return self.classifier(
            combined
        )


# =========================================================
# EVALUATION
# =========================================================

def evaluate(
    model,
    loader,
    criterion=None
):

    model.eval()

    predictions = []

    actual = []

    total_loss = 0.0

    with torch.no_grad():

        for texts, lengths, labels in loader:

            texts = texts.to(
                DEVICE
            )

            lengths = lengths.to(
                DEVICE
            )

            labels = labels.to(
                DEVICE
            )

            outputs = model(
                texts,
                lengths
            )

            if criterion is not None:

                loss = criterion(
                    outputs,
                    labels
                )

                total_loss += loss.item()

            predicted = torch.argmax(
                outputs,
                dim=1
            )

            predictions.extend(
                predicted.cpu().numpy()
            )

            actual.extend(
                labels.cpu().numpy()
            )

    accuracy = accuracy_score(
        actual,
        predictions
    )

    macro_f1 = f1_score(
        actual,
        predictions,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        actual,
        predictions,
        average="weighted",
        zero_division=0
    )

    average_loss = (
        total_loss / len(loader)
        if criterion is not None
        else 0.0
    )

    return (
        accuracy,
        macro_f1,
        weighted_f1,
        average_loss
    )


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 70)
    print("TRAINING ADVANCED BIDIRECTIONAL LSTM + ATTENTION")
    print("=" * 70)

    print(
        f"Device: {DEVICE}"
    )

    # -----------------------------------------------------
    # Load data
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # Label encoder
    # -----------------------------------------------------

    label_encoder = LabelEncoder()

    label_encoder.fit(
        train_df["disease"]
    )

    num_classes = len(
        label_encoder.classes_
    )

    print(
        f"Number of diseases: {num_classes}"
    )

    # -----------------------------------------------------
    # Vocabulary
    # -----------------------------------------------------

    print()
    print("Building vocabulary...")

    vocabulary = build_vocabulary(
        train_df["text"]
    )

    print(
        f"Vocabulary size: {len(vocabulary)}"
    )

    # -----------------------------------------------------
    # Datasets
    # -----------------------------------------------------

    train_dataset = MedicalDataset(
        train_df,
        vocabulary,
        label_encoder
    )

    validation_dataset = MedicalDataset(
        validation_df,
        vocabulary,
        label_encoder
    )

    test_dataset = MedicalDataset(
        test_df,
        vocabulary,
        label_encoder
    )

    # -----------------------------------------------------
    # DataLoaders
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # Class weights
    # -----------------------------------------------------

    print()
    print("Calculating class weights...")

    train_labels = label_encoder.transform(
        train_df["disease"]
    )

    classes = np.unique(
        train_labels
    )

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=train_labels
    )

    class_weights = np.ones(
        num_classes,
        dtype=np.float32
    )

    for class_index, weight in zip(
        classes,
        weights
    ):

        class_weights[
            class_index
        ] = weight

    class_weights = torch.tensor(
        class_weights,
        dtype=torch.float32
    ).to(DEVICE)

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    model = LSTMClassifier(
        vocab_size=len(vocabulary),
        num_classes=num_classes
    ).to(DEVICE)

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"Trainable parameters: {total_parameters:,}"
    )

    # -----------------------------------------------------
    # Loss
    # -----------------------------------------------------

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=0.05
    )

    # -----------------------------------------------------
    # Optimizer
    # -----------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # -----------------------------------------------------
    # Scheduler
    # -----------------------------------------------------

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2
    )

    # -----------------------------------------------------
    # Model directory
    # -----------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # -----------------------------------------------------
    # Training state
    # -----------------------------------------------------

    best_validation_f1 = -1.0

    best_validation_accuracy = 0.0

    epochs_without_improvement = 0

    history = []

    # -----------------------------------------------------
    # Training loop
    # -----------------------------------------------------

    for epoch in range(
        EPOCHS
    ):

        model.train()

        total_loss = 0.0

        for texts, lengths, labels in train_loader:

            texts = texts.to(
                DEVICE
            )

            lengths = lengths.to(
                DEVICE
            )

            labels = labels.to(
                DEVICE
            )

            optimizer.zero_grad()

            outputs = model(
                texts,
                lengths
            )

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRADIENT_CLIP
            )

            optimizer.step()

            total_loss += loss.item()

        train_loss = (
            total_loss /
            len(train_loader)
        )

        (
            validation_accuracy,
            validation_macro_f1,
            validation_weighted_f1,
            validation_loss
        ) = evaluate(
            model,
            validation_loader,
            criterion
        )

        scheduler.step(
            validation_macro_f1
        )

        current_lr = optimizer.param_groups[0]["lr"]

        print()
        print(
            f"Epoch {epoch + 1}/{EPOCHS}"
        )

        print(
            f"Train Loss: {train_loss:.4f}"
        )

        print(
            f"Validation Loss: {validation_loss:.4f}"
        )

        print(
            f"Validation Accuracy: "
            f"{validation_accuracy:.4f}"
        )

        print(
            f"Validation Macro F1: "
            f"{validation_macro_f1:.4f}"
        )

        print(
            f"Validation Weighted F1: "
            f"{validation_weighted_f1:.4f}"
        )

        print(
            f"Learning Rate: {current_lr:.7f}"
        )

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(train_loss),
                "validation_loss": float(validation_loss),
                "validation_accuracy": float(
                    validation_accuracy
                ),
                "validation_macro_f1": float(
                    validation_macro_f1
                ),
                "validation_weighted_f1": float(
                    validation_weighted_f1
                ),
                "learning_rate": float(
                    current_lr
                )
            }
        )

        # -------------------------------------------------
        # Save best model
        # -------------------------------------------------

        if validation_macro_f1 > best_validation_f1:

            best_validation_f1 = (
                validation_macro_f1
            )

            best_validation_accuracy = (
                validation_accuracy
            )

            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "vocab_size":
                        len(vocabulary),

                    "num_classes":
                        num_classes,

                    "embedding_dim":
                        EMBEDDING_DIM,

                    "hidden_dim":
                        HIDDEN_DIM,

                    "num_layers":
                        NUM_LAYERS,

                    "max_length":
                        MAX_LENGTH
                },
                MODEL_FILE
            )

            print(
                "✓ Best LSTM model saved."
            )

        else:

            epochs_without_improvement += 1

            print(
                f"No improvement: "
                f"{epochs_without_improvement}/"
                f"{PATIENCE}"
            )

        # -------------------------------------------------
        # Early stopping
        # -------------------------------------------------

        if (
            epochs_without_improvement
            >= PATIENCE
        ):

            print()
            print(
                "Early stopping triggered."
            )

            break

    # =====================================================
    # LOAD BEST MODEL
    # =====================================================

    print()
    print("=" * 70)
    print("LOADING BEST LSTM MODEL")
    print("=" * 70)

    checkpoint = torch.load(
        MODEL_FILE,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    # =====================================================
    # FINAL TEST
    # =====================================================

    (
        test_accuracy,
        test_macro_f1,
        test_weighted_f1,
        test_loss
    ) = evaluate(
        model,
        test_loader,
        criterion
    )

    # =====================================================
    # SAVE LABELS
    # =====================================================

    labels = {
        str(index): label
        for index, label
        in enumerate(
            label_encoder.classes_
        )
    }

    with open(
        LABEL_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            {
                "vocabulary": vocabulary,
                "labels": labels,
                "max_length": MAX_LENGTH
            },
            file,
            ensure_ascii=False,
            indent=4
        )

    # =====================================================
    # SAVE METRICS
    # =====================================================

    metrics = {

        "model":
            "Advanced Bidirectional LSTM with Attention",

        "test_accuracy":
            float(test_accuracy),

        "test_macro_f1":
            float(test_macro_f1),

        "test_weighted_f1":
            float(test_weighted_f1),

        "test_loss":
            float(test_loss),

        "best_validation_accuracy":
            float(best_validation_accuracy),

        "best_validation_macro_f1":
            float(best_validation_f1),

        "num_classes":
            int(num_classes),

        "vocabulary_size":
            int(len(vocabulary)),

        "embedding_dim":
            int(EMBEDDING_DIM),

        "hidden_dim":
            int(HIDDEN_DIM),

        "epochs":
            int(len(history)),

        "history":
            history
    }

    with open(
        METRICS_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

    # =====================================================
    # FINAL OUTPUT
    # =====================================================

    print()
    print("=" * 70)
    print("LSTM TRAINING COMPLETED")
    print("=" * 70)

    print(
        f"Best Validation Accuracy: "
        f"{best_validation_accuracy:.4f}"
    )

    print(
        f"Best Validation Macro F1: "
        f"{best_validation_f1:.4f}"
    )

    print(
        f"Test Accuracy: "
        f"{test_accuracy:.4f}"
    )

    print(
        f"Test Macro F1: "
        f"{test_macro_f1:.4f}"
    )

    print(
        f"Test Weighted F1: "
        f"{test_weighted_f1:.4f}"
    )

    print()
    print(
        f"Model saved to:"
    )

    print(
        MODEL_FILE
    )

    print(
        "Labels saved to:"
    )

    print(
        LABEL_FILE
    )

    print(
        "Metrics saved to:"
    )

    print(
        METRICS_FILE
    )


if __name__ == "__main__":
    main()