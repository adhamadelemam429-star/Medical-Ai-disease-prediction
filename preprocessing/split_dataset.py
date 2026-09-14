from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "data" / "training_dataset.csv"

OUTPUT_DIR = PROJECT_ROOT / "data"

TRAIN_FILE = OUTPUT_DIR / "train.csv"
VALIDATION_FILE = OUTPUT_DIR / "validation.csv"
TEST_FILE = OUTPUT_DIR / "test.csv"

RANDOM_STATE = 42


def split_disease_data(
    disease_df
):
    """
    Split samples belonging to one disease.

    Rules:

    1 sample  -> train
    2 samples -> train
    3 samples -> 2 train + 1 test
    4 samples -> 2 train + 1 validation + 1 test
    5 samples -> 3 train + 1 validation + 1 test
    6+ samples -> approximately 80% train,
                  10% validation,
                  10% test
    """

    disease_df = disease_df.sample(
        frac=1,
        random_state=RANDOM_STATE
    ).reset_index(
        drop=True
    )

    count = len(
        disease_df
    )

    if count == 1:

        train = disease_df

        validation = disease_df.iloc[0:0]

        test = disease_df.iloc[0:0]

    elif count == 2:

        train = disease_df

        validation = disease_df.iloc[0:0]

        test = disease_df.iloc[0:0]

    elif count == 3:

        train = disease_df.iloc[:2]

        validation = disease_df.iloc[0:0]

        test = disease_df.iloc[2:]

    elif count == 4:

        train = disease_df.iloc[:2]

        validation = disease_df.iloc[2:3]

        test = disease_df.iloc[3:]

    elif count == 5:

        train = disease_df.iloc[:3]

        validation = disease_df.iloc[3:4]

        test = disease_df.iloc[4:]

    else:

        test_count = max(
            1,
            round(
                count * 0.10
            )
        )

        validation_count = max(
            1,
            round(
                count * 0.10
            )
        )

        train_count = (
            count
            - validation_count
            - test_count
        )

        if train_count < 1:

            train_count = 1

            remaining = count - 1

            validation_count = (
                remaining // 2
            )

            test_count = (
                remaining
                - validation_count
            )

        train = disease_df.iloc[
            :train_count
        ]

        validation = disease_df.iloc[
            train_count:
            train_count + validation_count
        ]

        test = disease_df.iloc[
            train_count + validation_count:
        ]

    return (
        train,
        validation,
        test
    )


def main():

    print("=" * 60)
    print(
        "CREATING TRAIN / VALIDATION / TEST SPLITS"
    )
    print("=" * 60)

    # ---------------------------------------------------------
    # Load dataset
    # ---------------------------------------------------------

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}"
        )

    dataframe = pd.read_csv(
        INPUT_FILE
    )

    # ---------------------------------------------------------
    # Basic cleaning
    # ---------------------------------------------------------

    dataframe = dataframe.dropna(
        subset=[
            "text",
            "disease"
        ]
    )

    required_columns = [
        "text",
        "disease",
        "section"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing required columns: "
            + ", ".join(
                missing_columns
            )
        )

    dataframe = dataframe[
        required_columns
    ]

    dataframe = dataframe.drop_duplicates(
        subset=[
            "text",
            "disease"
        ]
    )

    # ---------------------------------------------------------
    # Dataset information
    # ---------------------------------------------------------

    print(
        f"Total samples: {len(dataframe)}"
    )

    print(
        f"Total diseases: "
        f"{dataframe['disease'].nunique()}"
    )

    # ---------------------------------------------------------
    # Split each disease separately
    # ---------------------------------------------------------

    train_parts = []

    validation_parts = []

    test_parts = []

    disease_counts = (
        dataframe[
            "disease"
        ]
        .value_counts()
        .sort_index()
    )

    print()

    print(
        f"Minimum samples per disease: "
        f"{disease_counts.min()}"
    )

    print(
        f"Maximum samples per disease: "
        f"{disease_counts.max()}"
    )

    print()

    for disease in disease_counts.index:

        disease_df = dataframe[
            dataframe["disease"]
            == disease
        ].copy()

        (
            train,
            validation,
            test
        ) = split_disease_data(
            disease_df
        )

        if len(train) > 0:

            train_parts.append(
                train
            )

        if len(validation) > 0:

            validation_parts.append(
                validation
            )

        if len(test) > 0:

            test_parts.append(
                test
            )

    # ---------------------------------------------------------
    # Combine all diseases
    # ---------------------------------------------------------

    train_df = pd.concat(
        train_parts,
        ignore_index=True
    )

    if validation_parts:

        validation_df = pd.concat(
            validation_parts,
            ignore_index=True
        )

    else:

        validation_df = pd.DataFrame(
            columns=dataframe.columns
        )

    if test_parts:

        test_df = pd.concat(
            test_parts,
            ignore_index=True
        )

    else:

        test_df = pd.DataFrame(
            columns=dataframe.columns
        )

    # ---------------------------------------------------------
    # Shuffle datasets
    # ---------------------------------------------------------

    train_df = train_df.sample(
        frac=1,
        random_state=RANDOM_STATE
    ).reset_index(
        drop=True
    )

    validation_df = validation_df.sample(
        frac=1,
        random_state=RANDOM_STATE
    ).reset_index(
        drop=True
    )

    test_df = test_df.sample(
        frac=1,
        random_state=RANDOM_STATE
    ).reset_index(
        drop=True
    )

    # ---------------------------------------------------------
    # Save files
    # ---------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    train_df.to_csv(
        TRAIN_FILE,
        index=False,
        encoding="utf-8"
    )

    validation_df.to_csv(
        VALIDATION_FILE,
        index=False,
        encoding="utf-8"
    )

    test_df.to_csv(
        TEST_FILE,
        index=False,
        encoding="utf-8"
    )

    # ---------------------------------------------------------
    # Verification
    # ---------------------------------------------------------

    print()
    print("=" * 60)
    print("SPLIT COMPLETED")
    print("=" * 60)

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

    print()

    print(
        f"Training diseases: "
        f"{train_df['disease'].nunique()}"
    )

    print(
        f"Validation diseases: "
        f"{validation_df['disease'].nunique()}"
    )

    print(
        f"Test diseases: "
        f"{test_df['disease'].nunique()}"
    )

    # ---------------------------------------------------------
    # Verify no sample was lost
    # ---------------------------------------------------------

    total_after_split = (
        len(train_df)
        + len(validation_df)
        + len(test_df)
    )

    original_total = len(
        dataframe
    )

    print()

    print(
        f"Total after split: "
        f"{total_after_split}"
    )

    print(
        f"Original total: "
        f"{original_total}"
    )

    if total_after_split == original_total:

        print()
        print(
            "SUCCESS: No samples were lost."
        )

    else:

        print()
        print(
            "ERROR: Sample count mismatch."
        )

        raise RuntimeError(
            "Dataset samples were lost "
            "during splitting."
        )

    # ---------------------------------------------------------
    # Verify every disease exists in training
    # ---------------------------------------------------------

    original_diseases = set(
        dataframe["disease"]
    )

    training_diseases = set(
        train_df["disease"]
    )

    missing_from_training = (
        original_diseases
        - training_diseases
    )

    print()

    if not missing_from_training:

        print(
            "SUCCESS: Every disease "
            "has training samples."
        )

    else:

        print(
            "ERROR: Some diseases are "
            "missing from training."
        )

        print(
            sorted(
                missing_from_training
            )
        )

        raise RuntimeError(
            "Some diseases are missing "
            "from the training dataset."
        )

    # ---------------------------------------------------------
    # Verify files
    # ---------------------------------------------------------

    print()

    print(
        f"Saved: {TRAIN_FILE}"
    )

    print(
        f"Saved: {VALIDATION_FILE}"
    )

    print(
        f"Saved: {TEST_FILE}"
    )

    print()

    print("=" * 60)
    print(
        "DATASET SPLITTING FINISHED SUCCESSFULLY"
    )
    print("=" * 60)


if __name__ == "__main__":

    main()