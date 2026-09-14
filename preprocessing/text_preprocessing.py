import re
import unicodedata


def normalize_text(text):
    """
    Normalize medical text before training.

    Steps:
    1. Convert to string.
    2. Normalize Unicode.
    3. Convert to lowercase.
    4. Remove section labels.
    5. Remove excessive whitespace.
    """

    if text is None:
        return ""

    text = str(text)

    text = unicodedata.normalize("NFKC", text)

    text = text.lower()

    text = re.sub(
        r"\[[^\]]+\]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def clean_medical_text(text):
    """
    Additional cleaning for medical text.

    Keeps useful medical words and punctuation while
    removing obvious formatting noise.
    """

    text = normalize_text(text)

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    return text.strip()


def combine_fields(disease, section, text):
    """
    Combine metadata and medical content into one training text.
    """

    disease = normalize_text(disease)
    section = normalize_text(section)
    text = clean_medical_text(text)

    parts = []

    if section:
        parts.append(section)

    if text:
        parts.append(text)

    return " ".join(parts).strip()