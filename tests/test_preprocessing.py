from preprocessing.text_preprocessing import (
    normalize_text,
    clean_medical_text,
    combine_fields
)


def test_normalization():

    result = normalize_text(
        "  HELLO   WORLD  "
    )

    assert result == "hello world"


def test_section_removal():

    result = normalize_text(
        "[Symptoms] Fever"
    )

    assert "[symptoms]" not in result


def test_combining():

    result = combine_fields(
        "Disease",
        "Symptoms",
        "Fever and cough"
    )

    assert "symptoms" in result

    assert "fever" in result