from prediction.predict_disease import (
    predict_disease
)


def main():

    print("=" * 70)
    print("MEDICAL AI DISEASE PREDICTION SYSTEM")
    print("=" * 70)

    print(
        "Enter patient symptoms."
    )

    print(
        "Type 'exit' to close."
    )

    while True:

        symptoms = input(
            "\nSymptoms: "
        ).strip()

        if symptoms.lower() == "exit":
            break

        if not symptoms:

            print(
                "Please enter symptoms."
            )

            continue

        try:

            result = predict_disease(
                symptoms,
                top_k=5
            )

            print()
            print(
                "TOP PREDICTIONS"
            )

            print(
                "-" * 70
            )

            for index, item in enumerate(
                result["predictions"],
                start=1
            ):

                print(
                    f"{index}. "
                    f"{item['disease']} "
                    f"-> "
                    f"{item['confidence_percent']}"
                )

            print()
            print(
                "Confidence level:",
                result["confidence_level"]
            )

            print()
            print(
                "WARNING:"
            )

            print(
                result["warning"]
            )

        except Exception as error:

            print(
                f"Prediction error: {error}"
            )


if __name__ == "__main__":

    main()