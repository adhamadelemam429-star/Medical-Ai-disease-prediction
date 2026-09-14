from prediction.predict_disease import predict


def main():

    print("=" * 60)
    print("MEDICAL DIAGNOSIS AI")
    print("=" * 60)

    print(
        "Enter symptoms."
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

            result = predict(
                symptoms,
                top_k=5
            )

            print(
                "\nTop predictions:"
            )

            for index, item in enumerate(
                result["predictions"],
                start=1
            ):

                print(
                    f"{index}. "
                    f"{item['condition']} "
                    f"({item['probability']:.2%})"
                )

        except Exception as error:

            print(
                f"\nPrediction error: {error}"
            )


if __name__ == "__main__":
    main()