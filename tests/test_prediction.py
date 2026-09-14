from prediction.predict_disease import predict_disease


def test_prediction():

    result = predict_disease(
        "fever cough headache",
        top_k=3
    )

    assert isinstance(
        result,
        dict
    )

    assert "predictions" in result

    assert len(
        result["predictions"]
    ) == 3

    for prediction in result["predictions"]:

        assert "rank" in prediction
        assert "disease" in prediction
        assert "confidence" in prediction
        assert "warnings" in prediction
        assert "recommendations" in prediction
        assert "emergency" in prediction

    assert "disclaimer" in result