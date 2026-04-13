import pandas as pd


def make_future_predictions(model):
    """
    Make sample future energy predictions.
    """
    future_data = pd.DataFrame(
        {
            "hour": [6, 9, 12, 15, 18, 21],
            "day_of_week": [0, 0, 0, 0, 0, 0],
            "month": [1, 1, 1, 1, 1, 1],
            "is_weekend": [0, 0, 0, 0, 0, 0],
        }
    )

    predictions = model.predict(future_data)
    future_data["Predicted_Energy"] = predictions

    return future_data