import pandas as pd


def load_and_preprocess_data(file_path: str) -> pd.DataFrame:
    """
    Load energy dataset and create time-based features.
    """
    data = pd.read_csv(file_path)

    # Convert Datetime column to datetime format
    data["Datetime"] = pd.to_datetime(data["Datetime"])

    # Create time-based features
    data["hour"] = data["Datetime"].dt.hour
    data["day_of_week"] = data["Datetime"].dt.dayofweek
    data["month"] = data["Datetime"].dt.month
    data["is_weekend"] = data["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)

    return data