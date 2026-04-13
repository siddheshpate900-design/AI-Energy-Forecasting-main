import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def train_model(data):
    """
    Train Random Forest model on energy dataset.
    """
    features = ["hour", "day_of_week", "month", "is_weekend"]
    target = "Energy"

    X = data[features]
    y = data[target]

    # Split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)

    print(f"RMSE: {rmse:.2f}")
    print(f"R2 Score: {r2:.2f}")

    # Save model
    joblib.dump(model, "models/energy_model.pkl")
    print("Model saved to models/energy_model.pkl")

    return model, X_test, y_test, y_pred