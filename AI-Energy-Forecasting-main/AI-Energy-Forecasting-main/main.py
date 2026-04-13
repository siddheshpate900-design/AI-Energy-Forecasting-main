import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no GUI issues
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# ======================
# 0. Create folders
# ======================
os.makedirs("outputs/graphs", exist_ok=True)

# ======================
# 1. Load dataset
# ======================
data = pd.read_csv("data/energy.csv")

print("Columns in dataset:", data.columns)

# ======================
# 2. Handle timestamp
# ======================
if "Datetime" in data.columns:
    data["timestamp"] = pd.to_datetime(data["Datetime"])
elif "timestamp" in data.columns:
    data["timestamp"] = pd.to_datetime(data["timestamp"])
elif "date" in data.columns:
    data["timestamp"] = pd.to_datetime(data["date"])
else:
    print("⚠ No time column → creating dummy")
    data["timestamp"] = pd.date_range(start="1/1/2024", periods=len(data), freq="h")

# ======================
# 3. Feature Engineering
# ======================
data["hour"] = data["timestamp"].dt.hour
data["day"] = data["timestamp"].dt.day
data["month"] = data["timestamp"].dt.month

# ======================
# 4. Target column
# ======================
if "Energy" in data.columns:
    y = data["Energy"]
elif "energy_usage" in data.columns:
    y = data["energy_usage"]
else:
    raise ValueError("❌ No energy column found!")

# Features
X = data[["hour", "day", "month"]]

# ======================
# 5. Train-Test Split
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ======================
# 6. Model Training
# ======================
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

print("✅ Model trained successfully!")

# ======================
# 7. Prediction
# ======================
pred = model.predict(X_test)

# ======================
# 8. Graph (IMPORTANT)
# ======================
plt.figure(figsize=(10, 6))

plt.plot(y_test.values[:100], label="Actual Energy", marker='o')
plt.plot(pred[:100], label="Predicted Energy", linestyle='--')

plt.title("Actual vs Predicted Energy Consumption")
plt.xlabel("Samples")
plt.ylabel("Energy Usage")

plt.legend()
plt.tight_layout()

save_path = "outputs/graphs/prediction.png"
plt.savefig(save_path, dpi=300)
plt.close()

print("📊 Graph saved at:", save_path)