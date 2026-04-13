from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("models/energy_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    features = np.array([[
        data["hour"],
        data["day_of_week"],
        data["month"],
        data["is_weekend"]
    ]])

    prediction = model.predict(features)

    return jsonify({
        "predicted_energy": float(prediction[0])
    })

if __name__ == "__main__":
    app.run(debug=True)