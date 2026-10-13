from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error

app = Flask(__name__)
CORS(app)

SEQUENCE_LENGTH = 60

# ===============================
# Load Models
# ===============================
cnn_model = tf.keras.models.load_model("models/cnn_model.h5", compile=False)
lstm_model = tf.keras.models.load_model("models/lstm_model.h5", compile=False)

scaler = joblib.load("models/scaler.pkl")


# ===============================
# Utility: Inverse Transform Close Column
# ===============================
def inverse_close(pred_scaled):
    temp = np.zeros((len(pred_scaled), 5))
    temp[:, 3] = pred_scaled.flatten()
    return scaler.inverse_transform(temp)[:, 3]


# ===============================
# Evaluation Function
# ===============================
def evaluate_predictions(preds, actual):
    rmse = np.sqrt(mean_squared_error(actual, preds))
    mae = mean_absolute_error(actual, preds)
    mape = np.mean(np.abs((actual - preds) / actual)) * 100
    accuracy = 100 - mape
    return round(rmse, 2), round(mae, 2), round(accuracy, 2)


# ===============================
# Prediction Route
# ===============================
@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    df = pd.read_csv(file)

    df.columns = df.columns.str.strip()

    required_cols = ['Open','High','Low','Close','Volume']
    for col in required_cols:
        if col not in df.columns:
            return jsonify({"error": f"Missing column: {col}"}), 400

    if len(df) < SEQUENCE_LENGTH:
        return jsonify({"error": f"Need at least {SEQUENCE_LENGTH} rows"}), 400

    # ===============================
    # Prepare Data
    # ===============================
    features = df[['Open','High','Low','Close','Volume']].values
    scaled_data = scaler.transform(features)

    close_prices = df["Close"].values

    X = []
    for i in range(SEQUENCE_LENGTH, len(scaled_data)):
        X.append(scaled_data[i-SEQUENCE_LENGTH:i])

    X = np.array(X)

    actual = close_prices[SEQUENCE_LENGTH:]

    # ===============================
    # Individual Predictions
    # ===============================
    cnn_scaled = cnn_model.predict(X, verbose=0)
    lstm_scaled = lstm_model.predict(X, verbose=0)

    cnn_preds = inverse_close(cnn_scaled)
    lstm_preds = inverse_close(lstm_scaled)

    # ===============================
    # RESIDUAL CORRECTION HYBRID
    # ===============================
    residual = lstm_scaled - cnn_scaled
    hybrid_scaled = lstm_scaled + 0.05 * residual
    hybrid_preds = inverse_close(hybrid_scaled)

    # ===============================
    # Evaluate Models
    # ===============================
    cnn_rmse, cnn_mae, cnn_acc = evaluate_predictions(cnn_preds, actual)
    lstm_rmse, lstm_mae, lstm_acc = evaluate_predictions(lstm_preds, actual)
    hybrid_rmse, hybrid_mae, hybrid_acc = evaluate_predictions(hybrid_preds, actual)

    # ===============================
    # Next Day Prediction
    # ===============================
    last_sequence = scaled_data[-SEQUENCE_LENGTH:]
    last_sequence = np.reshape(last_sequence, (1, SEQUENCE_LENGTH, 5))

    cnn_next = cnn_model.predict(last_sequence, verbose=0)
    lstm_next = lstm_model.predict(last_sequence, verbose=0)

    residual_next = lstm_next - cnn_next
    hybrid_next_scaled = lstm_next + 0.05 * residual_next
    hybrid_next = inverse_close(hybrid_next_scaled)[0]

    # ===============================
    # Return JSON
    # ===============================
    return jsonify({
        "next_prediction": round(float(hybrid_next), 2),
        "actual": actual.tolist(),
        "cnn": cnn_preds.tolist(),
        "lstm": lstm_preds.tolist(),
        "hybrid": hybrid_preds.tolist(),
        "metrics": {
            "cnn": {"rmse": cnn_rmse, "mae": cnn_mae, "accuracy": cnn_acc},
            "lstm": {"rmse": lstm_rmse, "mae": lstm_mae, "accuracy": lstm_acc},
            "hybrid": {"rmse": hybrid_rmse, "mae": hybrid_mae, "accuracy": hybrid_acc}
        }
    })


# ===============================
# Run Server
# ===============================
if __name__ == "__main__":
    print("🚀 Starting Stock Market Prediction Server...")
    app.run(debug=True)