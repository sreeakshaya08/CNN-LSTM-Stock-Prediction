import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error
import math


from preprocess import X_train, X_test, y_train, y_test

model = Sequential()

model.add(LSTM(50, return_sequences=True,
               input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))

model.add(LSTM(50))
model.add(Dropout(0.2))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

model.summary()


model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test)
)


predictions = model.predict(X_test)


rmse = math.sqrt(mean_squared_error(y_test, predictions))
rmse = round(rmse ,2)
print("LSTM RMSE:", rmse)

correct = 0

for i in range(1, len(y_test)):
    actual_direction = y_test[i] - y_test[i-1]
    predicted_direction = predictions[i] - predictions[i-1]

    if actual_direction * predicted_direction > 0:
        correct += 1

directional_accuracy = (correct / (len(y_test) - 1)) * 100
directional_accuracy = round(directional_accuracy, 2)

print("LSTM Directional Accuracy:", directional_accuracy, "%")
import os
os.makedirs("models", exist_ok=True)
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
models_path = os.path.join(BASE_DIR, "models")
os.makedirs(models_path, exist_ok=True)

model.save(os.path.join(models_path, "lstm_model.h5"))
print("Saved at:", os.path.abspath(os.path.join(models_path, "lstm_model.h5")))
print("LSTM Model Saved Successfully.")
model.save("models/lstm_model.h5")
print("LSTM Model Saved Successfully.")
import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
plt.plot(y_test, label="Actual Price", color='blue')
plt.plot(predictions, label="Predicted Price (LSTM)", color='green')
plt.title("LSTM: Actual vs Predicted Stock Price")
plt.xlabel("Time")
plt.ylabel("Normalized Price")
plt.legend()
plt.tight_layout()
plt.show()
