import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from sklearn.metrics import mean_squared_error
import math


from preprocess import X_train, X_test, y_train, y_test


model = Sequential()

model.add(Conv1D(filters=64, kernel_size=3, activation='relu',
                 input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(50, activation='relu'))
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
print("CNN RMSE:", rmse)


correct = 0

for i in range(1, len(y_test)):
    actual_direction = y_test[i] - y_test[i-1]
    predicted_direction = predictions[i] - predictions[i-1]

    if actual_direction * predicted_direction > 0:
        correct += 1

directional_accuracy = (correct / (len(y_test) - 1)) * 100
directional_accuracy = round(directional_accuracy, 2)

print("CNN Directional Accuracy:", directional_accuracy, "%")
import os
os.makedirs("models", exist_ok=True)

model.save("models/cnn_model.h5")
import os
print("Saved at:", os.path.abspath("models/cnn_model.h5"))
print("LSTM Model Saved Successfully.")
import matplotlib.pyplot as plt

