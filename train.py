from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from data_pipelines import pipeline_demo
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Ensure X and Y are NumPy arrays
X = np.load("X.npy")
Y = np.load("Y.npy")
print(X.shape, Y.shape)
# Split into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Reshape X for LSTM (samples, time steps, features)
n_features = X_train.shape[1] // 5  # Assuming 5 stations, reshape accordingly
X_train = X_train.reshape((X_train.shape[0], 5, n_features))
X_test = X_test.reshape((X_test.shape[0], 5, n_features))

# Define LSTM model
model = Sequential([
    LSTM(128, activation='relu', input_shape=(5, n_features
                                              )),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(Y_train.shape[1])  # Output layer matches the target size
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Early Stopping Callback
early_stopping = EarlyStopping(
    monitor='val_loss',  # Metric to monitor
    patience=25,         # Number of epochs with no improvement before stopping
    restore_best_weights=True,  # Restore model weights from the epoch with the best value
    verbose=1            # Verbosity mode
)

# Model Checkpoint Callback
model_checkpoint = ModelCheckpoint(
    filepath='best_model.keras',  # Path to save the best model
    monitor='val_loss',        # Metric to monitor
    save_best_only=True,       # Save only the best model
    mode='min',                # Save the model with minimum validation loss
    verbose=1                  # Verbosity mode
)

history = model.fit(
    X_train, Y_train,
    validation_data=(X_test, Y_test),
    epochs=500,
    batch_size=32,
    callbacks=[early_stopping, model_checkpoint]
)

# Evaluate on the test set
test_loss, test_mae = model.evaluate(X_test, Y_test, verbose=0)
print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")

Y_pred = model.predict(X_test)

# Metrics
rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
mae = mean_absolute_error(Y_test, Y_pred)
print(f"RMSE: {rmse}, MAE: {mae}")
import matplotlib.pyplot as plt

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
# Plot actual vs. predicted
plt.scatter(Y_test.flatten(), Y_pred.flatten(), color='orange', alpha=0.6)
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], color='blue', linestyle='--', label='Ideal Fit')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs. Predicted')
plt.legend(['Ideal Fit', 'Predicted'])
plt.show()