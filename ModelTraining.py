import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from reservoirpy.nodes import ESN

# Load dataset
file_path = "ExtractedData/ExtractedData_SmoothSine.csv"  # Replace with actual file path
data = pd.read_csv(file_path)

# Extract features and labels
sensor_data = data.iloc[:, 1:7].values  # Columns 2-7 (zero-indexed as 1:7)
coefficients = data.iloc[:, 7:13].values  # Columns 8-13
actuator_inputs = data.iloc[:, 13:15].values  # Columns 14-15

# Concatenate sensor data and actuator inputs as input features
X = np.hstack((sensor_data, actuator_inputs))
y = coefficients  # Ground truth coefficients

# Split dataset into training (90%) and validation (10%)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

# Define Echo State Network
input_dim = X_train.shape[1]
output_dim = y_train.shape[1]
n_reservoir = 500  # Number of reservoir neurons

reservoir = ESN(n_reservoir=n_reservoir, spectral_radius=0.9, input_scaling=0.5)

# Train the ESN on training data
reservoir.fit(X_train, y_train)

# Validate the model
y_pred = reservoir.run(X_val)

# Compute validation error
mse = np.mean((y_pred - y_val) ** 2)
print(f"Validation MSE: {mse:.6f}")

# Save the trained model
import joblib
joblib.dump(reservoir, "trained_esn_model.pkl")
