import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from reservoirpy.nodes import Reservoir, Ridge, Input, Output
import matplotlib.pyplot as plt


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

# reservoir = ESN(n_reservoir=n_reservoir, spectral_radius=0.9, input_scaling=0.5)

reservoir1 = Reservoir(200, name="res1-1")
reservoir2 = Reservoir(500, name="res2-1")

readout1 = Ridge(ridge=1e-5, name="readout1-1")
readout2 = Ridge(ridge=1e-5, name="readout2-1")

# path1 = Input() >> [reservoir1, reservoir2]
# path2 =  reservoir1 >> readout1 >> reservoir2 >> readout2 >> Output()
model = reservoir1 >> readout1
# model = reservoir1 >> readout1
# Train the ESN on training data
model.fit(X_train, y_train)

# Validate the model
y_pred = model.run(X_val)

# Compute validation error
mse = np.mean((y_pred - y_val) ** 2)
print(f"Validation MSE: {mse:.6f}")

# Save the trained model
import joblib
joblib.dump(model, "TrainedModels/trained_esn_model.pkl")



#######################################################
# graph 5 random curved
#######################################################

# Select 5 random samples from the validation set
NumGraphs = 1
np.random.seed(42)
random_indices = np.random.choice(len(X_val), NumGraphs, replace=False)
X_sample = X_val[random_indices]
y_sample_true = y_val[random_indices]

# Get model predictions
y_sample_pred = model.run(X_sample)

for i in range(NumGraphs):

    poly_func_true = np.poly1d(y_sample_true[i])  # Create a polynomial function
    poly_func_Pred = np.poly1d(y_sample_pred[i])

    # Generate smooth curve points
    x_smooth = np.linspace(100, 775, 100)
    y_smooth_true = poly_func_true(x_smooth)
    y_smooth_Pred = poly_func_Pred(x_smooth)

    plt.plot(x_smooth,y_smooth_true)
    plt.plot(x_smooth, y_smooth_Pred,'--')
plt.show()
