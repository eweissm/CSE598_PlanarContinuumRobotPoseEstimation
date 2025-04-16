import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from reservoirpy.nodes import Reservoir, Ridge, Input, Concat
import joblib
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time

# Constants
NUM_CENTERS = 3
NUM_MARKERS = 3
# POLY_DEGREE = 2
# N_RESERVOIR = 2000
# RIDGE_ALPHA = 1e-7
NUM_TRIALS = 5  # Number of times to train and average
MODEL_PATH = "../TrainedModels/trained_Transformer_model_3d.pkl"
CSV_PATH = "../ExtractedData/ManualManipulation_3D_edited.csv"  # Replace with actual file path

# Define marker names and column indices
CENTER_X_COLUMNS = [6, 22, 38, 9, 12, 15, 28, 25, 31, 47, 44, 41]  # X column of each center point
MARKER_NAMES = [f"Center{i}" for i in range(NUM_CENTERS)] + \
               [f"M{i}_{j}" for i in range(NUM_CENTERS) for j in range(NUM_MARKERS)]


class TransformerRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim, d_model=128, nhead=8, num_layers=4, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)

        # Learnable positional embedding (sequence length is 1, but we simulate a sequence)
        self.pos_embedding = nn.Parameter(torch.randn(1, 1, d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Deeper regression head with residual connection
        self.output_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

        # Optional projection if dimensions don't match
        self.residual_proj = nn.Linear(d_model, output_dim) if d_model != output_dim else nn.Identity()

    def forward(self, x):
        # x shape: (batch_size, 1, input_dim)
        x_proj = self.input_projection(x)  # -> (batch_size, 1, d_model)
        x_proj += self.pos_embedding  # Add positional embedding

        encoded = self.transformer_encoder(x_proj)  # -> (batch_size, 1, d_model)
        encoded = encoded[:, 0, :]  # Use the first (and only) token

        # Residual connection: output = head(encoded) + projection(encoded)
        output = self.output_head(encoded) + self.residual_proj(encoded)
        return output

def plot_scatter_fit_vs_prediction(y_val, y_pred, frame_index=None):
    """
    Visualize a 3D point ground truth vs prediction for one frame,
    with projections onto XY, YZ, and XZ planes and origin shifted to the first ground truth point.
    """
    if frame_index is None:
        frame_index = np.random.randint(len(y_val))

    # Extract and reshape points
    center_points = np.array([y_val[frame_index, i*3:i*3+3] for i in range(3)])
    prediction = np.array([y_pred[frame_index, i*3:i*3+3] for i in range(3)])

    # Shift everything so the first ground truth point is at the origin
    origin = center_points[0]
    center_points_shifted = center_points - origin
    prediction_shifted = prediction - origin

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter points
    ax.scatter(center_points_shifted[:, 0], center_points_shifted[:, 1], center_points_shifted[:, 2], color='black', label='Center Points', s=50)
    ax.scatter(prediction_shifted[:, 0], prediction_shifted[:, 1], prediction_shifted[:, 2], color='red', label='Prediction', s=50)

    try:
        # Fit splines in shifted space
        tck_gt, _ = splprep(center_points_shifted.T, s=0, k=2)
        spline_gt = splev(np.linspace(0, 1, 100), tck_gt)

        tck_pred, _ = splprep(prediction_shifted.T, s=0, k=2)
        spline_pred = splev(np.linspace(0, 1, 100), tck_pred)

        # Plot splines
        ax.plot(*spline_gt, color='green', linestyle='--', label='Ground Truth', linewidth=2)
        ax.plot(*spline_pred, color='blue', linestyle='--', label='Prediction Fit', linewidth=2)

        # Projections with slight offset
        offset = 0.01

        # XY projection
        ax.plot(spline_gt[0], spline_gt[1], np.full_like(spline_gt[0], np.min(spline_gt[2]) - offset),
                color='green', alpha=0.3, label='GT XY Projection')
        ax.plot(spline_pred[0], spline_pred[1], np.full_like(spline_pred[0], np.min(spline_pred[2]) - offset),
                color='blue', alpha=0.3, label='Pred XY Projection')

        # YZ projection
        ax.plot(np.full_like(spline_gt[1], np.min(spline_gt[0]) - offset), spline_gt[1], spline_gt[2],
                color='green', alpha=0.3, label='GT YZ Projection')
        ax.plot(np.full_like(spline_pred[1], np.min(spline_pred[0]) - offset), spline_pred[1], spline_pred[2],
                color='blue', alpha=0.3, label='Pred YZ Projection')

        # XZ projection
        ax.plot(spline_gt[0], np.full_like(spline_gt[1], np.min(spline_gt[1]) - offset), spline_gt[2],
                color='green', alpha=0.3, label='GT XZ Projection')
        ax.plot(spline_pred[0], np.full_like(spline_pred[1], np.min(spline_pred[1]) - offset), spline_pred[2],
                color='blue', alpha=0.3, label='Pred XZ Projection')

    except Exception as e:
        print("Spline fitting failed:", e)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.tight_layout()
    ax.set_aspect('equal', adjustable='box')
    plt.show()

def remove_nan_frames_from_features_and_targets(X, y):
    """
    Removes any rows where either X or y has NaNs.
    Returns filtered (X, y).
    """
    valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y).any(axis=1)
    return X[valid_mask], y[valid_mask]

def load_marker_data(file_path, marker_names, column_indices):
    """
    Loads and organizes marker data from a CSV.
    """
    data = pd.read_csv(file_path, header=[1, 2, 3, 4, 5, 6, 7], dtype='float32', low_memory=False)
    marker_dict = {
        name: data.iloc[:, column_indices[i]:column_indices[i] + 3].values
        for i, name in enumerate(marker_names)
    }
    return marker_dict, data

def compute_distances(marker_dict, marker1, marker2):
    """
    Computes Euclidean distances between corresponding points of two markers.
    """
    coords1 = marker_dict[marker1]
    coords2 = marker_dict[marker2]
    return np.linalg.norm(coords1 - coords2, axis=1)

def compute_sensor_values(marker_dict, num_centers, num_markers):
    """
    Computes pairwise distances between markers on adjacent centers.
    """
    sensor_values = np.zeros((len(next(iter(marker_dict.values()))), num_markers * (num_centers - 1)))
    for i in range(num_centers - 1):
        for j in range(num_markers):
            idx = i * num_markers + j
            marker1 = f"M{i}_{j}"
            marker2 = f"M{i+1}_{j}"
            sensor_values[:, idx] = compute_distances(marker_dict, marker1, marker2)

    return sensor_values

def fit_3d_polynomial_centers(marker_dict, degree=2):
    """
    Fits a 3D polynomial to the center markers for each frame.
    Returns coefficients stacked as [x_coeffs, y_coeffs, z_coeffs] per frame.
    """
    num_frames = marker_dict["Center0"].shape[0]
    t = np.arange(3)  # parameter values for the centers (0,1,2)

    coeffs = np.zeros((num_frames, (degree + 1) * 3))

    for frame in range(num_frames):
        points = np.array([marker_dict[f"Center{i}"][frame] for i in range(3)])  # shape: (3, 3)
        coeffs_x = np.polyfit(t, points[:, 0], degree)
        coeffs_y = np.polyfit(t, points[:, 1], degree)
        coeffs_z = np.polyfit(t, points[:, 2], degree)
        coeffs[frame] = np.hstack([coeffs_x, coeffs_y, coeffs_z])

    return coeffs

def train_transformer(X_train, y_train, input_dim, output_dim, seed=42, epochs=100, batch_size=64, lr=1e-4):
    torch.manual_seed(seed)
    model = TransformerRegressionModel(input_dim=input_dim, output_dim=output_dim)
    criterion = nn.MSELoss()
    # criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32).unsqueeze(1),
                            torch.tensor(y_train, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {epoch_loss / len(loader):.6f}")
    return model

def run_trial(seed=42, plot_one=False):
    marker_dict, _ = load_marker_data(CSV_PATH, MARKER_NAMES, CENTER_X_COLUMNS)
    X_raw = compute_sensor_values(marker_dict, NUM_CENTERS, NUM_MARKERS)
    y_raw = np.hstack([marker_dict[f"Center{i}"] for i in range(NUM_CENTERS)])
    X, y = remove_nan_frames_from_features_and_targets(X_raw, y_raw)

    # Normalize data
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_scaled = X_scaler.fit_transform(X)
    y_scaled = y_scaler.fit_transform(y)

    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_scaled, test_size=0.1, random_state=seed)

    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]

    model = train_transformer(X_train, y_train, input_dim=input_dim, output_dim=output_dim, seed=seed)

    model.eval()
    with torch.no_grad():
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)
        y_pred_scaled = model(X_val_tensor).numpy()

    y_pred = y_scaler.inverse_transform(y_pred_scaled)
    y_val_orig = y_scaler.inverse_transform(y_val)

    mse = np.mean((y_pred - y_val_orig) ** 2)

    if plot_one:
        plot_scatter_fit_vs_prediction(y_val_orig, y_pred)

    return mse

def main():
    mse_list = []
    time_list =[]
    for i in range(NUM_TRIALS):
        tic = time.time()
        seed = 42 + i
        mse = run_trial(seed=seed, plot_one=(1))  # Plot only the first run
        toc = time.time()
        print(f"[Trial {i + 1}] MSE: {mse:.6f}, Time: {toc - tic:.6f}")
        mse_list.append(mse)
        time_list.append(toc-tic)

    time_mean = np.mean(time_list)
    time_std = np.std(time_list)

    mse_mean = np.mean(mse_list)
    mse_std = np.std(mse_list)
    print(f"\nAverage MSE over {NUM_TRIALS} trials: {mse_mean:.6f} ± {mse_std:.6f}")
    print(f"\nAverage Time over {NUM_TRIALS} trials: {time_mean:.6f} ± {time_std:.6f}")

if __name__ == "__main__":
    main()
