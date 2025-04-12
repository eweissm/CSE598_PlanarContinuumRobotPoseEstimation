import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from reservoirpy.nodes import Reservoir, Ridge
import joblib

# Constants
NUM_CENTERS = 3
NUM_MARKERS = 3
POLY_DEGREE = 2
N_RESERVOIR = 200
MODEL_PATH = "../TrainedModels/trained_esn_model_3d.pkl"
CSV_PATH = "../ExtractedData/ManualManipulation_3D_edited.csv"  # Replace with actual file path

# Define marker names and column indices
CENTER_X_COLUMNS = [6, 22, 38, 9, 12, 15, 28, 25, 31, 47, 44, 41]  # X column of each center point
MARKER_NAMES = [f"Center{i}" for i in range(NUM_CENTERS)] + \
               [f"M{i}_{j}" for i in range(NUM_CENTERS) for j in range(NUM_MARKERS)]

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting

def plot_polynomial_fit_vs_prediction(y_val, y_pred, marker_dict, X_val, X_raw, frame_index=None):
    """
    Visualize a 3D polynomial ground truth vs prediction for one frame.
    """
    # Pick a frame (random if not provided)
    if frame_index is None:
        frame_index = np.random.randint(len(y_val))

    # Extract the coefficients for the frame
    gt_coeffs = y_val[frame_index]
    pred_coeffs = y_pred[frame_index]

    # Extract original center points
    valid_frame_indices = np.where(np.all(~np.isnan(X_raw), axis=1))[0]
    original_frame_index = valid_frame_indices[frame_index]
    center_points = np.array([marker_dict[f"Center{i}"][original_frame_index] for i in range(3)])

    # Time vector (same used in polyfit)
    t = np.linspace(0, 2, 100)

    # Evaluate the polynomials
    def eval_poly(coeffs, t):
        x = np.polyval(coeffs[0:3], t)
        y = np.polyval(coeffs[3:6], t)
        z = np.polyval(coeffs[6:9], t)
        return x, y, z

    gt_x, gt_y, gt_z = eval_poly(gt_coeffs, t)
    pred_x, pred_y, pred_z = eval_poly(pred_coeffs, t)

    # Plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Center points
    ax.scatter(center_points[:, 0], center_points[:, 1], center_points[:, 2],
               color='black', label='Center Points', s=50)

    # Ground truth polynomial fit
    ax.plot(gt_x, gt_y, gt_z, label='Ground Truth Fit', linewidth=2)

    # Predicted polynomial fit
    ax.plot(pred_x, pred_y, pred_z, '--', label='Predicted Fit', linewidth=2)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"3D Polynomial Fit vs Prediction (Frame {original_frame_index})")
    ax.legend()
    plt.tight_layout()
    plt.show()

def plot_scatter_fit_vs_prediction(y_val, y_pred, frame_index=None):
    """
    Visualize a 3D point ground truth vs prediction for one frame.
    """
    # Pick a frame (random if not provided)
    if frame_index is None:
        frame_index = np.random.randint(len(y_val))

    # Extract original center points
    center_points = np.array(y_val[frame_index,:])

    # Plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Center points
    ax.plot(y_val[frame_index,[0,3,6]],y_val[frame_index,[1,4,7]],y_val[frame_index,[2,5,8]], color='black', label='Center Points')

    ax.plot(y_pred[frame_index,[0,3,6]],y_pred[frame_index,[1,4,7]],y_pred[frame_index,[2,5,8]], color='red', label='Prediction')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.tight_layout()
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


def train_esn(X_train, y_train, n_reservoir=200, ridge_alpha=1e-5):
    """
    Trains an Echo State Network with Ridge readout.
    """
    reservoir = Reservoir(n_reservoir, name="reservoir")
    readout = Ridge(ridge=ridge_alpha, name="readout")
    model = reservoir >> readout
    model.fit(X_train, y_train)
    return model


def main():
    # Load data
    marker_dict, data = load_marker_data(CSV_PATH, MARKER_NAMES, CENTER_X_COLUMNS)

    # Prepare features and targets
    X_raw = compute_sensor_values(marker_dict, NUM_CENTERS, NUM_MARKERS)
    # y_raw = fit_3d_polynomial_centers(marker_dict, degree=POLY_DEGREE)
    y_raw= np.hstack([marker_dict["Center0"], marker_dict["Center1"], marker_dict["Center2"]] )

    # Remove frames where either X or y has NaNs
    X, y = remove_nan_frames_from_features_and_targets(X_raw, y_raw)

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

    # Train model
    model = train_esn(X_train, y_train, n_reservoir=N_RESERVOIR)

    # Validate model
    y_pred = model.run(X_val)
    mse = np.mean((y_pred - y_val) ** 2)
    print(f"Validation MSE: {mse:.6f}")

    # Save model
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    plot_scatter_fit_vs_prediction(y_val, y_pred)

if __name__ == "__main__":
    main()
