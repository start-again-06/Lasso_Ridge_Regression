import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error

# Generate synthetic dataset for droplet sorting
np.random.seed(42)
n_samples = 500

# Features: Fluorescence intensity, velocity, pressure, surface tension
fluorescence_intensity = np.random.uniform(10, 100, n_samples)
velocity = np.random.uniform(0.5, 5, n_samples)
pressure = np.random.uniform(1, 10, n_samples)
surface_tension = np.random.uniform(20, 50, n_samples)

# Droplet size (target) with some noise
droplet_size = 2.5 * fluorescence_intensity + 1.2 * velocity - 0.8 * pressure + 3.1 * surface_tension + np.random.normal(0, 5, n_samples)

# Create DataFrame
df = pd.DataFrame({
    "Fluorescence Intensity": fluorescence_intensity,
    "Velocity": velocity,
    "Pressure": pressure,
    "Surface Tension": surface_tension,
    "Droplet Size": droplet_size
})

# Split dataset into training and testing sets
X = df.drop(columns=["Droplet Size"])
y = df["Droplet Size"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply Lasso Regression
lasso = Lasso(alpha=1.0)  # Adjust alpha for tuning
lasso.fit(X_train_scaled, y_train)
y_pred_lasso = lasso.predict(X_test_scaled)

# Apply Ridge Regression
ridge = Ridge(alpha=1.0)  # Adjust alpha for tuning
ridge.fit(X_train_scaled, y_train)
y_pred_ridge = ridge.predict(X_test_scaled)

# Evaluate models
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)

print(f"Lasso Regression MSE: {mse_lasso:.4f}")
print(f"Ridge Regression MSE: {mse_ridge:.4f}")

# Plot actual vs predicted values
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred_lasso, label="Lasso Predictions", color="blue", alpha=0.5)
plt.scatter(y_test, y_pred_ridge, label="Ridge Predictions", color="red", alpha=0.5)
plt.plot(y_test, y_test, "k--", label="Ideal Fit")
plt.xlabel("Actual Droplet Size")
plt.ylabel("Predicted Droplet Size")
plt.title("Lasso & Ridge Regression: Droplet Size Prediction")
plt.legend()
plt.show()
