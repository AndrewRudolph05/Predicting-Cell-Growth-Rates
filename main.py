# =====================================================================
# MAIN PROGRAM
# Your code starts here. This is the file that will be run.
# =====================================================================


import src.functions as sf  # This allows you to access functions from functions.py using sf.function_name()
from pathlib import Path
from src.constants import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    TEST_SIZE,
    RANDOM_STATE,
    POLYNOMIAL_DEGREES,
    CV_FOLDS,
    KNN_NEIGHBORS,
    KNN_WEIGHTS
)

# Locate the dataset relative to this file so the project can be moved
# without needing to rewrite a computer-specific file path.
project_folder = Path(__file__).resolve().parent
dataset_path = project_folder / "datasets" / "dataset_cell_growth_4factor.csv"

# Load the dataset using the function defined in src/functions.py.
data = sf.load_data(dataset_path)

# Display a preview and check the table size and missing values.
print("First five samples:")
print(data.head())

print("\nDataset dimensions (rows, columns):")
print(data.shape)

print("\nMissing values in each column:")
print(data.isna().sum())

# Separate the culture conditions from the growth rate to be predicted.
X, y = sf.separate_features_target(
    data, FEATURE_COLUMNS, TARGET_COLUMN
)

# Confirm that each sample has four inputs and one target value.
print("\nFeature dimensions (samples, features):")
print(X.shape)

print("\nTarget dimensions:")
print(y.shape)

# Split the data before fitting any preprocessing or model.
# Keep the test samples aside while developing and selecting models.
X_train, X_test, y_train, y_test = sf.split_data(
    X, y, TEST_SIZE, RANDOM_STATE
)

# Check how many samples were assigned to each set.
print("\nTraining features:", X_train.shape)
print("Test features:", X_test.shape)
print("Training targets:", y_train.shape)
print("Test targets:", y_test.shape)

# Create an initial second-degree polynomial model.
# We will compare polynomial degrees using grid search later.
model_poly = sf.create_polynomial_model(degree=2)

# Fit the entire pipeline using only the training samples.
# This learns both the scaling values and the regression coefficients.
model_poly.fit(X_train, y_train)

# Confirm the pipeline structure and successful training.
print("\nPolynomial pipeline:")
print(model_poly)
print("\nModel training completed.")

# Assess the initial polynomial model on unseen validation folds.
# The held-out test set is not used during this comparison.
cv_results_poly = sf.evaluate_cross_validation(
    model_poly, X_train, y_train
)

# Compare polynomial complexity using the training data.
# Keep the held-out test set for the final evaluation.
model_gscv_poly, results_poly = sf.tune_polynomial_model(
    X_train,
    y_train,
    POLYNOMIAL_DEGREES,
    CV_FOLDS
)

# Find the strongest KNN candidate using the same validation approach
# as the polynomial model.
model_gscv_knn, results_knn = sf.tune_knn_model(
    X_train,
    y_train,
    KNN_NEIGHBORS,
    KNN_WEIGHTS,
    CV_FOLDS
)

# Compare the best mean validation RMSE from each model family.
# Negate the scores to report positive RMSE.
print("\nModel comparison — cross-validation RMSE:")
print(f"Polynomial: {-model_gscv_poly.best_score_:.4f} d^-1")
print(f"KNN:        {-model_gscv_knn.best_score_:.4f} d^-1")

# Export tables and figures to support the model architecture comparison.
sf.export_model_comparisons(
    results_poly, results_knn, project_folder
)