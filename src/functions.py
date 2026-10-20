# =======================================================================
# FUNCTIONS
# Define reusable helper functions in this file.
# =======================================================================
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def load_data(file_path):
    # Read the CSV into a DataFrame: a table with named columns.
    data = pd.read_csv(file_path)

    return data


def separate_features_target(data, feature_columns, target_column):
    # Separate the input features from the growth-rate target.
    X = data[feature_columns]
    y = data[target_column]

    return X, y

def split_data(X, y, test_size, random_state):
    # Randomly split samples while keeping each sample's features
    # paired with its corresponding growth rate.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

    return X_train, X_test, y_train, y_test

def create_polynomial_model(degree):
    # Combine input scaling, polynomial feature generation,
    # and regression into a single model workflow.
    model = make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree=degree, include_bias=False),
        LinearRegression()
    )

    return model

def evaluate_cross_validation(model, X_train, y_train):
    # Evaluate the whole pipeline across five folds.
    # Scaling and regression are fitted using only each fold's training samples.
    cv_results = cross_validate(
        model,
        X_train,
        y_train,
        cv=5,
        scoring=("neg_root_mean_squared_error", "r2"),
        return_train_score=True
    )

    # Convert negative RMSE scores back to positive prediction errors.
    # "test" in these result keys refers to validation folds, not our test set.
    rmse_train = -cv_results["train_neg_root_mean_squared_error"]
    rmse_valid = -cv_results["test_neg_root_mean_squared_error"]
    r2_train = cv_results["train_r2"]
    r2_valid = cv_results["test_r2"]

    # Summarize average performance and variation across the five folds.
    print("\nFive-fold cross-validation results:")
    print(
        f"Training   | RMSE: {np.mean(rmse_train):.4f}"
        f" +/- {np.std(rmse_train):.4f} d^-1"
    )
    print(
        f"Validation | RMSE: {np.mean(rmse_valid):.4f}"
        f" +/- {np.std(rmse_valid):.4f} d^-1"
    )
    print(
        f"Training   | R²:   {np.mean(r2_train):.4f}"
        f" +/- {np.std(r2_train):.4f}"
    )
    print(
        f"Validation | R²:   {np.mean(r2_valid):.4f}"
        f" +/- {np.std(r2_valid):.4f}"
    )

    return cv_results

def tune_polynomial_model(X_train, y_train, degrees, cv_folds):
    # Create the pipeline
    # Grid search will replace the initial degree with each candidate value.
    model = create_polynomial_model(degree=2)

    # The double underscore identifies a setting within a pipeline step.
    param_grid = {
        "polynomialfeatures__degree": degrees
    }

    # Compare degrees using validation RMSE and retain training scores
    # so we can assess underfitting and overfitting.
    search = GridSearchCV(
        model,
        param_grid=param_grid,
        cv=cv_folds,
        scoring="neg_root_mean_squared_error",
        return_train_score=True
    )

    # Use only the training portion for model selection.
    search.fit(X_train, y_train)

    # Summarize the cross-validation results as positive RMSE values.
    results = pd.DataFrame(search.cv_results_)

    summary = pd.DataFrame({
        "Degree": results["param_polynomialfeatures__degree"],
        "Training RMSE": -results["mean_train_score"],
        "Validation RMSE": -results["mean_test_score"],
        "Validation SD": results["std_test_score"],
        "Rank": results["rank_test_score"]
    })

    print("\nPolynomial degree comparison — RMSE in d^-1:")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    print("\nBest degree by mean validation RMSE:")
    print(search.best_params_["polynomialfeatures__degree"])

    return search, summary

def tune_knn_model(X_train, y_train, neighbors, weights, cv_folds):
    # Standardize the inputs before measuring distances between samples.
    model = make_pipeline(
        StandardScaler(),
        KNeighborsRegressor()
    )

    # Compare combinations of neighbor count and weighting method.
    param_grid = {
        "kneighborsregressor__n_neighbors": neighbors,
        "kneighborsregressor__weights": weights
    }

    # Use the same folds and scoring method as the polynomial search.
    search = GridSearchCV(
        model,
        param_grid=param_grid,
        cv=cv_folds,
        scoring="neg_root_mean_squared_error",
        return_train_score=True
    )

    # Select settings using only the training portion of the dataset.
    search.fit(X_train, y_train)

    # Convert the search results into a readable table with positive RMSE.
    results = pd.DataFrame(search.cv_results_)

    summary = pd.DataFrame({
        "Neighbors": results["param_kneighborsregressor__n_neighbors"],
        "Weights": results["param_kneighborsregressor__weights"],
        "Training RMSE": -results["mean_train_score"],
        "Validation RMSE": -results["mean_test_score"],
        "Validation SD": results["std_test_score"],
        "Rank": results["rank_test_score"]
    })

    # Display the ten best combinations while retaining all results.
    summary = summary.sort_values(by="Rank")

    print("\nTop 10 KNN combinations — RMSE in d^-1:")
    print(
        summary.head(10).to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}"
        )
    )

    print("\nBest KNN settings by mean validation RMSE:")
    print("Neighbors:", search.best_params_["kneighborsregressor__n_neighbors"])
    print("Weights:", search.best_params_["kneighborsregressor__weights"])

    return search, summary

def export_model_comparisons(results_poly, results_knn, project_folder):
    # Create output folders if they do not already exist.
    figures_folder = project_folder / "figures"
    tables_folder = project_folder / "tables"
    figures_folder.mkdir(exist_ok=True)
    tables_folder.mkdir(exist_ok=True)

    # Save all candidate results as CSV tables for the README analysis.
    results_poly.to_csv(
        tables_folder / "polynomial_comparison.csv", index=False
    )
    results_knn.to_csv(
        tables_folder / "knn_comparison.csv", index=False
    )

    # Plot polynomial errors in order of increasing model complexity.
    poly = results_poly.sort_values("Degree")

    fig, ax = plt.subplots(figsize=(7, 5), dpi=200, layout="constrained")

    ax.plot(
        poly["Degree"], poly["Training RMSE"],
        marker="o", linestyle="-", color="tab:blue", label="Training"
    )
    ax.plot(
        poly["Degree"], poly["Validation RMSE"],
        marker="o", linestyle="--", color="tab:blue", label="Validation"
    )

    ax.set_xlabel("Polynomial degree")
    ax.set_ylabel("RMSE [d⁻¹]")
    ax.set_title("Polynomial regression: effect of degree")
    ax.set_xticks(poly["Degree"].tolist())
    ax.set_ylim(bottom=0)
    ax.legend()

    fig.savefig(figures_folder / "polynomial_validation_curve.png")
    plt.close(fig)

    # Plot KNN errors separately for each weighting method.
    # Sort by neighbor count because the results table is ranked by score.
    fig, ax = plt.subplots(figsize=(7, 5), dpi=200, layout="constrained")

    for weight, color in [
        ("uniform", "tab:blue"),
        ("distance", "tab:orange")
    ]:
        knn = results_knn[
            results_knn["Weights"] == weight
        ].sort_values("Neighbors")

        ax.plot(
            knn["Neighbors"], knn["Training RMSE"],
            linestyle="-", color=color, label=f"Training ({weight})"
        )
        ax.plot(
            knn["Neighbors"], knn["Validation RMSE"],
            linestyle="--", color=color, label=f"Validation ({weight})"
        )

    ax.set_xlabel("Number of neighbors (k)")
    ax.set_ylabel("RMSE [d⁻¹]")
    ax.set_title("KNN regression: effect of neighbor count")
    ax.set_ylim(bottom=0)
    ax.legend()

    fig.savefig(figures_folder / "knn_validation_curve.png")
    plt.close(fig)

    print("\nComparison tables saved in:", tables_folder)
    print("Validation plots saved in:", figures_folder)