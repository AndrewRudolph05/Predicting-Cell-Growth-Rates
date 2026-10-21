# =======================================================================
# FUNCTIONS
# Define reusable helper functions in this file.
# =======================================================================
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
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
    # Grid search stores negative RMSE, so abs() converts it to positive RMSE
    # and avoids a "-0.0" result when an error is exactly zero.
    summary = pd.DataFrame({
        "Degree": results["param_polynomialfeatures__degree"],
        "Training RMSE": results["mean_train_score"].abs(),
        "Validation RMSE": results["mean_test_score"].abs(),
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
    # Grid search stores negative RMSE, so abs() converts it to positive RMSE
    # and avoids a "-0.0" result when an error is exactly zero.
    summary = pd.DataFrame({
        "Neighbors": results["param_kneighborsregressor__n_neighbors"],
        "Weights": results["param_kneighborsregressor__weights"],
        "Training RMSE": results["mean_train_score"].abs(),
        "Validation RMSE": results["mean_test_score"].abs(),
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

def evaluate_final_models(models, X_train, y_train, X_test, y_test,
                          project_folder):
    # Score each fitted model once on the training and held-out test samples.
    # Grid search has already refit each best model on all training samples.
    tables_folder = project_folder / "tables"
    tables_folder.mkdir(exist_ok=True)

    rows = []

    for name, model in models.items():
        predictions_train = model.predict(X_train)
        predictions_test = model.predict(X_test)

        rows.append({
            "Model": name,
            "Training RMSE": np.sqrt(
                mean_squared_error(y_train, predictions_train)
            ),
            "Test RMSE": np.sqrt(
                mean_squared_error(y_test, predictions_test)
            ),
            "Training R2": r2_score(y_train, predictions_train),
            "Test R2": r2_score(y_test, predictions_test),
            # Counted over the test samples only.
            "Negative test predictions": int((predictions_test < 0).sum())
        })

    summary = pd.DataFrame(rows)

    print("\nFinal evaluation — RMSE in d^-1:")
    print(
        summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}"
        )
    )

    summary.to_csv(tables_folder / "final_model_evaluation.csv", index=False)
    print("\nFinal evaluation table saved in: tables/")

    return summary

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

    print("\nComparison tables saved in: tables/")
    print("Validation plots saved in: figures/")

def plot_feature_effects(model_poly, model_knn, X_train, project_folder):
        # Use training-set means as the reference culture conditions.
        reference = X_train.mean()

        figures_folder = project_folder / "figures"
        tables_folder = project_folder / "tables"
        figures_folder.mkdir(exist_ok=True)
        tables_folder.mkdir(exist_ok=True)

        # Save the fixed reference conditions so the plots can be explained.
        reference.rename("Reference value").to_csv(
            tables_folder / "reference_conditions.csv",
            index_label="Feature"
        )

        fig, axes = plt.subplots(
            2, 2, figsize=(11, 8), dpi=200, layout="constrained"
        )

        effect_tables = []

        for feature, ax in zip(X_train.columns, axes.flat):
            # Vary one feature across its observed training range.
            values = np.linspace(
                X_train[feature].min(),
                X_train[feature].max(),
                100
            )

            # Build hypothetical samples with the other features held constant.
            X_plot = pd.DataFrame(
                np.tile(reference.to_numpy(), (len(values), 1)),
                columns=X_train.columns
            )
            X_plot[feature] = values

            # Predict using the fitted pipelines returned by grid search.
            predictions_poly = model_poly.predict(X_plot)
            predictions_knn = model_knn.predict(X_plot)

            ax.plot(values, predictions_poly, label="Polynomial", color="tab:blue")
            ax.plot(
                values, predictions_knn,
                label="KNN", color="tab:orange", linestyle="--"
            )

            # Show zero without hiding any negative model predictions.
            ax.axhline(0, color="grey", linewidth=0.8, linestyle=":")
            ax.set_xlabel(feature)
            ax.set_ylabel("Predicted specific growth rate [d⁻¹]")
            ax.legend()

            # Save the conditions and predictions underlying each curve.
            effect_table = X_plot.copy()
            effect_table.insert(0, "Varied feature", feature)
            effect_table["Polynomial prediction [d^-1]"] = predictions_poly
            effect_table["KNN prediction [d^-1]"] = predictions_knn
            effect_tables.append(effect_table)

        fig.suptitle("Input effects with other features held at training means")
        fig.savefig(figures_folder / "feature_effects.png")
        plt.close(fig)

        pd.concat(effect_tables, ignore_index=True).to_csv(
            tables_folder / "feature_effects.csv", index=False
        )

        print("\nReference culture conditions:")
        print(reference.to_string())
        print("\nFeature-effect figure and tables saved.")