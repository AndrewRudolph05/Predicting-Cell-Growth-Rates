# Cell Growth Rate Prediction

This project uses temperature, pH, glucose concentration, and lactate concentration to predict the specific growth rate of a cell culture. It compares polynomial regression and k-nearest neighbors regression using scikit-learn pipelines and grid search cross-validation. The results help explore how culture conditions relate to growth and where predictions become less reliable. The program exports tables and figures to support the model comparison and interpretation of this simulated dataset.

## Highlights

- Built reproducible pipelines for cell growth prediction
- Compared regression models using grid search cross-validation
- Identified underfitting and overfitting through validation curves
- Evaluated selected models on held-out test samples
- Visualized predicted responses to changing culture conditions

## Environment

- Python 3.14.7
- matplotlib 3.11.0
- numpy 2.5.2
- pandas 3.0.5
- scikit-learn 1.9.0

## Code Design

Running `main.py` carries out the full workflow, from loading the dataset to saving the results. Shared settings, such as the random state and hyperparameter ranges, are stored in `src/constants.py`. The reusable steps are defined in `src/functions.py`.

### Loading and checking the dataset

The program loads the CSV file from the `datasets/` folder into a pandas DataFrame. It prints the first five samples, the table dimensions, and the number of missing values in each column. These checks help confirm that the correct file was loaded before model training begins.

The dataset contains 318 samples, four input features, and one target: specific growth rate in d⁻¹. There are no missing values, so no imputation is needed. The file path is built relative to `main.py`, allowing the project folder to be moved without changing a computer-specific path.

### Separating features and splitting the samples

Temperature, pH, glucose, and lactate form the feature matrix, `X`. Specific growth rate forms the target, `y`. Keeping these separate tells the model which values are available as inputs and which value it must learn to predict.

The program reserves 20% of the samples for testing, leaving 254 training samples and 64 test samples. Setting `random_state=0` makes this split reproducible. The test set is kept separate while the models are fitted and their hyperparameters are compared. This gives the final evaluation samples that were not used for training or tuning.

### Building the model pipelines

A pipeline keeps preprocessing and regression together. The polynomial pipeline applies `StandardScaler`, generates polynomial features, and fits `LinearRegression`. The KNN pipeline applies `StandardScaler` followed by `KNeighborsRegressor`.

Standardization gives each original feature a mean of zero and a standard deviation of one on the data used to fit the scaler. This is especially important for KNN: without it, a feature with a larger numerical range could dominate the distance between samples. For polynomial regression, it expresses the original inputs on a common scale before creating squared and interaction terms.

Keeping the scaler inside the pipeline also prevents data leakage. During cross-validation, the scaler learns its mean and standard deviation from the training folds only. It then uses those same values to transform the validation fold.

### Comparing hyperparameters with cross-validation

Both model families use four-fold cross-validation on the training set. For each candidate, three folds are used for fitting and the remaining fold is used for validation. This repeats four times so every fold takes a turn as the validation fold. The program records the mean training RMSE, mean validation RMSE, and variation in validation RMSE across folds. Lower RMSE indicates smaller prediction errors.

Grid search compares polynomial degrees 1 through 4. For KNN, it compares 1 through 80 neighbors with uniform and distance weighting. These are hyperparameters: they define how the model works, rather than being fitted like regression weights. Both searches use the same samples, folds, and scoring method, making their results comparable.

After each search, the best pipeline is automatically refitted on all 254 training samples. The training and validation scores also show how model complexity affects underfitting and overfitting.

### Evaluating the fixed models

The degree-2 polynomial is the selected architecture. The best KNN candidate is retained for comparison. The program evaluates both fitted pipelines on the training and test sets, reporting RMSE and the coefficient of determination, R². It also counts negative predictions among the 64 test samples.

The test scores describe performance after tuning is complete; they are not used to change the hyperparameters or select a different architecture. The results are saved in `tables/final_model_evaluation.csv`. Training scores in this table use the full training set, so they differ slightly from the averages calculated during cross-validation.

### Exporting results and exploring input effects

The program saves the grid-search results as CSV tables and plots training and validation RMSE against polynomial degree or neighbor count. These validation curves help explain the architecture choice.

A separate figure varies each input across its training-data range while holding the other three at their training-set means. Each curve contains predictions for 100 hypothetical samples. The reference conditions and predictions are also exported as tables. These plots show how each fitted model responds to changing inputs, but they do not establish cause and effect or identify a combined optimum. All outputs are saved in `tables/` and `figures/`.

## Analysis

### Choosing the polynomial architecture

The selected architecture is a degree-2 polynomial regression pipeline: standardization, polynomial feature generation, and linear regression. The expansion uses `include_bias=False` because the regression step already fits an intercept. With four inputs, degree 2 produces 14 terms: four linear terms, four squared terms, and six interactions between pairs of inputs.

This architecture can represent curved responses and interactions while keeping a fixed, relatively small number of fitted weights. The choice is supported by the cross-validation results below.

| Degree | Training RMSE [d⁻¹] | Validation RMSE [d⁻¹] | Validation SD [d⁻¹] |
|---|---|---|---|
| 1 | 0.1717 | 0.1739 | 0.0080 |
| 2 | 0.1430 | 0.1542 | 0.0104 |
| 3 | 0.1299 | 0.1645 | 0.0180 |
| 4 | 0.1111 | 0.2289 | 0.0580 |

![Training and validation RMSE for polynomial degrees 1 to 4](figures/polynomial_validation_curve.png)

Degree 1 has relatively high training and validation errors. This suggests underfitting: a model without curvature or interactions misses useful patterns. Moving to degree 2 reduces both errors and gives the lowest mean validation RMSE.

Degrees 3 and 4 reduce training error further, but validation error increases. This is evidence of overfitting: the extra complexity improves the fit to training samples without improving predictions on validation samples. Degree 4 also has much greater variation between folds. Since degree 2 is inside the searched range, the results do not suggest a need to extend the grid toward higher degrees.

The best KNN model has a slightly lower validation RMSE, 0.1515 d⁻¹ compared with 0.1542 d⁻¹. This difference is small relative to the variation across folds. That does not prove a statistical tie, but it supports treating their validation performance as similar. The polynomial was chosen for its explicit equation and ability to describe curvature and interactions. Its interpretability is useful here, although a fitted equation does not establish the underlying biology.

Full results are available in the [polynomial comparison table](tables/polynomial_comparison.csv).

### Comparing with k-nearest neighbors regression

KNN predicts growth from samples with similar standardized culture conditions. With uniform weights, each selected neighbor contributes equally. With distance weights, closer neighbors have more influence. Grid search compared 160 combinations and selected nine neighbors with distance weighting.

![Training and validation RMSE across KNN neighbor counts](figures/knn_validation_curve.png)

With one neighbor, validation RMSE is 0.1854 d⁻¹. Predictions depend heavily on a single training sample, making them sensitive to local variation. Increasing the neighbor count improves validation performance until the best result at nine neighbors. At 80 neighbors, validation RMSE rises to approximately 0.169–0.173 d⁻¹, depending on the weighting method. Averaging over many neighbors can smooth away useful differences between culture conditions.

Distance-weighted KNN has zero training error in this run because each training sample is its own neighbor at zero distance. This is expected behavior and does not, by itself, prove that every distance-weighted candidate overfits. Validation error is the more useful measure for comparing these settings.

Nine neighbors with uniform weights is another reasonable candidate. Its training RMSE is 0.1439 d⁻¹ and validation RMSE is 0.1533 d⁻¹. Its validation performance is close to the best distance-weighted candidate, although grid search ranks the latter first. The complete [KNN comparison table](tables/knn_comparison.csv) records all settings.

### Performance on the test set

After tuning, the two fixed candidates give the following results on the full training set and 64 test samples.

| Model | Training RMSE [d⁻¹] | Test RMSE [d⁻¹] | Training R² | Test R² | Negative test predictions |
|---|---|---|---|---|---|
| Polynomial, degree 2 (selected) | 0.1444 | 0.1570 | 0.381 | 0.384 | 0 |
| KNN, 9 neighbors, distance weights | 0.0000 | 0.1587 | 1.000 | 0.371 | 0 |

The polynomial's training and test RMSE values are reasonably close, suggesting no large loss of performance on this test split. KNN fits its training samples exactly but has a similar test error. The difference between test RMSE values is only about 0.0017 d⁻¹ and does not provide a strong basis for preferring either architecture. The architecture justification rests on validation performance and interpretability, rather than this test comparison.

A polynomial test R² of 0.384 means its squared prediction error is about 38% lower than that of a constant prediction equal to the test-set mean. Considerable variation remains unexplained. These scores come from one split with a small test set, so performance on another set of samples could differ. The exported [final evaluation table](tables/final_model_evaluation.csv) contains the unrounded results.

### How the culture conditions affect predictions

The response plots hold three inputs at their training-set means while varying the fourth. The reference conditions are approximately 37.0 °C, pH 7.0, 15.1 mM glucose, and 15.0 mM lactate.

![Predicted growth as each input changes, with other inputs fixed at their training means](figures/feature_effects.png)

**pH produces the largest change along these curves.** The polynomial peaks near pH 6.96, while KNN peaks near pH 6.8. Both predict lower growth toward the ends of the range. At pH 6.0, however, the polynomial predicts approximately −0.134 d⁻¹ and KNN predicts 0.119 d⁻¹. This disagreement is an important limitation of the polynomial response.

**Higher lactate generally corresponds to lower predicted growth.** The polynomial falls from approximately 0.475 d⁻¹ at zero lactate to 0.278 d⁻¹ near 33.5 mM. KNN also shows an overall decline, with more local fluctuations. The slight recovery at high lactate should be treated cautiously rather than interpreted as evidence that adding more lactate helps growth.

**Temperature has a curved response.** The polynomial peaks near 36 °C at approximately 0.343 d⁻¹ and declines toward both ends of the range. KNN gives a less smooth response, but both models predict lower growth at 42 °C. They disagree more strongly at 30 °C, where the polynomial predicts approximately 0.167 d⁻¹ and KNN predicts 0.322 d⁻¹.

**Glucose has the smallest predicted range of effects in this comparison.** The polynomial rises from approximately 0.304 d⁻¹ at zero glucose to 0.435 d⁻¹ at 39.9 mM. KNN rises unevenly and then declines near the upper end. The models therefore agree less clearly on what happens at high glucose concentrations.

For both models, the prediction ranges rank the inputs as pH, lactate, temperature, then glucose. This ranking applies only to these reference conditions and ranges; it is not a general feature-importance score. Squared and interaction terms also mean that a polynomial coefficient's sign alone cannot explain an input's overall effect. Changing the reference conditions could change the curves.

### Conditions associated with higher and lower growth

The one-input-at-a-time plots suggest higher growth around pH 6.8–7.0, moderate temperatures of roughly 32–38 °C, low lactate, and moderate to high glucose. These are candidate conditions to explore further. The program has not tested whether combining all these settings gives the highest growth, and interactions mean their separate effects cannot simply be added together.

Lower growth is generally predicted at extreme pH, high lactate, and temperatures near 42 °C. The polynomial also predicts lower growth at low glucose. These patterns describe the models' responses within the supplied simulated dataset, rather than established operating rules for a real culture.

The largest predictions along these reference curves are below the dataset's maximum growth rate of 0.75 d⁻¹. Each curve changes only one input while fixing the others, so it does not cover every observed combination or the full range of possible model predictions.

### Where predictions are less reliable

The polynomial makes 11 negative predictions among the 100 hypothetical pH values plotted, but none among the 64 test samples. The two counts describe different sets of conditions. Negative predictions fall outside the nonnegative target range in this dataset and show that the regression has no lower-bound constraint. They should be flagged for review; automatically replacing them with zero would change the prediction rule and is not implemented in this program.

Model disagreement also identifies conditions worth investigating, such as low temperature and extreme pH. Disagreement is a reason for caution, not a measured uncertainty interval. Even when each feature is inside its observed range, a particular combination of all four inputs may have few nearby training samples.

Extrapolation is another limitation. Observed ranges are 30–42 °C, pH 6.0–7.8, glucose 0–39.9 mM, and lactate 0–40 mM. Polynomial predictions can grow without bound outside those ranges. KNN predictions remain within the training target range because they average neighboring targets, but that bound does not guarantee accurate predictions.

The dataset includes zero-growth samples. Distance-weighted KNN can reproduce their targets when predicting those training samples, so it would be incorrect to say that KNN cannot predict zero. The relevant question is how accurately either model predicts growth under new stressful conditions; the current aggregate scores do not answer that separately.

### Advantages, limitations, and practical use

The polynomial pipeline is quick to fit and provides a compact equation with curvature and interactions. Cross-validation gives a clear reason for choosing degree 2 over more complex polynomials. Keeping standardization inside the pipeline makes the preprocessing consistent during fitting, validation, and prediction.

The model still has limited predictive accuracy. A quadratic response may miss sharp changes, and the fitted curves are not guaranteed to follow biological constraints. Evaluation uses one train–test split, only four inputs, and simulated data. Neither model provides a prediction interval in the current program, and the response plots explore only one reference scenario.

The model could help identify culture conditions for follow-up experiments or provide a baseline for comparing other regression approaches. It could also support exploratory process simulations. New samples should be passed through the complete fitted pipeline so they receive the same standardization. Before using predictions to guide a real process, the model would need evaluation on measured culture data, especially near stressful conditions and sparsely sampled combinations.
