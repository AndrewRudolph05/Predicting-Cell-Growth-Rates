# =======================================================================
# CONSTANTS
# Define constant values that do not change throughout your program.
# =======================================================================

# Column names for the four inputs used to predict cell growth.
FEATURE_COLUMNS = [
    "Temperature [C]",
    "pH",
    "Glucose [mM]",
    "Lactate [mM]"
]

# Column name for the output we want the model to predict.
TARGET_COLUMN = "Specific Growth Rate [d^-1]"

# Reserve 20% of samples for testing.
TEST_SIZE = 0.20

# Keep the random split reproducible across runs.
RANDOM_STATE = 0

# Compare increasingly complex polynomial models.
POLYNOMIAL_DEGREES = [1, 2, 3, 4]

# Use four validation folds, following the lecture grid-search examples.
CV_FOLDS = 4

# Compare neighbor counts and the two weighting methods.
KNN_NEIGHBORS = list(range(1, 81))
KNN_WEIGHTS = ["uniform", "distance"]