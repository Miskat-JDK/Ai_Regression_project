import os
import sys
import warnings
from pathlib import Path

import joblib
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_object_dtype, is_string_dtype
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# ─────── CONFIGURATION ───────
BASE_DIR = Path(__file__).resolve().parent
CSV_FILE = BASE_DIR / "insurance.csv"
TEST_SIZE = 0.20
RANDOM_STATE = 42
OUTPUT_PKL = BASE_DIR / "all_models_and_meta.pkl"


# ─────── HELPER FUNCTIONS ───────

def print_health_report(df: pd.DataFrame) -> None:
    """Show the dataset size, column types, and missing values."""
    print("\n📊 Dataset Health Report")
    print("=" * 60)
    print(f"Shape: {df.shape}")
    print("\nData types:")
    print(df.dtypes.to_string())
    print("\nNull counts:")
    print(df.isnull().sum().to_string())
    print("=" * 60)


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Measure how well the model performs on the test data."""
    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    return {"r2": float(r2), "mae": float(mae)}


# ─────── MAIN TRAINING PIPELINE ───────

def main() -> None:
    # Check whether the CSV file exists before doing anything else.
    if not os.path.exists(CSV_FILE):
        print(f"⚠️ File not found: {CSV_FILE}")
        print("Please place the CSV file in this folder and run the script again.")
        sys.exit(1)

    # Load the dataset from the CSV file.
    try:
        df = pd.read_csv(CSV_FILE)
    except Exception as exc:
        print(f"⚠️ Could not read {CSV_FILE}: {exc}")
        sys.exit(1)

    # Print a quick summary of the dataset for beginners to understand.
    print_health_report(df)

    # The last column is treated as the target value to predict.
    target_col = df.columns[-1]

    # Text columns are treated as categorical features.
    cat_cols = [
        col
        for col in df.columns[:-1]
        if is_object_dtype(df[col])
        or is_string_dtype(df[col])
        or isinstance(df[col].dtype, pd.CategoricalDtype)
    ]

    # Number columns are treated as numeric features.
    num_cols = [col for col in df.columns[:-1] if col not in cat_cols and is_numeric_dtype(df[col])]

    print(f"\n🎯 Target column detected: {target_col}")
    print(f"📚 Categorical features: {cat_cols if cat_cols else 'None'}")
    print(f"🔢 Numeric features: {num_cols if num_cols else 'None'}")

    # Remove rows where the target value is missing.
    missing_target_rows = df[target_col].isnull().sum()
    if missing_target_rows > 0:
        warnings.warn(
            f"Dropping {missing_target_rows} row(s) with missing target values in '{target_col}'.",
            UserWarning,
        )
        df = df.dropna(subset=[target_col]).reset_index(drop=True)

    # Convert text columns into numbers using one-hot encoding.
    # drop_first=True helps avoid a common machine learning problem called the dummy variable trap.
    encoded_df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Split the table into input features (X) and the answer (y).
    X = encoded_df.drop(columns=[target_col])
    y = encoded_df[target_col]

    # Save the exact feature names so the app uses the same input format.
    feature_columns = list(X.columns)

    # Split the dataset into training and testing parts.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    # Scale numeric values so distance-based models work better.
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    print(f"\n✅ Data ready — X_train shape: {X_train_scaled.shape}")

    # Train five regression models and compare their results.
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
        "K-Nearest Neighbors": KNeighborsRegressor(n_neighbors=5),
    }

    metrics = {}

    for name, model in models.items():
        # Train each model on the same training data.
        model.fit(X_train_scaled, y_train)

        # Test each model on the same unseen data.
        model_metrics = evaluate_model(model, X_test_scaled, y_test)
        metrics[name] = model_metrics

        print(
            f"✅ {name} — R²: {model_metrics['r2']:.4f} | "
            f"MAE: ${model_metrics['mae']:,.2f}"
        )

    # Save the trained models and their results into one file.
    artifact = {
        "models": models,
        "metrics": metrics,
        "feature_columns": feature_columns,
        "target_column": target_col,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "scaler": scaler,
    }

    try:
        joblib.dump(artifact, OUTPUT_PKL)
        print(f"\n💾 Saved → {OUTPUT_PKL}")
    except Exception as exc:
        print(f"⚠️ Failed to save {OUTPUT_PKL}: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
