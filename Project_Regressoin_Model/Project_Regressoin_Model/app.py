import joblib
from pathlib import Path

import pandas as pd
import streamlit as st

# ─────── PAGE CONFIGURATION ───────
st.set_page_config(
    page_title="Insurance Cost Predictor",
    page_icon="🏥",
    layout="wide",
)


# ─────── MODEL LOADING ───────
BASE_DIR = Path(__file__).resolve().parent


@st.cache_resource
def load_artifacts(path: str = str(BASE_DIR / "all_models_and_meta.pkl")):
    """Load the saved model file once so the app can use it."""
    try:
        return joblib.load(path)
    except FileNotFoundError:
        st.error("⚠️ Run train_model.py first.")
        st.stop()


artifacts = load_artifacts()
models = artifacts["models"]
metrics = artifacts["metrics"]
feature_columns = artifacts["feature_columns"]
cat_cols = artifacts["cat_cols"]
scaler = artifacts.get("scaler")


# ─────── SIDEBAR ───────
st.sidebar.header("⚙️ Configuration")
model_name = st.sidebar.selectbox("Choose Model", list(models.keys()))
model = models[model_name]

st.sidebar.divider()
current_metrics = metrics.get(model_name, {"r2": 0.0, "mae": 0.0})
st.sidebar.metric("R² Score", f"{current_metrics['r2']:.4f}")
st.sidebar.metric("MAE", f"${current_metrics['mae']:,.2f}")
st.sidebar.caption("Higher R² and lower MAE = better model")


# ─────── MAIN PANEL ───────
st.title("🏥 Medical Insurance Cost Predictor")
st.caption("Adjust the patient details below to get a live cost estimate.")

left_col, right_col = st.columns([0.55, 0.45])

with left_col:
    # These are the numeric values the model uses.
    age = st.slider("Age", 18, 100, value=30)
    bmi = st.slider("BMI", 15.0, 50.0, value=27.5, step=0.1)
    children = st.selectbox("Number of Children", [0, 1, 2, 3, 4, 5])

with right_col:
    # These are the text categories the model was trained on.
    sex = st.radio("Biological Sex", ["male", "female"])
    smoker = st.radio("Smoker?", ["no", "yes"])
    region = st.selectbox(
        "US Region",
        ["northeast", "northwest", "southeast", "southwest"],
    )

# Create one small table from the user inputs.
raw = {
    "age": age,
    "sex": sex,
    "bmi": bmi,
    "children": children,
    "smoker": smoker,
    "region": region,
}

input_df = pd.DataFrame([raw])

# Turn the text categories into the same dummy columns used during training.
input_df = pd.get_dummies(input_df, columns=cat_cols, drop_first=True)

# Make sure the input has exactly the same columns as the trained model expects.
input_df = input_df.reindex(columns=feature_columns, fill_value=0)

# Scale the input values if the model needs them in a standard form.
if scaler is not None:
    input_df = pd.DataFrame(scaler.transform(input_df), columns=input_df.columns, index=input_df.index)

# Use the selected model to predict the insurance cost.
prediction = model.predict(input_df)[0]

# Make sure the prediction is never negative.
prediction = max(0.0, float(prediction))

st.divider()
st.success(f"💰 Predicted Insurance Cost: **${prediction:,.2f}**")

smoker_label = "smoker" if smoker == "yes" else "non-smoker"
st.info(
    "This estimate is for a "
    f"{age}-year-old {sex} ({smoker_label}) with BMI {bmi:.1f} "
    f"and {children} child(ren) in the {region} region, using the {model_name} model."
)
