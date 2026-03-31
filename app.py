import streamlit as st
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_NAME = "ML InnovateX Hackathon"
AUTHOR_NAME = "Khushi Donga"
AUTHOR_ID = "23AIML016"

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "fraud_detection_model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"
ANN_MODEL_PATH = BASE_DIR / "fraud_detection_ann.h5"
IMAGE_FILES = {
    "Class Distribution": BASE_DIR / "class_distribution.png",
    "Model Comparison": BASE_DIR / "model_comparison.png",
    "ROC Curves": BASE_DIR / "roc_curves.png",
    "Confusion Matrices": BASE_DIR / "confusion_matrices.png",
    "Training History": BASE_DIR / "training_history.png",
    "Overfitting Analysis": BASE_DIR / "overfitting_analysis.png",
}


def find_dataset():
    paths = [BASE_DIR / "creditcard.csv", BASE_DIR / "creditcard.csv (1)" / "creditcard.csv"]
    for path in paths:
        if path.exists():
            return path
    return None


@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        raise FileNotFoundError(
            "Required model artifacts not found. Place fraud_detection_model.pkl and scaler.pkl in the project folder."
        )

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    ann_model = None
    if ANN_MODEL_PATH.exists():
        try:
            import importlib

            tensorflow = importlib.import_module("tensorflow")
            from tensorflow.keras.models import load_model

            ann_model = load_model(ANN_MODEL_PATH)
        except ModuleNotFoundError:
            ann_model = None
        except Exception:
            ann_model = None

    return model, scaler, ann_model


@st.cache_data
def load_image(image_path):
    if image_path.exists():
        return image_path
    return None


@st.cache_data
def load_dataset():
    csv_path = find_dataset()
    if csv_path is None:
        return None
    return pd.read_csv(csv_path)


def get_model_label(model):
    name = type(model).__name__
    if "XGB" in name:
        return "XGBoost"
    if "RandomForest" in name:
        return "Random Forest"
    if "LogisticRegression" in name:
        return "Logistic Regression"
    if "Sequential" in name or "keras" in name.lower():
        return "ANN"
    return name


def predict_transaction(model, scaler, time_val, amount, v_values):
    scaled_time = scaler.transform(np.array([[time_val]]))[0, 0]
    scaled_amount = scaler.transform(np.array([[amount]]))[0, 0]
    feature_vector = [scaled_time, scaled_amount] + [v_values[f"V{i}"] for i in range(1, 29)]
    features = np.array(feature_vector).reshape(1, -1)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)[0]
        prediction = int(proba[1] > 0.5)
    else:
        raw = model.predict(features)
        prediction = int(raw[0] > 0.5) if hasattr(raw[0], "__float__") else int(raw[0])
        proba = None

    return prediction, proba, features


def format_probability(prediction, proba):
    if proba is None:
        return "Probability is not available for this model."
    fraud_pct = proba[1] * 100
    legit_pct = proba[0] * 100
    if prediction == 1:
        return f"Fraud risk: {fraud_pct:.2f}% | Legitimate: {legit_pct:.2f}%"
    return f"Legitimate risk: {legit_pct:.2f}% | Fraud: {fraud_pct:.2f}%"


def build_feature_frame(values):
    columns = ["scaled_Time", "scaled_Amount"] + [f"V{i}" for i in range(1, 29)]
    return pd.DataFrame([values], columns=columns)


def display_header(title: str, subtitle: str):
    st.title(title)
    st.markdown(subtitle)
    st.divider()


model, scaler, ann_model = load_artifacts()
available_models = {get_model_label(model): model}
if ann_model is not None:
    available_models["ANN"] = ann_model

selected_page = st.sidebar.selectbox("Navigation", ["Home", "Predict", "Dashboard", "About"])
selected_model_name = st.sidebar.selectbox("Choose model", list(available_models.keys()))
selected_model = available_models[selected_model_name]

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Project:** {PROJECT_NAME}")
st.sidebar.markdown(f"**Author:** {AUTHOR_NAME}")
st.sidebar.markdown(f"**ID:** {AUTHOR_ID}")
st.sidebar.markdown("**Model loaded:** " + selected_model_name)

if selected_page == "Home":
    display_header(
        "Credit Card Fraud Detection",
        "A production-ready dashboard that loads your saved classifier, provides transaction scoring, and displays training metrics.",
    )

    st.markdown(
        f"**Project:** {PROJECT_NAME}  \\n**Author:** {AUTHOR_NAME}  \\n**ID:** {AUTHOR_ID}"
    )

    st.subheader("Model status")
    st.success(f"Loaded model: {selected_model_name}")

    dataset = load_dataset()
    if dataset is not None:
        total = len(dataset)
        fraud = int(dataset[dataset["Class"] == 1].shape[0])
        legit = total - fraud
        cols = st.columns(3)
        cols[0].metric("Total Transactions", f"{total:,}")
        cols[1].metric("Legitimate", f"{legit:,}")
        cols[2].metric("Fraud", f"{fraud:,}")

        st.markdown("### Dataset snapshot")
        st.dataframe(dataset.head(5), use_container_width=True)
    else:
        st.warning("Transaction dataset not found in the project folder.")

    st.markdown("---")
    st.subheader("Project highlights")
    st.info(
        "**ML InnovateX Hackathon entry** built by Khushi Donga (ID: 23AIML016).\n"
        "This dashboard provides an intuitive experience for fraud scoring, model comparison, and training insights."
    )
    st.write(
        "- Fully local inference with saved model artifacts.\n"
        "- Real-time scoring UI for transaction risk assessment.\n"
        "- Visual dashboards for model performance and confusion analysis."
    )

elif selected_page == "Predict":
    display_header(
        "Predict Transaction Fraud",
        "Enter transaction values and get an immediate fraud risk prediction using the selected model.",
    )

    with st.expander("Transaction inputs", expanded=True):
        time_val = st.number_input("Time since first transaction (seconds)", min_value=0.0, value=50000.0, step=1.0)
        amount = st.number_input("Transaction amount ($)", min_value=0.0, value=50.0, step=1.0)

        st.markdown("### PCA-derived features (V1 - V28)")
        v_inputs = {}
        cols = st.columns(4)
        for i in range(1, 29):
            column = cols[(i - 1) % 4]
            v_inputs[f"V{i}"] = column.number_input(f"V{i}", value=0.0, format="%.4f", key=f"V{i}")

    if st.button("Run prediction"):
        prediction, proba, values = predict_transaction(selected_model, scaler, time_val, amount, v_inputs)

        st.markdown("### Prediction result")
        if prediction == 1:
            st.error("🚨 Transaction predicted as FRAUDULENT")
        else:
            st.success("✅ Transaction predicted as LEGITIMATE")

        st.write(format_probability(prediction, proba))
        if proba is not None:
            st.progress(int(proba[1] * 100))

        st.markdown("### Feature input preview")
        st.dataframe(build_feature_frame(values[0]).T, use_container_width=True)

elif selected_page == "Dashboard":
    display_header(
        "Training performance dashboard",
        "Review saved evaluation charts and compare classifier performance from the training run.",
    )

    if load_image(IMAGE_FILES["Class Distribution"]):
        st.image(str(IMAGE_FILES["Class Distribution"]), caption="Class Distribution", use_column_width=True)

    chart_cols = st.columns(2)
    with chart_cols[0]:
        image = load_image(IMAGE_FILES["Model Comparison"])
        if image is not None:
            st.image(str(image), caption="Model Comparison", use_column_width=True)
    with chart_cols[1]:
        image = load_image(IMAGE_FILES["ROC Curves"])
        if image is not None:
            st.image(str(image), caption="ROC Curves", use_column_width=True)

    st.markdown("---")
    st.subheader("Confusion matrix")
    image = load_image(IMAGE_FILES["Confusion Matrices"])
    if image is not None:
        st.image(str(image), caption="Confusion Matrices", use_column_width=True)

    st.markdown("---")
    overfit_cols = st.columns(2)
    with overfit_cols[0]:
        image = load_image(IMAGE_FILES["Training History"])
        if image is not None:
            st.image(str(image), caption="Training History", use_column_width=True)
    with overfit_cols[1]:
        image = load_image(IMAGE_FILES["Overfitting Analysis"])
        if image is not None:
            st.image(str(image), caption="Overfitting Analysis", use_column_width=True)

else:
    display_header(
        "About this App",
        "A Streamlit dashboard built for credit card fraud detection using your saved model artifacts.",
    )
    st.markdown("### How to run")
    st.code("python -m streamlit run app.py")
    st.markdown("### Notes")
    st.write(
        "- The app loads the saved model from fraud_detection_model.pkl and scaler.pkl.\n"
        "- If fraud_detection_ann.h5 is available, the app will also display it as an optional model choice.\n"
        "- Use the Predict page to score individual transactions."
    )
    st.markdown("### Project details")
    st.write(
        f"**Project:** {PROJECT_NAME}\n"
        f"**Author:** {AUTHOR_NAME}\n"
        f"**ID:** {AUTHOR_ID}\n"
    )
    st.markdown("### Model files found")
    st.write(
        f"- Classifier: {'found' if MODEL_PATH.exists() else 'missing'}\n"
        f"- Scaler: {'found' if SCALER_PATH.exists() else 'missing'}\n"
        f"- ANN: {'found' if ANN_MODEL_PATH.exists() else 'not found'}"
    )
