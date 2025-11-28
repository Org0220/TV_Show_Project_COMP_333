import streamlit as st
import pandas as pd
import numpy as np
import ast
import joblib
from pathlib import Path

from sklearn.preprocessing import MultiLabelBinarizer


# ============================================================
# CONFIG
# ============================================================
DATA_PATH = "data/processed/transformed_tv_shows.csv"
MODEL_PATH = Path("data/processed/models/transformed_dataset__multilabel__gradient_boosting.joblib")


# ============================================================
# HELPERS
# ============================================================
def parse_genres(x):
    try:
        return ast.literal_eval(x)
    except:
        return []


def certainty_bucket(p):
    """Return certainty category based on probability."""
    if p >= 0.90: return 0
    if p >= 0.80: return 1
    if p >= 0.70: return 2
    if p >= 0.60: return 3
    if p >= 0.50: return 4
    return 5  # low confidence


# Colors per certainty category
CERTAINTY_COLORS = {
    0: "#00FF00",  # bright green
    1: "#66FF66",
    2: "#CCFF66",
    3: "#FFCC66",
    4: "#FF9966",
    5: "#FF6666"   # red
}


# ============================================================
# DATA LOADING + PREPROCESSING (MULTILABEL)
# ============================================================
@st.cache_resource
def load_model_and_metadata():
    """Load the pre-trained pipeline plus supporting metadata."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Pre-trained model not found at {MODEL_PATH}")

    df = pd.read_csv(DATA_PATH)

    # Parse genre list to recover label order
    genre_col = "genres_parsed" if "genres_parsed" in df.columns else "genres"
    df["genres_list"] = df[genre_col].apply(parse_genres)
    mlb = MultiLabelBinarizer()
    mlb.fit(df["genres_list"])
    genre_classes = mlb.classes_

    # Build combined text column matching the training pipeline
    title_bits = df["title"].fillna("")
    if "title_normalized" in df.columns:
        title_bits = (title_bits + " " + df["title_normalized"].fillna("")).str.strip()
    df["text_all"] = (title_bits + " " + df["description"].fillna("")).str.strip()

    numeric_candidates = [
        "rating_avg_minmax_01",
        "runtime_minutes_zscore",
        "release_year_decscale",
        "premiere_month",
        "premiere_dayofweek",
        "premiere_year",
        "premiere_decade",
        "rating_avg",
        "runtime_minutes",
        "release_year",
    ]
    categorical_candidates = [
        "language",
        "type",
        "status",
        "on_netflix",
        "on_disney",
        "on_amazon",
        "on_hulu",
    ]

    numeric_features = [c for c in numeric_candidates if c in df.columns]
    categorical_features = [c for c in categorical_candidates if c in df.columns]
    text_feature = "text_all"

    numeric_defaults = df[numeric_features].median(numeric_only=True).to_dict()
    categorical_defaults = {}
    for col in categorical_features:
        mode_series = df[col].mode(dropna=True)
        categorical_defaults[col] = mode_series.iloc[0] if not mode_series.empty else "Unknown"

    model = joblib.load(MODEL_PATH)

    metadata = {
        "genre_classes": genre_classes,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "text_feature": text_feature,
        "numeric_defaults": numeric_defaults,
        "categorical_defaults": categorical_defaults,
    }
    return model, metadata


# ============================================================
# STREAMLIT UI
# ============================================================
st.title("Multi-Genre TV Show Classifier")
st.write("Enter details about a TV show, and the model will predict **all possible genres** with a certainty score.")

with st.spinner("Loading pre-trained multi-label model..."):
    model, METADATA = load_model_and_metadata()
    GENRES = METADATA["genre_classes"]

st.success("Model loaded & ready!")


# ============================================================
# INPUT FIELDS
# ============================================================
st.header("Enter Show Information")

release_year = st.number_input("Release Year", min_value=1900, max_value=2030)
runtime = st.number_input("Runtime (minutes)", min_value=1, max_value=500)
rating_avg = st.number_input("Rating (1-10)", min_value=0.0, max_value=10.0, step=0.1)

language = st.text_input("Language", "English")
type_ = st.text_input("Type", "Scripted")
status = st.text_input("Status", "Ended")
on_netflix = st.checkbox("Available on Netflix", value=False)
on_disney = st.checkbox("Available on Disney+", value=False)
on_amazon = st.checkbox("Available on Amazon Prime", value=False)
on_hulu = st.checkbox("Available on Hulu", value=False)

description = st.text_area("Description")
title = st.text_input("Title")
title_norm = st.text_input("Normalized Title (optional)")


if st.button("Predict Genres"):
    # Build input row
    text_combined = (title + " " + title_norm + " " + description).strip()

    feature_row = {}
    for col in METADATA["numeric_features"]:
        feature_row[col] = METADATA["numeric_defaults"].get(col, np.nan)
    for col in METADATA["categorical_features"]:
        feature_row[col] = METADATA["categorical_defaults"].get(col, "Unknown")

    feature_row.update({
        "release_year": release_year,
        "runtime_minutes": runtime,
        "rating_avg": rating_avg,
        "language": language,
        "type": type_,
        "status": status,
        "on_netflix": int(on_netflix),
        "on_disney": int(on_disney),
        "on_amazon": int(on_amazon),
        "on_hulu": int(on_hulu),
        METADATA["text_feature"]: text_combined or " ",
    })

    df_input = pd.DataFrame([feature_row])

    # Predict probabilities
    Y_proba = model.predict_proba(df_input)

    # Build certainty list
    results = []
    for i, genre in enumerate(GENRES):
        p = float(Y_proba[i][0][1])  # probability genre==1
        bucket = certainty_bucket(p)
        results.append((genre, bucket, p))

    # Sort: score 0 first, higher numbers later
    results.sort(key=lambda x: x[1])

    st.subheader("Predicted Genres (Sorted by Confidence)")

    for genre, bucket, prob in results:
        color = CERTAINTY_COLORS[bucket]
        st.markdown(
            f"""
            <div style='padding:8px; border-radius:8px; background-color:{color}; margin-bottom:6px'>
                <b>{genre}</b>  
                - Certainty Level: <b>{bucket}</b>  
                - Probability: <b>{prob:.2f}</b>
            </div>
            """,
            unsafe_allow_html=True
        )
