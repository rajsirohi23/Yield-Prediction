from flask import Flask, render_template, request
import pickle
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

app = Flask(__name__)

MODEL_PATH = Path(__file__).with_name("yield_prediction_model.pkl")
model = None
model_load_error = None
model_info = {}
try:
    with MODEL_PATH.open("rb") as f:
        model = pickle.load(f)
    st = MODEL_PATH.stat()
    model_info = {
        "path": str(MODEL_PATH),
        "size_bytes": st.st_size,
        "modified": datetime.fromtimestamp(st.st_mtime).isoformat(sep=" ", timespec="seconds"),
        "type": type(model).__name__,
    }
except Exception as e:
    model_load_error = f"{type(e).__name__}: {e}"


def _infer_feature_names(m):
    if m is None:
        return []

    # sklearn estimators often expose training feature names
    names = getattr(m, "feature_names_in_", None)
    if names is not None:
        return [str(x) for x in list(names)]

    # fall back to number of features, if available
    n = getattr(m, "n_features_in_", None)
    if isinstance(n, (int, np.integer)) and int(n) > 0:
        return [f"x{i}" for i in range(1, int(n) + 1)]

    return ["x1"]


FEATURES = _infer_feature_names(model)

# Adjust these based on how your model was trained
CATEGORICAL_FEATURES = {
    "State_Name",
    "District_Name",
    "Season",
    "Crop",
}


def _extract_category_options(m, feature_names):
    """
    If the pickled model is a sklearn Pipeline that contains a OneHotEncoder,
    extract the known categories so the UI can offer valid choices.
    """
    try:
        preprocess = getattr(m, "named_steps", {}).get("preprocess")
        if preprocess is None:
            return {}

        cat = None
        try:
            cat = preprocess.named_transformers_.get("cat")
        except Exception:
            cat = None

        categories = getattr(cat, "categories_", None)
        if categories is None:
            return {}

        # Try to determine which input columns the encoder corresponds to
        cat_features = None
        for name, transformer, cols in getattr(preprocess, "transformers_", []):
            if name == "cat":
                cat_features = list(cols) if cols is not None else None
                break
        if not cat_features:
            # Fall back to intersection with our known categorical set
            cat_features = [f for f in feature_names if f in CATEGORICAL_FEATURES]

        options = {}
        for f, cats in zip(cat_features, categories, strict=False):
            if f in feature_names:
                options[f] = [str(x) for x in list(cats)]
        return options
    except Exception:
        return {}


CATEGORY_OPTIONS = _extract_category_options(model, FEATURES)


def _sanity_predictions(m, options):
    """
    Quick check to confirm predictions change with inputs.
    Uses the first known category for each categorical feature and varies Area.
    """
    try:
        if m is None or not options:
            return {}
        required = ["State_Name", "District_Name", "Season", "Crop"]
        if not all(options.get(k) for k in required):
            return {}

        row = {
            "State_Name": options["State_Name"][0],
            "District_Name": options["District_Name"][0],
            "Season": options["Season"][0],
            "Crop": options["Crop"][0],
        }
        X1 = pd.DataFrame([{**row, "Area": 10.0}])
        X2 = pd.DataFrame([{**row, "Area": 200.0}])
        p1 = float(np.asarray(m.predict(X1)).ravel()[0])
        p2 = float(np.asarray(m.predict(X2)).ravel()[0])
        return {"example_row": row, "pred_area_10": p1, "pred_area_200": p2}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


SANITY = _sanity_predictions(model, CATEGORY_OPTIONS)


@app.route("/")
def home():
    if model_load_error:
        return (
            "Flask is working, but the model failed to load.<br><br>"
            f"Expected model at: {MODEL_PATH}<br>"
            f"Error: {model_load_error}"
        ), 500
    return render_template(
        "index.html",
        features=FEATURES,
        prediction=None,
        error=None,
        options=CATEGORY_OPTIONS,
        values={},
        model_info=model_info,
        sanity=SANITY,
    )


@app.route("/predict", methods=["POST"])
def predict():
    if model_load_error:
        return (
            "Model failed to load.<br><br>"
            f"Expected model at: {MODEL_PATH}<br>"
            f"Error: {model_load_error}"
        ), 500

    row = {}
    values = {}
    for name in FEATURES:
        raw = request.form.get(name, "")
        values[name] = raw

        # Basic required check
        if raw == "":
            return render_template(
                "index.html",
                features=FEATURES,
                prediction=None,
                error=f"Missing value for '{name}'",
            ), 400

        if name in CATEGORICAL_FEATURES:
            # Keep categorical features as strings for encoders
            allowed = CATEGORY_OPTIONS.get(name)
            if allowed is not None and raw not in allowed:
                return render_template(
                    "index.html",
                    features=FEATURES,
                    prediction=None,
                    error=(
                        f"'{name}' value {raw!r} is not in the model's known categories. "
                        "Please select a value from the suggestions list."
                    ),
                    options=CATEGORY_OPTIONS,
                    values=values,
                ), 400
            row[name] = raw
        else:
            # Numeric features are parsed as floats
            try:
                row[name] = float(raw)
            except ValueError:
                return render_template(
                    "index.html",
                    features=FEATURES,
                    prediction=None,
                    error=f"Invalid number for '{name}': {raw!r}",
                ), 400

    # Use a DataFrame so sklearn pipelines with column transformers can
    # handle mixed numeric/categorical inputs based on column names.
    X = pd.DataFrame([row])
    try:
        y = model.predict(X)
        pred = float(np.asarray(y).ravel()[0])
    except Exception as e:
        return render_template(
            "index.html",
            features=FEATURES,
            prediction=None,
            error=f"Prediction failed: {type(e).__name__}: {e}",
            options=CATEGORY_OPTIONS,
            values=values,
        ), 500

    return render_template(
        "index.html",
        features=FEATURES,
        prediction=pred,
        error=None,
        options=CATEGORY_OPTIONS,
        values=values,
        model_info=model_info,
        sanity=SANITY,
    )

if __name__ == "__main__":
    app.run(debug=True)