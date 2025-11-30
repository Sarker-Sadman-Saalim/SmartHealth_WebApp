# app.py

from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import joblib
import traceback

from sklearn.base import BaseEstimator, TransformerMixin

app = Flask(__name__)
app.secret_key = "change_this_to_something_random"  # any random string


# =========================================================
# 1. Custom transformer used in your saved pipeline
#    (must match the class definition from your notebook)
# =========================================================
class CorrelationDropper(BaseEstimator, TransformerMixin):
    def __init__(self, features_to_drop):
        self.features_to_drop = features_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # feature_names will be defined after we load the pipeline
        X_df = pd.DataFrame(X, columns=feature_names)
        return X_df.drop(columns=self.features_to_drop, errors="ignore").values


# =========================================================
# 2. Load pipeline and label encoder
# =========================================================
print("Loading model pipeline and label encoder...")

pipeline = joblib.load("models/obesity_best_pipeline.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

# names coming out of the ColumnTransformer (preprocessor)
feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()

print("Loaded successfully!")
print("Classes:", list(label_encoder.classes_))

# store last prediction + chat in memory
last_result = {}
chat_history = []


# =========================================================
# 3. Helper: build input DataFrame from HTML form
# =========================================================
def build_input_dataframe(form):
    """Turn form fields into a single-row DataFrame + raw dict for display."""
    try:
        age = float(form.get("Age"))
        height = float(form.get("Height"))
        weight = float(form.get("Weight"))
        fcvc = float(form.get("FCVC"))
        ncp = float(form.get("NCP"))
        ch2o = float(form.get("CH2O"))
        faf = float(form.get("FAF"))
        tue = float(form.get("TUE"))
    except (TypeError, ValueError) as e:
        raise ValueError(f"Numeric field missing or invalid: {e}")

    gender = form.get("Gender")
    fam_hist = form.get("family_history_with_overweight")
    favc = form.get("FAVC")
    caec = form.get("CAEC")
    smoke = form.get("SMOKE")
    scc = form.get("SCC")
    calc = form.get("CALC")
    mtrans = form.get("MTRANS")

    # base raw fields
    row = {
        "Age": age,
        "Height": height,
        "Weight": weight,
        "FCVC": fcvc,
        "NCP": ncp,
        "CH2O": ch2o,
        "FAF": faf,
        "TUE": tue,
        "Gender": gender,
        "family_history_with_overweight": fam_hist,
        "FAVC": favc,
        "CAEC": caec,
        "SMOKE": smoke,
        "SCC": scc,
        "CALC": calc,
        "MTRANS": mtrans,
    }

    df = pd.DataFrame([row])

    # engineered features (same as in your ML notebook)
    df["BMI"] = df["Weight"] / (df["Height"] ** 2)
    df["FCVC_FAF_Interact"] = df["FCVC"] * df["FAF"]
    df["CH2O_per_NCP"] = df["CH2O"] / (df["NCP"] + 1e-6)

    df["Age_Group"] = pd.cut(
        df["Age"],
        bins=[0, 18, 30, 45, 60, 120],
        labels=["Child", "Young Adult", "Adult", "Middle Age", "Senior"],
    )

    return df, row


# =========================================================
# 4. Helper: simple text reasons (can be replaced by SHAP later)
# =========================================================
def generate_simple_reasons(df_row, prediction_label):
    reasons = []

    bmi = float(df_row["BMI"].iloc[0])
    faf = float(df_row["FAF"].iloc[0])
    ch2o = float(df_row["CH2O"].iloc[0])
    ncp = float(df_row["NCP"].iloc[0])
    fam_hist = df_row["family_history_with_overweight"].iloc[0]

    if bmi >= 30:
        reasons.append("Your BMI is in a high range, which strongly contributes to obesity risk.")
    elif bmi >= 25:
        reasons.append("Your BMI is in the overweight range, which increases obesity risk.")

    if faf == 0:
        reasons.append("You reported no physical activity (FAF), which increases obesity risk.")
    elif faf < 2:
        reasons.append("Your physical activity level is relatively low; more movement can help reduce risk.")

    if ch2o < 2:
        reasons.append("Low daily water consumption may be associated with less healthy habits overall.")

    if ncp > 3:
        reasons.append("Having many main meals per day (NCP) may contribute to higher calorie intake.")

    if fam_hist == "yes":
        reasons.append("Family history with overweight increases your predisposition to obesity.")

    if not reasons:
        reasons.append(
            "Your result is influenced by a combination of your weight, activity level, and eating habits."
        )

    reasons.append(f"The final predicted category is {prediction_label} based on these combined factors.")
    return reasons


# =========================================================
# 5. Routes
# =========================================================
@app.route("/", methods=["GET", "POST"])
def index():
    global last_result, chat_history

    if request.method == "GET":
        # show form
        last_result = {}
        chat_history = []
        return render_template("index.html")

    # POST: user submitted form → make prediction
    try:
        df_input, raw_input = build_input_dataframe(request.form)

        # main prediction
        encoded_pred = pipeline.predict(df_input)[0]
        prediction = label_encoder.inverse_transform([encoded_pred])[0]

        # probabilities from underlying model inside pipeline
        preprocessed = pipeline.named_steps["preprocessor"].transform(df_input)
        dropped = pipeline.named_steps["corr_drop"].transform(preprocessed)
        probs = pipeline.named_steps["model"].predict_proba(dropped)[0]

        class_probs = list(zip(label_encoder.classes_, probs))
        reasons = generate_simple_reasons(df_input, prediction)

        last_result = {
            "prediction": prediction,
            "class_probs": class_probs,
            "user_input": raw_input,
            "reasons": reasons,
        }
        chat_history = []

        return render_template(
            "result.html",
            prediction=prediction,
            class_probs=class_probs,
            user_input=raw_input,
            reasons=reasons,
            chat_history=chat_history,
        )

    except Exception:
        # Show error nicely in browser + log to console
        traceback.print_exc()
        return f"<h2>Something went wrong while making the prediction.</h2><pre>{traceback.format_exc()}</pre>"


@app.route("/chat", methods=["POST"])
def chat():
    global last_result, chat_history

    if not last_result:
        # If user tries chat before prediction, send them back to form
        return redirect(url_for("index"))

    user_message = request.form.get("user_message", "").strip()
    prediction = last_result["prediction"]

    if user_message:
        # add user's message
        chat_history.append({"role": "user", "message": user_message})

        # very simple, rule-based reply for now
        reply = (
            f"Thank you for your question. Your current predicted category is {prediction}. "
            "In general, focusing on balanced meals, regular physical activity, better sleep, "
            "and reducing sugary and highly processed foods can help improve your health. "
            "For personalised medical advice, please consult a healthcare professional."
        )
        chat_history.append({"role": "assistant", "message": reply})

    return render_template(
        "result.html",
        prediction=last_result["prediction"],
        class_probs=last_result["class_probs"],
        user_input=last_result["user_input"],
        reasons=last_result["reasons"],
        chat_history=chat_history,
    )


# =========================================================
# 6. Run app
# =========================================================
if __name__ == "__main__":
    print("Starting Flask app on http://127.0.0.1:5000/")
    app.run(debug=True)
