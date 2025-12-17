from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import joblib
import traceback
from sklearn.base import BaseEstimator, TransformerMixin
from obesity_chatbot import ObesityChatbot

app = Flask(__name__)
app.secret_key = "change_this_to_something_random_and_secure_in_production"


# =========================================================
# Custom transformer - must exist BEFORE loading pipeline
# =========================================================
class CorrelationDropper(BaseEstimator, TransformerMixin):
    def __init__(self, features_to_drop):
        self.features_to_drop = features_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X, columns=feature_names)
        return X_df.drop(columns=self.features_to_drop, errors="ignore").values


# =========================================================
# Load model pipeline + label encoder
# =========================================================
print("Loading model pipeline and label encoder...")

pipeline = joblib.load("models/obesity_best_pipeline.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()

print("✓ Models loaded successfully!")
print(f"✓ Classes: {list(label_encoder.classes_)}")


# =========================================================
# Initialize Phi-3 chatbot
# =========================================================
try:
    chatbot = ObesityChatbot(model_name="microsoft/Phi-3-mini-4k-instruct")
    print("✓ Chatbot initialized successfully")
except Exception as e:
    print(f"⚠ Chatbot init failed: {e}")
    chatbot = None


# =========================================================
# Helper: build input DataFrame from form
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

    row = {
        "Age": age,
        "Height": height,
        "Weight": weight,
        "FCVC": fcvc,
        "NCP": ncp,
        "CH2O": ch2o,
        "FAF": faf,
        "TUE": tue,
        "Gender": form.get("Gender"),
        "family_history_with_overweight": form.get("family_history_with_overweight"),
        "FAVC": form.get("FAVC"),
        "CAEC": form.get("CAEC"),
        "SMOKE": form.get("SMOKE"),
        "SCC": form.get("SCC"),
        "CALC": form.get("CALC"),
        "MTRANS": form.get("MTRANS"),
    }

    df = pd.DataFrame([row])

    # Engineered features (must match training)
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
# Helper: generate top factors (keep short + real)
# =========================================================
def generate_top_factors(df_row):
    factors = []

    bmi = float(df_row["BMI"].iloc[0])
    faf = float(df_row["FAF"].iloc[0])
    ch2o = float(df_row["CH2O"].iloc[0])
    fam_hist = str(df_row["family_history_with_overweight"].iloc[0]).lower()

    if bmi >= 30:
        factors.append("High BMI (obesity range)")
    elif bmi >= 25:
        factors.append("BMI in overweight range")
    else:
        factors.append("BMI in normal/healthy range")

    if faf == 0:
        factors.append("No physical activity reported")
    elif faf < 2:
        factors.append("Low physical activity level")

    if ch2o < 2:
        factors.append("Low water intake")

    if fam_hist == "yes":
        factors.append("Family history of overweight")

    return factors[:3]


# =========================================================
# Routes
# =========================================================
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        # Start fresh only when user first arrives
        if "prediction" not in session:
            session.clear()
        return render_template("index.html")

    # POST: user submitted form → prediction
    try:
        df_input, raw_input = build_input_dataframe(request.form)

        # Predict encoded then decode
        encoded_pred = pipeline.predict(df_input)[0]
        prediction = label_encoder.inverse_transform([encoded_pred])[0]

        # Probabilities (optional)
        preprocessed = pipeline.named_steps["preprocessor"].transform(df_input)
        dropped = pipeline.named_steps["corr_drop"].transform(preprocessed)
        probs = pipeline.named_steps["model"].predict_proba(dropped)[0]
        class_probs = list(zip(label_encoder.classes_, probs))

        bmi_value = float(df_input["BMI"].iloc[0])
        top_factors = generate_top_factors(df_input)

        # Store in session
        session["prediction"] = str(prediction)
        session["class_probs"] = [(str(c), float(p)) for c, p in class_probs]
        session["user_input"] = raw_input
        session["bmi"] = bmi_value
        session["top_factors"] = top_factors

        session["chat_history"] = []
        session["chat_mode"] = None
        session["chat_expect"] = None  # <-- IMPORTANT for numeric follow-ups

        return render_template(
            "result.html",
            prediction=prediction,
            class_probs=class_probs,
            user_input=raw_input,
            reasons=top_factors,
            chat_history=[],
        )

    except Exception:
        traceback.print_exc()
        return f"<h2>Prediction error.</h2><pre>{traceback.format_exc()}</pre>"


@app.route("/chat", methods=["POST"])
def chat():
    if "prediction" not in session:
        return redirect(url_for("index"))

    user_message = request.form.get("user_message", "").strip()
    chat_history = session.get("chat_history", [])

    if user_message:
        chat_history.append({"role": "user", "message": user_message})

        # Build chatbot context
        ui = session.get("user_input", {})
        ctx = {
            "prediction": session.get("prediction"),
            "bmi": session.get("bmi"),
            "age": ui.get("Age"),
            "height": ui.get("Height"),
            "weight": ui.get("Weight"),
            "faf": ui.get("FAF"),
            "fcvc": ui.get("FCVC"),
            "ch2o": ui.get("CH2O"),
            "top_factors": session.get("top_factors", []),
            "mode": session.get("chat_mode"),
            "expect": session.get("chat_expect"),  # <-- PASS EXPECTATION
        }

        # Generate response
        try:
            if chatbot:
                reply = chatbot.generate_response(user_message, ctx)
            else:
                reply = "Chatbot is not available right now."
        except Exception as e:
            print(f"Chatbot error: {e}")
            traceback.print_exc()
            reply = "Sorry — something went wrong. Try asking for a diet plan or exercise plan."

        # --------- Update chat mode (optional, helps coaching flow) ----------
        lm = user_message.lower()
        if any(w in lm for w in ["diet", "meal", "food", "nutrition"]):
            session["chat_mode"] = "diet_coach"
        elif any(w in lm for w in ["exercise", "workout", "activity", "gym"]):
            session["chat_mode"] = "exercise_coach"

        # --------- Expectation logic (fixes the '2' reply issue) ----------
        # If user just said foods (diet follow-up), next message is expected to be 1/2/3+
        # (You can tune this list as needed)
        food_keywords = ["rice", "chicken", "bread", "pasta", "pizza", "burger", "fries", "soda", "coke"]
        if any(k in lm for k in food_keywords):
            session["chat_expect"] = "EXPECT_MEALS_PER_DAY"

        # If user gives numeric follow-up, clear expectation
        if user_message.strip().lower() in {"1", "2", "3", "3+", "once", "twice", "three"}:
            session["chat_expect"] = None

        chat_history.append({"role": "assistant", "message": reply})
        session["chat_history"] = chat_history
        session.modified = True

    return render_template(
        "result.html",
        prediction=session["prediction"],
        class_probs=session["class_probs"],
        user_input=session["user_input"],
        reasons=session.get("top_factors", []),
        chat_history=chat_history,
    )


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚀 Starting Flask app on http://127.0.0.1:5000/")
    print("=" * 60 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
