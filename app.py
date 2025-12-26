from flask import Flask, request, jsonify, render_template, redirect, url_for, session, flash, send_file
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import joblib
import numpy as np
import pandas as pd
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import traceback
import os
import warnings

warnings.filterwarnings("ignore")

# -----------------------------------------------------------
# FLASK SETUP
# -----------------------------------------------------------
app = Flask(__name__)
CORS(app)
app.secret_key = "your_secret_key"

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///database.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

# -----------------------------------------------------------
# USER MODEL
# -----------------------------------------------------------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

with app.app_context():
    db.create_all()

# -----------------------------------------------------------
# MODEL LOADING
# -----------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATASET_PATH = os.path.join(MODEL_DIR, "agri_dataset_5000.csv")

MODEL_PATHS = {
    "label_encoder": os.path.join(MODEL_DIR, "label_encoder_retrained.pkl"),
    "rf_crop": os.path.join(MODEL_DIR, "rf_crop_retrained_v2.pkl"),
    "rf_yield": os.path.join(MODEL_DIR, "rf_yield_retrained.pkl"),
    "rf_expense": os.path.join(MODEL_DIR, "rf_expense_retrained.pkl"),
}

def safe_load(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model missing: {path}")
    return joblib.load(path)

# Load ML Models
label_encoder = safe_load(MODEL_PATHS["label_encoder"])
rf_crop = safe_load(MODEL_PATHS["rf_crop"])
rf_yield = safe_load(MODEL_PATHS["rf_yield"])
rf_expense = safe_load(MODEL_PATHS["rf_expense"])

print("✅ Models Loaded Successfully")

# Load dataset (for crop list only)
if os.path.exists(DATASET_PATH):
    market_df = pd.read_csv(DATASET_PATH)
    market_df["label"] = market_df["label"].astype(str).str.lower()
    ALL_CROPS = sorted(market_df["label"].unique())
    print("✅ Dataset Loaded")
else:
    market_df = None
    ALL_CROPS = []
    print("⚠ Dataset Missing")

# -----------------------------------------------------------
# HELPERS
# -----------------------------------------------------------
def to_float(x):
    try:
        return float(x)
    except:
        return 0.0

def adjust_to_valid_crop(pred_crop, valid_list):
    pred_crop = pred_crop.lower().strip()
    if pred_crop in valid_list:
        return pred_crop
    for v in valid_list:
        if pred_crop[:3] in v:
            return v
    return valid_list[0]

# -----------------------------------------------------------
# CROP FAILURE CHECK
# -----------------------------------------------------------
def check_crop_failure(data):
    temp = to_float(data["temperature"])
    ph = to_float(data["ph"])
    rain = to_float(data["rainfall"])
    N = to_float(data["N"])
    P = to_float(data["P"])
    K = to_float(data["K"])

    extreme = 0

    # Temperature very high
    if temp > 45:
        extreme += 1

    # Very low rainfall
    if rain < 20:
        extreme += 1

    # Very acidic soil
    if ph < 5:
        extreme += 1

    # Extremely low nutrients
    if N < 20:
        extreme += 1
    if P < 15:
        extreme += 1
    if K < 15:
        extreme += 1

    return extreme >= 2  # lower threshold


# -----------------------------------------------------------
# YIELD CORRECTION
# -----------------------------------------------------------
def apply_yield_correction(raw_yield, data):
    temp = to_float(data["temperature"])
    ph = to_float(data["ph"])
    rain = to_float(data["rainfall"])
    humidity = to_float(data["humidity"])
    N = to_float(data["N"])
    P = to_float(data["P"])
    K = to_float(data["K"])

    corr = 1.0

    if temp > 45: corr *= 0.30
    elif temp > 38: corr *= 0.55
    if rain < 20: corr *= 0.40
    elif rain < 40: corr *= 0.65
    if ph < 5 or ph > 8: corr *= 0.50
    if N < 40: corr *= 0.60
    if P < 30: corr *= 0.70
    if K < 30: corr *= 0.60
    if humidity > 85:
        corr *= 0.80

    return raw_yield * corr

# -----------------------------------------------------------
# EXPENSE CORRECTION
# -----------------------------------------------------------
def apply_expense_correction(expense, data):
    temp = to_float(data["temperature"])
    rain = to_float(data["rainfall"])
    humidity = to_float(data["humidity"])

    corr = 1.0

    if temp > 40: corr *= 1.20
    if rain < 20: corr *= 1.30
    if humidity > 90: corr *= 1.15

    return expense * corr

# -----------------------------------------------------------
# SUGGESTION ENGINE
# -----------------------------------------------------------
# FIXED YIELD FACTOR FOR EACH CROP (PUT HERE)
CROP_FACTOR = {
    "rice": 0.78,
    "wheat": 0.74,
    "maize": 0.72,
    "banana": 0.70,
    "barley": 0.69,
    "blackgram": 0.68,
    "brinjal": 0.71,
    "sesame": 0.67,
    "chickpea": 0.73,
    "onion": 0.66,
    "chilli": 0.65,
    "cauliflower": 0.70,
    "pigeonpeas": 0.74,
    "potato": 0.76,
    "sorghum": 0.69,
    "sugarcane": 0.64
}
def suggest_best_crops(data, user_price, base_yield):

    suggestions = []

    for crop in ALL_CROPS:

        # FIXED FACTOR – NEVER CHANGES
        factor = CROP_FACTOR.get(crop, 0.75)

        approx_yield = base_yield * factor
        revenue = approx_yield * user_price

        # FIXED EXPENSE (NO RANDOM FOREST)
        fixed_expense = (
            to_float(data.get("fertilizer")) * 40 +
            to_float(data.get("pesticide")) * 120 +
            to_float(data.get("seed")) +
            to_float(data.get("other"))
        )

        profit = revenue - fixed_expense

        suggestions.append({
            "Crop": crop,
            "Yield": approx_yield,
            "Profit": profit
        })

    # deterministic sorting (ALWAYS SAME)
    suggestions = sorted(suggestions, key=lambda x: x["Profit"], reverse=True)

    profitable = [c for c in suggestions if c["Profit"] > 0]

    if profitable:
        return profitable[:3]

    return []

# -----------------------------------------------------------
# MAIN PREDICTION API
# -----------------------------------------------------------
@app.route("/api/predict_all", methods=["POST"])
def api_predict_all():

    try:
        data = request.get_json()

        # Normalize input keys
        data["fertilizer"] = to_float(data.get("fertilizer") or data.get("Fertilizer_Usage_kg_per_hectare"))
        data["pesticide"]  = to_float(data.get("pesticide")  or data.get("Pesticide_Usage_litre_per_hectare"))
        data["seed"]       = to_float(data.get("seed")       or data.get("Seed_Expense_per_hectare(INR)"))
        data["other"]      = to_float(data.get("other")      or data.get("Other_Expense(INR)"))

        # FIX: ensure seed exists
        if "seed" not in data or data["seed"] == "" or data["seed"] is None:
            data["seed"] = 0
        # ML INPUT
        X = pd.DataFrame([{
            "N": to_float(data["N"]),
            "P": to_float(data["P"]),
            "K": to_float(data["K"]),
            "temperature": to_float(data["temperature"]),
            "humidity": to_float(data["humidity"]),
            "ph": to_float(data["ph"]),
            "rainfall": to_float(data["rainfall"])
        }])

        # Predict crop
        crop_encoded = rf_crop.predict(X)[0]
        crop_raw = label_encoder.inverse_transform([crop_encoded])[0].lower()

        if check_crop_failure(data):
            crop_final = "Crop Failure"
        else:
            crop_final = adjust_to_valid_crop(crop_raw, ALL_CROPS)

        # Predict yield
        raw_yield = float(rf_yield.predict(X)[0])
        predicted_yield = 1.0 if check_crop_failure(data) else apply_yield_correction(raw_yield, data)

        # Predict expense
        raw_exp = float(rf_expense.predict(pd.DataFrame([{
            "Fertilizer_Usage_kg_per_hectare": data["fertilizer"],
            "Pesticide_Usage_litre_per_hectare": data["pesticide"],
            "Seed_Expense_per_hectare(INR)": data["seed"],
            "Other_Expense(INR)": data["other"]
        }])))

        predicted_expense = apply_expense_correction(raw_exp, data)

        # Revenue & Profit
        price = to_float(data["market_price"])
        revenue = predicted_yield * price
        profit = max(revenue - predicted_expense, 0)
        loss = max(predicted_expense - revenue, 0)

        # Suggestions
        # Suggestions only if profit is positive
        best = []

        if profit > 0:
            best = suggest_best_crops(data, price, predicted_yield)

        else:
            best = []   # No suggestions when main prediction is loss


        return jsonify({
            "Predicted_Crop": crop_final,
            "Predicted_Yield": f"{predicted_yield:.2f} Kg/ha",
            "Total_Expense": f"{predicted_expense:.2f}",
            "Predicted_Revenue": f"{revenue:.2f}",
            "Profit": f"{profit:.2f}",
            "Loss": f"{loss:.2f}",
            "Best_Crops": [
                {
                    "Crop": c["Crop"],
                    "Yield": f"{float(c['Yield']):.2f} Kg/ha",
                    "Profit": f"{float(c['Profit']):.2f}"
                }
                for c in best
            ],
            "show_suggestions": profit > 0   
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# -----------------------------------------------------------
# PDF REPORT
# -----------------------------------------------------------
@app.route("/download_report", methods=["POST"])
def download_report():
    try:
        data = request.get_json()

        buffer = BytesIO()
        pdf = canvas.Canvas(buffer, pagesize=letter)

        y = 760

        # -------------------------
        # HEADER
        # -------------------------
        pdf.setFont("Helvetica-Bold", 20)
        pdf.drawString(50, y, "AgriProfit AI Report")
        y -= 40

        pdf.setFont("Helvetica", 12)
        pdf.drawString(50, y, f"Predicted Crop: {data['Predicted_Crop']}")
        y -= 20
        pdf.drawString(50, y, f"Predicted Yield: {data['Predicted_Yield']}")
        y -= 20
        pdf.drawString(50, y, f"Total Expense: ₹{data['Total_Expense']}")
        y -= 20
        pdf.drawString(50, y, f"Predicted Revenue: ₹{data['Predicted_Revenue']}")
        y -= 20
        pdf.drawString(50, y, f"Profit: ₹{data['Profit']}")
        y -= 20
        pdf.drawString(50, y, f"Loss: ₹{data['Loss']}")
        y -= 30

        # -------------------------
        # TOP 3 CROPS TABLE
        # -------------------------
        pdf.setFont("Helvetica-Bold", 14)
        pdf.drawString(50, y, "Top 3 Recommended Crops:")
        y -= 25

        pdf.setFont("Helvetica", 12)
        for idx, c in enumerate(data["Best_Crops"], start=1):
            pdf.drawString(50, y, f"{idx}. {c['Crop']}  | Yield: {c['Yield']}  | Profit: ₹{c['Profit']}")
            y -= 20

        y -= 30

        # -------------------------
        # INSERT CHART IMAGE
        # -------------------------
        if "chart_image" in data:
            import base64
            from PIL import Image
            import io

            img_data = data["chart_image"].split(",")[1]
            img_bytes = base64.b64decode(img_data)
            img = Image.open(io.BytesIO(img_bytes))

            img_path = "chart_temp.png"
            img.save(img_path)

            pdf.drawImage(img_path, 50, y-220, width=500, height=220)

        pdf.save()
        buffer.seek(0)

        return send_file(buffer, as_attachment=True,
                         download_name="AgriProfit_Report.pdf",
                         mimetype="application/pdf")

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# -----------------------------------------------------------
# AUTH ROUTES
# -----------------------------------------------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password, password):
            session["user_id"] = user.id
            session["username"] = user.username
            flash("Login successful", "success")
            return redirect(url_for("prediction"))
        else:
            flash("Invalid email or password", "error")

    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username")
        email = request.form.get("email")
        password = request.form.get("password")

        if User.query.filter_by(email=email).first():
            flash("Email already exists", "error")
            return redirect(url_for("register"))

        hashed_pw = generate_password_hash(password)
        new_user = User(username=username, email=email, password=hashed_pw)

        db.session.add(new_user)
        db.session.commit()

        flash("Registration successful, please login.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully", "success")
    return redirect(url_for("login"))

@app.route("/send_message", methods=["POST"])
def send_message():
    name = request.form.get("name")
    email = request.form.get("email")
    subject = request.form.get("subject")
    message = request.form.get("message")

    # Example: store or send email (you can modify)
    print("New Message:")
    print("Name:", name)
    print("Email:", email)
    print("Subject:", subject)
    print("Message:", message)

    flash("Your message has been sent successfully!", "success")
    return redirect(url_for("contact"))

# -----------------------------------------------------------
# ROUTES
# -----------------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/prediction")
def prediction():
    if "user_id" not in session:
        return redirect(url_for("login"))
    return render_template("prediction.html")

# -----------------------------------------------------------
# RUN
# -----------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
