from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load("fraud_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    premium = float(request.form["premium"])
    age = int(request.form["age"])
    severity = int(request.form["severity"])
    vehicles = int(request.form["vehicles"])
    damage = int(request.form["damage"])

    data = np.array([[premium, age, severity, vehicles, damage]])

    prediction = model.predict(data)

    result = "Fraud Claim 🚨" if prediction[0] == 1 else "Genuine Claim ✅"

    return render_template("index.html", prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)