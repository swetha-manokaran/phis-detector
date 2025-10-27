from flask import Flask, render_template, request
import pickle
import pandas as pd
import os
from datetime import datetime
from .features import URLFeatures  # relative import to access features.py

app = Flask(__name__)

# --- Load the trained pipeline ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), "pipeline.pkl")
with open(MODEL_PATH, "rb") as f:
    pipeline = pickle.load(f)

# --- Log file ---
LOG_FILE = os.path.join(os.path.dirname(__file__), "predictions_log.csv")

# --- Home route ---
@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    url = None

    # If POST request (user submits a URL)
    if request.method == "POST":
        url = request.form["url"].strip()
        feature_extractor = URLFeatures()
        features = feature_extractor.get_features(url)
        features_df = pd.DataFrame([features])

        prediction = pipeline.predict(features_df)[0]
        result = "Phishing 🚨" if prediction == 1 else "Legitimate ✅"

        # Log results to CSV file
        log_entry = pd.DataFrame([{
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "url": url,
            "result": result
        }])
        log_entry.to_csv(LOG_FILE, mode="a", header=not os.path.exists(LOG_FILE), index=False)

    # Stats for dashboard
    stats = {"total": 0, "phishing": 0, "legit": 0}
    if os.path.exists(LOG_FILE) and os.path.getsize(LOG_FILE) > 0:
        df = pd.read_csv(LOG_FILE)
        stats["total"] = len(df)
        stats["phishing"] = len(df[df["result"].str.contains("Phishing")])
        stats["legit"] = len(df[df["result"].str.contains("Legitimate")])

    return render_template("index.html", result=result, url=url, stats=stats)


if __name__ == "__main__":
    app.run(debug=True)
