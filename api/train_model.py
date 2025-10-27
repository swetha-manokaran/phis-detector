import os
import sys
sys.path.append(os.path.dirname(__file__))  # ✅ ensures features.py can be imported

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from features import URLFeatures

# ✅ Create synthetic data if dataset not found
data_path = os.path.join(os.path.dirname(__file__), '../data/raw/urls.csv')
if not os.path.exists(data_path):
    print("⚠️ No dataset found. Creating synthetic training data...")
    df = pd.DataFrame({
        "url": ["https://google.com", "http://free-gift.ru", "https://paypal.com/login", "http://clickme.win"],
        "label": [0, 1, 0, 1],
    })
else:
    df = pd.read_csv(data_path)

# ✅ Extract features using the same logic as URLFeatures
feature_extractor = URLFeatures()
features = df["url"].apply(lambda u: feature_extractor.get_features(u))
features_df = pd.DataFrame(list(features))
X = features_df
y = df["label"]

# ✅ Define pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(n_estimators=100, random_state=42))
])

# ✅ Train
pipeline.fit(X, y)

# ✅ Save model
model_path = os.path.join(os.path.dirname(__file__), "pipeline.pkl")
with open(model_path, "wb") as f:
    pickle.dump(pipeline, f)

print(f"✅ Model trained and saved successfully → {model_path}")
