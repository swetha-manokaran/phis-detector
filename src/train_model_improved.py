import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import os

from api.features import URLFeatures  # ✅ import from api

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load your data
data_path = os.path.join(BASE_DIR, 'data', 'raw', 'urls.csv')
df = pd.read_csv(data_path)

X = df['url']
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create pipeline
pipeline = Pipeline([
    ('features', URLFeatures()),
    ('scaler', StandardScaler()),
    ('classifier', GaussianNB())
])

# Train model
pipeline.fit(X_train, y_train)

# Save pipeline
joblib.dump(pipeline, os.path.join(BASE_DIR, 'api', 'pipeline.pkl'))
print("✅ Pipeline saved as api/pipeline.pkl")
