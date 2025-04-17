import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib

# Load your preprocessed training features
df = pd.read_csv("voice_features_dataframe.csv")
X = df.drop(columns=["filename", "anomaly_score"], errors="ignore")

# Train scaler and model
scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)
model = IsolationForest(contamination=0.2, random_state=42)
model.fit(X_scaled)

# Save both
joblib.dump(scaler, "scaler.joblib")
joblib.dump(model, "isolation_forest_model.joblib")

print("Model and scaler saved.")