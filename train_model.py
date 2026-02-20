import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

print("Loading dataset...")

data = pd.read_csv("insurance.csv")

le = LabelEncoder()
data["incident_severity"] = le.fit_transform(data["incident_severity"])
data["property_damage"] = le.fit_transform(data["property_damage"])
data["fraud_reported"] = le.fit_transform(data["fraud_reported"])

X = data.drop("fraud_reported", axis=1)
y = data["fraud_reported"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Model Accuracy:", accuracy_score(y_test, predictions))

joblib.dump(model, "fraud_model.pkl")

print("Model Saved Successfully!")