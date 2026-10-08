import joblib

data = joblib.load("models/random_forest.joblib")

print(type(data))

if isinstance(data, dict):
    print("Keys:", data.keys())