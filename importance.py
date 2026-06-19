import pandas as pd
import joblib

model = joblib.load("titanic_model.pkl")

features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]

for name, importance in zip(features, model.feature_importances_):
    print(f"{name}: {importance:.3f}")