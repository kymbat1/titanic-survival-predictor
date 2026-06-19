import joblib
import pandas as pd
import matplotlib.pyplot as plt

model = joblib.load("titanic_model.pkl")

features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]
importances = model.feature_importances_

importance_data = pd.DataFrame({
    "Feature": features,
    "Importance": importances
})

importance_data = importance_data.sort_values(
    by="Importance",
    ascending=False
)

print(importance_data)

plt.figure(figsize=(8, 5))
plt.bar(importance_data["Feature"], importance_data["Importance"])

plt.title("Feature Importance in Titanic Survival Model")
plt.xlabel("Feature")
plt.ylabel("Importance")

plt.show()