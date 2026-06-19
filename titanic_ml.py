import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

data = pd.read_csv("train.csv")

features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]
target = "Survived"

data["Age"] = data["Age"].fillna(data["Age"].median())
data["Sex"] = data["Sex"].map({"male": 0, "female": 1})

X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)

joblib.dump(model, "titanic_model.pkl")
print("Model saved!")