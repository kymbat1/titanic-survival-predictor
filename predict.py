import joblib
import pandas as pd

model = joblib.load("titanic_model.pkl")

person = pd.DataFrame([{
    "Pclass": 3,
    "Sex": 0,      # 0 = male, 1 = female
    "Age": 40,
    "SibSp": 0,
    "Parch": 0,
    "Fare": 7
}])

prediction = model.predict(person)[0]
probability = model.predict_proba(person)[0][1]

if prediction == 1:
    print("Результат: Выжил")
else:
    print("Результат: Не выжил")

print(f"Вероятность выживания: {probability * 100:.2f}%")
