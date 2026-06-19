import pandas as pd

data = pd.read_csv("train.csv")

print("Всего пассажиров:", len(data))
print()

print("Выживаемость:")
print(data["Survived"].value_counts())
print()

print("Выживаемость по полу:")
print(data.groupby("Sex")["Survived"].mean())
print()

print("Выживаемость по классу:")
print(data.groupby("Pclass")["Survived"].mean())