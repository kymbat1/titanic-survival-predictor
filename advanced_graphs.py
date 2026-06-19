import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("train.csv")

# График 1
plt.figure(figsize=(8, 5))
data.groupby("Sex")["Survived"].mean().plot(kind="bar")
plt.title("Survival Rate by Sex")
plt.ylabel("Survival Rate")
plt.show()

# График 2
plt.figure(figsize=(8, 5))
data.groupby("Pclass")["Survived"].mean().plot(kind="bar")
plt.title("Survival Rate by Passenger Class")
plt.ylabel("Survival Rate")
plt.show()

# График 3
plt.figure(figsize=(8, 5))
data["Age"].hist(bins=20)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Passengers")
plt.show()