import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("train.csv")

survival_by_class = data.groupby("Pclass")["Survived"].mean()

survival_by_class.plot(kind="bar")

plt.title("Survival Rate by Passenger Class")
plt.xlabel("Class")
plt.ylabel("Survival Rate")

plt.show()