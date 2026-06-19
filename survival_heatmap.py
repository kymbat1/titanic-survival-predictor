import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("train.csv")

table = data.pivot_table(
    values="Survived",
    index="Sex",
    columns="Pclass",
    aggfunc="mean"
)

print(table)

plt.imshow(table)

plt.colorbar(label="Survival Rate")

plt.xticks([0, 1, 2], ["1 Class", "2 Class", "3 Class"])
plt.yticks([0, 1], table.index)

plt.title("Survival Rate by Sex and Class")

plt.show()