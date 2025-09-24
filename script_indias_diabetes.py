import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

data = pd.read_csv('diabetes.csv')

print(data.info)
print(data.describe())

plt.style.use("seaborn-darkgrid")
plt.rcParams["figure.figsize"] = (10, 6)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

sns.histplot(data["Glucose"], bins=30, kde=True, ax=axes[0,0])
axes[0,0].set_title("Distribution du Glucose")

sns.histplot(data["BMI"], bins=30, kde=True, ax=axes[0,1])
axes[0,1].set_title("Distribution du BMI")

sns.boxplot(x="Outcome", y="Glucose", data=data, ax=axes[1,0])
axes[1,0].set_title("Glucose par Outcome")

sns.boxplot(x="Outcome", y="BMI", data=data, ax=axes[1,1])
axes[1,1].set_title("BMI par Outcome")

plt.tight_layout()
plt.show()

