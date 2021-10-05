import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

age_to_getting_pension = pd.read_csv("https://github.com/Ceyron/machine-learning-and-simulation/files/7251110/age_to_getting_pension.csv")

classifier = LogisticRegression()
classifier.fit(age_to_getting_pension[["age", ]], age_to_getting_pension["get_pension"])

age_set = np.linspace(18, 99, 100)
prob = classifier.predict_proba(age_set.reshape((-1, 1)))[:, 1]

age_to_getting_pension.plot.scatter(x="age", y="get_pension")
plt.plot(age_set, prob, color="orange")
plt.grid()
plt.title("Logistic Regression on whether someone is receiving pension")
plt.show()
