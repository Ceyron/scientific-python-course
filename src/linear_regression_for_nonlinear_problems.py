import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.linear_model import LinearRegression

msci_world_monthly = pd.read_csv("https://github.com/Ceyron/machine-learning-and-simulation/files/7250723/msci_world_monthly.csv")
print(msci_world_monthly)

months_passed = np.array(list(msci_world_monthly.index)).reshape((-1, 1))
y = msci_world_monthly["value"]

y_log = np.log(y)

lin_regr = LinearRegression()
lin_regr.fit(months_passed, y_log)

# Once fitted, the lin_regr instance contains attributes corresponding to the
# fitted weights and intercept. Only the slope is interest for the exponential
# growth
log_theta = lin_regr.coef_[0]
theta = np.exp(log_theta)
print(f"Theta: {theta:1.4f}")

# Since we used monthly data, theta is corresponding to the monthly growth, the
# yearly growth is just 12 consecutive applications of theta
theta_yearly = theta**12
print(f"Theta yearly: {theta_yearly:1.4f}")

# The yearly percentage growth
percentage_growth = (theta_yearly - 1.0) * 100.0
print(f"Yearly growth in percent: {percentage_growth:1.4f} %")

y_pred_log = lin_regr.predict(months_passed)
y_pred = np.exp(y_pred_log)

# Add the prediction to the DataFrame as a series to simply use pandas
# integrated plotting routines (this prettily formats the apparent dates)
msci_world_monthly["value_predicted"] = y_pred
msci_world_monthly.plot.line(x="date", y=["value", "value_predicted"])

plt.title("End of Month Value for MSCI World, truth and predicted")
plt.grid()
plt.show()

