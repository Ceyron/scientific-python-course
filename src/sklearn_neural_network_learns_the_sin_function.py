from scipy.sparse.construct import rand, random
from sklearn.neural_network import MLPRegressor
import numpy as np
import matplotlib.pyplot as plt

# For reproducibility
np.random.seed(42)

x = np.random.uniform(size=(50,), low=0.0, high=2*3.141)
y = np.sin(x) + 0.2 * np.random.normal(size=(50, ))
x = np.reshape(x, (-1, 1))
# y = np.reshape(y, (-1, 1))

plt.scatter(x.flatten(), y, color="black")

model = MLPRegressor(max_iter=1, random_state=42)
model.fit(x, y)

x_set = np.linspace(0.0, 2.0*3.141, 100)
x_set = np.reshape(x_set, (-1, 1))
y_pred = model.predict(x_set)
plt.plot(x_set, y_pred)


for i in range(10):
    model = MLPRegressor(hidden_layer_sizes=(1000, ), max_iter=100*(i+1), random_state=42)
    model.fit(x, y)
    y_pred = model.predict(x_set)
    plt.plot(x_set, y_pred)


plt.show()
