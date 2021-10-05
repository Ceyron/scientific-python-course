import tensorflow as tf
import matplotlib.pyplot as plt

# For reproducibility
tf.random.set_seed(42)

x = tf.random.uniform((50,), minval=0.0, maxval=2*3.141)
y = tf.sin(x) + 0.2 * tf.random.normal((50, ))
x = tf.reshape(x, (-1, 1))
y = tf.reshape(y, (-1, 1))

plt.scatter(x.numpy(), y.numpy(), color="black")

model = tf.keras.Sequential([
    # Play around with the model complexity
    tf.keras.layers.Dense(
        1000,
        activation="relu",
        kernel_regularizer="l2"
    ),
    tf.keras.layers.Dense(
        1000,
        activation="relu",
        kernel_regularizer="l2"
    ),
    tf.keras.layers.Dense(1, activation=None)
])

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.MeanSquaredError(),
)

x_set = tf.linspace(0.0, 2.0*3.141, 100)
x_set = tf.reshape(x_set, (-1, 1))
y_pred = model(x_set)
plt.plot(x_set, y_pred)


for i in range(10):
    model.fit(x, y, epochs=50, validation_split=0.2)
    y_pred = model(x_set)
    plt.plot(x_set, y_pred)


plt.show()
