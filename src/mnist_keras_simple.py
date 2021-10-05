import tensorflow as tf
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

# plt.imshow(x_train[0], cmap="gray")
# plt.show()

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Softmax()
])

# print(tf.keras.losses.SparseCategoricalCrossentropy()(y_train[:1], model(x_train[:1])))

model.compile(optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"])
model.fit(x_train, y_train, epochs=5)

print(model(x_test[:5]))
print(tf.argmax(model(x_test[:5]), axis=1))
print(y_test[:5])