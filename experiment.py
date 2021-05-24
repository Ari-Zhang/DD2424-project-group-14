import tensorflow as tf
import numpy as np

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(3, input_shape = (3,)),
    tf.keras.layers.Dense(3),
    tf.keras.layers.Dropout(0.05),
    tf.keras.layers.Dense(2)
])
model.compile(
    loss = 'mse',
    metrics = 'mae'
)
model.summary()
x = np.zeros((3,3))

with tf.GradientTape() as tape:
    preds = model(x)
    print(type(preds))
    print(preds)
