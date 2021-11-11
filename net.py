import tensorflow as tf
import tensorflow.keras.layers as layers

input_shape = (1,54,1)

model = tf.keras.models.Sequential()
model.add(layers.Conv1D(31, 7, activation='relu', input_shape=input_shape[1:]))
model.add(layers.MaxPooling1D(7, data_format='channels_first'))
model.add(layers.Conv1D(31, 7, activation='relu'))
model.add(layers.MaxPooling1D(7, data_format='channels_first'))
model.add(layers.Conv1D(31, 7, activation='relu'))
model.add(layers.MaxPooling1D(7, data_format='channels_first'))
model.add(layers.GRU(7))
model.add(layers.Dense(18))

model.compile(optimizer='sgd', loss='mse')

print("\n===============\n")

x = tf.random.normal(input_shape)
#print("x: ", x)

y = model.evaluate(x)
#print("y: ", y)

model = layers.Conv1D(31, 7, activation='relu', input_shape=input_shape[1:])(x)
print(model.shape)
model = layers.MaxPooling1D(7, data_format='channels_first')(model)
print(model.shape)
model = layers.Conv1D(31, 7, activation='relu')(model)
print(model.shape)
model = layers.MaxPooling1D(7, data_format='channels_first')(model)
print(model.shape)
model = layers.Conv1D(31, 7, activation='relu')(model)
print(model.shape)
model = layers.MaxPooling1D(7, data_format='channels_first')(model)
print(model.shape)
model = layers.GRU(7)(model)
print(model.shape)

