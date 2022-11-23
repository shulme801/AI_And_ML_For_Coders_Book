import os
import tensorflow as tf
import numpy as np
# importing Sequential and Dense from tensorflow.keras gets an IDE error, though the code still runs correctly.
# but ignoring tensorflow.keras and importing directly from keras gets no IDE errors, and also works.
# Code tested 11/23/2022 with python3.9.13, tensorflow-macos and tensorflow-metal, as built with miniconda.
from keras import Sequential
from keras.layers import Dense
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def tensor_test1(num_epochs):
    l0 = Dense(units=1, input_shape=[1])
    model = Sequential([l0])
    model.compile(optimizer='sgd', loss='mean_squared_error')

    xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
    ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

    model.fit(xs, ys, epochs=num_epochs)

    print(model.predict([10.0]))
    print('Here is what I learned: {}'.format(l0.get_weights()))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    tensor_test1(500)
    exit()
# That's all folks!
