import numpy as np
from keras.models import Model, Sequential
from keras.layers import Dense, RepeatVector, TimeDistributed
from keras.initializers import Constant
from keras import losses, metrics, optimizers

input_size = 3

def test_installation():
    model = Sequential()
    model.add(Dense(2, input_shape=(input_size,)))
    model.add(RepeatVector(input_size))
    model.add(TimeDistributed(Dense(input_size)))
    model.compile(loss=losses.MSE,
                  optimizer=optimizers.RMSprop(lr=0.0001),
                  metrics=[metrics.categorical_accuracy],
                  sample_weight_mode='temporal')
    x = np.random.random((1, input_size))
    y = np.random.random((1, input_size, input_size))
    model.train_on_batch(x, y)

    out = model.predict(x)
    success = out.shape == y.shape
    return success

if __name__ == '__main__':
    success = test_installation()
    if success:
        print('Succesfully installed keras!')
    else:
        raise AssertionError('Did not properly install keras')

