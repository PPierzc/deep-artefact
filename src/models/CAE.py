from keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Reshape
from keras.models import Sequential


def CAE(
        input_shape=(28, 28, 1),
        filters=(32, 64, 128, 10)
):
    model = Sequential()

    if input_shape[0] % 8 == 0:
        pad3 = 'same'
    else:
        pad3 = 'valid'

    model.add(Conv2D(filters[0], 3, strides=2, padding=pad3, activation='relu', name='conv1', input_shape=input_shape))
    model.add(Conv2D(filters[1], 3, strides=2, padding=pad3, activation='relu', name='conv2'))
    model.add(Conv2D(filters[2], 2, strides=2, padding=pad3, activation='relu', name='conv3'))

    model.add(Flatten(name='flatten'))
    model.add(Dense(units=filters[3], name='embedding'))

    model.add(Dense(units=filters[2] * int(input_shape[0] / 8) * int(input_shape[0] / 8), activation='relu', name='FC'))
    model.add(Reshape((int(input_shape[0] / 8), int(input_shape[0] / 8), filters[2]), name='reshape'))

    model.add(Conv2DTranspose(filters[1], 2, strides=2, padding=pad3, activation='relu', name='deconv3'))
    model.add(Conv2DTranspose(filters[0], 3, strides=2, padding=pad3, activation='relu', name='deconv2'))
    model.add(Conv2DTranspose(input_shape[2], 3, strides=2, padding=pad3, name='deconv1'))

    return model
