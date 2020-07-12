from tensorflow import keras
from tensorflow.keras import layers

import config


def vgg_style(input_tensor):
    """
    The original feature extraction structure from CRNN paper.
    Related paper: https://ieeexplore.ieee.org/abstract/document/7801919
    """
    # return input_tensor
    # TODO change strides...
    x = layers.Conv2D(4, 3, padding='same', activation='relu')(input_tensor)
    x = layers.MaxPool2D(pool_size=2, padding='same')(x)

    x = layers.Conv2D(8, 3, padding='same', activation='relu')(x)
    x = layers.MaxPool2D(pool_size=2, padding='same')(x)

    x = layers.Conv2D(16, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(16, 3, padding='same', activation='relu')(x)
    x = layers.MaxPool2D(pool_size=2, strides=(2, 1), padding='same')(x)

    x = layers.Conv2D(32, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.MaxPool2D(pool_size=2, strides=(2, 1), padding='same')(x)

    x = layers.Conv2D(32, 2, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x


def build_model(num_classes, image_width=None, channels=1):
    """build CNN-RNN model"""

    img_input = keras.Input(shape=(32, image_width, channels))
    x = vgg_style(img_input)
    x = layers.Reshape((-1, 32))(x)

    x = layers.Bidirectional(layers.LSTM(units=64, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(units=64, return_sequences=True))(x)
    x = layers.Dense(units=num_classes)(x)
    return keras.Model(inputs=img_input, outputs=x, name='CRNN')
