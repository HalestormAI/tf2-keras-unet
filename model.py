import tensorflow as tf
from tensorflow.keras import layers


def left_layer(x, num_filters, dropout_rate=None, downsample=True, pad_convs=True):
    # Note: The paper says unpadded, but all the online implementations seem to pad.
    padding = "same" if pad_convs else "valid"
    conv1 = layers.Conv2D(num_filters, 3, padding=padding, activation="relu", kernel_initializer='he_normal')
    conv2 = layers.Conv2D(num_filters, 3, padding=padding, activation="relu", kernel_initializer='he_normal')

    x = conv1(x)
    x = conv2(x)

    if dropout_rate is not None:
        x = layers.Dropout(dropout_rate)(x)

    # We'll downsample every contraction layer except the last one
    if downsample:
        maxpool = layers.MaxPooling2D((2, 2))
        return maxpool(x), x

    return x


def right_layer(x, r, num_filters, pad_convs=True):
    x = layers.UpSampling2D(size=(2, 2))(x)

    padding = "same" if pad_convs else "valid"
    conv1 = layers.Conv2D(num_filters, 2, padding=padding, activation="relu", kernel_initializer='he_normal')
    conv2 = layers.Conv2D(num_filters, 3, padding=padding, activation="relu", kernel_initializer='he_normal')
    conv3 = layers.Conv2D(num_filters, 3, padding=padding, activation="relu", kernel_initializer='he_normal')

    x = layers.concatenate([r, conv1(x)], axis=3)
    x = conv2(x)
    return conv3(x)


def unet(input_size=572, input_channels=1, num_classes=1):
    inputs = tf.keras.Input((input_size, input_size, input_channels))

    dropout_rate = 0.5
    filter_sizes = [64, 128, 256, 512, 1024]

    residuals = {}

    x = inputs
    # Run the contraction layers
    for i, f in enumerate(filter_sizes[:-1]):
        d = dropout_rate if i > 2 else None
        x, residuals[f] = left_layer(x, f, d, downsample=True)

    # Final layer before going back up
    x = left_layer(x, filter_sizes[-1], dropout_rate, False)

    # Run the expansion layers
    for i, f in enumerate(filter_sizes[-2::-1]):
        x = right_layer(x, residuals[f], f)

    # Classification head
    outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)
    return tf.keras.Model(inputs, outputs)
