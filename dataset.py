import tensorflow as tf
import tensorflow_datasets as tfds


def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask -= 1
    return input_image, input_mask


@tf.function
def load_image_train(datapoint, img_resize):
    input_image = tf.keras.preprocessing.image.smart_resize(
        datapoint['image'], (img_resize, img_resize), interpolation='nearest'
    )
    input_mask = tf.keras.preprocessing.image.smart_resize(
        datapoint['segmentation_mask'], (img_resize, img_resize), interpolation='nearest'
    )

    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask


def load_image_test(datapoint, img_resize):
    input_image = tf.keras.preprocessing.image.smart_resize(
        datapoint['image'], (img_resize, img_resize), interpolation='nearest'
    )
    input_mask = tf.keras.preprocessing.image.smart_resize(
        datapoint['segmentation_mask'], (img_resize, img_resize), interpolation='nearest'
    )

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask


def get_dataset(img_size, batch_size=32, shuffle_buff=1024):
    # Construct a tf.data.Dataset
    dataset, ds_info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)

    ds_train = dataset['train'].map(lambda d: load_image_train(d, img_size), num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.cache().shuffle(shuffle_buff).batch(batch_size).repeat()
    ds_train = ds_train.prefetch(buffer_size=tf.data.AUTOTUNE)

    ds_test = dataset['test'].map(lambda d: load_image_test(d, img_size))
    ds_test = ds_test.batch(batch_size)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)
    return ds_train, ds_test, ds_info
