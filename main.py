import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import tensorflow as tf

from dataset import get_dataset
from model import unet


def show_history(model_history, num_epochs):
    loss = model_history.history['loss']
    val_loss = model_history.history['val_loss']

    epochs = range(num_epochs)

    plt.figure()
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'bo', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.ylim([0, 1])
    plt.legend()
    plt.show()


def train():
    img_size = 512
    batch_size = 32
    buffer_size = 1024

    num_epochs = 6

    ds_train, ds_test, ds_info = get_dataset(img_size, batch_size, buffer_size)
    model = unet(img_size, 3, num_classes=3)
    model.summary()

    steps_per_epoch = ds_info.splits['train'].num_examples // batch_size

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss="sparse_categorical_crossentropy",
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint("pets_unet_model_ckpt.{epoch:02d}-{val_loss:.2f}.hdf5", save_best_only=True)
    ]

    model_history = model.fit(
        ds_train,
        epochs=num_epochs,
        validation_data=ds_test,
        steps_per_epoch=steps_per_epoch,
        callbacks=callbacks
    )

    model.save("pets_unet_model")

    show_history(model_history, num_epochs)


if __name__ == "__main__":
    train()
