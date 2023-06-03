"""Train a model"""
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from absl import app
from absl import flags

import flatbuffers
import tensorflow as tf

from utils import connect_to_tpu


FLAGS = flags.FLAGS


def define_flags():
    # TODO: REMOVE DEFAULTS
    flags.DEFINE_string("dataset_path", "/mnt/disks/persist/RiceLeafs/train", "The path to the dataset.")
    flags.DEFINE_bool("augment_data", True, "Whether to augment the data.")
    flags.DEFINE_string("tpu_address", "local", "The address of the TPU to connect to.")

    flags.mark_flag_as_required("dataset_path")


def get_image_dataset_from_directory(directory, seed=42):
    IMAGE_SIZE = (224, 224)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        directory,
        image_size=IMAGE_SIZE,
        validation_split=0.2,
        subset="training",
        seed=seed,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        directory,
        image_size=IMAGE_SIZE,
        validation_split=0.2,
        subset="validation",
        seed=seed,
    )
    return train_ds, val_ds


def augment_data(train_ds, val_ds):
    augment_layer = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomRotation(0.2),
            tf.keras.layers.RandomZoom(0.2),
        ]
    )
    train_ds = train_ds.map(lambda x, y: (augment_layer(x), y))
    if val_ds is not None:
        val_ds = val_ds.map(lambda x, y: (augment_layer(x), y))
    return train_ds, val_ds


def main(_):
    tf.keras.backend.clear_session()

    cluster_resolver, strategy = connect_to_tpu(tpu_address=FLAGS.tpu_address)

    train_ds, val_ds = get_image_dataset_from_directory(FLAGS.dataset_path)

    class_names = train_ds.class_names
    num_classes = len(class_names)

    if FLAGS.augment_data:
        train_ds, val_ds = augment_data(train_ds, val_ds)

    # TODO: Batch
    # TODO: Prefetch
    # TODO: Define and compile model
    # TODO: Fitting Model
    # TODO: Save model


if __name__ == "__main__":
    define_flags()
    app.run(main)
