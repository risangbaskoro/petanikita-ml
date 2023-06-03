"""Train a model"""
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from absl import app
from absl import flags

import flatbuffers
import tensorflow as tf

from model import LeafDiseaseClassifier
from utils import connect_to_tpu


FLAGS = flags.FLAGS


def define_flags():
    flags.DEFINE_string(
        "dataset_path", "/mnt/disks/persist/RiceLeafs/train", "The path to the dataset."
    )# TODO: REMOVE DEFAULTS to None
    flags.DEFINE_string("model_export_name", "rice_leaf_disease_classifier", "The name of the model.") # TODO: REMOVE DEFAULTS to None
    flags.DEFINE_string("model_export_path", "model", "The path to the model.") # TODO: REMOVE DEFAULTS to None
    flags.DEFINE_bool("augment_data", True, "Whether to augment the data.")
    flags.DEFINE_integer("batch_size", 200, "The batch size.")
    flags.DEFINE_string("tpu_address", "local", "The address of the TPU to connect to.") # TODO: REMOVE DEFAULTS to ""
    flags.DEFINE_integer("num_epochs", 10, "The number of epochs to train for.")

    flags.mark_flag_as_required("dataset_path")
    flags.mark_flag_as_required("model_export_name")
    flags.mark_flag_as_required("model_export_path")


def get_image_dataset_from_directory(directory, image_size=(224, 224), validation_split=0.2, seed=42): # TODO: Default value can be removed, use flags instead
    train_ds = tf.keras.utils.image_dataset_from_directory(
        directory,
        image_size=image_size,
        validation_split=validation_split,
        subset="training",
        seed=seed,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        directory,
        image_size=image_size,
        validation_split=validation_split,
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

def get_callbacks():
    callbacks = []
    return callbacks


def main(_):
    tf.keras.backend.clear_session()

    _, strategy = connect_to_tpu(tpu_address=FLAGS.tpu_address)

    train_ds, val_ds = get_image_dataset_from_directory(FLAGS.dataset_path)

    class_names = train_ds.class_names
    num_classes = len(class_names)

    if FLAGS.augment_data:
        train_ds, val_ds = augment_data(train_ds, val_ds)

    batch_size = FLAGS.batch_size

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE).repeat()
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE).repeat()

    with strategy.scope():
        model = LeafDiseaseClassifier(num_classes=num_classes, model="MobileNetV2", weights="imagenet") # TODO: Default value can be removed, use flags
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

    with strategy.scope():
        model.fit(
            train_ds,
            epochs=FLAGS.num_epochs,
            batch_size=batch_size,
            validation_data=val_ds,
            validation_steps=10000 // batch_size, # TODO: Default value can be removed, use flags
            steps_per_epoch=40000 // batch_size, # TODO: Default value can be removed, use flags
            callbacks=get_callbacks(),
        )

    if FLAGS.model_export_path is not None:
        model_path = os.path.join(FLAGS.model_export_path, FLAGS.model_export_name)
        model.save(model_path)


if __name__ == "__main__":
    define_flags()
    app.run(main)
