import os

from absl import app
from absl import flags

import tensorflow as tf

FLAGS = flags.FLAGS


def define_flags():
    flags.DEFINE_string(
        "image_path",
        "/mnt/disks/persist/RiceLeafs/validation/LeafBlast/IMG_4837.jpg",
        "The path to the image.",
    )  # TODO: Remove defaults
    flags.DEFINE_string(
        "model_path",
        "/home/risangbaskoro/petanikita-ml/model/coba3",
        "The path to the model.",
    )  # TODO: Remove defaults

    flags.mark_flag_as_required("image_path")
    flags.mark_flag_as_required("model_path")


def prepare_image(image):
    image = tf.expand_dims(image, axis=0)
    image = tf.cast(image, tf.float32)
    return image


def main(_):
    model = tf.keras.models.load_model(FLAGS.model_path)
    image = tf.keras.preprocessing.image.load_img(
        FLAGS.image_path, target_size=(224, 224)
    )

    image = prepare_image(image)

    pred = model.predict(image)

    print(pred)


if __name__ == "__main__":
    define_flags()
    app.run(main)
