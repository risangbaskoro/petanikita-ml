import os

from absl import app
from absl import flags

FLAGS = flags.FLAGS

import tensorflow as tf


def define_flags():
    flags.DEFINE_string(
        "model_path", "model/rice_leaf_disease_classifier", "The path to the model."
    )
    flags.DEFINE_string("export_path", "converted_model", "The path to the model.")


def main(_):
    model_path = FLAGS.model_path
    model_basename = os.path.basename(model_path)

    converter = tf.lite.TFLiteConverter.from_saved_model(model_path)

    tflite_model = converter.convert()

    if not os.path.exists(FLAGS.export_path):
        os.mkdir(FLAGS.export_path)

    with open(os.path.join(FLAGS.export_path, model_basename + ".tflite"), "wb") as f:
        f.write(tflite_model)


if __name__ == "__main__":
    define_flags()
    app.run(main)
