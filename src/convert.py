import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from absl import app
from absl import flags

FLAGS = flags.FLAGS

import tensorflow as tf


def define_flags():
    flags.DEFINE_string(
        "model_path", "model/rldc_mobilenet_v1_1_default_1", "The path to the model."
    )  # TODO: Remove defaults
    flags.DEFINE_string("export_path", "converted_model", "The path to the model.")

def representative_dataset_gen():
    img_path = "/mnt/disks/persist/rice-leafs-1000px/train/Healthy"
    img_name = os.listdir(img_path)[0]
    img = tf.keras.utils.load_img(os.path.join(img_path, img_name), target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array = tf.cast(img_array, tf.float32) / 255.0
    yield [img_array]


def main(_):
    model_path = FLAGS.model_path
    model_basename = os.path.basename(model_path)

    converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    tflite_model = converter.convert()

    if not os.path.exists(FLAGS.export_path):
        os.mkdir(FLAGS.export_path)

    with open(os.path.join(FLAGS.export_path, model_basename + ".tflite"), "wb") as f:
        f.write(tflite_model)
    print("Model successfully converted to TFLite.")


if __name__ == "__main__":
    define_flags()
    app.run(main)
