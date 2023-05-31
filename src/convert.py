import os

import tensorflow as tf


def convert_to_tflite(saved_model_dir, convert_name='model.tflite', same_name=False):
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)  # path to the SavedModel directory
    tflite_model = converter.convert()

    if same_name:
        convert_name = os.path.basename(saved_model_dir) + '.tflite'

    with open(os.path.join(os.getcwd(), 'converted_model', convert_name), 'wb') as f:
        f.write(tflite_model)


if __name__ == '__main__':
    model_dir = os.path.join(os.getcwd(), 'model')
    model_name = 'rice_leaf_disease_classifier'
    model_path = os.path.join(model_dir, model_name)
    convert_to_tflite(saved_model_dir=model_path, same_name=True)
    print("Model Converted")
