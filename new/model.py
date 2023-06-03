import tensorflow as tf


class LeafDiseaseClassifier(tf.keras.Model):
    """A basic model for classifying leaf diseases, uses MobileNetV2 as a backbone.
    """

    def __init__(self, num_classes, model="MobileNetV2", weights="imagenet"):
        super(LeafDiseaseClassifier, self).__init__(name="LeafDiseaseClassifier")

        self.num_classes = num_classes

        self.pre_trained_model, self.preprocess_function = self._get_pre_trained_model(model, weights)
        self.pre_trained_model.trainable = False

        self.dropout = tf.keras.layers.Dropout(0.2)
        self.conv1 = tf.keras.layers.Conv2D(2000, 3, name="top_conv1")
        self.conv2 = tf.keras.layers.Conv2D(1000, 3, name="top_conv2")
        self.flatten = tf.keras.layers.Flatten(name="flatten")
        self.fc1 = tf.keras.layers.Dense(1000, activation="relu", name="fc1")
        self.fc2 = tf.keras.layers.Dense(500, activation="relu", name="fc2")
        self.prediction = tf.keras.layers.Dense(
            self.num_classes, activation="softmax", name="prediction"
        )

    def call(self, inputs):
        if self.pre_trained_model is None:
            raise NotImplementedError("You must implement this method.")

        x = self.preprocess_function(inputs)
        x = self.pre_trained_model(x)
        x = self.dropout(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return self.prediction(x)

    def _get_pre_trained_model(self, model="MobileNetV2", weights="imagenet"):
        supported_models = ["MobileNetV2", "ResNet50V2", "InceptionV3"]
        if model == "MobileNetV2":
            return tf.keras.applications.MobileNetV2(
                input_shape=(224, 224, 3), include_top=False, weights=weights
            ), tf.keras.applications.mobilenet_v2.preprocess_input

        if model not in supported_models:
            raise ValueError(
                f"Unsupported model {model}, supported models are {supported_models}"
            )
