import constants
import tensorflow as tf

def load_facenet():
    return tf.keras.models.load_model(constants.FACENET_MODEL_PATH)