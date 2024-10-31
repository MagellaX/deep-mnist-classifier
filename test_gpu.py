import tensorflow as tf
print("TensorFlow:", tf.__version__)
devices = tf.config.list_physical_devices()
print("Available devices:", devices)
