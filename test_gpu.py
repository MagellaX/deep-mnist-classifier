import tensorflow as tf

# Simple GPU check
print("TensorFlow version:", tf.__version__)
print("\nGPUs Available:", len(tf.config.list_physical_devices('GPU')))
print("GPU Device:", tf.test.gpu_device_name())

# Basic GPU test
if tf.config.list_physical_devices('GPU'):
    print("\nGPU was found and is ready to use!")
else:
    print("\nNo GPU found, using CPU!")
