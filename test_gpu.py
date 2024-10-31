import tensorflow as tf

<<<<<<< HEAD
# Simple GPU check
print("TensorFlow version:", tf.__version__)
print("\nGPUs Available:", len(tf.config.list_physical_devices('GPU')))
print("GPU Device:", tf.test.gpu_device_name())

# Basic GPU test
if tf.config.list_physical_devices('GPU'):
    print("\nGPU was found and is ready to use!")
else:
    print("\nNo GPU found, using CPU!")
=======
print("TensorFlow version:", tf.__version__)
print("GPU devices:", tf.config.list_physical_devices('GPU'))

try:
    with tf.device('/GPU:0'):
        a = tf.random.normal([1000, 1000])
        print("\nGPU test successful!")
except RuntimeError as e:
    print("\nError using GPU:", e)
>>>>>>> 1b1d31a03b8188a21c901881dbb515d79d526d29
