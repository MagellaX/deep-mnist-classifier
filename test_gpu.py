import tensorflow as tf

print("TensorFlow version:", tf.__version__)
print("GPU devices:", tf.config.list_physical_devices('GPU'))

try:
    with tf.device('/GPU:0'):
        a = tf.random.normal([1000, 1000])
        print("\nGPU test successful!")
except RuntimeError as e:
    print("\nError using GPU:", e)
