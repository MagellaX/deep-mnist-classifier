import tensorflow as tf
import numpy as np

# Print TensorFlow version
print("TensorFlow version:", tf.__version__)

# Check GPU availability
gpus = tf.config.list_physical_devices('GPU')
print("\nGPU Devices:", gpus)

if gpus:
    try:
        # Try a simple GPU operation
        with tf.device('/GPU:0'):
            a = tf.random.normal([1000, 1000])
            b = tf.random.normal([1000, 1000])
            c = tf.matmul(a, b)
            print("\nGPU test successful: Matrix multiplication completed")
            print("Using GPU:", tf.test.gpu_device_name())
    except RuntimeError as e:
        print("\nError using GPU:", e)
else:
    print("\nNo GPU devices found")

# Additional system info
print("\nCUDA Available:", tf.test.is_built_with_cuda())