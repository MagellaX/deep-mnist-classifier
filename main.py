import tensorflow as tf
import numpy as np
import time

# Load and preprocess MNIST dataset
def load_data():
    print("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Normalize and reshape the data
    x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 784).astype('float32') / 255.0
    
    # Convert labels to categorical format
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    
    return (x_train, y_train), (x_test, y_test)

def create_model():
    """
    Create a neural network with three hidden layers (500 nodes each)
    """
    model = tf.keras.Sequential([
        # Input layer
        tf.keras.layers.Input(shape=(784,)),
        
        # Hidden layers
        tf.keras.layers.Dense(500, activation='relu'),
        tf.keras.layers.Dense(500, activation='relu'),
        tf.keras.layers.Dense(500, activation='relu'),
        
        # Output layer
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    return model

def train_model():
    """
    Train the neural network using the MNIST dataset
    """
    print("Starting training...")
    start_time = time.time()
    
    # Load data
    (x_train, y_train), (x_test, y_test) = load_data()
    
    # Create model
    model = create_model()
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Print model summary
    model.summary()
    
    # Train model
    batch_size = 100
    n_epochs = 10
    
    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=n_epochs,
        validation_data=(x_test, y_test),
        verbose=1
    )
    
    # Evaluate model
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    
    training_time = time.time() - start_time
    print(f'\nTraining completed in {training_time:.2f} seconds')
    print(f'Final Test Accuracy: {test_accuracy * 100:.2f}%')
    
    return model, history

def plot_history(history):
    """
    Plot training history (if matplotlib is installed)
    """
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 4))
        
        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("Matplotlib is not installed. Skipping plotting.")

def main():
    """
    Main function to run the MNIST neural network
    """
    print("MNIST Deep Neural Network - TensorFlow 2.x Version")
    print("-----------------------------------------------")
    
    # Force GPU usage
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("GPU is available!")
else:
    print("No GPU found. Running on CPU")
    
    # Train the model
    model, history = train_model()
    
    # Plot training history
    plot_history(history)
    
    # Optional: Save the model
    try:
        model.save('mnist_model')
        print("\nModel saved successfully!")
    except:
        print("\nFailed to save model.")

if __name__ == '__main__':
    main()