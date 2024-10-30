# MNIST Deep Neural Network

A simple implementation of a deep neural network for MNIST digit recognition using TensorFlow.

## Network Architecture
- Input Layer: 784 nodes (28x28 pixels)
- Hidden Layer 1: 500 nodes
- Hidden Layer 2: 500 nodes
- Hidden Layer 3: 500 nodes
- Output Layer: 10 nodes (digits 0-9)

## Setup & Run
1. Install requirements:
```bash
pip install -r requirements.txt
```

2. Run the model:
```bash
python main.py
```

## Features
- Uses TensorFlow
- Three hidden layers
- ReLU activation
- Adam optimizer
- Batch processing


## Troubleshooting

### Common Installation Issues

#### Windows Permission Error
If you encounter permission errors while installing TensorFlow (e.g., "Could not install packages due to an OSError: [WinError 2]"), follow these steps:

1. **Close all Python processes**
   - Close any running Python processes or IDEs
   - Close any command prompts or PowerShell windows

2. **Try Administrator Installation**
   ```bash
   # Run PowerShell as Administrator
   # Navigate to your project directory
   cd your-project-path
   pip install tensorflow
   ```

3. **If that doesn't work, try Python 3.10 Installation:**
   1. Uninstall current Python (from Windows Control Panel)
   2. Download Python 3.10 from [python.org](https://www.python.org/downloads/)
   3. During installation:
      - ✅ Check "Add Python to PATH"
      - ✅ Check "Install for all users"
   4. After installation:
      ```bash
      # Open new PowerShell and try
      pip install tensorflow
      ```

   > **Note:** TensorFlow may have compatibility issues with Python 3.12. Python 3.10 is recommended for better compatibility.

#### Alternative Installation Method
If you're still experiencing issues, try installing with the `--user` flag:
```bash
pip install tensorflow --user
```
