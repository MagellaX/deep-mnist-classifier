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



## GPU Setup Guide

### Prerequisites
- NVIDIA GPU (CUDA-capable)
- NVIDIA Graphics Drivers installed
- Anaconda or Miniconda installed

### Step-by-Step Setup

1. **Create Fresh Environment**
```bash
conda deactivate
conda env remove -n tf    # If it exists
conda create -n tf python=3.9
conda activate tf


# Install NumPy first to avoid version conflicts
pip install numpy==1.23.5

# Install CUDA toolkit and cuDNN
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0

# Install TensorFlow
pip install "tensorflow<2.11"


### Common GPU Setup Issues & Solutions

1. **NumPy Version Conflicts**
   - Error: "Module compiled with NumPy 1.x cannot be run in NumPy 2.x"
   - Solution: Install NumPy 1.23.5 before TensorFlow

2. **CUDA Not Found**
   - Error: "Could not load dynamic library 'cudart64_110.dll'"
   - Solution: Install correct CUDA toolkit version (11.2)

3. **GPU Not Detected**
   - Solution: Make sure NVIDIA drivers are up to date
   - Use `nvidia-smi` to verify GPU is recognized

4. **TensorFlow Version Issues**
   - Use TensorFlow < 2.11 for best compatibility
   - Avoid mixing conda and pip installations


## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/deep-mnist-classifier.git
cd deep-mnist-classifier

# Create and activate environment
conda create -n tf python=3.9
conda activate tf

# Install dependencies
pip install numpy==1.23.5
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
pip install "tensorflow<2.11"

# Optional: Install matplotlib for visualizations
pip install matplotlib
```


## Troubleshooting

### Common Issues

1. **"No module named 'tensorflow'"**
   - Ensure you're in the correct conda environment
   - Verify installation: `pip list | grep tensorflow`

2. **GPU Not Detected**
   - Check NVIDIA drivers: `nvidia-smi`
   - Verify CUDA installation
   - Make sure TensorFlow version matches CUDA version

3. **Memory Errors**
   - Reduce batch size in main.py
   - Close other GPU-intensive applications

### Version Compatibility Matrix

| Component    | Version   | Notes                               |
|-------------|-----------|-------------------------------------|
| Python      | 3.9       | Most stable for TensorFlow          |
| NumPy       | 1.23.5    | Prevents version conflicts          |
| TensorFlow  | <2.11     | Best GPU support for Windows        |
| CUDA        | 11.2      | Matches TensorFlow requirements     |
| cuDNN       | 8.1.0     | Matches CUDA version               |


## License

MIT
