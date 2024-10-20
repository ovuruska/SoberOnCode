
# CUDA-Optimized RMSNorm Layer in PyTorch

This repository provides an example of how to utilize CUDA to optimize the **RMSNorm** layer in a deep learning architecture using PyTorch. The project demonstrates how we can
design and implement a custom CUDA kernel to accelerate the computation of the RMSNorm operation by executing it in parallel for each dimension of every sample in the input data.

## Table of Contents
- [Introduction](#introduction)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Performance Benefits](#performance-benefits)

## Introduction
In this project, we implement a CUDA kernel to optimize the **Root Mean Square Normalization (RMSNorm)** layer. This layer is commonly used in 
deep learning models for normalization. Our custom CUDA kernel enables us to perform the normalization operation for each dimension of the input data in parallel,
thus improving computational efficiency when working with large datasets.

The RMSNorm layer performs the following key steps:
1. Calculate the root mean square (RMS) for each data sample and its dimentions.
2. Normalize each sample using the calculated RMS.
3. Execute these operations in parallel for multiple dimensions across multiple data entries.

## Requirements
To run this project, you need the following:
- **NVIDIA GPU** with CUDA support
- **CUDA Toolkit** installed (version 10.1 or later)
- **Python 3.8+**
- **PyTorch 2.x+** with CUDA support

## Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/cuda-optimized-rmsnorm.git
   cd cuda-optimized-rmsnorm
   ```

2. **Set up a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   Ensure that PyTorch is installed with CUDA support. If it is not, you can install the appropriate version from [PyTorch's official site](https://pytorch.org/get-started/locally/).

4. **Compile the CUDA Kernel**:
   The custom CUDA kernel needs to be compiled before use. Use the following command:
   ```bash
   python setup_cuda.py build_ext --inplace
   ```

## Usage
Once the environment is set up, you can run the program using the provided script.

1. **Run the example**:
   ```bash
   python run_rmsnorm.py
   ```

   This script demonstrates how the RMSNorm layer is applied to a sample dataset using the custom CUDA kernel. It compares the performance of the
   CUDA-optimized implementation with the standard PyTorch version.

3. **Modify for your own use**:
   You can modify the code to fit your specific deep learning model by integrating the RMSNorm layer into your architecture.

## How It Works
The code leverages CUDA to parallelize the normalization computation across multiple GPU threads. The key steps involved are:
1. **CUDA Kernel Design**: A custom CUDA kernel is created to perform the RMSNorm operation in parallel. Each thread in the kernel processes one dimension of a data sample,
   leading to significant performance improvements over a CPU or non-parallelized implementation.
   
3. **PyTorch Integration**: The CUDA kernel is seamlessly integrated with PyTorch using the `torch.utils.cpp_extension.load_inline` function to execute the kernel from Python code.

4. **Parallel Execution**: By utilizing the GPU, the RMSNorm operation is distributed across multiple threads, allowing for highly
    efficient processing of large datasets, especially in high-dimensional data.

## Performance Benefits
This CUDA-optimized version of the RMSNorm layer provides several key benefits:
- **Speed**: The CUDA kernel allows for parallel execution, reducing the time required for normalization in high-dimensional data.
- **Scalability**: This implementation can scale with large datasets and models, making it suitable for training and inference in deep learning architectures.
- **Seamless PyTorch Integration**: The implementation integrates smoothly with PyTorch, making it easy to use in any PyTorch-based deep learning project.
