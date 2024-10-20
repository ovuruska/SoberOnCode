
# CUDA-Optimized RMSNorm Layer in PyTorch

In this project, we implement a CUDA kernel to optimize the **Root Mean Square Normalization (RMSNorm)** layer. This layer is commonly used in 
deep learning models for normalization. Our custom CUDA kernel enables us to perform the normalization operation for each dimension of the input data in parallel,
thus improving computational efficiency when working with large datasets.

The RMSNorm layer performs the following key steps:
1. Calculate the root mean square (RMS) for each data sample and its dimentions.
2. Normalize each sample using the calculated RMS.
3. Execute these operations in parallel for multiple dimensions across multiple data entries.
