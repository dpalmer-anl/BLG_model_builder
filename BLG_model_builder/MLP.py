import torch
import torch.nn as nn
import numpy as np
try:
    import cupy
    import cupyx as cpx
    if cupy.cuda.is_available():
        np = cupy
        
        gpu_avail = True
    else:
        gpu_avail = False
except:
    gpu_avail = False

class MLP(nn.Module):
    """
    General MLP-based function that predicts a value from a vector of descriptors.
    
    Input: descriptors (vector)
    Output: predicted value
    """
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, descriptors):
        """
        Args:
            descriptors: descriptor values
        Returns:
            output value: 
        """
        if descriptors.dim() == 1:
            descriptors = descriptors.unsqueeze(1)  
        return self.mlp(descriptors)  

class MLP_numpy:
    """
    General MLP-based function reconstructed from PyTorch logic.
    Supports 2 hidden layers with ReLU activation.
    """
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=2):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Initialization (He initialization for ReLU)
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2./input_dim)
        self.b1 = np.zeros((1, hidden_dim))
        
        self.W2 = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2./hidden_dim)
        self.b2 = np.zeros((1, hidden_dim))
        
        self.W3 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2./hidden_dim)
        self.b3 = np.zeros((1, output_dim))

    def relu(self, x):
        return np.maximum(0, x)

    def forward(self, descriptors):
        """
        Args:
            descriptors: (N,) or (N, input_dim) array
        Returns:
            output value: (N, output_dim)
        """
        # Equivalent to descriptors.unsqueeze(1) in PyTorch
        if descriptors.ndim == 1:
            descriptors = descriptors[:, np.newaxis]
        
        # Layer 1
        self.z1 = np.dot(descriptors, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        
        # Layer 2
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.relu(self.z2)
        
        # Layer 3 (Output)
        self.z3 = np.dot(self.a2, self.W3) + self.b3
        
        return self.z3

    def get_parameters(self):
        """Returns all parameters as a single 1D vector."""
        return np.concatenate([
            self.W1.ravel(), self.b1.ravel(),
            self.W2.ravel(), self.b2.ravel(),
            self.W3.ravel(), self.b3.ravel()
        ])

    def set_parameters(self, parameters):
        """Reconstructs matrices from a 1D vector."""
        idx = 0
        
        # W1, b1
        size = self.input_dim * self.hidden_dim
        self.W1 = parameters[idx:idx+size].reshape(self.input_dim, self.hidden_dim)
        idx += size
        size = self.hidden_dim
        self.b1 = parameters[idx:idx+size].reshape(1, self.hidden_dim)
        idx += size
        
        # W2, b2
        size = self.hidden_dim * self.hidden_dim
        self.W2 = parameters[idx:idx+size].reshape(self.hidden_dim, self.hidden_dim)
        idx += size
        size = self.hidden_dim
        self.b2 = parameters[idx:idx+size].reshape(1, self.hidden_dim)
        idx += size
        
        # W3, b3
        size = self.hidden_dim * self.output_dim
        self.W3 = parameters[idx:idx+size].reshape(self.hidden_dim, self.output_dim)
        idx += size
        size = self.output_dim
        self.b3 = parameters[idx:idx+size].reshape(1, self.output_dim)

    def backward(self, descriptors, targets, output):
        """
        Manual backprop for 2-hidden layer MLP.
        Assumes Mean Squared Error Loss.
        """
        m = descriptors.shape[0]  # Batch size
        
        # --- 1. Output Layer Gradient (z3) ---
        # dLoss/dz3 = (dLoss/dOutput) * (dOutput/dz3)
        # For MSE: 2/m * (output - targets)
        dz3 = (2.0 / m) * (output - targets)
        
        # Gradients for W3 and b3
        dW3 = np.dot(self.a2.T, dz3)
        db3 = np.sum(dz3, axis=0, keepdims=True)
        
        # --- 2. Second Hidden Layer Gradient (z2) ---
        # dLoss/da2 = dz3 * W3^T
        da2 = np.dot(dz3, self.W3.T)
        # dLoss/dz2 = da2 * ReLU'(z2)
        dz2 = da2 * (self.z2 > 0) 
        
        dW2 = np.dot(self.a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)
        
        # --- 3. First Hidden Layer Gradient (z1) ---
        # dLoss/da1 = dz2 * W2^T
        da1 = np.dot(dz2, self.W2.T)
        # dLoss/dz1 = da1 * ReLU'(z1)
        dz1 = da1 * (self.z1 > 0)
        
        # descriptors.T is the 'a0' layer
        dW1 = np.dot(descriptors.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)
        
        return dW1, db1, dW2, db2, dW3, db3