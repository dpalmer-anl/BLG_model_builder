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
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=2, num_layers=1):
        super().__init__()
        if num_layers < 1:
            raise ValueError("Num_layers must be at least 1")

        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]

        # Additional hidden layers with size (hidden_dim, hidden_dim)
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, output_dim))
        self.mlp = nn.Sequential(*layers)
    
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
    Supports configurable hidden layers with ReLU activation.
    """
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=2, num_layers=1):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        if self.num_layers < 1:
            raise ValueError("Num_layers must be at least 1")
        
        # Initialization (He initialization for ReLU)
        self.weights = []
        self.biases = []

        # First layer maps input_dim -> hidden_dim
        self.weights.append(np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim))
        self.biases.append(np.zeros((1, hidden_dim)))

        # Additional hidden layers (hidden_dim -> hidden_dim)
        for _ in range(self.num_layers - 1):
            self.weights.append(np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim))
            self.biases.append(np.zeros((1, hidden_dim)))

        # Output layer
        self.W_out = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.b_out = np.zeros((1, output_dim))

        # Keep shapes for parameter reconstruction
        self.weight_shapes = [w.shape for w in self.weights]
        self.bias_shapes = [b.shape for b in self.biases]

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

        self.activations = [descriptors]
        self.zs = []

        # Hidden layers
        a = descriptors
        for W, b in zip(self.weights, self.biases):
            z = np.dot(a, W) + b
            a = self.relu(z)
            self.zs.append(z)
            self.activations.append(a)

        # Output layer (no activation)
        self.z_out = np.dot(a, self.W_out) + self.b_out

        return self.z_out

    def get_parameters(self):
        """Returns all parameters as a single 1D vector."""
        params = []
        for W, b in zip(self.weights, self.biases):
            params.append(W.ravel())
            params.append(b.ravel())
        params.append(self.W_out.ravel())
        params.append(self.b_out.ravel())
        return np.concatenate(params)

    def set_parameters(self, parameters):
        """Reconstructs matrices from a 1D vector."""
        idx = 0

        for i, shape in enumerate(self.weight_shapes):
            size = np.prod(shape)
            self.weights[i] = parameters[idx:idx + size].reshape(shape)
            idx += size

            size = np.prod(self.bias_shapes[i])
            self.biases[i] = parameters[idx:idx + size].reshape(self.bias_shapes[i])
            idx += size

        # Output layer
        size = self.hidden_dim * self.output_dim
        self.W_out = parameters[idx:idx + size].reshape(self.hidden_dim, self.output_dim)
        idx += size
        size = self.output_dim
        self.b_out = parameters[idx:idx + size].reshape(1, self.output_dim)

    def backward(self, descriptors, targets, output):
        """
        Manual backprop for configurable hidden-layer MLP.
        Assumes Mean Squared Error Loss.
        """
        if descriptors.ndim == 1:
            descriptors = descriptors[:, np.newaxis]

        m = descriptors.shape[0]  # Batch size

        # --- Output Layer Gradient ---
        dz = (2.0 / m) * (output - targets)  # dLoss/dz_out
        dW_out = np.dot(self.activations[-1].T, dz)
        db_out = np.sum(dz, axis=0, keepdims=True)

        dWs = []
        dBs = []

        # Backprop through hidden layers
        da = np.dot(dz, self.W_out.T)
        for i in reversed(range(len(self.weights))):
            z = self.zs[i]
            dz_hidden = da * (z > 0)  # ReLU derivative
            a_prev = self.activations[i]

            dW = np.dot(a_prev.T, dz_hidden)
            db = np.sum(dz_hidden, axis=0, keepdims=True)

            dWs.insert(0, dW)
            dBs.insert(0, db)

            da = np.dot(dz_hidden, self.weights[i].T)

        return dWs, dBs, dW_out, db_out

    def backward(self, descriptors):
        """
        Compute the Jacobian of the network outputs with respect to the inputs.
        Returns an array with shape (N, output_dim, input_dim) where
        jac[n, o, d] = d output_o / d descriptor_d for sample n.
        """
        # Ensure caches are populated for the provided descriptors
        _ = self.forward(descriptors)

        batch = descriptors.shape[0] if descriptors.ndim > 1 else 1
        jac = np.zeros((batch, self.output_dim, self.input_dim))

        for n in range(batch):
            # Start from output layer: W_out^T has shape (output_dim, hidden_dim)
            g = self.W_out.T.copy()  # (output_dim, hidden_dim)

            # Propagate through hidden layers in reverse
            for i in reversed(range(len(self.weights))):
                z = self.zs[i][n]  # pre-activation for sample n at layer i
                relu_grad = (z > 0).astype(float)  # (hidden_dim,)

                # Apply elementwise ReLU derivative to each column of g
                g = g * relu_grad  # broadcast over rows of g

                # Move gradient to previous layer/input
                g = g @ self.weights[i].T  # (output_dim, prev_dim)

            jac[n] = g

        return jac