import numpy as np

class BatchNorm:
    """Batch Normalization layer."""
    
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        self.eps = eps
        self.momentum = momentum
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Apply batch normalization.
        """
        if training:
            mean = x.mean(axis=0)
            var = x.var(axis=0)

            x_hat = (x - mean) / np.sqrt(var + self.eps)
        else:
            x_hat = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
        return x_hat * self.gamma + self.beta

def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation."""
    return np.maximum(0, x)

def post_activation_block(x: np.ndarray, W1: np.ndarray, W2: np.ndarray, bn1: BatchNorm, bn2: BatchNorm) -> np.ndarray:
    """
    Post-activation ResNet block: Conv -> BN -> ReLU -> Conv -> BN -> ReLU
    Uses x @ W for "convolution" (simplified as linear transform).
    """
    x = relu(bn1.forward(x @ W1))
    x = relu(bn2.forward(x @ W2))
    return x

def pre_activation_block(x: np.ndarray, W1: np.ndarray, W2: np.ndarray, bn1: BatchNorm, bn2: BatchNorm) -> np.ndarray:
    """
    Pre-activation ResNet block: BN -> ReLU -> Conv -> BN -> ReLU -> Conv
    This ordering often works better for very deep networks.
    """
    x = relu(bn1.forward(x)) @ W1
    x = relu(bn2.forward(x)) @ W2
    return x
    