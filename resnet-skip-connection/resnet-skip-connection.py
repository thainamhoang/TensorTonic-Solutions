import numpy as np

import numpy as np

def compute_gradient_with_skip(gradients_F: list, x: np.ndarray) -> np.ndarray:
    """
    Compute gradient flow through L layers WITH skip connections.
    Gradient at layer l = sum of paths through network
    """
    g = np.array(x, dtype=float).flatten()  # ensure 1D
    for J in reversed(gradients_F):
        if np.isscalar(J) or (isinstance(J, np.ndarray) and J.ndim == 0):
            g = (1.0 + float(J)) * g
        else:
            J = np.asarray(J, dtype=float)
            g = (np.eye(J.shape[0]) + J) @ g
    return g


def compute_gradient_without_skip(gradients_F: list, x: np.ndarray) -> np.ndarray:
    """
    Compute gradient flow through L layers WITHOUT skip connections.
    """
    g = np.array(x, dtype=float).flatten()  # ensure 1D
    for J in reversed(gradients_F):
        if np.isscalar(J) or (isinstance(J, np.ndarray) and J.ndim == 0):
            g = float(J) * g
        else:
            g = np.asarray(J, dtype=float) @ g
    return g
