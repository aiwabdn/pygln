import numpy as np
from typing import Callable, Optional, Sequence


def GLN(backend: str,
        layer_sizes: Sequence[int],
        input_size: int,
        num_classes: int = 2,
        context_map_size: int = 4,
        bias: bool = True,
        context_bias: bool = False,
        base_predictor: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        learning_rate: float = 1e-3,
        pred_clipping: float = 1e-3,
        weight_clipping: float = 5.0):
    """
    Backend wrapper for Gated Linear Network implementations (https://arxiv.org/abs/1910.01526).

    Args:
        backend ("jax", "numpy", "pytorch", "tf"): Which backend implementation to use.
        layer_sizes (list[int >= 1]): List of layer output sizes, excluding last classification
            layer which is added implicitly.
        input_size (int >= 1): Input vector size.
        num_classes (int >= 2): For values >2, turns GLN into a multi-class classifier by internally
            creating a one-vs-all binary GLN classifier per class and return the argmax as output.
        context_map_size (int >= 1): Context dimension, i.e. number of context halfspaces.
        bias (bool): Whether to add a bias prediction in each layer.
        context_bias (bool): Whether to use a random non-zero bias for context halfspace gating.
        base_predictor (np.array[N] -> np.array[K]): If given, maps the N-dim input vector to a
            corresponding K-dim vector of base predictions (could be a constant prior), instead of
            simply using the clipped input vector itself.
        learning_rate (float > 0.0): Update learning rate.
        pred_clipping (0.0 < float < 0.5): Clip predictions into [p, 1 - p] at each layer.
        weight_clipping (float > 0.0): Clip weights into [-w, w] after each update.
    """

    if backend == 'jax':
        from pygln.jax import GLN
    elif backend in ['numpy', 'np']:
        from pygln.numpy import GLN
    elif backend in ['pytorch', 'torch']:
        from pygln.pytorch import GLN
    elif backend in ['tensorflow', 'tf']:
        from pygln.tf import GLN
    else:
        raise NotImplementedError(f"No implementation for backend {backend}.")

    return GLN(layer_sizes, input_size, num_classes, context_map_size, bias,
               context_bias, base_predictor, learning_rate, pred_clipping,
               weight_clipping)
