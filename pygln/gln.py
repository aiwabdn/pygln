from typing import Callable, Optional, Sequence


def GLN(backend: str,
        layer_sizes: Sequence[int],
        input_size: int,
        context_map_size: int = 4,
        num_classes: int = 2,
        base_predictor: Optional[Callable] = None,
        learning_rate: float = 1e-4,
        pred_clipping: float = 1e-3,
        weight_clipping: float = 5.0,
        bias: bool = True,
        context_bias: bool = True):
    """
    Backend wrapper for Gated Linear Network implementations (https://arxiv.org/abs/1910.01526).

    Args:
        backend ("jax", "numpy", "pytorch", "tf"): Which backend implementation to use.
        layer_sizes (list[int >= 1]): List of layer output sizes.
        input_size (int >= 1): Input vector size.
        context_map_size (int >= 1): Context dimension, i.e. number of context halfspaces.
        num_classes (int >= 2): For values >2, turns GLN into a multi-class classifier by internally
            creating a one-vs-all binary GLN classifier per class and return the argmax as output.
        base_predictor (np.array[N] -> np.array[K]): If given, maps the N-dim input vector to a
            corresponding K-dim vector of base predictions (could be a constant prior), instead of
            simply using the clipped input vector itself.
        learning_rate (float > 0.0): Update learning rate.
        pred_clipping (0.0 < float < 0.5): Clip predictions into [p, 1 - p] at each layer.
        weight_clipping (float > 0.0): Clip weights into [-w, w] after each update.
        bias (bool): Whether to add a bias prediction in each layer.
        context_bias (bool): Whether to use a random non-zero bias for context halfspace gating.
    """

    if backend == 'jax':
        from pygln.jax import GLN
        return GLN(layer_sizes, input_size, context_map_size, num_classes,
                   base_predictor, learning_rate, pred_clipping,
                   weight_clipping, bias, context_bias)

    elif backend == 'numpy':
        from pygln.numpy import GLN
        return GLN(layer_sizes, input_size, context_map_size, num_classes,
                   base_predictor, learning_rate, pred_clipping,
                   weight_clipping, bias, context_bias)

    elif backend == 'pytorch':
        from pygln.pytorch import GLN
        return GLN(layer_sizes, input_size, context_map_size, num_classes,
                   base_predictor, learning_rate, pred_clipping,
                   weight_clipping, bias, context_bias)

    elif backend == 'tf':
        from pygln.tf import GLN
        return GLN(layer_sizes, input_size, context_map_size, num_classes,
                   base_predictor, learning_rate, pred_clipping,
                   weight_clipping, bias, context_bias)

    else:
        raise NotImplementedError(f"No implementation for backend {backend}.")
