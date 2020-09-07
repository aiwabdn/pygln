from abc import ABC, abstractmethod
import numpy as np
from typing import Callable, Optional, Sequence


class OnlineUpdateModel(ABC):
    """Base class for online-update models, shared by all backend implementations."""

    @abstractmethod
    def predict(self,
                input: np.ndarray,
                target: Optional[np.ndarray] = None,
                return_probs: bool = False) -> np.ndarray:
        """
        Predict the class for the given inputs, and optionally update the weights.

        Args:
            input (np.array[B, N]): Batch of B N-dim float input vectors.
            target (np.array[B]): Optional batch of B bool/int target class labels which, if given,
                triggers an online update if given.
            return_probs (bool): Whether to return the classification probability (for each
                one-vs-all classifier if num_classes given) instead of the class.

        Returns:
            Predicted class per input instance, or classification probabilities if return_probs set.
        """
        raise NotImplementedError()


class GLNBase(OnlineUpdateModel):
    """
    Base class for Gated Linear Network implementations (https://arxiv.org/abs/1910.01526).

    Args:
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
    def __init__(self,
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
        super().__init__()

        assert all(size > 0 for size in layer_sizes)
        self.layer_sizes = tuple(layer_sizes)

        assert input_size > 0
        self.input_size = int(input_size)

        assert num_classes >= 2
        self.num_classes = int(num_classes)

        assert context_map_size >= 0
        self.context_map_size = int(context_map_size)

        self.bias = bool(bias)

        self.context_bias = bool(context_bias)

        if base_predictor is None:
            self.base_predictor = (
                lambda x: (x - x.min(axis=1, keepdims=True)) /
                (x.max(axis=1, keepdims=True) - x.min(axis=1, keepdims=True))
            )
            self.base_pred_size = self.input_size
        else:
            self.base_predictor = base_predictor
            dummy_input = np.zeros(shape=(1, self.input_size))
            dummy_pred = self.base_predictor(dummy_input)
            assert dummy_pred.dtype in (np.float32, np.float64)
            assert dummy_pred.ndim == 2 and dummy_pred.shape[0] == 1
            self.base_pred_size = dummy_pred.shape[1]

        assert not isinstance(learning_rate, float) or learning_rate > 0.0
        self.learning_rate = learning_rate

        assert 0.0 < pred_clipping < 1.0
        self.pred_clipping = float(pred_clipping)

        assert weight_clipping > 0.0
        self.weight_clipping = float(weight_clipping)
