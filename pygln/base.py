from abc import ABC, abstractmethod
import numpy as np
from typing import Callable, Optional, Sequence


class OnlineUpdateModel(ABC):
    """Base class for online-update models, shared by all backend implementations."""
    @abstractmethod
    def predict(self,
                input: np.ndarray,
                target: np.ndarray = None,
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
        layer_sizes (list[int >= 1]): List of layer output sizes.
        input_size (int >= 1): Input vector size.
        context_map_size (int >= 1): Context dimension, i.e. number of context halfspaces.
        num_classes (int >= 2): For values >2, turns GLN into a multi-class classifier by internally
            creating N one-vs-all binary GLN classifiers and return the argmax as output class.
        base_predictor (np.array[n] -> np.array[k]): If given, maps the n-dim input vector to a
            corresponding k-dim vector of base predictions (could be a constant prior), instead of
            simply using the clipped input vector itself.
        learning_rate (float > 0.0): Update learning rate.
        pred_clipping (0.0 < float < 0.5): Clip predictions into [p, 1 - p] at each layer.
        weight_clipping (float > 0.0): Clip weights into [-w, w] after each update.
        bias (bool): Whether to add a bias prediction in each layer.
        context_bias (bool): Whether to use a random non-zero bias for context halfspace gating.
    """
    def __init__(self,
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
        super().__init__()

        assert len(layer_sizes) > 0 and layer_sizes[-1] == 1
        self.layer_sizes = tuple(layer_sizes)

        assert input_size > 0
        self.input_size = input_size

        assert context_map_size >= 2
        self.context_map_size = context_map_size

        assert num_classes >= 2
        self.num_classes = num_classes

        if base_predictor is None:
            self.base_predictor = (lambda x: x)
            self.base_pred_size = self.input_size
        else:
            self.base_predictor = base_predictor
            dummy_input = np.zeros(shape=(1, input_size))
            dummy_pred = self.base_predictor(dummy_input)
            assert dummy_pred.dtype in (np.float32, np.float64)
            assert dummy_pred.ndim == 2 and dummy_pred.shape[0] == 1
            self.base_pred_size = dummy_pred.shape[1]

        if isinstance(learning_rate, float):
            assert learning_rate > 0.0
        self.learning_rate = learning_rate

        assert 0.0 < pred_clipping < 1.0
        self.pred_clipping = pred_clipping

        assert weight_clipping >= 1.0
        self.weight_clipping = weight_clipping

        self.bias = bias
        self.context_bias = context_bias
