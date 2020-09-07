import numpy as np
import torch
from scipy.special import logit as slogit
from torch import nn
from typing import Callable, Optional, Sequence, Union

from ..base import GLNBase


class DynamicParameter(nn.Module):
    def __init__(self, name: Optional[str] = None):
        super().__init__()
        self.step = 0
        self.name = name

    @property
    def value(self):
        self.step += 1
        return self.step


class ConstantParameter(DynamicParameter):
    def __init__(self, constant_value: float, name: Optional[str] = None):
        super().__init__(name)

        assert isinstance(constant_value, float)
        self.constant_value = constant_value

    @property
    def value(self):
        return self.constant_value


class PaperLearningRate(DynamicParameter):
    def __init__(self):
        DynamicParameter.__init__(self, 'paper_learning_rate')

    @property
    def value(self):
        return min(100 / super().value, 0.01)


class Linear(nn.Module):
    def __init__(self,
                 size: int,
                 input_size: int,
                 context_size: int,
                 context_map_size: int,
                 num_classes: int,
                 learning_rate: DynamicParameter,
                 pred_clipping: float,
                 weight_clipping: float,
                 bias: bool,
                 context_bias: bool):
        super().__init__()

        assert size > 0 and input_size > 0 and context_size > 0
        assert context_map_size >= 0
        assert num_classes >= 2

        self.context_map_size = context_map_size
        self.num_classes = num_classes if num_classes > 2 else 1
        self.learning_rate = learning_rate
        # clipping value for predictions
        self.pred_clipping = pred_clipping
        # clipping value for weights of layer
        self.weight_clipping = weight_clipping

        if bias and size > 1:
            self.bias = torch.empty(
                (1, 1,
                 self.num_classes)).uniform_(slogit(self.pred_clipping),
                                             slogit(1 - self.pred_clipping))
            self.size = size - 1
        else:
            self.bias = None
            self.size = size

        if context_map_size > 0:
            self._context_maps = torch.as_tensor(
                np.random.normal(size=(self.num_classes, self.size,
                                       context_map_size, context_size)),
                dtype=torch.float32)

        # constant values for halfspace gating
        if context_map_size == 0:
            pass
        elif context_bias:
            context_bias_shape = (self.num_classes, self.size,
                                  context_map_size, 1)
            self._context_bias = torch.tensor(
                np.random.normal(size=context_bias_shape), dtype=torch.float32)
            self._context_maps /= torch.norm(self._context_maps,
                                             dim=-1,
                                             keepdim=True)
        else:
            self._context_bias = torch.tensor(0.0)

        self.bias = nn.Parameter(self.bias, requires_grad=False)

        if context_map_size > 0:
            self._context_maps = nn.Parameter(self._context_maps,
                                              requires_grad=False)
            self._context_bias = nn.Parameter(self._context_bias,
                                              requires_grad=False)

            # array to convert mapped_context_binary context to index
            self._boolean_converter = nn.Parameter(torch.as_tensor(
                np.array([[2**i] for i in range(context_map_size)])),
                                                   requires_grad=False)

        # weights for the whole layer
        weights_shape = (self.num_classes, self.size, 2**context_map_size,
                         input_size)
        self._weights = nn.Parameter(torch.full(size=weights_shape,
                                                fill_value=1 / input_size,
                                                dtype=torch.float32),
                                     requires_grad=False)

    def predict(self, logit, context, target=None):
        if self.context_map_size > 0:
            # project side information and determine context index
            distances = torch.matmul(self._context_maps, context.T)
            mapped_context_binary = (distances > self._context_bias).int()
            current_context_indices = torch.sum(mapped_context_binary *
                                                self._boolean_converter,
                                                dim=-2)
        else:
            current_context_indices = torch.zeros(
                self.num_classes, self.size, 1, dtype=torch.int64
            )

        # select all context across all neurons in layer
        current_selected_weights = self._weights[
            torch.arange(self.num_classes).reshape(-1, 1, 1),
            torch.arange(self.size).reshape(1, -1, 1
                                            ), current_context_indices, :]

        if logit.ndim == 2:
            logit = torch.unsqueeze(logit, dim=-1)

        output_logits = torch.clamp(torch.matmul(
            current_selected_weights,
            torch.unsqueeze(logit.T, dim=-3)).diagonal(dim1=-2, dim2=-1),
                                    min=slogit(self.pred_clipping),
                                    max=slogit(1 - self.pred_clipping)).T

        if target is not None:
            sigmoids = torch.sigmoid(output_logits)
            # compute update
            diff = sigmoids - torch.unsqueeze(target, dim=1)
            update_values = self.learning_rate.value * torch.unsqueeze(
                diff, dim=-1) * torch.unsqueeze(logit.permute(0, 2, 1), dim=1)
            self._weights[torch.arange(self.num_classes).reshape(-1, 1, 1),
                          torch.arange(self.size).reshape(1, -1, 1),
                          current_context_indices, :] = torch.clamp(
                              current_selected_weights -
                              update_values.permute(2, 1, 0, 3),
                              -self.weight_clipping, self.weight_clipping)

        if self.bias is not None:
            bias_append = torch.cat([self.bias] * output_logits.shape[0],
                                    dim=0)
            output_logits = torch.cat([bias_append, output_logits], dim=1)

        return output_logits

    def extra_repr(self):
        return 'input_size={}, neurons={}, context_map_size={}, bias={}'.format(
            self._weights.size(3), self._weights.size(1),
            self._context_maps.size(2), self.bias)


class GLN(nn.Module, GLNBase):
    """
    PyTorch implementation of Gated Linear Networks (https://arxiv.org/abs/1910.01526).

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
                 base_predictor: Optional[
                     Callable[[np.ndarray], np.ndarray]] = None,
                 learning_rate: Union[float, DynamicParameter] = 1e-3,
                 pred_clipping: float = 1e-3,
                 weight_clipping: float = 5.0):

        nn.Module.__init__(self)
        GLNBase.__init__(self, layer_sizes, input_size, num_classes,
                         context_map_size, bias, context_bias, base_predictor,
                         learning_rate, pred_clipping, weight_clipping)

        # Initialize layers
        self.layers = nn.ModuleList()
        previous_size = self.base_pred_size

        if isinstance(learning_rate, float):
            self.learning_rate = ConstantParameter(learning_rate,
                                                   'learning_rate')
        elif isinstance(learning_rate, DynamicParameter):
            self.learning_rate = learning_rate
        else:
            raise ValueError('Invalid learning rate')

        if bias:
            self.base_bias = np.random.uniform(low=slogit(pred_clipping),
                                               high=slogit(1 - pred_clipping))
        for size in (self.layer_sizes + (1,)):
            layer = Linear(size, previous_size, self.input_size,
                           self.context_map_size, self.num_classes,
                           self.learning_rate, self.pred_clipping,
                           self.weight_clipping, self.bias, self.context_bias)
            self.layers.append(layer)
            previous_size = size

        if torch.cuda.is_available():
            self.cuda()

    def predict(self,
                input: np.ndarray,
                target: Optional[np.ndarray] = None,
                return_probs: bool = False) -> np.ndarray:
        """
        Predict the class for the given inputs, and optionally update the weights.

        Args:
            input (np.array[B, N]): Batch of B N-dim float input vectors.
            target (np.array[B]): Optional batch of B target class labels (bool, or int if
                num_classes given) which, if given, triggers an online update if given.
            return_probs (bool): Whether to return the classification probability (for each
                one-vs-all classifier if num_classes given) instead of the class.

        Returns:
            Predicted class per input instance (bool, or int if num_classes given),
            or classification probabilities if return_probs set.
        """
        if input.ndim == 1:
            input = torch.unsqueeze(input, dim=0)

        # Base predictions
        base_preds = self.base_predictor(input)

        # Default data transform
        input = torch.tensor(input, dtype=torch.float32)
        base_preds = torch.tensor(base_preds, dtype=torch.float32)
        if target is not None:
            target = torch.tensor(target)
        if torch.cuda.is_available():
            input = input.cuda()
            base_preds = base_preds.cuda()
            if target is not None:
                target = target.cuda()

        # Context
        context = input

        # Target
        if target is not None:
            target = nn.functional.one_hot(target.long(), self.num_classes)
            if self.num_classes == 2:
                target = target[:, 1:]

        # Base logits
        base_preds = torch.clamp(base_preds,
                                 min=self.pred_clipping,
                                 max=(1.0 - self.pred_clipping))
        logits = torch.log(base_preds / (1.0 - base_preds))
        if self.bias:
            logits[:, 0] = self.base_bias

        # Layers
        for layer in self.layers:
            logits = layer.predict(logit=logits,
                                   context=context,
                                   target=target)

        logits = torch.squeeze(logits, dim=1)
        if self.num_classes == 2:
            logits = logits.squeeze(dim=1)

        if return_probs:
            output = torch.sigmoid(logits)
        elif self.num_classes == 2:
            output = logits > 0
        else:
            output = torch.argmax(logits, dim=1)

        if torch.cuda.is_available():
            output = output.cpu()
        return output.numpy()

    def extra_repr(self):
        return 'num_classes={}, num_layers={}'.format(self.num_classes,
                                                      len(self.layers))
