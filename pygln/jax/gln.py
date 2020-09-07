import jax
from jax import lax, nn as jnn, numpy as jnp, random as jnr, scipy as jsp
from numpy import ndarray
from random import randrange
from typing import Callable, Optional, Sequence, Union

from ..base import GLNBase


jax.config.update("jax_debug_nans", True)
jax.config.update("jax_numpy_rank_promotion", "raise")


class DynamicParameter(object):
    def initialize(self):
        return 0.0

    def value(self, step):
        return step + 1.0


class ConstantParameter(DynamicParameter):
    def __init__(self, constant_value):
        DynamicParameter.__init__(self)

        assert isinstance(constant_value, float)
        self.constant_value = constant_value

    def value(self, step):
        return super().value(step), self.constant_value


class PaperLearningRate(DynamicParameter):
    def value(self, step):
        step = super().value(step)
        return step, jnp.minimum(100.0 / step, 0.01)


class OnlineUpdateModule(object):
    def __init__(self, learning_rate: DynamicParameter, pred_clipping: float,
                 weight_clipping: float):
        assert isinstance(learning_rate, DynamicParameter)
        assert 0.0 < pred_clipping < 1.0
        assert weight_clipping >= 1.0

        self.learning_rate = learning_rate
        self.pred_clipping = pred_clipping
        self.weight_clipping = weight_clipping

    def initialize(self, rng):
        params = dict()
        params['lr_step'] = self.learning_rate.initialize()
        return params

    def predict(self, params, logits, context, target=None):
        raise NotImplementedError()


class Linear(OnlineUpdateModule):
    def __init__(self, size: int, input_size: int, context_size: int,
                 context_map_size: int, num_classes: int,
                 learning_rate: DynamicParameter, pred_clipping: float,
                 weight_clipping: float, bias: bool, context_bias: bool):
        super().__init__(learning_rate, pred_clipping, weight_clipping)

        assert size > 0 and input_size > 0 and context_size > 0
        assert context_map_size >= 1
        assert num_classes >= 2

        self.size = size
        self.input_size = input_size
        self.context_size = context_size
        self.context_map_size = context_map_size
        self.num_classes = num_classes if num_classes > 2 else 1
        self.bias = bias
        self.context_bias = context_bias

    def initialize(self, rng):
        rng, rng1 = jnr.split(key=rng, num=2)
        params = super().initialize(rng=rng1)

        logits_size = self.input_size + int(self.bias)
        num_context_indices = 1 << self.context_map_size
        weights_shape = (self.num_classes, self.size, num_context_indices,
                         logits_size)
        params['weights'] = jnp.full(shape=weights_shape,
                                     fill_value=(1.0 / logits_size))

        if self.bias:
            rng, rng1 = jnr.split(key=rng, num=2)
            bias_shape = (1, self.num_classes, 1)
            params['bias'] = jnr.uniform(
                key=rng1,
                shape=bias_shape,
                minval=jsp.special.logit(self.pred_clipping),
                maxval=jsp.special.logit(1.0 - self.pred_clipping))

        context_maps_shape = (1, self.num_classes, self.size,
                              self.context_map_size, self.context_size)
        if self.context_bias:
            rng1, rng2 = jnr.split(key=rng, num=2)
            context_maps = jnr.normal(key=rng1, shape=context_maps_shape)
            norm = jnp.linalg.norm(context_maps, axis=-1, keepdims=True)
            params['context_maps'] = context_maps / norm

            context_bias_shape = (1, self.num_classes, self.size,
                                  self.context_map_size)
            params['context_bias'] = jnr.normal(key=rng2,
                                                shape=context_bias_shape)

        else:
            params['context_maps'] = jnr.normal(key=rng,
                                                shape=context_maps_shape)

        return params

    def predict(self, params, logits, context, target=None):
        context = jnp.expand_dims(jnp.expand_dims(jnp.expand_dims(context,
                                                                  axis=1),
                                                  axis=1),
                                  axis=1)
        context_bias = params.get('context_bias', 0.0)
        context_index = (params['context_maps'] *
                         context).sum(axis=-1) > context_bias

        context_map_values = jnp.asarray(
            [[[[1 << n for n in range(self.context_map_size)]]]])
        context_index = jnp.where(context_index, context_map_values, 0)
        context_index = context_index.sum(axis=-1, keepdims=True)

        batch_size = logits.shape[0]
        class_neuron_index = jnp.asarray([[[[c, n] for n in range(self.size)]
                                           for c in range(self.num_classes)]])
        class_neuron_index = jnp.tile(class_neuron_index,
                                      reps=(batch_size, 1, 1, 1))
        context_index = jnp.concatenate([class_neuron_index, context_index],
                                        axis=-1)

        dims = lax.GatherDimensionNumbers(offset_dims=(3, ),
                                          collapsed_slice_dims=(0, 1, 2),
                                          start_index_map=(0, 1, 2))
        weights = lax.gather(operand=params['weights'],
                             start_indices=context_index,
                             dimension_numbers=dims,
                             slice_sizes=(1, 1, 1,
                                          self.input_size + int(self.bias)))

        if self.bias:
            bias = jnp.tile(params['bias'], reps=(batch_size, 1, 1))
            logits = jnp.concatenate([logits, bias], axis=-1)
        logits = jnp.expand_dims(logits, axis=-1)

        output_logits = jnp.matmul(weights, logits)
        output_logits = jnp.clip(output_logits,
                                 a_min=jsp.special.logit(self.pred_clipping),
                                 a_max=jsp.special.logit(1.0 -
                                                         self.pred_clipping))

        if target is None:
            return jnp.squeeze(output_logits, axis=-1)

        else:
            logits = jnp.expand_dims(jnp.squeeze(logits, axis=-1), axis=-2)
            output_preds = jnn.sigmoid(output_logits)
            target = jnp.expand_dims(jnp.expand_dims(target, axis=-1), axis=-1)
            params['lr_step'], learning_rate = self.learning_rate.value(
                params['lr_step'])
            delta = learning_rate * (target - output_preds) * logits

            dims = lax.ScatterDimensionNumbers(
                update_window_dims=(3, ),
                inserted_window_dims=(0, 1, 2),
                scatter_dims_to_operand_dims=(0, 1, 2))

            if self.weight_clipping is None:
                params['weights'] = lax.scatter_add(
                    operand=params['weights'],
                    scatter_indices=context_index,
                    updates=delta,
                    dimension_numbers=dims)
            else:
                weights = jnp.clip(weights + delta,
                                   a_min=-self.weight_clipping,
                                   a_max=self.weight_clipping)
                params['weights'] = lax.scatter(operand=params['weights'],
                                                scatter_indices=context_index,
                                                updates=weights,
                                                dimension_numbers=dims)

            return params, jnp.squeeze(output_logits, axis=-1)


class GLN(GLNBase):
    """
    JAX implementation of Gated Linear Networks (https://arxiv.org/abs/1910.01526).

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
        seed (int): Random seed.
    """
    def __init__(self,
                 layer_sizes: Sequence[int],
                 input_size: int,
                 num_classes: int = 2,
                 context_map_size: int = 4,
                 bias: bool = True,
                 context_bias: bool = False,
                 base_predictor: Optional[Callable[[ndarray], ndarray]] = None,
                 learning_rate: Union[float, DynamicParameter] = 1e-3,
                 pred_clipping: float = 1e-3,
                 weight_clipping: float = 5.0,
                 seed: Optional[int] = None):

        super().__init__(layer_sizes, input_size, num_classes,
                         context_map_size, bias, context_bias, base_predictor,
                         learning_rate, pred_clipping, weight_clipping)

        # Learning rate as dynamic parameter
        if self.learning_rate == 'paper':
            self.learning_rate = PaperLearningRate()
        else:
            self.learning_rate = ConstantParameter(self.learning_rate)

        # Random seed
        if seed is None:
            self.seed = randrange(1000000)
        else:
            self.seed = seed
        self.params = dict()
        self.params['rng'] = jnr.PRNGKey(seed=self.seed)

        # Initialize layers
        self.layers = list()
        self.params['rng'], *rngs = jnr.split(key=self.params['rng'],
                                              num=(len(self.layer_sizes) + 2))
        previous_size = self.base_pred_size
        for n, (size, rng) in enumerate(zip(self.layer_sizes + (1,), rngs)):
            layer = Linear(size=size,
                           input_size=previous_size,
                           context_size=self.input_size,
                           context_map_size=self.context_map_size,
                           num_classes=self.num_classes,
                           learning_rate=self.learning_rate,
                           pred_clipping=self.pred_clipping,
                           weight_clipping=self.weight_clipping,
                           bias=self.bias,
                           context_bias=self.context_bias)
            self.layers.append(layer)
            self.params[f'layer{n}'] = layer.initialize(rng=rng)
            previous_size = size

        # JAX-compiled predict function
        self._jax_predict = jax.jit(fun=self._predict, static_argnums=(3,))

        # JAX-compiled update function
        self._jax_update = jax.jit(fun=self._predict, static_argnums=(3,))

    def predict(
        self, input: ndarray, target: Optional[ndarray] = None, return_probs: bool = False
    ) -> ndarray:
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

        # Base predictions
        base_preds = self.base_predictor(input)
        base_preds = jnp.asarray(base_preds, dtype=jnp.float32)

        # Context
        context = jnp.asarray(input, dtype=jnp.float32)

        if target is None:
            # Predict without update
            prediction = self._jax_predict(self.params, base_preds, context,
                                           return_probs)

        else:
            target = jnp.asarray(target, dtype=jnp.int32)

            # Predict with update
            self.params, prediction = self._jax_update(self.params,
                                                       base_preds,
                                                       input,
                                                       return_probs,
                                                       target=target)

        return prediction

    def _predict(self, params, base_preds, context, return_probs, target=None):
        # Base logits
        base_preds = jnp.clip(base_preds,
                              a_min=self.pred_clipping,
                              a_max=(1.0 - self.pred_clipping))
        logits = jsp.special.logit(base_preds)
        logits = jnp.expand_dims(logits, axis=1)
        if self.num_classes == 2:
            logits = jnp.tile(logits, reps=(1, 1, 1))
        else:
            logits = jnp.tile(logits, reps=(1, self.num_classes, 1))

        # Turn target class into one-hot
        if target is not None:
            target = jnn.one_hot(target, num_classes=self.num_classes)
            if self.num_classes == 2:
                target = target[:, 1:]

        # Layers
        if target is None:
            for n, layer in enumerate(self.layers):
                logits = layer.predict(params=params[f'layer{n}'],
                                       logits=logits,
                                       context=context)
        else:
            for n, layer in enumerate(self.layers):
                params[f'layer{n}'], logits = layer.predict(
                    params=params[f'layer{n}'],
                    logits=logits,
                    context=context,
                    target=target)

        logits = jnp.squeeze(logits, axis=-1)
        if self.num_classes == 2:
            logits = jnp.squeeze(logits, axis=1)

        # Output prediction
        if return_probs:
            prediction = jnn.sigmoid(logits)
        elif self.num_classes == 2:
            prediction = logits > 0.0
        else:
            prediction = jnp.argmax(logits, axis=1)

        if target is None:
            return prediction
        else:
            return params, prediction
