import warnings

from typing import Optional

import numpy as np
import scipy.stats
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from compressai._CXX import pmf_to_quantized_cdf as _pmf_to_quantized_cdf
from ops.parametrizers import LowerBound



class _EntropyCoder:
    """Proxy class to an actual entropy coder class."""

    def __init__(self, method):
        if not isinstance(method, str):
            raise ValueError(f'Invalid method type "{type(method)}"')

        from compressai import available_entropy_coders

        if method not in available_entropy_coders():
            methods = ", ".join(available_entropy_coders())
            raise ValueError(
                f'Unknown entropy coder "{method}"' f" (available: {methods})"
            )

        if method == "ans":
            from compressai import ans  # pylint: disable=E0611

            encoder = ans.RansEncoder()
            decoder = ans.RansDecoder()
        elif method == "rangecoder":
            import range_coder  # pylint: disable=E0401

            encoder = range_coder.RangeEncoder()
            decoder = range_coder.RangeDecoder()

        self._encoder = encoder
        self._decoder = decoder

    def encode_with_indexes(self, *args, **kwargs):
        return self._encoder.encode_with_indexes(*args, **kwargs)

    def decode_with_indexes(self, *args, **kwargs):
        return self._decoder.decode_with_indexes(*args, **kwargs)


def default_entropy_coder():
    from compressai import get_entropy_coder

    return get_entropy_coder()


def pmf_to_quantized_cdf(pmf, precision=16):
    cdf = _pmf_to_quantized_cdf(pmf.tolist(), precision)
    cdf = torch.IntTensor(cdf)
    return cdf


class EntropyModel(nn.Module):
    r"""Entropy model base class.

    Args:
        likelihood_bound (float): minimum likelihood bound
        entropy_coder (str, optional): set the entropy coder to use, use default
            one if None
        entropy_coder_precision (int): set the entropy coder precision
    """

    def __init__(
        self, likelihood_bound=1e-9, entropy_coder=None, entropy_coder_precision=16
    ):
        super().__init__()

        if entropy_coder is None:
            entropy_coder = default_entropy_coder()
        self.entropy_coder = _EntropyCoder(entropy_coder)
        self.entropy_coder_precision = int(entropy_coder_precision)

        self.use_likelihood_bound = likelihood_bound > 0
        if self.use_likelihood_bound:
            self.likelihood_lower_bound = LowerBound(likelihood_bound)

        # to be filled on update()
        self.register_buffer("_offset", torch.IntTensor())
        self.register_buffer("_quantized_cdf", torch.IntTensor())
        self.register_buffer("_cdf_length", torch.IntTensor())

    @property
    def offset(self):
        return self._offset

    @property
    def quantized_cdf(self):
        return self._quantized_cdf

    @property
    def cdf_length(self):
        return self._cdf_length

    def forward(self, *args):
        raise NotImplementedError()

    @torch.jit.unused
    def _get_noise_cached(self, x):
        # use simple caching method to avoid creating a new tensor every call
        half = float(0.5)
        if not hasattr(self, "_noise"):
            setattr(self, "_noise", x.new(x.size()))
        self._noise.resize_(x.size())
        self._noise.uniform_(-half, half)
        return self._noise

    def quantize(
        self, inputs: Tensor, mode: str, means: Optional[Tensor] = None
    ) -> Tensor:
        if mode not in ("noise", "dequantize", "symbols"):
            raise ValueError(f'Invalid quantization mode: "{mode}"')

        if mode == "noise":
            if torch.jit.is_scripting():
                half = float(0.5)
                noise = torch.empty_like(inputs).uniform_(-half, half)  # (C, 1, HWB)
            else:
                noise = self._get_noise_cached(inputs)
            inputs = inputs + noise
            return inputs

        outputs = inputs.clone()
        if means is not None:
            outputs -= means

        # outputs = torch.round(outputs)  # gradient becomes 0
        # preserve the gradient
        with torch.no_grad():
            tmp = torch.round(outputs) - outputs
        outputs = tmp + outputs

        if mode == "dequantize":
            if means is not None:
                outputs += means
            return outputs

        assert mode == "symbols", mode
        outputs = outputs.int()
        return outputs

    def _quantize(self, inputs, mode, means=None):
        warnings.warn("_quantize is deprecated. Use quantize instead.")
        return self.quantize(inputs, mode, means)

    @staticmethod
    def dequantize(inputs, means=None):
        if means is not None:
            outputs = inputs.type_as(means)
            outputs += means
        else:
            outputs = inputs.float()
        return outputs

    @classmethod
    def _dequantize(cls, inputs, means=None):
        warnings.warn("_dequantize. Use dequantize instead.")
        return cls.dequantize(inputs, means)

    def _pmf_to_cdf(self, pmf, tail_mass, pmf_length, max_length):
        cdf = torch.zeros((len(pmf_length), max_length + 2), dtype=torch.int32, device=pmf.device)
        for i, p in enumerate(pmf):
            prob = torch.cat((p[: pmf_length[i]], tail_mass[i]), dim=0)
            _cdf = pmf_to_quantized_cdf(prob, self.entropy_coder_precision)
            cdf[i, : _cdf.size(0)] = _cdf
        return cdf

    def _check_cdf_size(self):
        if self._quantized_cdf.numel() == 0:
            raise ValueError("Uninitialized CDFs. Run update() first")

        if len(self._quantized_cdf.size()) != 2:
            raise ValueError(f"Invalid CDF size {self._quantized_cdf.size()}")

    def _check_offsets_size(self):
        if self._offset.numel() == 0:
            raise ValueError("Uninitialized offsets. Run update() first")

        if len(self._offset.size()) != 1:
            raise ValueError(f"Invalid offsets size {self._offset.size()}")

    def _check_cdf_length(self):
        if self._cdf_length.numel() == 0:
            raise ValueError("Uninitialized CDF lengths. Run update() first")

        if len(self._cdf_length.size()) != 1:
            raise ValueError(f"Invalid offsets size {self._cdf_length.size()}")

    def compress(self, inputs, indexes, means=None):
        """
        Compress input tensors to char strings.

        Args:
            inputs (torch.Tensor): input tensors (B, M, H, W)
            indexes (torch.IntTensor): tensors CDF indexes (B, M, H, W)
            means (torch.Tensor, optional): optional tensor means
        """
        symbols = self.quantize(inputs, "symbols", means)

        if len(inputs.size()) != 4:
            raise ValueError("Invalid `inputs` size. Expected a 4-D tensor.")

        if inputs.size() != indexes.size():
            raise ValueError("`inputs` and `indexes` should have the same size.")

        self._check_cdf_size()
        self._check_cdf_length()
        self._check_offsets_size()

        strings = []
        for i in range(symbols.size(0)):
            rv = self.entropy_coder.encode_with_indexes(
                symbols[i].reshape(-1).int().tolist(),
                indexes[i].reshape(-1).int().tolist(),
                self._quantized_cdf.tolist(),
                self._cdf_length.reshape(-1).int().tolist(),
                self._offset.reshape(-1).int().tolist(),
            )
            strings.append(rv)
        return strings

    def decompress(self, strings, indexes, means=None):
        """
        Decompress char strings to tensors.

        Args:
            strings (str): compressed tensors
            indexes (torch.IntTensor): tensors CDF indexes
            means (torch.Tensor, optional): optional tensor means
        """

        if not isinstance(strings, (tuple, list)):
            raise ValueError("Invalid `strings` parameter type.")

        if not len(strings) == indexes.size(0):
            raise ValueError("Invalid strings or indexes parameters")

        if len(indexes.size()) != 4:
            raise ValueError("Invalid `indexes` size. Expected a 4-D tensor.")

        self._check_cdf_size()
        self._check_cdf_length()
        self._check_offsets_size()

        if means is not None:
            if means.size()[:-2] != indexes.size()[:-2]:
                raise ValueError("Invalid means or indexes parameters")
            if means.size() != indexes.size() and (
                means.size(2) != 1 or means.size(3) != 1
            ):
                raise ValueError("Invalid means parameters")

        cdf = self._quantized_cdf
        outputs = cdf.new(indexes.size())

        for i, s in enumerate(strings):
            values = self.entropy_coder.decode_with_indexes(
                s,
                indexes[i].reshape(-1).int().tolist(),
                cdf.tolist(),
                self._cdf_length.reshape(-1).int().tolist(),
                self._offset.reshape(-1).int().tolist(),
            )
            outputs[i] = torch.Tensor(values).reshape(outputs[i].size())
        outputs = self.dequantize(outputs, means)
        return outputs


class EntropyBottleneck(EntropyModel):
    r"""Entropy bottleneck layer, introduced by J. Ballé, D. Minnen, S. Singh,
    S. J. Hwang, N. Johnston, in `"Variational image compression with a scale
    hyperprior" <https://arxiv.org/abs/1802.01436>`_.

    This is a re-implementation of the entropy bottleneck layer in
    *tensorflow/compression*. See the original paper and the `tensorflow
    documentation
    <https://tensorflow.github.io/compression/docs/entropy_bottleneck.html>`__
    for an introduction.

    <https://github.com/tensorflow/compression/blob/fce05bce0a7255e33767b4e2c810598aa67f6e8a/tensorflow_compression/python/layers/entropy_models.py#L297>
    The layer trains an independent probability density model for each channel, but
    assumes that across all other dimensions, the inputs are i.i.d. (independent
    and identically distributed).

    Because data compression always involves discretization, the outputs of the
    layer are generally only approximations of its inputs. During training,
    discretization is modeled using additive uniform noise to ensure
    differentiability. The entropies computed during training are differential
    entropies. During evaluation, the data is actually quantized, and the
    entropies are discrete (Shannon entropies). To make sure the approximated
    tensor values are good enough for practical purposes, the training phase must
    be used to balance the quality of the approximation with the entropy, by
    adding an entropy term to the training loss. See the example in the package
    documentation to get started.

    Note: the layer always produces exactly one auxiliary loss and one update op,
    which are only significant for compression and decompression. To use the
    compression feature, the auxiliary loss must be minimized during or after
    training. After that, the update op must be executed at least once.

    """

    def __init__(
        self,
        channels,
        *args,
        tail_mass=1e-9,
        init_scale=10,
        filters=(3, 3, 3, 3),
        **kwargs,
    ):
        """Initializer.

        Arguments:
          init_scale: Float. A scaling factor determining the initial width of the
            probability densities. This should be chosen big enough so that the
            range of values of the layer inputs roughly falls within the interval
            [`-init_scale`, `init_scale`] at the beginning of training.

          filters: An iterable of ints, giving the number of filters at each layer
            of the density model. Generally, the more filters and layers, the more
            expressive is the density model in terms of modeling more complicated
            distributions of the layer inputs. For details, refer to the paper
            referenced above. The default is `[3, 3, 3]`, which should be sufficient
            for most practical purposes.
          data_format: Either `'channels_first'` or `'channels_last'` (default).
          **kwargs: Other keyword arguments passed to superclass (`EntropyModel`).

        Creates the variables for the network modeling the densities, creates the
        auxiliary loss estimating the median and tail quantiles of the densities,
        and then uses that to create the probability mass functions and the discrete
        cumulative density functions used by the range coder.
        """
        super().__init__(*args, **kwargs)

        self.channels = int(channels)
        self.filters = tuple(int(f) for f in filters)
        self.init_scale = float(init_scale)
        self.tail_mass = float(tail_mass)

        # Create parameters
        self._biases = nn.ParameterList()
        self._factors = nn.ParameterList()
        self._matrices = nn.ParameterList()

        filters = (1,) + self.filters + (1,)
        scale = self.init_scale ** (1 / (len(self.filters) + 1))
        channels = self.channels

        for i in range(len(self.filters) + 1):  # 5
            init = np.log(np.expm1(1 / scale / filters[i + 1]))

            matrix = torch.Tensor(channels, filters[i + 1], filters[i])
            matrix.data.fill_(init)
            self._matrices.append(nn.Parameter(matrix))

            bias = torch.Tensor(channels, filters[i + 1], 1)
            nn.init.uniform_(bias, -0.5, 0.5)
            self._biases.append(nn.Parameter(bias))

            if i < len(self.filters):
                factor = torch.Tensor(channels, filters[i + 1], 1)
                nn.init.zeros_(factor)
                self._factors.append(nn.Parameter(factor))

        # To figure out what range of the densities to sample, we need to compute
        # the quantiles given by `tail_mass / 2` and `1 - tail_mass / 2`. Since we
        # can't take inverses of the cumulative directly, we make it an optimization
        # problem:
        # `quantiles = argmin(|logit(cumulative) - target|)`
        # where `target` is `logit(tail_mass / 2)` or `logit(1 - tail_mass / 2)`.
        # Taking the logit (inverse of sigmoid) of the cumulative makes the
        # representation of the right target more numerically stable.

        self.quantiles = nn.Parameter(torch.Tensor(channels, 1, 3))
        init = torch.Tensor([-self.init_scale, 0, self.init_scale])
        self.quantiles.data = init.repeat(self.quantiles.size(0), 1, 1)

        # Numerically stable way of computing logits of `tail_mass / 2` and `1 - tail_mass / 2`.
        target = np.log(2 / self.tail_mass - 1)
        # Compute lower and upper tail quantile as well as median.
        self.register_buffer("target", torch.Tensor([-target, 0, target]))

    def _medians(self):
        medians = self.quantiles[:, :, 1:2]
        return medians

    def update(self, force=False):
        # Check if we need to update the bottleneck parameters, the offsets are
        # only computed and stored when the conditonal model is update()'d.
        if self._offset.numel() > 0 and not force:  # pylint: disable=E0203
            return False

        medians = self.quantiles[:, 0, 1]

        minima = medians - self.quantiles[:, 0, 0]
        minima = torch.ceil(minima).int()
        minima = torch.clamp(minima, min=0)

        maxima = self.quantiles[:, 0, 2] - medians
        maxima = torch.ceil(maxima).int()
        maxima = torch.clamp(maxima, min=0)

        self._offset = -minima

        pmf_start = medians - minima
        pmf_length = maxima + minima + 1

        max_length = pmf_length.max()
        device = pmf_start.device
        samples = torch.arange(max_length, device=device)

        samples = samples[None, :] + pmf_start[:, None, None]

        half = float(0.5)

        lower = self._logits_cumulative(samples - half, stop_gradient=True)
        upper = self._logits_cumulative(samples + half, stop_gradient=True)
        sign = -torch.sign(lower + upper)
        pmf = torch.abs(torch.sigmoid(sign * upper) - torch.sigmoid(sign * lower))

        # Compute out-of-range (tail) masses.
        pmf = pmf[:, 0, :]
        tail_mass = torch.sigmoid(lower[:, 0, :1]) + torch.sigmoid(-upper[:, 0, -1:])

        quantized_cdf = self._pmf_to_cdf(pmf, tail_mass, pmf_length, max_length)
        self._quantized_cdf = quantized_cdf
        self._cdf_length = pmf_length + 2
        return True

    def loss(self):
        # update quantiles
        logits = self._logits_cumulative(self.quantiles, stop_gradient=True)
        loss = torch.abs(logits - self.target).sum()
        return loss

    def _logits_cumulative(self, inputs, stop_gradient):
        # TorchScript not yet working (nn.Mmodule indexing not supported)
        """Evaluate logits of the cumulative densities.
        Arguments:
          inputs: The values at which to evaluate the cumulative densities, expected
            to be a `Tensor` of shape `(channels, 1, batch)`.

          stop_gradient: Boolean. Whether to add `tf.stop_gradient` calls so
            that the gradient of the output with respect to the density model
            parameters is disconnected (the gradient with respect to `inputs` is
            left untouched).

        Returns:
          A `Tensor` of the same shape as `inputs`, containing the logits of the
          cumulative densities evaluated at the given inputs.
        """

        logits = inputs
        for i in range(len(self.filters) + 1):
            matrix = self._matrices[i]
            if stop_gradient:
                matrix = matrix.detach()
            logits = torch.matmul(F.softplus(matrix), logits)

            bias = self._biases[i]
            if stop_gradient:
                bias = bias.detach()
            logits += bias

            if i < len(self._factors):
                factor = self._factors[i]
                if stop_gradient:
                    factor = factor.detach()
                logits += torch.tanh(factor) * torch.tanh(logits)
        return logits

    @torch.jit.unused
    def _likelihood(self, inputs):
        half = float(0.5)
        # Evaluate densities.
        # We can use the special rule below to only compute differences in the left
        # tail of the sigmoid. This increases numerical stability: sigmoid(x) is 1
        # for large x, 0 for small x. Subtracting two numbers close to 0 can be done
        # with much higher precision than subtracting two numbers close to 1.
        lower = self._logits_cumulative(inputs - half, stop_gradient=False)  # stop_gradient: False
        upper = self._logits_cumulative(inputs + half, stop_gradient=False)  # stop_gradient: False
        # Flip signs if we can move more towards the left tail of the sigmoid.
        sign = -torch.sign(lower + upper)
        sign = sign.detach()
        likelihood = torch.abs(
            torch.sigmoid(sign * upper) - torch.sigmoid(sign * lower)
        )
        return likelihood

    def forward(self, x):
        # Convert to (channels, ... , batch) format
        x = x.permute(1, 2, 3, 0).contiguous()
        shape = x.size()
        values = x.reshape(x.size(0), 1, -1)

        # Add noise or quantize
        outputs = self.quantize(
            values, "noise" if self.training else "dequantize", self._medians()
        )

        if not torch.jit.is_scripting():
            likelihood = self._likelihood(outputs)
            if self.use_likelihood_bound:
                likelihood = self.likelihood_lower_bound(likelihood)
        else:
            # TorchScript not yet supported
            likelihood = torch.zeros_like(outputs)

        # Convert back to input tensor shape
        outputs = outputs.reshape(shape)
        outputs = outputs.permute(3, 0, 1, 2).contiguous()

        likelihood = likelihood.reshape(shape)
        likelihood = likelihood.permute(3, 0, 1, 2).contiguous()

        return outputs, likelihood

    @staticmethod
    def _build_indexes(size):
        N, C, H, W = size
        indexes = torch.arange(C).view(1, -1, 1, 1)
        indexes = indexes.int()
        return indexes.repeat(N, 1, H, W)

    def compress(self, x):
        # x: (B, M, H, W)
        indexes = self._build_indexes(x.size())
        medians = self._medians().detach().expand(x.size(0), -1, 1, 1)
        return super().compress(x, indexes, medians)

    def decompress(self, strings, size):
        output_size = (len(strings), self._quantized_cdf.size(0), size[0], size[1])
        indexes = self._build_indexes(output_size).to(self._quantized_cdf.device)
        medians = self._medians().detach().expand(len(strings), -1, 1, 1)
        return super().decompress(strings, indexes, medians)


class GaussianConditional(EntropyModel):
    r"""Gaussian conditional layer, introduced by J. Ballé, D. Minnen, S. Singh,
    S. J. Hwang, N. Johnston, in `"Variational image compression with a scale
    hyperprior" <https://arxiv.org/abs/1802.01436>`_.

    This is a re-implementation of the Gaussian conditional layer in
    *tensorflow/compression*. See the `tensorflow documentation
    <https://tensorflow.github.io/compression/docs/api_docs/python/tfc/GaussianConditional.html>`__
    for more information.
    """

    def __init__(self, scale_table, *args, scale_bound=0.11, tail_mass=1e-9, **kwargs):
        super().__init__(*args, **kwargs)

        if not isinstance(scale_table, (type(None), list, tuple)):
            raise ValueError(f'Invalid type for scale_table "{type(scale_table)}"')

        if isinstance(scale_table, (list, tuple)) and len(scale_table) < 1:
            raise ValueError(f'Invalid scale_table length "{len(scale_table)}"')

        if scale_table and (
            scale_table != sorted(scale_table) or any(s <= 0 for s in scale_table)
        ):
            raise ValueError(f'Invalid scale_table "({scale_table})"')

        self.tail_mass = float(tail_mass)
        if scale_bound is None and scale_table:
            self.lower_bound_scale = LowerBound(self.scale_table[0])
        elif scale_bound > 0:
            self.lower_bound_scale = LowerBound(scale_bound)
        else:
            raise ValueError("Invalid parameters")

        self.register_buffer(
            "scale_table",
            self._prepare_scale_table(scale_table) if scale_table else torch.Tensor(),
        )

        self.register_buffer(
            "scale_bound",
            torch.Tensor([float(scale_bound)]) if scale_bound is not None else None,
        )

    @staticmethod
    def _prepare_scale_table(scale_table):
        return torch.Tensor(tuple(float(s) for s in scale_table))

    def _standardized_cumulative(self, inputs):
        # type: (Tensor) -> Tensor
        half = float(0.5)
        const = float(-(2 ** -0.5))  # - 1/sqrt(2)
        # Using the complementary error function maximizes numerical precision.
        return half * torch.erfc(const * inputs)  # cdf of gaussian

    @staticmethod
    def _standardized_quantile(quantile):
        return scipy.stats.norm.ppf(quantile)  # ppf: inverse of cdf

    def update_scale_table(self, scale_table, force=False):
        # Check if we need to update the gaussian conditional parameters, the
        # offsets are only computed and stored when the conditonal model is
        # updated.
        if self._offset.numel() > 0 and not force:
            return False
        device = self.scale_table.device
        self.scale_table = self._prepare_scale_table(scale_table).to(device)
        self.update()
        return True

    def update(self):
        multiplier = -self._standardized_quantile(self.tail_mass / 2)
        pmf_center = torch.ceil(self.scale_table * multiplier).int()
        pmf_length = 2 * pmf_center + 1
        max_length = torch.max(pmf_length).item()

        device = pmf_center.device
        samples = torch.abs(
            torch.arange(max_length, device=device).int() - pmf_center[:, None]
        )
        samples_scale = self.scale_table.unsqueeze(1)
        samples = samples.float()
        samples_scale = samples_scale.float()
        upper = self._standardized_cumulative((0.5 - samples) / samples_scale)
        lower = self._standardized_cumulative((-0.5 - samples) / samples_scale)
        pmf = upper - lower

        tail_mass = 2 * lower[:, :1]

        quantized_cdf = torch.Tensor(len(pmf_length), max_length + 2)
        quantized_cdf = self._pmf_to_cdf(pmf, tail_mass, pmf_length, max_length)
        self._quantized_cdf = quantized_cdf
        self._offset = -pmf_center
        self._cdf_length = pmf_length + 2

    def _likelihood(self, inputs, scales, means=None):
        # type: (Tensor, Tensor, Optional[Tensor]) -> Tensor
        half = float(0.5)

        if means is not None:
            values = inputs - means
        else:
            values = inputs

        scales = self.lower_bound_scale(scales)

        values = torch.abs(values)
        upper = self._standardized_cumulative((half - values) / scales)
        lower = self._standardized_cumulative((-half - values) / scales)
        likelihood = upper - lower

        return likelihood

    def forward(self, inputs, scales, means=None):
        # type: (Tensor, Tensor, Optional[Tensor]) -> Tuple[Tensor, Tensor]
        outputs = self.quantize(
            inputs, "noise" if self.training else "dequantize", means
        )
        likelihood = self._likelihood(outputs, scales, means)
        if self.use_likelihood_bound:
            likelihood = self.likelihood_lower_bound(likelihood)
        return outputs, likelihood

    def build_indexes(self, scales):
        scales = self.lower_bound_scale(scales)
        indexes = scales.new_full(scales.size(), len(self.scale_table) - 1).int()
        for s in self.scale_table[:-1]:
            indexes -= (scales <= s).int()
        return indexes
