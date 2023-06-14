
import functools
from typing import Any, Dict, List, MutableMapping, Tuple, Union

import os
import subprocess
import time
import numpy as np
import re
import shutil
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import transformers
import deepspeed

from pathlib import Path
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from dataclasses import is_dataclass
from enum import Enum
from accelerate import Accelerator
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR

try:
    from opendelta import (
        AdapterModel,
        BitFitModel,
        LoraModel,
        PrefixModel,
        SoftPromptModel,
    )

    HAS_OPENDELTA = True
except ModuleNotFoundError:
    HAS_OPENDELTA = False

from src.utils.logger import logger


def get_distributed_config(accelerator: Accelerator):
    """
    Return accelerator distributed config
    """

    dist_config = {
        "mixed_precision": accelerator.mixed_precision,
        "num_gpus": accelerator.num_processes,
    }

    if accelerator.state.deepspeed_plugin is not None:
        ds_plugin = accelerator.state.deepspeed_plugin
        dist_config.update(
            {
                "gradient_accumulation_steps": ds_plugin.gradient_accumulation_steps,
                "gradient_clipping": ds_plugin.gradient_clipping,
                "zero_stage": ds_plugin.zero_stage,
                "offload_optimizer_device": ds_plugin.offload_optimizer_device,
                "offload_param_device": ds_plugin.offload_param_device,
            }
        )

    return dist_config


class OptimizerName(str, Enum):
    """Supported optimizer names"""

    ADAM: str = "adam"
    ADAMW: str = "adamw"
    ADAM_8BIT_BNB: str = "adam_8bit_bnb"
    ADAMW_8BIT_BNB: str = "adamw_8bit_bnb"
    SGD: str = "sgd"


def get_optimizer_class(name: OptimizerName):
    """
    Returns the optimizer class with the given name

    Args:
        name (str): Name of the optimizer as found in `OptimizerNames`
    """
    if name == OptimizerName.ADAM:
        return torch.optim.Adam
    if name == OptimizerName.ADAMW:
        return torch.optim.AdamW
    if name == OptimizerName.ADAM_8BIT_BNB.value:
        try:
            from bitsandbytes.optim import Adam8bit

            return Adam8bit
        except ImportError:
            raise ImportError(
                "You must install the `bitsandbytes` package to use the 8-bit Adam. "
                "Install with: `pip install bitsandbytes`"
            )
    if name == OptimizerName.ADAMW_8BIT_BNB.value:
        try:
            from bitsandbytes.optim import AdamW8bit

            return AdamW8bit
        except ImportError:
            raise ImportError(
                "You must install the `bitsandbytes` package to use 8-bit AdamW. "
                "Install with: `pip install bitsandbytes`"
            )
    if name == OptimizerName.SGD.value:
        return torch.optim.SGD
    supported_optimizers = [o.value for o in OptimizerName]
    raise ValueError(f"`{name}` is not a supported optimizer. " f"Supported optimizers are: {supported_optimizers}")


class SchedulerName(str, Enum):
    """Supported scheduler names"""

    COSINE_ANNEALING = "cosine_annealing"
    LINEAR = "linear"


def get_scheduler_class(name: SchedulerName):
    """
    Returns the scheduler class with the given name
    """
    if name == SchedulerName.COSINE_ANNEALING:
        return CosineAnnealingLR
    if name == SchedulerName.LINEAR:
        return LinearLR
    supported_schedulers = [s.value for s in SchedulerName]
    raise ValueError(f"`{name}` is not a supported scheduler. " f"Supported schedulers are: {supported_schedulers}")


class Clock:
    """
    Helper object for keeping track of time for computations.
    """

    def __init__(self):
        self.start = time.time()
        self.total_time = 0
        self.total_samples = 0

    def tick(self, samples: int = 0) -> float:
        """
        Returns time (s) since last call to tick(). Also records samples processed since last call.

        :param samples: number of samples that have been processed since last call
        """
        end = time.time()
        delta = end - self.start
        self.start = end

        if samples != 0:
            self.total_time += delta
            self.total_samples += samples

        return delta

    def get_stat(self, n_samp: int = 1000, reset: bool = False):
        """
        Returns average time (s) per n_samp samples processed

        :param reset: Reset counts?
        """
        sec_per_samp = self.total_time / self.total_samples

        if reset:
            self.total_samples = 0
            self.total_time = 0

        return sec_per_samp * n_samp


def tree_map(f, tree: Any) -> Any:
    """
    Apply function f to all leaves in tree
    """
    if is_dataclass(tree):
        return tree.__class__(**{k: tree_map(f, v) for k, v in tree.__dict__.items()})
    elif isinstance(tree, dict):
        return {k: tree_map(f, v) for k, v in tree.items()}
    elif isinstance(tree, (list, tuple)):
        return tree.__class__(tree_map(f, v) for v in tree)
    else:
        return f(tree)


def to_device(tree, device, non_blocking=False):
    """
    Move all tensors in tree to device
    """
    return tree_map(lambda x: x.to(device, non_blocking=non_blocking), tree)


def filter_non_scalars(xs: Dict) -> Dict:
    """
    Trims everything that can't be casted to float
    """
    ys = {}
    for k, v in xs.items():
        try:
            ys[k] = float(v)
        except TypeError:
            continue

    return ys


def get_git_tag() -> Tuple[str, str]:
    """
    Returns commit's short hash and date
    """
    try:
        output = subprocess.check_output("git log --format='%h/%as' -n1".split())
        branch = subprocess.check_output("git rev-parse --abbrev-ref HEAD".split())
        return branch.decode()[:-1], output.decode()[1:-2]
    except subprocess.CalledProcessError:
        return "unknown", "unknown"


def make_head(n_embd: int, out: int, dtype: type = torch.float32) -> nn.Sequential:
    """Returns a generic sequential MLP head."""
    return nn.Sequential(
        nn.Linear(n_embd, n_embd * 2, dtype=dtype),
        nn.ReLU(),
        nn.Linear(n_embd * 2, out, dtype=dtype),
    )


def freeze_bottom_causal_layers(model: nn.Module, num_layers_unfrozen: int = 0):
    """Freezes the bottom transformer block layers of the specified model."""
    hidden_layers = hf_get_decoder_blocks(model)
    if num_layers_unfrozen == 0:
        hidden_layers_to_freeze = list(hidden_layers)
    elif num_layers_unfrozen > 0:
        hidden_layers_to_freeze = list(hidden_layers)[:-num_layers_unfrozen]
    else:
        hidden_layers_to_freeze = []
    for layer in hidden_layers_to_freeze:
        layer.requires_grad_(False)


def freeze_bottom_seq2seq_layers(model: nn.Module, num_layers_unfrozen: int = 0):
    """Freezes the bottom transformer block layers of the specified model."""
    if num_layers_unfrozen == -1:
        return
    shared_embed = model.shared
    decoder_embed = model.decoder.embed_tokens
    encoder_blocks = model.encoder.block
    encoder_norm_layer = model.encoder.final_layer_norm
    decoder_norm_layer = model.decoder.final_layer_norm
    decoder_blocks = model.decoder.block[:-num_layers_unfrozen]
    blocks_to_freeze = (
        list(encoder_blocks)
        + list(decoder_blocks)
        + [shared_embed]
        + [encoder_norm_layer]
        + [decoder_norm_layer]
        + [decoder_embed]
    )
    for block in blocks_to_freeze:
        block.requires_grad_(False)


def rhasattr(obj, attr):
    """A chain-able attribute version of hasattr. For example, to check if
    `obj` has the attribute `foo.bar.baz`, you can use:
        `rhasattr(obj, "foo.bar.baz")`
    Reference: https://stackoverflow.com/a/67303315
    """
    _nested_attrs = attr.split(".")
    _curr_obj = obj
    for _a in _nested_attrs[:-1]:
        if hasattr(_curr_obj, _a):
            _curr_obj = getattr(_curr_obj, _a)
        else:
            return False
    return hasattr(_curr_obj, _nested_attrs[-1])


def rgetattr(obj, attr: str, *args) -> object:
    """A chain-able attribute version of getattr. For example, to get the
    attribute `foo.bar.baz` from `obj`, you can use:
        `rgetattr(obj, "foo.bar.baz")`
    Reference: https://stackoverflow.com/a/31174427
    """

    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def findattr(obj, attrs: Tuple[str]) -> Union[object, None]:
    for attr in attrs:
        if rhasattr(obj, attr):
            return rgetattr(obj, attr)
    raise ValueError(f"Could not find an attribute from `{attrs}` in `{obj}`")


def hf_get_decoder(model: nn.Module) -> nn.Module:
    """Returns the causal decoder backbone of the specified HuggingFace transformers
    model.
    NOTE: Different model configurations have different causal decoder attribute
    names.
        - transformer: (GPT2LMHeadModel, GPTJConfig)
        - model.decoder: (OPTConfig, BloomConfig)
        - gpt_neox: (GPTNeoXConfig)
    """
    decoder_attrs = ("transformer", "model.decoder", "gpt_neox", "decoder")
    return findattr(model, decoder_attrs)


def hf_get_decoder_final_norm(model: nn.Module) -> float:
    """Returns the final (layer) norm of the specified decoder.
    NOTE: Different model configurations have different final norm attribute names.
        - transformer.ln_f: (GPT2LMHeadModel, GPTJForCausalLM)
        - model.decoder.final_layer_norm: (OPTForCausalLM)
        - gpt_neox.layers.final_layer_norm: (GPTNeoXForCausalLM)
    """
    norm_attrs = (
        "transformer.ln_f",
        "model.decoder.final_layer_norm",
        "decoder.final_layer_norm",
        "gpt_neox.final_layer_norm",
    )
    return findattr(model, norm_attrs)


def hf_get_decoder_blocks(model: nn.Module) -> Tuple[nn.Module]:
    """Returns the decoder hidden layers of the specified model.
    NOTE: Different model configurations have different hidden layer attribute names.
        - transformer.h: (BloomForCausalLM, GPT2LMHeadModel, GPTJForCausalLM)
        - model.decoder.layers: (OPTForCausalLM)
        - gpt_neox.layers: (GPTNeoXForCausalLM)
        - decoder.block: (T5ForConditionalGeneration)
    """
    hidden_layers_attrs = (
        "h",
        "layers",
        "decoder.layers",
        "transformer.h",
        "model.decoder.layers",
        "gpt_neox.layers",
        "decoder.block",
        "glm.transformer.layers"
    )
    return findattr(model, hidden_layers_attrs)


def hf_get_lm_head(model: nn.Module) -> nn.Module:
    """Returns the language modeling (lm) head of the specified HuggingFace
    transformers model.
    NOTE: Different model configurations have different `lm_head` attribute names.
        - lm_head: (GPT2LMHeadModel, BloomForCausalLM)
        - embed_out: (GPTNeoXForCausalLM)
    """
    return model.get_output_embeddings()


def hf_get_hidden_size(config: transformers.PretrainedConfig) -> int:
    """Returns the hidden layer dimensionality of the model architecture specified
    by the HuggingFace transformers config.
    NOTE: Different model configurations have different hidden size attribute names.
        - hidden_size: (OPTConfig, BloomConfig)
        - n_embd: (GPT2Config, GPTJConfig)
        - d_model: (PegasusConfig, XLNetConfig)
    """
    hidden_size_attrs = ("hidden_size", "n_embd", "d_model")
    return findattr(config, hidden_size_attrs)


def hf_get_num_hidden_layers(config: transformers.PretrainedConfig) -> int:
    """Returns the number of hidden layers in the model architecture specified
    by the HuggingFace transformers config.
    NOTE: Different model configurations have different number-of-layers attribute
    names.
        - num_hidden_layers: (GPTNeoXConfig, OPTConfig)
        - n_layer: (GPT2Config, GPTJConfig, BloomConfig)
    """
    num_hidden_layers_attrs = ("num_hidden_layers", "n_layer")
    return findattr(config, num_hidden_layers_attrs)


def get_global_statistics(xs: torch.Tensor) -> Tuple[float, float, int]:
    """
    Computes element-wise mean and variance of the tensor across processes
    """
    sum_and_count = torch.tensor([xs.sum(), xs.numel()], device=xs.device)
    dist.all_reduce(sum_and_count, dist.ReduceOp.SUM)
    global_sum, count = sum_and_count
    global_mean = global_sum / count

    sum_var = torch.sum((xs - global_mean) ** 2)
    dist.all_reduce(sum_var, dist.ReduceOp.SUM)
    global_var = sum_var / count
    return global_mean, global_var, count


def whiten(xs: torch.Tensor, shift_mean=True, distributed=True) -> torch.Tensor:
    """Whitens values"""
    if distributed and dist.is_initialized():
        mean, var, _ = get_global_statistics(xs)
    else:
        var, mean = torch.var_mean(xs)

    whitened = (xs - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened


def logprobs_of_labels(logits, labels):
    """Log probabilities of the labels

    These are calculated from the logits."""
    logprobs = F.log_softmax(logits, dim=-1)
    logprobs_labels = torch.gather(logprobs, dim=-1, index=labels.unsqueeze(-1))
    return logprobs_labels.squeeze(-1)


def flatten_dict(
    d: Union[dict, MutableMapping],
    parent_key: str = "",
    sep: str = "/",
) -> dict:
    # From: https://stackoverflow.com/a/6027615
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def get_tensor_stats(xs: torch.Tensor, mask: torch.Tensor, n: int):
    mean = (xs * mask).sum() / n
    return dict(
        mean=mean,
        min=torch.where(mask.bool(), xs, np.inf).min(),
        max=torch.where(mask.bool(), xs, -np.inf).max(),
        std=torch.sqrt(((xs - mean) * mask).pow(2).sum() / n),
    )


class RunningMoments:
    def __init__(self):
        """
        Calculates the running mean and standard deviation of a data stream. Modified version of
        https://github.com/DLR-RM/stable-baselines3/blob/a6f5049a99a4c21a6f0bcce458ca3306cef310e0/stable_baselines3/common/running_mean_std.py
        """
        self.mean = 0
        self.std = 1
        self.var = 1
        self.count = 1e-24

    def update(self, xs: torch.Tensor) -> Tuple[float, float]:
        """Updates running moments from batch's moments computed across ranks"""
        if dist.is_initialized():
            xs_mean, xs_var, xs_count = get_global_statistics(xs)
        else:
            xs_count = xs.numel()
            xs_var, xs_mean = torch.var_mean(xs, unbiased=False)

        delta = xs_mean - self.mean
        tot_count = self.count + xs_count

        new_sum = xs_var * xs_count
        # correct old_sum deviation accounting for the new mean
        old_sum = self.var * self.count + delta**2 * self.count * xs_count / tot_count
        tot_sum = old_sum + new_sum

        self.mean += delta * xs_count / tot_count
        self.var = tot_sum / tot_count
        self.std = (self.var * tot_count / (tot_count - 1)).sqrt()
        self.count = tot_count

        return xs_mean, (xs_var * xs_count / (xs_count - 1)).sqrt()


# OpenDelta utilities


MODIFIED_MODULES_DICT = {
    "gptj": {
        "attention": ["attn.q_proj", "attn.k_proj", "attn.v_proj"],
        "mlp": ["mlp.fc_in", "mlp.fc_out"],
        "all": [
            "attn.q_proj",
            "attn.k_proj",
            "attn.v_proj",
            "attn.out_proj",
            "mlp.fc_in",
            "mlp.fc_out",
        ],
    },
    "gpt_neox": {
        "attention": ["attention.query_key_value"],
        "mlp": ["mlp.dense_h_to_4h", "mlp.dense_4h_to_h"],
        "all": [
            "attention.query_key_value",
            "attention.dense",
            "mlp.dense_h_to_4h",
            "mlp.dense_4h_to_h",
        ],
    },
    "opt": {
        "attention": [
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.q_proj",
            "self_attn.out_proj",
        ],
        "mlp": ["fc1", "fc2"],
        "all": [
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.q_proj",
            "self_attn.out_proj",
            "fc1",
            "fc2",
        ],
    },
    "bloom": {
        "attention": ["self_attention.query_key_value", "self_attention.dense"],
        "mlp": ["mlp.dense_h_to_4h", "mlp.dense_4h_to_h"],
        "all": [
            "self_attention.query_key_value",
            "self_attention.dense",
            "mlp.dense_h_to_4h",
            "mlp.dense_4h_to_h",
        ],
    },
    "t5": {
        "attention": [
            "layer.0.SelfAttention.q",
            "layer.0.SelfAttention.k",
            "layer.0.SelfAttention.v",
            "layer.0.SelfAttention.o",
            "layer.1.EncDecAttention.q",
            "layer.1.EncDecAttention.k",
            "layer.1.EncDecAttention.v",
            "layer.1.EncDecAttention.o",
        ],
        "mlp": [
            "layer.2.DenseReluDense.wo",
            "layer.2.DenseReluDense.wi_0",
            "layer.2.DenseReluDense.wi_1",
        ],
        "all": [
            "layer.0.SelfAttention.q",
            "layer.0.SelfAttention.k",
            "layer.0.SelfAttention.v",
            "layer.0.SelfAttention.o",
            "layer.1.EncDecAttention.q",
            "layer.1.EncDecAttention.k",
            "layer.1.EncDecAttention.v",
            "layer.1.EncDecAttention.o",
            "layer.2.DenseReluDense.wo",
            "layer.2.DenseReluDense.wi_0",
            "layer.2.DenseReluDense.wi_1",
        ],
    },
}


def generate_layer_regex(config: transformers.PretrainedConfig, num_layers_unfrozen: int = -1) -> str:
    """Generates a regex range for the specified number of learnable layers."""
    if num_layers_unfrozen == -1:
        return "(\d)+."
    num_hidden_layers = hf_get_num_hidden_layers(config)
    start_layer = num_hidden_layers - num_layers_unfrozen
    if start_layer < 0:
        raise Exception("Number of layers unfrozen cannot be greater than number of layers in the model")
    pattern = f"(?:{regex_for_range(start_layer, num_hidden_layers - 1)})."
    return f"{pattern}"


def get_delta_modified_modules(
    config: transformers.PretrainedConfig,
    modified_modules: List[str],
    num_layers_unfrozen: int = -1,
) -> List[str]:
    """Returns a list of module names to be modified for a given delta method with
    the specified number of learnable layers."""
    unfrozen_layers_pattern = generate_layer_regex(config, num_layers_unfrozen)

    # [r] for regex as per https://github.com/thunlp/OpenDelta/blob/main/opendelta/utils/name_based_addressing.py#L20
    regex_prefix = "[r]"
    # TODO (jon-tow): `decoder.block.` is hardcoded to support T5 layer naming.
    decoder_prefix = "decoder.block." if config.is_encoder_decoder else ""
    module_list = [regex_prefix + decoder_prefix + unfrozen_layers_pattern + module for module in modified_modules]
    return module_list


def get_delta_model_class(model_type: str):
    if not HAS_OPENDELTA:
        raise ValueError("OpenDelta package required to train with delta models. https://github.com/thunlp/OpenDelta.")
    delta_models = {
        "bitfit": BitFitModel,
        "adapter": AdapterModel,
        "prefix": PrefixModel,
        "lora": LoraModel,
        "softprompt": SoftPromptModel,
    }
    return delta_models[model_type]


def parse_delta_kwargs(
    config: transformers.PretrainedConfig,
    delta_kwargs: Dict[str, Any],
    num_layers_unfrozen: int = -1,
) -> Tuple[str, Dict[str, Any]]:
    """Parses through delta kwargs to get delta type and proper modified modules."""
    # This function is needed to parse through the `delta_kwargs` in order to:
    # 1) Get the `delta_type` method name to access the correct `delta_model_class`
    # 2a) Accept user specified `modified_modules` and if not provided use the `trlx` default mapping
    # 2b) Convert the list of `modified_modules` to a range of layers that fit within the range
    #    of learnable layers as specified by `num_layers_unfrozen`

    # Pop `delta_type` to allow passing the kwargs to the model constructor since
    # `delta_type` is not a valid argument of the constructor
    delta_type = delta_kwargs.pop("delta_type")
    assert delta_type in ["lora"], "Only `LoRA` based delta models are supported"

    # Use `trlx` default modified modules if none are specified
    modified_modules = delta_kwargs.get("modified_modules", "all")
    if modified_modules in ["all", "attention", "mlp"]:
        if config.model_type not in MODIFIED_MODULES_DICT:
            raise ValueError(
                f"Model type `{config.model_type}` is not currently supported for "
                "delta training with default modified modules."
            )
        modified_modules = MODIFIED_MODULES_DICT[config.model_type][modified_modules]
    # Update the `modified_modules` with the correct layer ranges
    delta_kwargs["modified_modules"] = get_delta_modified_modules(
        config, modified_modules, num_layers_unfrozen=num_layers_unfrozen
    )

    return delta_type, delta_kwargs


def regex_for_range(min_: int, max_: int) -> str:  # noqa
    """Returns a regex that matches all numbers in the given range.

    Example: regex_for_range(12, 34) -> "1[2-9]|2\d|3[0-4]"

    Copyright (c) 2013, Dmitry Voronin. All rights reserved.
    Reference: https://github.com/voronind/range-regex
    """

    def split_to_patterns(min_, max_):
        subpatterns = []
        start = min_
        for stop in split_to_ranges(min_, max_):
            subpatterns.append(range_to_pattern(start, stop))
            start = stop + 1
        return subpatterns

    def split_to_ranges(min_, max_):
        stops = {max_}
        nines_count = 1
        stop = fill_by_nines(min_, nines_count)
        while min_ <= stop < max_:
            stops.add(stop)
            nines_count += 1
            stop = fill_by_nines(min_, nines_count)
        zeros_count = 1
        stop = fill_by_zeros(max_ + 1, zeros_count) - 1
        while min_ < stop <= max_:
            stops.add(stop)
            zeros_count += 1
            stop = fill_by_zeros(max_ + 1, zeros_count) - 1
        stops = list(stops)
        stops.sort()
        return stops

    def fill_by_nines(integer, nines_count):
        return int(str(integer)[:-nines_count] + "9" * nines_count)

    def fill_by_zeros(integer, zeros_count):
        return integer - integer % 10**zeros_count

    def range_to_pattern(start, stop):
        pattern = ""
        any_digit_count = 0
        for start_digit, stop_digit in zip(str(start), str(stop)):
            if start_digit == stop_digit:
                pattern += start_digit
            elif start_digit != "0" or stop_digit != "9":
                pattern += "[{}-{}]".format(start_digit, stop_digit)
            else:
                any_digit_count += 1
        if any_digit_count:
            pattern += r"\d"
        if any_digit_count > 1:
            pattern += "{{{}}}".format(any_digit_count)
        return pattern

    positive_subpatterns = []
    negative_subpatterns = []

    if min_ < 0:
        min__ = 1
        if max_ < 0:
            min__ = abs(max_)
        max__ = abs(min_)
        negative_subpatterns = split_to_patterns(min__, max__)
        min_ = 0
    if max_ >= 0:
        positive_subpatterns = split_to_patterns(min_, max_)

    negative_only_subpatterns = ["-" + val for val in negative_subpatterns if val not in positive_subpatterns]
    positive_only_subpatterns = [val for val in positive_subpatterns if val not in negative_subpatterns]
    intersected_subpatterns = ["-?" + val for val in negative_subpatterns if val in positive_subpatterns]
    subpatterns = negative_only_subpatterns + intersected_subpatterns + positive_only_subpatterns
    return "|".join(subpatterns)


def get_optimizer_grouped_parameters(model,
                                     weight_decay,
                                     no_decay_name_list=["bias", "LayerNorm.weight"]):
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if (not any(nd in n
                            for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay":
                weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if (any(nd in n
                        for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay":
                0.0,
        },
    ]
    return optimizer_grouped_parameters


def get_all_reduce_mean(tensor):
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    tensor = tensor / torch.distributed.get_world_size()
    return tensor


def moving_average(model, model_ema, beta=0.992, device=None, zero_stage=0):
    zero_stage_3 = (zero_stage == 3)
    with torch.no_grad():
        for param, param_ema in zip(model.parameters(),
                                    model_ema.parameters()):
            # TODO: use prefiltering for efficiency
            params_to_fetch = _z3_params_to_fetch([param, param_ema
                                                   ]) if zero_stage_3 else []
            should_gather_param = len(params_to_fetch) > 0
            with deepspeed.zero.GatheredParameters(
                    params_to_fetch, enabled=should_gather_param):
                data = param.data
                if device is not None:
                    data = data.to(device)
                param_ema.data.copy_(torch.lerp(data, param_ema.data, beta))


def save_hf_format(model, tokenizer, args, sub_folder=""):
    # used to save huggingface format, so we can use it for hf.from_pretrained
    model_to_save = model.module if hasattr(model, 'module') else model
    CONFIG_NAME = "config.json"
    WEIGHTS_NAME = "pytorch_model.bin"
    output_dir = os.path.join(args.output_dir, sub_folder)
    try:
        os.makedirs(output_dir)
    except FileExistsError:
        pass
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)
    save_dict = model_to_save.state_dict()
    for key in list(save_dict.keys()):
        if "lora" in key:
            del save_dict[key]
    torch.save(save_dict, output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(output_dir)


def _z3_params_to_fetch(param_list):
    return [
        p for p in param_list
        if hasattr(p, 'ds_id') and p.ds_status == ZeroParamStatus.NOT_AVAILABLE
    ]


def save_zero_three_model(model_ema, global_rank, save_dir, zero_stage=0):
    zero_stage_3 = (zero_stage == 3)
    os.makedirs(save_dir, exist_ok=True)
    WEIGHTS_NAME = "pytorch_model.bin"
    output_model_file = os.path.join(save_dir, WEIGHTS_NAME)

    model_to_save = model_ema.module if hasattr(model_ema, 'module') else model_ema
    if not zero_stage_3:
        if global_rank == 0:
            torch.save(model_to_save.state_dict(), output_model_file)
    else:
        output_state_dict = {}
        for k, v in model_to_save.named_parameters():

            if hasattr(v, 'ds_id'):
                with deepspeed.zero.GatheredParameters(_z3_params_to_fetch([v]),
                                                       enabled=zero_stage_3):
                    v_p = v.data.cpu()
            else:
                v_p = v.cpu()
            if global_rank == 0 and "lora" not in k:
                output_state_dict[k] = v_p
        if global_rank == 0:
            torch.save(output_state_dict, output_model_file)
        del output_state_dict


def sorted_checkpoints(output_dir=None, checkpoint_prefix="checkpoint", use_mtime=False) -> List[str]:
    ordering_and_checkpoint_path = []

    glob_checkpoints = [str(x) for x in Path(output_dir).glob(f"{checkpoint_prefix}-*") if os.path.isdir(x)]

    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(f".*{checkpoint_prefix}-([0-9]+)", path)
            if regex_match is not None and regex_match.groups() is not None:
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]

    return checkpoints_sorted


def rotate_checkpoints(save_total_limit, use_mtime=False, output_dir=None) -> None:
    if save_total_limit is None or save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    checkpoints_sorted = sorted_checkpoints(use_mtime=use_mtime, output_dir=output_dir)
    if len(checkpoints_sorted) <= save_total_limit:
        return

    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
        shutil.rmtree(checkpoint, ignore_errors=True)
