
import os
import yaml

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set

from src.utils.method_configs import MethodConfig, get_method, PPOConfig, SFTConfig, ILQLConfig


ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESOURCE_PATH = os.path.join(ROOT_PATH, "resources")


# -------- RLHF Config------- #

def merge(base: Dict, update: Dict, updated: Set) -> Dict:
    "Recursively updates a nested dictionary with new values"
    for k, v in base.items():
        if k in update and isinstance(v, dict):
            base[k] = merge(v, update[k], updated)
            updated.add(k)
        elif k in update:
            base[k] = update[k]
            updated.add(k)

    return base


def _merge_dicts(base: Dict, update: Dict) -> Dict:
    "Merge two dictionaries recursively, returning a new dictionary."

    base = deepcopy(base)

    for k, v in update.items():
        if isinstance(v, dict):
            base[k] = _merge_dicts(base.get(k, {}), v)
        else:
            base[k] = v

    return base


@dataclass
class ModelConfig:
    """
    Config for a model.

    :param model_path: Path or name of the model (local or on huggingface hub)
    :type model_path: str

    :param model_arch_type: Type of model architecture. Either "causal" or "seq2seq"
    :type model_arch_type: str

    :param num_layers_unfrozen: Number of layers to unfreeze for fine-tuning.
        -1 means all layers are unfrozen.
    :type num_layers_unfrozen: int

    :param delta_kwargs: Keyword arguments for instantiating OpenDelta models for delta-tuning.
        Follow the `OpenDelta.AutoDeltaConfig` specification, e.g. for LoRA style tuning, set
        the `delta_type` to `lora` and include the model specific hyper-parameters (e.g. `lora_r`)
            {"delta_type": "lora", "modified_modules": "all", "lora_r": 8, "lora_alpha": 16, "lora_dropout": 0.0}
        or in YAML format:
            delta_kwargs:
                delta_type: lora
                modified_modules: "all"
                lora_r: 8
                lora_alpha: 16
                lora_dropout: 0.0
        See: https://opendelta.readthedocs.io/en/latest/modules/auto_delta.html#opendelta.auto_delta.AutoDeltaConfig
    :type delta_kwargs: Optional[Dict[str, Any]]
    """

    model_path: str
    model_arch_type: str = "causal"
    num_layers_unfrozen: int = -1
    delta_kwargs: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(cls, config: Dict[str, Any]):
        return cls(**config)


@dataclass
class TokenizerConfig:
    """
    Config for a model.

    :param tokenizer_path: Path or name of the tokenizer (local or on huggingface hub)
    :type tokenizer_path: str

    :param padding_side: Padding side
    :type padding_path: str

    :param truncation_side: Truncation side
    :type truncation_side: str
    """

    tokenizer_path: str
    padding_side: str = "left"
    truncation_side: str = "right"

    @classmethod
    def from_dict(cls, config: Dict[str, Any]):
        return cls(**config)


@dataclass
class OptimizerConfig:
    """
    Config for an optimizer.

    :param name: Name of the optimizer
    :type name: str

    :param kwargs: Keyword arguments for the optimizer (e.g. lr, betas, eps, weight_decay)
    :type kwargs: Dict[str, Any]
    """

    name: str
    kwargs: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, config: Dict[str, Any]):
        return cls(**config)


@dataclass
class SchedulerConfig:
    """
    Config for a learning rate scheduler.

    :param name: Name of the scheduler
    :type name: str

    :param kwargs: Keyword arguments for the scheduler instance (e.g. warmup_steps, T_max)
    :type kwargs: Dict[str, Any]
    """

    name: str
    kwargs: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, config: Dict[str, Any]):
        return cls(**config)


@dataclass
class TrainConfig:
    """
    Config for train job on model.

    :param total_steps: Total number of training steps
    :type total_steps: int

    :param seq_length: Number of tokens to use as context (max length for tokenizer)
    :type seq_length: int

    :param epochs: Total number of passes through data
    :type epochs: int

    :param batch_size: Batch size for training
    :type batch_size: int

    :param tracker: Tracker to use for logging. Default: "wandb"
    :type tracker: str

    :param checkpoint_interval: Save model every checkpoint_interval steps.
        Each checkpoint is stored in a sub-directory of the `TrainConfig.checkpoint_dir`
        directory in the format `checkpoint_dir/checkpoint_{step}`.
    :type checkpoint_interval: int

    :param eval_interval: Evaluate model every eval_interval steps
    :type eval_interval: int

    :param pipeline: Pipeline to use for training. One of the registered pipelines present in trlx.pipeline
    :type pipeline: str

    :param trainer: Trainer to use for training. One of the registered trainers present in trlx.trainer
    :type trainer: str

    :param trainer_kwargs: Extra keyword arguments for the trainer
    :type trainer: Dict[str, Any]

    :param project_name: Project name for wandb
    :type project_name: str

    :param entity_name: Entity name for wandb
    :type entity_name: str

    :param group_name: Group name for wandb (used for grouping runs)
    :type group_name: str

    :param checkpoint_dir: Directory to save checkpoints
    :type checkpoint_dir: str

    :param rollout_logging_dir: Directory to store generated rollouts for use in Algorithm Distillation.
                                Only used by AcceleratePPOTrainer.
    :type rollout_logging_dir: Optional[str]

    :param save_best: Save best model based on mean reward
    :type save_best: bool

    :param seed: Random seed
    :type seed: int
    """

    total_steps: int
    seq_length: int
    epochs: int
    batch_size: int

    checkpoint_interval: int
    eval_interval: int

    pipeline: str  # One of the pipelines in framework.pipeline
    trainer: str  # One of the trainers
    trainer_kwargs: Dict[str, Any] = field(default_factory=dict)  # Extra keyword arguments for the trainer

    project_name: str = "trlx"
    entity_name: Optional[str] = None
    group_name: Optional[str] = None

    checkpoint_dir: str = "ckpts"
    rollout_logging_dir: Optional[str] = None
    save_best: bool = True

    tracker: Optional[str] = "wandb"
    logging_dir: Optional[str] = None

    lora_rank: Optional[int] = 0
    lora_alpha: Optional[int] = 1
    lora_train_bias: Optional[str] = "none"

    seed: int = 1000

    @classmethod
    def from_dict(cls, config: Dict[str, Any]):
        return cls(**config)


@dataclass
class TRLConfig:
    """
    Top level config for trlX. Loads configs and can be converted to dictionary.
    """

    method: MethodConfig
    model: ModelConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    tokenizer: TokenizerConfig
    train: TrainConfig

    @classmethod
    def load_yaml(cls, yml_fp: str):
        """
        Load yaml file as TRLConfig.

        :param yml_fp: Path to yaml file
        :type yml_fp: str
        """
        with open(yml_fp, mode="r") as file:
            config = yaml.safe_load(file)
        return cls.from_dict(config)

    def to_dict(self):
        """
        Convert TRLConfig to dictionary.
        """
        data = {
            "method": self.method.__dict__,
            "model": self.model.__dict__,
            "optimizer": self.optimizer.__dict__,
            "scheduler": self.scheduler.__dict__,
            "tokenizer": self.tokenizer.__dict__,
            "train": self.train.__dict__,
        }

        return data

    def evolve(self, **kwargs) -> "TRLConfig":
        """
        Evolve TRLConfig with new parameters. Can update nested parameters.
        >>> config = trlx.data.default_configs.default_ilql_config()
        >>> config = config.evolve(method=dict(gamma=0.99, gen_kwargs=dict(max_new_tokens=100))
        >>> config.method.gamma
        0.99
        """
        return TRLConfig.from_dict(_merge_dicts(self.to_dict(), kwargs))

    @classmethod
    def from_dict(cls, config: Dict):
        """
        Convert dictionary to TRLConfig.
        """
        return cls(
            method=get_method(config["method"]["name"]).from_dict(config["method"]),
            model=ModelConfig.from_dict(config["model"]),
            tokenizer=TokenizerConfig.from_dict(config["tokenizer"]),
            optimizer=OptimizerConfig.from_dict(config["optimizer"]),
            scheduler=SchedulerConfig.from_dict(config["scheduler"]),
            train=TrainConfig.from_dict(config["train"]),
        )

    @classmethod
    def update(cls, baseconfig: Dict, config: Dict):
        if not isinstance(baseconfig, Dict):
            baseconfig = baseconfig.to_dict()

        updates = set()
        merged = merge(baseconfig, config, updates)

        for param in config:
            if param not in updates:
                raise ValueError(f"parameter {param} is not present in the config (typo or a wrong config)")

        return cls.from_dict(merged)

    def __str__(self):
        """Returns a human-readable string representation of the config."""
        import json

        return json.dumps(self.to_dict(), indent=4)


def default_ppo_config():
    return TRLConfig(
        train=TrainConfig(
            seq_length=1024,
            epochs=100,
            total_steps=10000,
            batch_size=32,
            checkpoint_interval=10000,
            eval_interval=100,
            pipeline="PromptPipeline",
            trainer="AcceleratePPOTrainer",
        ),
        model=ModelConfig(model_path="lvwerra/gpt2-imdb", num_layers_unfrozen=2),
        tokenizer=TokenizerConfig(tokenizer_path="gpt2", truncation_side="right"),
        optimizer=OptimizerConfig(
            name="adamw", kwargs=dict(lr=1.0e-4, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=1.0e-6)
        ),
        scheduler=SchedulerConfig(name="cosine_annealing", kwargs=dict(T_max=10000, eta_min=1.0e-4)),
        method=PPOConfig(
            name="PPOConfig",
            num_rollouts=128,
            chunk_size=128,
            ppo_epochs=4,
            init_kl_coef=0.05,
            target=6,
            horizon=10000,
            gamma=1,
            lam=0.95,
            cliprange=0.2,
            cliprange_value=0.2,
            vf_coef=1,
            scale_reward="ignored",
            ref_mean=None,
            ref_std=None,
            cliprange_reward=10,
            gen_kwargs=dict(
                max_new_tokens=40,
                top_k=0,
                top_p=1.0,
                do_sample=True,
            ),
        ),
    )


def default_ilql_config():
    return TRLConfig(
        train=TrainConfig(
            seq_length=64,
            batch_size=32,
            epochs=100,
            total_steps=1000,
            checkpoint_interval=1000,
            eval_interval=100,
            pipeline="PromptPipeline",
            trainer="AccelerateILQLTrainer",
        ),
        model=ModelConfig(model_path="gpt2", num_layers_unfrozen=-1),
        tokenizer=TokenizerConfig(tokenizer_path="gpt2", truncation_side="right"),
        optimizer=OptimizerConfig(
            name="adamw", kwargs=dict(lr=5.0e-5, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=1.0e-6)
        ),
        scheduler=SchedulerConfig(
            name="cosine_annealing", kwargs=dict(T_max=1000, eta_min=5.0e-5)  # train.total_steps
        ),
        method=ILQLConfig(
            name="ilqlconfig",
            tau=0.7,
            gamma=0.99,
            cql_scale=0.1,
            awac_scale=1,
            alpha=0.001,
            beta=0,
            steps_for_target_q_sync=5,
            two_qs=True,
            gen_kwargs=dict(max_new_tokens=56, top_k=20, beta=4, temperature=1.0),
        ),
    )


def default_sft_config():
    return TRLConfig(
        train=TrainConfig(
            seq_length=1024,
            epochs=100,
            total_steps=1000,
            batch_size=8,
            checkpoint_interval=10000,
            eval_interval=100,
            pipeline="PromptPipeline",
            trainer="AccelerateSFTTrainer",
        ),
        model=ModelConfig(model_path="gpt2", num_layers_unfrozen=-1),
        tokenizer=TokenizerConfig(tokenizer_path="gpt2", truncation_side="right"),
        optimizer=OptimizerConfig(
            name="adamw", kwargs=dict(lr=1.0e-4, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=1.0e-6)
        ),
        scheduler=SchedulerConfig(
            name="cosine_annealing", kwargs=dict(T_max=10000, eta_min=1.0e-4)  # train.total_steps
        ),
        method=SFTConfig(
            name="sftconfig",
            gen_kwargs=dict(max_new_tokens=40, top_k=0, top_p=1.0, do_sample=True),
        ),
    )


def get_train_ds_config(global_batch_size=32,
                        micro_batch_size=4,
                        offload=False,
                        stage=2,
                        enable_hybrid_engine=False,
                        inference_tp_size=1,
                        release_inference_cache=False,
                        pin_parameters=True,
                        tp_gather_partition_size=8,
                        max_out_tokens=512):

    device = "cpu" if offload else "none"
    zero_opt_dict = {
        "stage": stage,
        "offload_param": {
            "device": device
        },
        "offload_optimizer": {
            "device": device
        },
        "stage3_param_persistence_threshold": 1e4,
        "stage3_max_live_parameters": 3e7,
        "stage3_prefetch_bucket_size": 3e7,
        "memory_efficient_linear": False
    }
    return {
        "train_batch_size": global_batch_size,
        "train_micro_batch_size_per_gpu": micro_batch_size,
        "steps_per_print": 10,
        "zero_optimization": zero_opt_dict,
        "fp16": {
            "enabled": True,
            "loss_scale_window": 100
        },
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
        "hybrid_engine": {
            "enabled": enable_hybrid_engine,
            "max_out_tokens": max_out_tokens,
            "inference_tp_size": inference_tp_size,
            "release_inference_cache": release_inference_cache,
            "pin_parameters": pin_parameters,
            "tp_gather_partition_size": tp_gather_partition_size,
        }
    }


def get_eval_ds_config(global_batch_size=32, micro_batch_size=4, offload=False, stage=0):
    device = "cpu" if offload else "none"
    zero_opt_dict = {
        "stage": stage,
        "stage3_param_persistence_threshold": 1e4,
        "offload_param": {
            "device": device
        },
        "memory_efficient_linear": False
    }
    return {
        "train_batch_size": global_batch_size,
        "train_micro_batch_size_per_gpu": micro_batch_size,
        "steps_per_print": 10,
        "zero_optimization": zero_opt_dict,
        "fp16": {
            "enabled": True
        },
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": False
    }
