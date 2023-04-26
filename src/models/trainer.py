
import sys
import json
import os
import ray
import torch
import logging
import uuid
import torch.nn.functional as F

from abc import abstractmethod
from time import time
from tqdm import tqdm
from typing import Any, Callable,  Iterable, Dict, List, Optional, Tuple
from torch.utils.data import DataLoader
from accelerate import Accelerator  # type: ignore
from ray.air import session
from ray.air.checkpoint import Checkpoint
from rich.console import Console
from rich.table import Table
from transformers import AutoTokenizer, PretrainedConfig

from src.utils.logger import logger
from src.utils.config import TRLConfig
from src.data.pipeline import BaseRolloutStore
from src.utils.file_utils import significant
from src.utils.modeling_utils import (
    filter_non_scalars,
    get_distributed_config,
    get_git_tag,
    get_optimizer_class,
    get_scheduler_class,
    flatten_dict,
    freeze_bottom_causal_layers,
    freeze_bottom_seq2seq_layers,
    get_delta_model_class,
    parse_delta_kwargs,
)

from src.data.data_types import PromptBatch, PPORLBatch, PPORLElement
from src.models.ppo import (
    AdaptiveKLController,
    AutoModelForCausalLMWithHydraValueHead,
    AutoModelForSeq2SeqLMWithHydraValueHead,
    FixedKLController,
)
from src.data.pipeline import BasePipeline, PPORolloutStorage
from src.utils.modeling_utils import Clock, RunningMoments, logprobs_of_labels
from src.utils.logger import logger

# specifies a dictionary of architectures
_TRAINERS: Dict[str, Any] = {}  # registry


def register_trainer(name):
    """Decorator used to register a trainer
    Args:
        name: Name of the trainer type to register
    """

    def register_class(cls, name):
        _TRAINERS[name] = cls
        setattr(sys.modules[__name__], name, cls)
        return cls

    if isinstance(name, str):
        name = name.lower()
        return lambda c: register_class(c, name)

    cls = name
    name = cls.__name__
    register_class(cls, name.lower())

    return cls


@register_trainer
class BaseRLTrainer:
    def __init__(
            self,
            config: TRLConfig,
            reward_fn=None,
            metric_fn=None,
            logit_mask=None,
            stop_sequences=None,
            train_mode=False,
    ):
        self.store: BaseRolloutStore = None
        self.config = config
        self.reward_fn = reward_fn
        self.metric_fn = metric_fn
        self.train_mode = train_mode
        self.logit_mask = logit_mask
        self.stop_sequences = stop_sequences

    def push_to_store(self, data):
        self.store.push(data)

    def add_eval_pipeline(self, eval_pipeline):
        """Adds pipeline for validation prompts"""
        self.eval_pipeline = eval_pipeline

    @abstractmethod
    def sample(self, prompts: Iterable[str], length: int, n_samples: int) -> Iterable[str]:
        """
        Sample from the language. Takes prompts and maximum length to generate.

        :param prompts: List of prompts to tokenize and use as context

        :param length: How many new tokens to genrate for each prompt
        :type length: int

        :param n_samples: Default behavior is to take number of prompts as this
        """
        pass

    @abstractmethod
    def learn(
            self,
            log_fn: Callable = None,
            save_fn: Callable = None,
            eval_fn: Callable = None,
    ):
        """
        Use experiences in RolloutStore to learn

        :param log_fn: Optional function that is called when logging and passed a dict of logging relevant values
        :type log_fn: Callable[Dict[str, any]]

        :param save_fn: Optional function to call after saving. Is passed the components.
        :type save_fn: Callable[Dict[str, any]]

        :param eval_fn: Optional function to call during evaluation. Eval doesn't do anything without this.
        :type eval_fn: Callable[BaseRLTrainer]
        """
        pass

    @abstractmethod
    def save(self, directory: Optional[str] = None):
        """Creates a checkpoint of training states"""
        pass

    @abstractmethod
    def load(self, directory=None):
        """Loads a checkpoint created from `save`"""
        pass


@register_trainer
class AccelerateRLTrainer(BaseRLTrainer):
    """
    RL model trainer with an `accelerate` based backend
    """

    def __init__(self, config, **kwargs):  # noqa: C901
        super().__init__(config, **kwargs)
        self.max_length = config.train.seq_length
        self.accelerator = Accelerator(log_with=config.train.tracker, logging_dir=config.train.logging_dir)

        if self.accelerator.state.deepspeed_plugin is not None:
            # by accelerate's default, arguments in `model.forward` would be casted to half
            if "fp16" in self.accelerator.state.deepspeed_plugin.deepspeed_config:
                self.accelerator.state.deepspeed_plugin.deepspeed_config["fp16"]["auto_cast"] = False

        if int(os.environ.get("WORLD_SIZE", 1)) > 1:
            torch.distributed.barrier(device_ids=[int(os.environ.get("LOCAL_RANK", 0))])

        self.model = self.setup_model()
        self.opt = self.setup_optimizer()
        self.scheduler = self.setup_scheduler()

        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer.tokenizer_path, trust_remote_code=True)
        self.tokenizer.padding_side = config.tokenizer.padding_side
        self.tokenizer.truncation_side = config.tokenizer.truncation_side
        self.padding_side = config.tokenizer.padding_side
        # self.tokenizer.sep_token = "<sep>"
        # if config.model.model_arch_type != "seq2seq":
        #     self.tokenizer.pad_token = self.tokenizer.eos_token

        script_name = os.path.basename(sys.argv[0]).rsplit(".", 1)[0]
        if not isinstance(config.model.model_path, str):
            model_name = str(config.model.model_path).split()[0]
        else:
            model_name = config.model.model_path.split("/")[-1]

        if self.accelerator.num_processes == 1:
            num_gpus = "1gpu"
        else:
            num_gpus = f"{self.accelerator.num_processes}gpus"
        branch = get_git_tag()[0]

        run_name = "/".join([script_name, model_name, num_gpus]) + f":{branch}"

        if self.accelerator.is_main_process and not ray.is_initialized():
            config_dict = self.config.to_dict()
            dist_config = get_distributed_config(self.accelerator)
            config_dict["distributed"] = dist_config
            init_trackers_kwargs = {}

            if config.train.tracker == "wandb":
                init_trackers_kwargs["wandb"] = {
                    "name": run_name,
                    "entity": self.config.train.entity_name,
                    "group": self.config.train.group_name,
                    "tags": ["/".join(get_git_tag())],
                    "mode": "disabled" if os.environ.get("debug", False) else "online",
                }

                self.accelerator.init_trackers(
                    project_name=self.config.train.project_name,
                    config=config_dict,
                    init_kwargs=init_trackers_kwargs,
                )
            elif config.train.tracker == "tensorboard":
                # flatten config for tensorboard, split list in hparams into flatten config
                config_dict_flat = flatten_dict(config_dict)
                config_dict_flat["optimizer/kwargs/beta_1"] = config_dict_flat["optimizer/kwargs/betas"][0]
                config_dict_flat["optimizer/kwargs/beta_2"] = config_dict_flat["optimizer/kwargs/betas"][1]
                config_dict_flat.pop("optimizer/kwargs/betas", None)
                self.accelerator.init_trackers(
                    project_name=self.config.train.project_name,
                    config=config_dict_flat,
                )
            elif config.train.tracker is None:
                self.accelerator.init_trackers(project_name=self.config.train.project_name)
            else:
                raise ValueError(
                    f"Only supported trackers are `wandb` and `tensorboard`. Got: `{config.train.tracker}`. "
                    "Set `tracker` to `None` to disable tracking."
                )

    def setup_model(self):
        """
        Returns a model derived from an instance's TRLConfig
        """
        logger.info(f"Initializing model: {self.config.model.model_path}")

        # Retrieves model equipped for ppo, ilql, etc
        model = self.get_arch(self.config)
        # if self.config.model.model_arch_type == "seq2seq":
        #     freeze_bottom_seq2seq_layers(model.base_model, self.config.model.num_layers_unfrozen)
        # else:
        freeze_bottom_causal_layers(model.base_model, self.config.model.num_layers_unfrozen)
        # Set the delta tuning strategies
        if self.config.model.delta_kwargs is not None:
            delta_type, delta_kwargs = parse_delta_kwargs(
                model.base_model.config,
                self.config.model.delta_kwargs,
                self.config.model.num_layers_unfrozen,
            )
            delta_model_class = get_delta_model_class(delta_type)
            delta_model = delta_model_class(model.base_model, **delta_kwargs)
            delta_model.freeze_module(exclude=["deltas"], set_state_dict=True)
            if self.accelerator.is_main_process:
                delta_model.log()
        return model

    def setup_optimizer(self):
        """
        Returns an optimizer derived from an instance's TRLConfig
        """
        optimizer_class = get_optimizer_class(self.config.optimizer.name)
        optimizer = optimizer_class(
            self.model.parameters(),
            **self.config.optimizer.kwargs,
        )

        if "bitsandbytes" in optimizer.__class__.__module__:
            # Force 32-bit `nn.Embedding` weights for stability. See discussion:
            # https://github.com/huggingface/transformers/issues/14819#issuecomment-1016017746
            from bitsandbytes.optim import GlobalOptimManager

            manager = GlobalOptimManager.get_instance()
            for module in self.model.modules():
                if isinstance(module, torch.nn.Embedding):
                    manager.register_module_override(module, "weight", {"optim_bits": 32})

        return optimizer

    def setup_scheduler(self):
        """
        Returns a learning rate scheduler derived from an instance's TRLConfig
        """
        scheduler_class = get_scheduler_class(self.config.scheduler.name)
        scheduler = scheduler_class(self.opt, **self.config.scheduler.kwargs)
        return scheduler

    def decode(
            self,
            prompts: List[torch.LongTensor],
            samples: List[torch.LongTensor],
            prompt_sizes: torch.LongTensor = None,
    ) -> Tuple[List[str], List[str], List[str], List[str], List[List[torch.Tensor]]]:
        """
        Decode tensor generations into lists of strings (`samples`: List[str], `prompts`: List[str], `outputs`: List[str])
        """
        # Assuming prompts were left-padded
        prompt_sizes = []
        prefix_indices = []
        for prompt in prompts:
            prefix_idx = None
            if "chatglm" in self.config.model.model_path:
                prompt_sizes.append(len(prompt))
            else:
                logger.debug(f"[decode] prompt: {prompt}")
                if isinstance(prompt, torch.Tensor):
                    prompt = prompt.cpu().detach().tolist()
                prompt_sizes.append(prompt.index(self.tokenizer.sep_token_id))
                if "glm" in self.config.model.model_path:
                    try:
                        prefix_idx = prompt.index(self.tokenizer.mask_token_id)
                    except IndexError:
                        pass
            prefix_indices.append(prefix_idx)

        str_samples, str_prompts, str_outputs, str_prefixes, sample_outputs = [], [], [], [], []
        for prompt, sample, prompt_size, prefix_idx in zip(prompts, samples, prompt_sizes, prefix_indices):
            # if self.config.model.model_arch_type == "seq2seq":
            #     output_start_ix = 0
            # else:
            output_start_ix = prompt_size

            str_prompt = self.tokenizer.decode(prompt[:prompt_size], skip_special_tokens=True)
            if prefix_idx is not None:
                str_prefix = self.tokenizer.decode(sample[output_start_ix:prefix_idx], skip_special_tokens=True)
                sample_output = sample[prefix_idx:]
                str_output = self.tokenizer.decode(sample_output, skip_special_tokens=True)
            else:
                str_prefix = None
                sample_output = sample[output_start_ix:]
                str_output = self.tokenizer.decode(sample_output, skip_special_tokens=True)

            # Trim outputs up to `self.stop_sequences` if any are present
            if self.stop_sequences:
                for stop in self.stop_sequences:
                    stop_ix = str_output.find(stop)
                    if stop_ix >= 0:
                        str_output = str_output[:stop_ix].rstrip()

            str_prompts.append(str_prompt)
            str_outputs.append(str_output)
            str_prefixes.append(str_prefix)
            sample_outputs.append(sample_output)

            if "chatglm" in self.config.model.model_path:
                sample = str_prompt + str_output
            else:
                sample = str_prompt + self.tokenizer.sep_token + str_output

            str_samples.append(sample)

        return str_samples, str_prompts, str_outputs, str_prefixes, sample_outputs

    def generate(self, input_ids, attention_mask=None, **kwargs):
        """Wraps hf's `generate` adding some specific method's defaults"""
        input_ids = input_ids.to(self.accelerator.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.accelerator.device)
        if self.generate_experience_kwargs is not None:
            kwargs = dict(self.generate_experience_kwargs, **kwargs)
        else:
            kwargs = dict(self.generate_kwargs, **kwargs)

        with torch.no_grad():
            return self.accelerator.unwrap_model(self.model).generate(
                input_ids=input_ids, attention_mask=attention_mask, **kwargs
            )

    def generate_eval(self, input_ids, attention_mask=None, **kwargs):
        """Wraps hf's `generate` adding some specific method's defaults"""
        input_ids = input_ids.to(self.accelerator.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.accelerator.device)

        kwargs = dict(self.generate_kwargs, **kwargs)

        with torch.no_grad():
            return self.accelerator.unwrap_model(self.model).generate(
                input_ids=input_ids, attention_mask=attention_mask, **kwargs
            )

    def save_pretrained(self, directory: Optional[str] = None, **kwargs):
        """Save the underlying Hugging Face model, tokenizer, and configuration files to a directory for
        later use.

        Args:
            directory (str, *optional*): The directory to save the trainer files to.
                NOTE: If not specified, the model will be saved to a directory named `hf_model` in the
                checkpoint directory as specified by the Trainer's config.
            **kwargs: Additional keyword arguments passed to the underlying Hugging Face model's
                `save_pretrained` method.
        """
        if directory is None:
            directory = os.path.join(self.config.train.checkpoint_dir, "hf_model")
        self.accelerator.wait_for_everyone()
        self.accelerator.unwrap_model(self.model).save_pretrained(directory, **kwargs)
        if self.accelerator.is_main_process:
            self.tokenizer.save_pretrained(directory)

    def save(self, directory: Optional[str] = None, **kwargs):
        """Creates a checkpoint of the optimizer, scheduler and model"""
        self.accelerator.save_state(directory or self.config.train.checkpoint_dir, **kwargs)

    def load(self, directory: Optional[str] = None, **kwargs):
        """Load checkpoint of optimizer, scheduler and a model"""
        self.accelerator.load_state(directory or self.config.train.checkpoint_dir, **kwargs)

    def add_eval_pipeline(self, eval_pipeline):
        """Adds pipeline from with validation prompts"""
        self.eval_pipeline = eval_pipeline

    def evaluate(self):  # noqa: C901
        """Samples model on `eval_prompts`, logs stats with `reward_fn` or `metric_fn` if provided"""
        logger.info("Evaluating model")

        # Do multiple evaluations over a single list in `gen_kwargs` if present
        if self.generate_sweep_kwarg is not None:
            gen_sweep_arg, gen_sweep_values = self.generate_sweep_kwarg
        else:
            gen_sweep_values = [None]

        desc = [
            f"generation sweep 0/{len(gen_sweep_values)}",
            f"eval batch 0/{len(self.eval_dataloader)}",
        ]
        tbar = tqdm(
            total=len(self.eval_dataloader) * len(gen_sweep_values),
            desc=f"[{' | '.join(desc)}]",
            disable=not self.accelerator.is_main_process,
            position=0,
            leave=True,
        )

        stats = {}
        table = []

        for i_sweep, gen_sweep_value in enumerate(gen_sweep_values):
            # A dedicated suffix for wandb logging
            if gen_sweep_value is not None:
                sweep_suffix = f"@{gen_sweep_arg}={gen_sweep_value}"
            else:
                sweep_suffix = ""

            all_samples = []
            all_prompts = []
            all_prompt_sizes = []
            generate_time = time()
            for i_prompt, prompts in enumerate(self.eval_dataloader):
                logger.debug(f"evaluate() - prompts keys: {prompts.keys()}, input_ids: {prompts['input_ids'].shape}")
                if self.generate_sweep_kwarg:
                    samples = self.generate_eval(**prompts, **{gen_sweep_arg: gen_sweep_value})
                else:
                    samples = self.generate_eval(**prompts)

                # if self.config.model.model_arch_type == "seq2seq":
                #     samples = samples[:, 1:].contiguous()

                prompt_sizes = torch.tensor(prompts['input_ids'].shape[1]).repeat(len(prompts['input_ids']))
                prompts, samples, prompt_sizes = self.accelerator.gather_for_metrics(
                    self.accelerator.pad_across_processes(
                        [prompts['input_ids'], samples, prompt_sizes.to(samples.device)],
                        dim=1,
                        pad_index=self.tokenizer.pad_token_id,
                    )
                )
                all_samples.extend(samples.tolist())
                all_prompts.extend(prompts.tolist())
                all_prompt_sizes.extend(prompt_sizes.tolist())

                desc = [
                    f"generation sweep {i_sweep + 1}/{len(gen_sweep_values)}",
                    f"eval batch {i_prompt + 1}/{len(self.eval_dataloader)}",
                ]
                tbar.set_description(f"[{' | '.join(desc)}]")
                tbar.update()
            tbar.close()

            stats["time/generate"] = time() - generate_time

            if self.accelerator.is_main_process:
                str_samples, str_prompts, str_outputs, str_prefixes, _ = self.decode(all_prompts, all_samples, all_prompt_sizes)

                columns = ["prompt", "output"]
                columns_data = [str_prompts, str_outputs]

                # in online setting, compute the reward for validation
                if self.reward_fn:
                    logger.info("Computing rewards")
                    rewards = torch.tensor(
                        self.reward_fn(
                            samples=str_samples,
                            prompts=str_prompts,
                            outputs=str_outputs,
                        ),
                        dtype=float,
                    )
                    mean_reward = rewards.mean().item()
                    columns.append("reward")
                    if not isinstance(rewards, list):
                        rewards = rewards.tolist()
                    columns_data.append(rewards)
                    stats[f"reward/mean{sweep_suffix}"] = mean_reward

                # additionally log any other metrics
                if self.metric_fn:
                    logger.info("Computing metrics")
                    metric_time = time()
                    metrics = self.metric_fn(
                        samples=str_samples,
                        prompts=str_prompts,
                        outputs=str_outputs,
                    )
                    stats["time/metric"] = time() - metric_time

                    mean_metrics = {
                        f"metrics/{k}{sweep_suffix}": torch.as_tensor(xs).mean(-1) for k, xs in metrics.items()
                    }

                    stats.update(mean_metrics)

                    for metric, values in metrics.items():
                        columns.append(metric)
                        if not isinstance(values, list):
                            values = values.tolist()
                        columns_data.append(values)

                # Prepend the sweep argument along with samples
                if self.generate_sweep_kwarg:
                    columns.insert(0, gen_sweep_arg)
                    columns_data.insert(0, [gen_sweep_value] * len(samples))

                table.append(list(zip(*columns_data)))

        # Log and display evaluation metrics
        logger.info("Summarizing evaluation")
        if self.accelerator.is_main_process:
            rows = sum(list(map(list, zip(*table))), [])

            # Add metrics/rewards to the table's title
            table_title = f"Evaluation #{self.nth_evaluation}"
            for k, x in stats.items():
                if k.startswith("reward") or k.startswith("metrics"):
                    table_title += f" {k}: {significant(x)}"

            rich_table = Table(*columns, title=table_title, show_lines=True)
            for ix in range(max(min(3, len(rows)), len(gen_sweep_values))):
                rich_table.add_row(*[str(significant(x)) for x in rows[ix]])
            Console().print(rich_table)

            if not ray.is_initialized():
                if self.config.train.tracker == "wandb":
                    import wandb

                    stats["samples"] = wandb.Table(columns, rows)

        self.nth_evaluation += 1
        return stats

    def learn(self):  # noqa: C901
        """
        Samples batches from `self.store`, updates model and periodically evaluates it on `self.eval_dataloader`
        """
        logger.info("Starting training")

        self.generate_sweep_kwarg = None
        for k, v in self.config.method.gen_kwargs.items():
            if isinstance(v, list):
                if self.generate_sweep_kwarg is not None:
                    logger.info("Only a single sweep is allowed, {k} is going to be set to {v[0]}")
                    self.generate_kwargs[k] = v[0]
                else:
                    self.generate_sweep_kwarg = (k, v)

        self.prepare_learning()
        self.iter_count = 0
        self.nth_evaluation = 0

        if ray.is_initialized():
            checkpoint = session.get_checkpoint()
            if checkpoint:
                with checkpoint.as_directory() as dir:
                    self.accelerator.load_state(dir)

                    with open(os.path.join(dir, "state.json")) as f:
                        state = json.load(f)
                        self.iter_count = state["iter_count"]
        else:
            results = self.evaluate()
            self.accelerator.log(results, step=self.iter_count)

        tbar = tqdm(
            initial=self.iter_count,
            total=self.total_steps,
            disable=not self.accelerator.is_local_main_process,
            position=0,
            leave=True,
        )

        best_reward = -float("inf")

        # For each epoch
        for _ in range(self.config.train.epochs):
            # For each batch
            for batch in self.train_dataloader:
                # For each update per batch
                for _ in range(self.n_updates_per_batch):
                    # Note that whereas standard policy gradient methods perform one
                    # gradient update per batch, PPO for example commonly performs
                    # multiple gradient updates on the same batch of data.
                    # https://arxiv.org/pdf/1707.06347.pdf
                    forward_time = time()
                    loss, stats = self.loss(batch)
                    forward_time = time() - forward_time
                    backward_time = time()
                    self.accelerator.backward(loss)
                    backward_time = time() - backward_time

                    self.opt.step()
                    self.opt.zero_grad()
                    self.scheduler.step()
                    self.iter_count += 1

                    if self.iter_count % self.config.train.checkpoint_interval == 0:
                        subfolder = f"checkpoint_{self.iter_count:0{len(str(self.total_steps))}d}"
                        directory = os.path.join(self.config.train.checkpoint_dir, subfolder)
                        self.save(directory)

                    stats["time/forward"] = forward_time
                    stats["time/backward"] = backward_time
                    for group_number, lr in enumerate(self.scheduler.get_last_lr()):
                        stats[f"learning_rate_group_{group_number}"] = lr

                    if self.iter_count % self.config.train.eval_interval == 0:
                        results = self.evaluate()
                        stats.update(results)

                        # always save checkpoint with the greatest mean reward
                        if self.config.train.save_best:
                            if stats.get("reward/mean", -float("inf")) > best_reward:
                                best_reward = stats.get("reward/mean")
                                do_save = True
                            # in case ILQL reports reward estimate as one of its metrics
                            elif stats.get("metrics/reward", -float("inf")) > best_reward:
                                best_reward = stats.get("metrics/reward")
                                do_save = True
                            else:
                                do_save = False
                            do_save = torch.tensor(do_save, device=self.accelerator.device)
                            if torch.distributed.is_initialized():
                                torch.distributed.all_reduce(do_save, torch.distributed.ReduceOp.MAX)
                            if do_save:
                                best_path = f"{self.config.train.checkpoint_dir}/best_checkpoint"
                                logger.info(f"Saving the best state so far into {best_path}")
                                self.save(best_path)

                        # Report the metrics to Ray Tune.
                        if ray.is_initialized():
                            self.save("state")
                            with open("state/state.json", "w") as f:
                                json.dump(dict(iter_count=self.iter_count), f)
                            checkpoint = Checkpoint.from_directory("state")
                            session.report(filter_non_scalars(stats), checkpoint=checkpoint)

                    if not ray.is_initialized():
                        self.accelerator.log(stats, step=self.iter_count)

                    desc = " | ".join(f"{k}: {v:.2f}" for k, v in stats.items() if k.startswith("loss"))
                    tbar.set_description(f"[{desc}]")
                    tbar.update()

                    if self.iter_count >= self.total_steps:
                        subfolder = f"checkpoint_{self.iter_count:0{len(str(self.total_steps))}d}"
                        directory = os.path.join(self.config.train.checkpoint_dir, subfolder)
                        self.save(directory)
                        return self.evaluate()

                self.post_backward_callback()

            self.post_epoch_callback()
        tbar.close()

    @abstractmethod
    def get_arch(self, config: TRLConfig):
        """Returns a specific wrapper of the decoder architecture"""
        pass

    @abstractmethod
    def loss(self, batch) -> Tuple[float, Dict]:
        """Compute loss on a batch from `store` and return some statistics"""
        pass

    @abstractmethod
    def post_backward_callback(self):
        """Do something after model update"""
        pass

    @abstractmethod
    def post_epoch_callback(self):
        """Do something after exhausting/single pass over `self.store`"""
        pass


@register_trainer
class AcceleratePPOTrainer(AccelerateRLTrainer):
    """PPO Accelerate Trainer"""

    reward_fn: Callable[[List[str], List[str], List[str]], List[float]]
    tokenizer: AutoTokenizer

    def __init__(self, config: TRLConfig, **kwargs):
        """PPO Accelerate Trainer initialization

        Args:
            config: Config
        """
        super().__init__(config, **kwargs)

        # Setup rollout logging
        if config.train.rollout_logging_dir is not None:
            self.log_rollouts = True
            self.setup_rollout_logging(config)
        else:
            self.log_rollouts = False

        # Setup the rollout store
        # Rollouts contain the prompt & response, log probs, values and rewards - from each rollout
        self.store = PPORolloutStorage(self.tokenizer.pad_token_id)

        # Create the rollout store dataloader (for batching up rollouts)
        # TODO (jon-tow): This is only used to satisfy to `accelerator.prepare` call constraint below - remove in future
        rollout_loader: DataLoader = self.store.create_loader(self.config.train.batch_size, shuffle=True)

        # Prepare multi-GPU acceleration
        self.model, self.opt, self.scheduler, rollout_loader = self.accelerator.prepare(
            self.model, self.opt, self.scheduler, rollout_loader
        )

        self.store.clear_history()  # Clear the rollout store

        # Setup a reference model when hydra heads are not used
        if not hasattr(self.model, "frozen_head"):
            self.ref_model = self.get_arch(self.config)
            self.ref_model.to(self.accelerator.device)
            self.ref_model.eval()

        # Setup the KL controller
        # This helps prevent large divergences in the controller (policy)
        if config.method.target is not None:
            self.kl_ctl = AdaptiveKLController(config.method.init_kl_coef, config.method.target, config.method.horizon)
        else:
            self.kl_ctl = FixedKLController(config.method.init_kl_coef)

        # Create the parameters for the Hugging Face language model's generator
        # method (that generates new tokens from a prompt).
        # https://huggingface.co/docs/transformers/v4.25.1/en/main_classes/text_generation#transformers.GenerationMixin.generate
        if config.model.model_arch_type == "seq2seq":
            self.generate_kwargs = dict(
                config.method.gen_kwargs,
                eos_token_id=self.tokenizer.eop_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            if config.method.gen_experience_kwargs is not None:
                self.generate_experience_kwargs = dict(
                    config.method.gen_experience_kwargs,
                    eos_token_id=self.tokenizer.eop_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            else:
                self.generate_experience_kwargs = None
        else:
            self.generate_kwargs = dict(
                config.method.gen_kwargs,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            if config.method.gen_experience_kwargs is not None:
                self.generate_experience_kwargs = dict(
                    config.method.gen_experience_kwargs,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            else:
                self.generate_experience_kwargs = None

        # Setup stats tracker
        self.running_moments = RunningMoments()
        self.ref_mean = self.config.method.ref_mean
        self.ref_std = self.config.method.ref_std

    def get_arch(self, config: TRLConfig):
        """Get the model"""
        model_class = AutoModelForCausalLMWithHydraValueHead
        if config.model.model_arch_type == "seq2seq":
            model_class = AutoModelForSeq2SeqLMWithHydraValueHead

        from_fn = model_class.from_pretrained
        # backward-compat: Try to create a randomly initialized architecture from a config
        if issubclass(type(config.model.model_path), PretrainedConfig):
            from_fn = model_class.from_config

        model = from_fn(
            config.model.model_path,
            trust_remote_code=True,
            num_layers_unfrozen=config.model.num_layers_unfrozen,
            config=config
        )

        return model

    def loss(self, batch: PPORLBatch):
        """Forward pass & loss

        Args:
            batch: Previous batch of episodes
        """
        # Move `batch` data to `accelerator` device
        input_ids = batch.query_tensors.to(self.accelerator.device)
        response_tensors = batch.response_tensors.to(self.accelerator.device)
        attention_mask = batch.attention_mask.to(self.accelerator.device)
        old_logprobs = batch.logprobs.to(self.accelerator.device)
        old_values = batch.values.to(self.accelerator.device)
        old_rewards = batch.rewards.to(self.accelerator.device)
        response_length = old_rewards.shape[1]
        logger.debug(f"loss() - input ids shape: {input_ids.shape}, attention mask shape: {attention_mask.shape}")

        advantages, returns = self.config.method.get_advantages_and_returns(old_values, old_rewards, response_length)

        if self.config.model.model_arch_type == "seq2seq":
            # TODO: To be modified for glm and chatglm
            # input_ids = query_tensors
            decoder_input_ids = response_tensors
            # attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long().to(self.accelerator.device)
            position_ids = torch.stack(batch.position_ids).to(self.accelerator.device)
            # decoder_attention_mask = (
            #     decoder_input_ids.ne(self.tokenizer.pad_token_id).long().to(self.accelerator.device)
            # )
            # decoder_attention_mask[:, 0] = 1
            logger.debug(f"loss() - position ids shape: {position_ids.shape}")

            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids
                # decoder_input_ids=decoder_input_ids,
                # decoder_attention_mask=decoder_attention_mask,
            )

            logits = outputs.logits
            values_pred = outputs.value
            logprobs = logprobs_of_labels(logits[:, :-1, :], decoder_input_ids[:, 1:])
            mask = decoder_input_ids.ne(self.tokenizer.pad_token_id).long().to(self.accelerator.device)
            start = 0
            end = start + response_length
            logprobs, values_pred, mask = (
                logprobs[:, start:end],
                values_pred[:, start:end],
                mask[:, start:end],
            )
        else:
            # tokens = torch.cat((query_tensors, response_tensors), dim=1)
            # attention_mask = tokens.not_equal(self.tokenizer.pad_token_id).long().to(tokens.device)
            outputs = self.model(input_ids, attention_mask, return_dict=True)
            logits = outputs.logits
            values_pred = outputs.value
            logger.info(f"loss() - s1 values_pred shape: {values_pred.shape}")
            values_pred = values_pred[:, :-1]
            logger.info(f"loss() - s2 values_pred shape: {values_pred.shape}")
            logprobs = logprobs_of_labels(logits[:, :-1, :], input_ids[:, 1:])

            start = input_ids.shape[1] - 1
            end = start + response_length
            logprobs, values_pred, mask = (
                logprobs[:, start:end],
                values_pred[:, start:end],
                attention_mask[:, start:end],
            )
            logger.info(f"loss() - s3 values_pred shape: {values_pred.shape}")

        # TODO: need debugging here
        loss, stats = self.config.method.loss(
            logprobs=logprobs,
            values=values_pred,
            old_logprobs=old_logprobs,
            old_values=old_values,
            advantages=advantages,
            returns=returns,
            mask=mask,
        )

        return loss, stats

    def setup_rollout_logging(self, config):
        # Make rollout logging dir for this run and store config
        exists = os.path.exists(config.train.rollout_logging_dir)
        isdir = os.path.isdir(config.train.rollout_logging_dir)
        assert exists and isdir

        self.run_id = f"run-{uuid.uuid4()}"
        self.rollout_logging_dir = os.path.join(config.train.rollout_logging_dir, self.run_id)
        os.mkdir(self.rollout_logging_dir)

        with open(os.path.join(self.rollout_logging_dir, "config.json"), "w") as f:
            f.write(json.dumps(config.to_dict(), indent=2))

    def post_epoch_callback(self):
        """Post epoch callback

        Clears the store and creates `num_rollouts` new episodes.
        """
        if self.log_rollouts:
            self.store.export_history(location=self.rollout_logging_dir)
        self.store.clear_history()
        # Collect more rollouts for training
        self.make_experience(self.config.method.num_rollouts, self.iter_count)

    def post_backward_callback(self):
        self.kl_ctl.update(self.mean_kl.item(), n_steps=self.config.train.batch_size)

    def prepare_learning(self):
        eval_dataloader = self.eval_pipeline.create_loader(self.config.method.chunk_size)
        self.eval_dataloader = self.accelerator.prepare_data_loader(eval_dataloader)
        self.train_dataloader = self.store.create_loader(self.config.train.batch_size, shuffle=True)

        self.n_updates_per_batch = self.config.method.ppo_epochs
        self.total_steps = self.config.train.epochs * self.n_updates_per_batch * len(self.train_dataloader)
        self.total_steps = min(self.total_steps, self.config.train.total_steps)

    def add_prompt_pipeline(self, pipeline: BasePipeline):
        """Add a prompt pipeline dataloader to a trainer instance for the `make_experience` stage"""
        prompt_dataloader = pipeline.create_loader(self.config.method.chunk_size, shuffle=True)
        self.prompt_dataloader = self.accelerator.prepare_data_loader(prompt_dataloader)
        self.prompt_iterator = iter(self.prompt_dataloader)

    def make_experience(self, num_rollouts: int = 1024, iter_count: int = 0):  # noqa:
        """Make experiences

        Takes `chunk_size` number of prompts from `prompt_iterator`, samples
        from the model and then computes the KL against a reference model. Finally it
        then appends PPOElements to trainer's `store`.

        Args:
            num_rollouts: Number of rollouts to generate
            iter_count: Total number of updates run (i.e. number of updates run for all batches & epochs)
        """
        logger.info("Collecting rollouts")
        tbar = tqdm(
            total=num_rollouts,
            disable=os.environ.get("RANK", 0) != "0",
            desc=f"[rollout 0 / {num_rollouts}]",
            # Lower progress bar by 1 if we're in WARNING mode or above to avoid hiding high priority progress
            # bars (e.g. loss progress in trainers)
            position=logger.level >= logging.WARNING,
            # Leave progress bar if we're in INFO mode or lower to avoid spamming in suppressed verbosity levels
            leave=logger.level < logging.WARNING,
        )

        ppo_rl_elements = []
        stats = {}
        clock = Clock()

        while len(ppo_rl_elements) < num_rollouts:
            # Get next batch in prompt dataset and refresh if exhausted
            # TOOD (jon-tow): Make `prompt_dataloader` a cyclic/infinite DataLoader to not require manually
            # "refreshing" the contents of the `prompt_iterator`
            try:
                batch: PromptBatch = next(self.prompt_iterator)
            except StopIteration:
                self.prompt_iterator = iter(self.prompt_dataloader)
                batch = next(self.prompt_iterator)

            exp_generate_time = time()

            # Generate samples from the language model (similar to using HuggingFace `generate` method)
            logger.debug(f"generate() input `batch` keys: {batch.keys()}")
            samples = self.generate(**batch)
            for i in range(len(batch['input_ids'])):
                p = self.tokenizer.decode(batch['input_ids'][i], skip_special_tokens=True)
                gt = self.tokenizer.decode(samples[i], skip_special_tokens=True)
                logger.debug(f"prompt: {p}, generated result: {gt}, samples: {samples[i]}")
            logger.debug(f"make_experience() - input ids shape: {batch['input_ids'].shape}, samples shape: {samples.shape}")
            stats["time/exp_generate"] = time() - exp_generate_time

            prompt_tensors = batch['input_ids']
            device = samples.device

            prompt_sizes = torch.tensor([prompt_tensors.shape[1]] * len(prompt_tensors), device=device)
            padded_samples = self.accelerator.pad_across_processes(
                samples, dim=1, pad_index=self.tokenizer.pad_token_id, pad_first=False
            )
            padded_prompts = self.accelerator.pad_across_processes(
                prompt_tensors, dim=1, pad_index=self.tokenizer.pad_token_id, pad_first=False
            )
            gathered_samples = self.accelerator.gather(padded_samples)
            gathered_prompts = self.accelerator.gather(padded_prompts)
            gathered_prompt_sizes = self.accelerator.gather(prompt_sizes)

            if self.accelerator.is_main_process:
                all_str_samples, all_str_prompts, all_str_outputs, all_str_prefixes, _ = self.decode(
                    gathered_prompts, gathered_samples, gathered_prompt_sizes
                )

                exp_score_time = time()
                all_scores = torch.tensor(
                    self.reward_fn(
                        samples=all_str_samples,
                        prompts=all_str_prompts,
                        outputs=all_str_outputs,
                    ),
                    dtype=torch.float,
                    device=device,
                )
                stats["time/exp_score"] = time() - exp_score_time

                all_scores = list(all_scores.reshape(self.accelerator.num_processes, -1).unbind())
            else:
                all_scores = None

            if torch.distributed.is_initialized():
                scores = torch.empty(len(samples), device=device)
                torch.distributed.scatter(scores, all_scores)
            else:
                scores = torch.tensor(all_scores[0])

            str_samples, str_prompts, str_outputs, str_prefixes, outputs = self.decode(prompt_tensors, samples)

            # Pad the sample outputs
            # outputs = self.tokenizer(str_outputs).input_ids
            # if self.config.model.model_arch_type == "seq2seq":
            #     # add <pad> to the start of the output
            #     for i in range(len(outputs)):
            #         outputs[i] = [self.tokenizer.pad_token_id] + outputs[i]
            # outputs = list(map(torch.LongTensor, outputs))
            maxsize = max(map(len, outputs))
            outputs = [
                F.pad(
                    output,
                    (0, maxsize - len(output)),
                    value=self.tokenizer.pad_token_id,
                )
                for output in outputs
            ]
            sample_outputs = torch.vstack(outputs).to(device)

            # store statistics of the initial rollout as reference
            if self.ref_mean is None:
                self.ref_mean, self.ref_std = scores.mean(), scores.std()
            all_scores_mean, all_scores_std = self.running_moments.update(scores)
            stats["exp_scores/mean"] = all_scores_mean
            stats["exp_scores/std"] = all_scores_std
            stats["exp_scores/running_mean"] = self.running_moments.mean
            stats["exp_scores/running_std"] = self.running_moments.std

            if self.config.method.scale_reward == "running":
                scores /= self.running_moments.std
            elif self.config.method.scale_reward == "ref":
                scores /= self.ref_std

            clip_reward = self.config.method.cliprange_reward
            if clip_reward:
                scores = torch.clip(scores, -clip_reward, clip_reward)

            # Precompute logprobs, values
            logger.debug(f"sample_outputs shape: {sample_outputs.shape}")
            logger.debug(f"str_prompts[0]: {str_prompts[0]}, str_outputs[0]: {str_outputs[0]}, input_ids[0]: {batch['input_ids'][0]}, sample_outputs[0]: {sample_outputs[0]}")
            # logger.debug(f"str_prompts[1]: {str_prompts[1]}, str_outputs[1]: {str_outputs[1]}, input_ids[1]: {batch['input_ids'][1]}, sample_outputs[1]: {sample_outputs[1]}")
            self.tokenizer.padding_side = "right"
            if self.config.model.model_arch_type == "seq2seq":
                input_ids, attention_mask, position_ids = [], [], []
                for str_prompt, str_output, str_prefix in zip(str_prompts, str_outputs, str_prefixes):
                    encoded_prompt = self.tokenizer(str_prompt, str_prefix + self.tokenizer.mask_token)
                    prompt_length = len(encoded_prompt['input_ids'])
                    label_length = len(self.tokenizer.tokenize(str_output)) + 1
                    if prompt_length + label_length > self.max_length:
                        num_tokens_to_remove = prompt_length + label_length - self.max_length
                        for _ in range(num_tokens_to_remove):
                            if prompt_length > label_length:
                                prompt_length -= 1
                            else:
                                label_length -= 1
                    else:
                        label_length = self.max_length - prompt_length
                    assert prompt_length > 0
                    assert label_length > 0
                    assert prompt_length + label_length <= self.max_length
                    encoded_dict = self.tokenizer(str_prompt, str_prefix + self.tokenizer.mask_token,
                                                  max_length=prompt_length,
                                                  truncation="only_first",
                                                  return_tensors="pt",
                                                  return_attention_mask=True,
                                                  return_token_type_ids=False)
                    encoded_dict = self.tokenizer.build_inputs_for_generation(encoded_dict, targets=str_output,
                                                                              max_gen_length=label_length, padding=True)
                    input_ids.append(encoded_dict['input_ids'])
                    attention_mask.append(encoded_dict['attention_mask'])
                    position_ids.append(encoded_dict['position_ids'])
                input_ids = torch.cat(input_ids).to(device)
                attention_mask = torch.cat(attention_mask).to(device)
                position_ids = torch.cat(position_ids).to(device)
                with torch.no_grad():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids
                    )
                    logits = outputs.logits
                    values = outputs.value
                    if hasattr(self.model, "frozen_head"):
                        ref_logits = self.model.forward_hydra(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            # decoder_input_ids=sample_outputs,
                            # decoder_attention_mask=decoder_attention_mask,
                            return_dict=True,
                        ).logits
                    else:
                        ref_logits = self.ref_model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            # decoder_input_ids=sample_outputs,
                            # decoder_attention_mask=decoder_attention_mask,
                            return_dict=True,
                        ).logits

            else:
                # all_tokens = torch.cat((prompt_tensors.to(device), sample_outputs), dim=1)
                # attention_mask = all_tokens.not_equal(self.tokenizer.pad_token_id).long().to(device)
                encoded_dict = self.tokenizer(str_prompts, str_outputs, max_length=self.max_length, return_tensors="pt",
                                              truncation="longest_first", padding="max_length", return_token_type_ids=False)
                input_ids = encoded_dict['input_ids'].to(device)
                attention_mask = encoded_dict['attention_mask'].to(device)
                position_ids = None
                with torch.no_grad():
                    logits, *_, values = self.model(
                        input_ids,
                        attention_mask=attention_mask,
                    )
                    # TODO(dahoas): When hydra model works need to also support generation on hydra head
                    if hasattr(self.model, "frozen_head"):
                        ref_logits = self.model.forward_hydra(
                            input_ids,
                            attention_mask=attention_mask,
                            return_dict=True,
                        ).logits
                    else:
                        ref_logits = self.ref_model(
                            input_ids,
                            attention_mask=attention_mask,
                            return_dict=True,
                        ).logits
                        ref_logits = ref_logits.to(device)
            self.tokenizer.padding_side = self.padding_side

            if self.config.model.model_arch_type == "seq2seq":
                # TODO: to be tested against glm and chatglm
                logprobs = logprobs_of_labels(logits[:, :-1, :], sample_outputs[:, 1:])
                ref_logprobs = logprobs_of_labels(ref_logits[:, :-1, :], sample_outputs[:, 1:])
            else:
                logprobs = logprobs_of_labels(logits[:, :-1, :], input_ids[:, 1:])
                ref_logprobs = logprobs_of_labels(ref_logits[:, :-1, :], input_ids[:, 1:])

            n_samples: int = samples.shape[0]
            logprobs = logprobs.cpu()
            ref_logprobs = ref_logprobs.cpu()
            # prompt_tensors = prompt_tensors.cpu()
            sample_outputs = sample_outputs.cpu()
            input_ids = input_ids.cpu()
            attention_mask = attention_mask.cpu()
            position_ids = position_ids.cpu() if position_ids is not None else None
            values = values.cpu()[:, :-1]

            # Estimate the KL divergence between the model and reference model
            if self.config.model.model_arch_type == "seq2seq":
                # TODO: to be modified for glm and chatglm
                attention_mask_tmp = sample_outputs != self.tokenizer.pad_token_id
                start = 0
            else:
                attention_mask_tmp = attention_mask
                start = prompt_tensors.shape[1] - 1

            ends = start + attention_mask_tmp[:, start:].sum(1)

            # Get the logprobs and values, for tokens that are not padding
            # or beginning of sequences tokens. These are from the model (not the reference model)
            all_values = [values[ix, start : ends[ix]] for ix in range(n_samples)]
            all_logprobs = [logprobs[ix, start : ends[ix]] for ix in range(n_samples)]

            log_ratio = (logprobs - ref_logprobs) * attention_mask_tmp[:, :-1].cpu()
            self.mean_kl = (log_ratio.exp() - 1 - log_ratio).mean().to(device)
            kl_penalty = self.kl_ctl.value * -log_ratio
            kl_penalty = [xs[start : ends[ix]] for ix, xs in enumerate(kl_penalty)]

            rollout_count = 0

            for sample_idx in range(n_samples):
                if len(kl_penalty[sample_idx]) == 0 or len(all_logprobs[sample_idx]) == 0:
                    continue

                rewards = kl_penalty[sample_idx]
                rewards[-1] += scores[sample_idx].cpu()

                logger.debug(f"make_experience() - attention mask shape: {attention_mask[sample_idx].shape}")

                ppo_rl_elements.append(
                    PPORLElement(
                        query_tensor=input_ids[sample_idx],
                        # query_tensor=prompt_tensors[sample_idx],
                        response_tensor=sample_outputs[sample_idx],
                        attention_mask=attention_mask[sample_idx],
                        position_ids=position_ids[sample_idx] if position_ids is not None else None,
                        logprobs=all_logprobs[sample_idx],
                        values=all_values[sample_idx],
                        rewards=rewards,
                    )
                )

                rollout_count += 1
            exp_time = clock.tick()
            tbar.set_description(f"[rollout {len(ppo_rl_elements)} / {num_rollouts}]")
            tbar.update(min(rollout_count, num_rollouts))
        tbar.close()

        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(self.mean_kl, torch.distributed.ReduceOp.AVG)

        stats["policy/sqrt_kl"] = torch.sqrt(self.mean_kl)
        stats["kl_ctl_value"] = self.kl_ctl.value
        stats["time/exp"] = exp_time

        if not ray.is_initialized():
            self.accelerator.log(stats, step=iter_count)

        # Push samples and rewards to trainer's rollout storage
        self.push_to_store(ppo_rl_elements)


def get_model_norm(model):
    with torch.no_grad():
        total = 0.0
        for param in model.parameters():
            should_gather = hasattr(
                param,
                'ds_id') and param.ds_status == ZeroParamStatus.NOT_AVAILABLE
            with deepspeed.zero.GatheredParameters(param,
                                                   enabled=should_gather):
                total += float(param.float().norm())

    return total


def gather_log_probs(logits, labels):
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)


class DeepSpeedPPOTrainer():

    def __init__(self, rlhf_engine, args):
        self.rlhf_engine = rlhf_engine
        self.actor_model = self.rlhf_engine.actor
        self.critic_model = self.rlhf_engine.critic
        self.ref_model = self.rlhf_engine.ref
        self.reward_model = self.rlhf_engine.reward
        self.tokenizer = self.rlhf_engine.tokenizer
        self.args = args
        self.max_answer_seq_len = args.max_gen_length
        # self.end_of_conversation_token_id = self.tokenizer(
        #     args.end_of_conversation_token)['input_ids'][-1]
        self.end_of_conversation_token_id = self.tokenizer.eos_token_id

        # Those value can be changed
        self.kl_ctl = 0.02
        self.clip_reward_value = 5
        self.cliprange = 0.2
        self.cliprange_value = 0.2
        self.gamma = 1.0
        self.lam = 0.95

    def _generate_sequence(self, prompts):

        max_min_length = self.max_answer_seq_len + prompts.shape[1]

        with torch.no_grad():
            logger.debug(f"[_generate_sequence] prompts: {prompts}")
            seq = self.actor_model.module.generate(prompts,
                                                   max_length=max_min_length,
                                                   min_length=max_min_length)
            logger.debug(f"[_generate_sequence] seq: {seq}")

        # Filter out seq with no asnwers (or very short). This happens when users directly use the pre-training ckpt without supervised finetuning
        # NOTE: this will causes each GPU has different number of examples
        batch_size = seq.shape[0]
        prompt_length = prompts.shape[1]
        ans = seq[:, prompt_length:]
        self.prompt_length = prompt_length
        valid_ans_len = (ans != self.tokenizer.pad_token_id).sum(dim=-1)
        out_seq = []
        for i in range(batch_size):
            # if the answer is shorter than 1 token, drop it
            if valid_ans_len[i] <= 1:
                continue
            else:
                out_seq.append(seq[i:i + 1])
        out_seq = torch.cat(out_seq, dim=0)  # concat output in the batch dim
        logger.debug(f"[_generate_sequence] out_seq: {out_seq}")

        return out_seq

    def generate_experience(self, prompts):
        self.eval()
        seq = self._generate_sequence(prompts)
        self.train()

        pad_token_id = self.tokenizer.pad_token_id
        # TODO: need to modify fo GLM-based models
        attention_mask = seq.not_equal(pad_token_id).long()

        with torch.no_grad():
            output = self.actor_model(seq, attention_mask=attention_mask)
            output_ref = self.ref_model(seq, attention_mask=attention_mask)
            reward_score = self.reward_model.reward(seq, attention_mask)[1].detach()
            values = self.critic_model.reward(seq, attention_mask)[0].detach()
            # reward_score = self.reward_model.forward_value(
            #     seq, attention_mask,
            #     prompt_length=self.prompt_length)['chosen_end_scores'].detach(
            # )
            # values = self.critic_model.forward_value(
            #     seq, attention_mask, return_value_only=True).detach()[:, :-1]

        logits = output.logits
        logits_ref = output_ref.logits

        return {
            'prompts': prompts,
            'logprobs': gather_log_probs(logits[:, :-1, :], seq[:, 1:]),
            'ref_logprobs': gather_log_probs(logits_ref[:, :-1, :], seq[:, 1:]),
            'value': values,
            'rewards': reward_score,
            'input_ids': seq,
            "attention_mask": attention_mask
        }

    def compute_rewards(self, prompts, log_probs, ref_log_probs, reward_score,
                        action_mask):

        kl_divergence_estimate = -self.kl_ctl * (log_probs - ref_log_probs)
        rewards = kl_divergence_estimate
        start = prompts.shape[1] - 1
        ends = start + action_mask[:, start:].sum(1)
        reward_clip = torch.clamp(reward_score, -self.clip_reward_value,
                                  self.clip_reward_value)
        batch_size = log_probs.shape[0]
        for j in range(batch_size):
            rewards[j, start:ends[j]][-1] += reward_clip[j]

        return rewards

    def train_rlhf(self, inputs):
        # train the rlhf mode here
        ### process the old outputs
        prompts = inputs['prompts']
        log_probs = inputs['logprobs']
        ref_log_probs = inputs['ref_logprobs']
        reward_score = inputs['rewards']
        values = inputs['value']
        attention_mask = inputs['attention_mask']
        seq = inputs['input_ids']

        start = prompts.size()[-1] - 1
        action_mask = attention_mask[:, 1:]

        old_values = values
        with torch.no_grad():
            old_rewards = self.compute_rewards(prompts, log_probs,
                                               ref_log_probs, reward_score,
                                               action_mask)
            advantages, returns = self.get_advantages_and_returns(
                old_values, old_rewards, start)

        ### process the new outputs
        batch = {'input_ids': seq, "attention_mask": attention_mask}
        actor_prob = self.actor_model(**batch, use_cache=False).logits
        actor_log_prob = gather_log_probs(actor_prob[:, :-1, :],
                                          inputs['input_ids'][:, 1:])
        actor_loss = self.actor_loss_fn(actor_log_prob[:, start:],
                                        log_probs[:, start:], advantages,
                                        action_mask[:, start:])
        self.actor_model.backward(actor_loss)
        self.actor_model.step()
        value = self.critic_model.forward_value(**batch,
                                                return_value_only=True,
                                                use_cache=False)[:, :-1]
        critic_loss = self.critic_loss_fn(value[:, start:], old_values[:,
                                                            start:],
                                          returns, action_mask[:, start:])
        self.critic_model.backward(critic_loss)
        self.critic_model.step()

        return actor_loss, critic_loss

    def actor_loss_fn(self, logprobs, old_logprobs, advantages, mask):
        ## policy gradient loss
        log_ratio = (logprobs - old_logprobs) * mask
        ratio = torch.exp(log_ratio)
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - self.cliprange,
                                             1.0 + self.cliprange)
        pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2) * mask) / mask.sum()
        return pg_loss

    def critic_loss_fn(self, values, old_values, returns, mask):
        ## value loss
        values_clipped = torch.clamp(
            values,
            old_values - self.cliprange_value,
            old_values + self.cliprange_value,
            )
        vf_loss1 = (values - returns)**2
        vf_loss2 = (values_clipped - returns)**2
        vf_loss = 0.5 * torch.sum(
            torch.max(vf_loss1, vf_loss2) * mask) / mask.sum()
        return vf_loss

    def get_advantages_and_returns(self, values, rewards, start):
        # Adopted from https://github.com/CarperAI/trlx/blob/main/trlx/models/modeling_ppo.py#L134
        lastgaelam = 0
        advantages_reversed = []
        length = rewards.size()[-1]
        for t in reversed(range(start, length)):
            nextvalues = values[:, t + 1] if t < length - 1 else 0.0
            delta = rewards[:, t] + self.gamma * nextvalues - values[:, t]
            lastgaelam = delta + self.gamma * self.lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values[:, start:]
        return advantages.detach(), returns

    def _validate_training_mode(self):
        assert self.actor_model.module.training
        assert self.critic_model.module.training

    def _validate_evaluation_mode(self):
        assert not self.actor_model.module.training
        assert not self.critic_model.module.training
        assert not self.ref_model.module.training
        assert not self.reward_model.module.training

    def train(self):
        self.actor_model.train()
        self.critic_model.train()

    def eval(self):
        self.actor_model.eval()
        self.critic_model.eval()
        self.reward_model.eval()
        self.ref_model.eval()

    def dump_model_norms(self, tag):
        actor_model_norm = get_model_norm(self.actor_model)
        ref_model_norm = get_model_norm(self.ref_model)
        critic_model_norm = get_model_norm(self.critic_model)
        reward_model_norm = get_model_norm(self.reward_model)
        if self.args.global_rank <= 0:
            logger.info(f'{tag} global_actor_model_norm', actor_model_norm,
                            self.args.local_rank)
            logger.info(f'{tag} global_ref_model_norm', ref_model_norm,
                            self.args.local_rank)
            logger.info(f'{tag} global_critic_model_norm', critic_model_norm,
                            self.args.local_rank)
            logger.info(f'{tag} global_reward_model_norm', reward_model_norm,
                            self.args.local_rank)


class DeepSpeedPPOPTXTrainer(DeepSpeedPPOTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train_unsupervised(self, inputs, unsup_coef):
        # Train the unsupervised model here
        self._validate_training_mode()

        outputs = self.actor_model(**inputs, use_cache=False)
        loss = outputs.loss
        self.actor_model.backward(unsup_coef * loss)
        self.actor_model.step()

        return loss
