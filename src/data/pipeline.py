
import os
import sys
import time
import json
import torch

from abc import abstractmethod
from typing import Optional, Any, Callable, Dict, Iterable, List, Union
from dataclasses import dataclass
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from torchtyping import TensorType
from transformers import PreTrainedTokenizer, PreTrainedTokenizerBase
from transformers.tokenization_utils_base import PaddingStrategy

from src.data.data_types import PPORLBatch, PPORLElement
from src.utils.config import TRLConfig


# specifies a dictionary of architectures
_DATAPIPELINE: Dict[str, any] = {}  # registry

@dataclass
class GeneralElement:
    """
    General element outputted by a data pipeline
    """

    pass


@dataclass
class RLElement:
    """
    Batch element for RL model
    """

    state: Iterable[str] = None  # Context/prompts
    action: TensorType["N"] = None  # Tokens generated by model given prompts
    reward: float = None  # Reward obtained for that generation


@dataclass
class BatchElement:
    """
    General batch element for any transformer to use in its forward pass
    """

    tokens: TensorType["BATCH", "SEQ_LEN"]
    masks: TensorType["BATCH", "SEQ_LEN"]


@dataclass
class GLMDataCollator:

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # batch = self.tokenizer.pad(
        #     features,
        #     padding=self.padding,
        #     max_length=self.max_length,
        #     pad_to_multiple_of=self.pad_to_multiple_of,
        #     return_tensors=self.return_tensors,
        # )
        max_length = max(map(lambda x: x['input_ids'].shape[0], features))
        input_ids_list = []
        attention_mask_list = []
        position_ids_list = []
        labels_list = []
        for feature in features:
            input_ids = feature['input_ids']
            seq_length = input_ids.shape[0]
            # padding for GLM generation: cls_token_id + prompt_tokens + mask_token_id + [eos_token_id]*N + sop_token_id
            input_ids = torch.cat((input_ids[:-1],
                                   torch.tensor([self.tokenizer.pad_token_id]*(max_length-seq_length), dtype=input_ids.dtype),
                                   input_ids[-1:]
                                   ), dim=0)
            input_ids_list.append(input_ids)
            attention_mask_list.append(feature['generation_attention_mask'])
            position_ids_list.append(feature['position_ids'])
            if "labels" in feature:
                labels_list.append(feature['labels'])

        batch = {
            "input_ids": torch.stack(input_ids_list, dim=0),
            "generation_attention_mask": torch.stack(attention_mask_list, dim=0),
            "position_ids": torch.stack(position_ids_list, dim=0)
        }

        if len(labels_list) > 0:
            batch['labels'] = torch.stack(labels_list, dim=0)

        return batch


def register_datapipeline(name):
    """Decorator used register a CARP architecture
    Args:
        name: Name of the architecture
    """

    def register_class(cls, name):
        _DATAPIPELINE[name] = cls
        setattr(sys.modules[__name__], name, cls)
        return cls

    if isinstance(name, str):
        name = name.lower()
        return lambda c: register_class(c, name)

    cls = name
    name = cls.__name__
    register_class(cls, name.lower())

    return cls


@register_datapipeline
class BasePipeline(Dataset):
    def __init__(self, path: str = "dataset"):
        super().__init__()

    @abstractmethod
    def __getitem__(self, index: int) -> GeneralElement:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def create_loader(
            self,
            batch_size: int,
            shuffle: bool,
            prep_fn: Callable = None,
            num_workers: int = 0,
    ) -> DataLoader:
        """
        Create a dataloader for the pipeline

        :param prep_fn: Typically a tokenizer. Applied to GeneralElement after collation.
        """
        pass


class BaseRolloutStore(Dataset):
    def __init__(self, capacity=-1):
        self.history: Iterable[Any] = None
        self.capacity = capacity

    @abstractmethod
    def push(self, exps: Iterable[Any]):
        """
        Push experiences to rollout storage
        """
        pass

    def __getitem__(self, index: int) -> RLElement:
        return self.history[index]

    def __len__(self) -> int:
        return len(self.history)

    @abstractmethod
    def create_loader(
            self,
            batch_size: int,
            shuffle: bool,
            prep_fn: Callable = None,
            num_workers: int = 0,
    ) -> DataLoader:
        """
        Create a dataloader for the rollout store

        :param prep_fn: Applied to RLElement after collation (typically tokenizer)
        :type prep_fn: Callable
        """
        pass


@register_datapipeline
class PanguPipeline(BasePipeline):
    def __init__(self, prompts: List[dict], config: TRLConfig, tokenizer: PreTrainedTokenizer):

        super().__init__()

        self.prompts = prompts
        self.tokenizer = tokenizer
        self.config = config
        self.max_prompt_length = config.train.seq_length - config.method.gen_kwargs["max_new_tokens"]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        data = self.prompts[idx]
        prompt = data['prompt']
        prefix = data['prefix']
        encoded_dict = self.tokenizer(prompt, self.tokenizer.sep_token + prefix,
                                      max_length=self.max_prompt_length,
                                      return_tensors="pt",
                                      truncation="only_first",
                                      # padding="max_length",
                                      add_special_tokens=False,
                                      return_token_type_ids=False)

        return {
            "input_ids": encoded_dict['input_ids'][0],
            "attention_mask": encoded_dict['attention_mask'][0],
        }

    def create_loader(self, batch_size: int, shuffle=False) -> DataLoader:
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)


@register_datapipeline
class GLMPipeline(BasePipeline):
    def __init__(self, prompts: List[dict], config: TRLConfig, tokenizer: PreTrainedTokenizer):

        super().__init__()

        self.prompts = prompts
        self.tokenizer = tokenizer
        # self.config = config
        self.max_generation_length = config.method.gen_kwargs["max_new_tokens"]
        self.max_prompt_length = config.train.seq_length - self.max_generation_length

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        data = self.prompts[idx]
        prompt = data['prompt']
        prefix = data['prefix']

        inputs = self.tokenizer(prompt, prefix + self.tokenizer.mask_token,
                                max_length=self.max_prompt_length,
                                truncation="only_first",
                                # padding="max_length",
                                return_tensors="pt",
                                return_token_type_ids=False)
        inputs_glm = self.tokenizer.build_inputs_for_generation(inputs, max_gen_length=self.max_generation_length,
                                                                padding=True)
        return {
            "input_ids": inputs_glm['input_ids'][0],
            "position_ids": inputs_glm['position_ids'][0],
            "generation_attention_mask": inputs_glm['generation_attention_mask'][0]
        }

    def create_loader(self, batch_size: int, shuffle=False) -> DataLoader:
        # collate_fn = GLMDataCollator(self.tokenizer)
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)#, collate_fn=collate_fn)


@register_datapipeline
class ChatGLMPipeline(BasePipeline):
    def __init__(self, prompts: List[dict], config: TRLConfig, tokenizer: PreTrainedTokenizer):

        super().__init__()

        self.prompts = prompts
        self.tokenizer = tokenizer
        self.config = config
        self.max_prompt_length = config.train.seq_length - config.method.gen_kwargs["max_new_tokens"]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        data = self.prompts[idx]
        prompt = data['prompt']
        encoded_dict = self.tokenizer(prompt, max_length=self.max_prompt_length, return_tensors="pt",
                                      truncation="only_first", padding="max_length")

        return {
            "input_ids": encoded_dict['input_ids'][0],
            # "attention_mask": encoded_dict['attention_mask'][0],
        }

    def create_loader(self, batch_size: int, shuffle=False) -> DataLoader:
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)


class PPORolloutStorage(BaseRolloutStore):
    """
    Rollout storage for training PPO
    """

    def __init__(self, pad_token_id):
        super().__init__()

        self.pad_token_id = pad_token_id
        self.history: Iterable[PPORLElement] = [None]

    def push(self, exps: Iterable[PPORLElement]):
        self.history += exps

    def clear_history(self):
        self.history = []

    def export_history(self, location: str):
        assert os.path.exists(location)

        fpath = os.path.join(location, f"epoch-{str(time.time())}.json")

        def exp_to_dict(exp):
            {k: v.cpu().tolist() for k, v in exp.__dict__.items()}

        data = [exp_to_dict(exp) for exp in self.history]
        with open(fpath, "w") as f:
            f.write(json.dumps(data, indent=2))

    def __getitem__(self, index: int) -> PPORLElement:
        return self.history[index]

    def __len__(self) -> int:
        return len(self.history)

    def create_loader(
            self,
            batch_size: int,
            shuffle: bool,
    ) -> DataLoader:
        def collate_fn(elems: Iterable[PPORLElement]):
            return PPORLBatch(
                torch.stack([elem.query_tensor for elem in elems]),
                # # Left padding of already left-padded queries
                # pad_sequence(
                #     [elem.query_tensor.flip(0) for elem in elems],
                #     padding_value=self.pad_token_id,
                #     batch_first=True,
                # ).flip(1),
                # Right pad the rest, to have a single horizontal query/response split
                pad_sequence(
                    [elem.response_tensor for elem in elems],
                    padding_value=self.pad_token_id,
                    batch_first=True,
                ),
                torch.stack([elem.attention_mask for elem in elems]),
                [elem.position_ids for elem in elems],
                pad_sequence(
                    [elem.logprobs for elem in elems],
                    padding_value=0.0,
                    batch_first=True,
                ),
                pad_sequence([elem.values for elem in elems], padding_value=0.0, batch_first=True),
                pad_sequence(
                    [elem.rewards for elem in elems],
                    padding_value=0.0,
                    batch_first=True,
                ),
            )

        return DataLoader(self, batch_size, shuffle=shuffle, collate_fn=collate_fn)