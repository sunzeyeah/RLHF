from typing import Optional

import torch
import torch.nn as nn

from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import ModelOutput

from src.models.lora import LoRAModule


class SFTModel(LoRAModule):
    """
    SFT model base class with LoRA enabled

    Args:
        config (PretrainedConfig): model config.
        model (nn.Module): SFT model.
    """

    def __init__(self,
                 config: PretrainedConfig,
                 model: nn.Module) -> None:
        super().__init__(lora_rank=config.lora_rank,
                         lora_alpha=config.lora_alpha,
                         lora_train_bias=config.lora_train_bias)
        self.model = model
        self.convert_to_lora()

        # if value_head is not None:
        #     if value_head.out_features != 1:
        #         raise ValueError("The value head of reward model's output dim should be 1!")
        #     self.value_head = value_head
        # else:
        #     self.value_head = nn.Linear(model.config.n_embd, 1)

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                **kwargs) -> ModelOutput:

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask,
                             position_ids=position_ids, labels=labels, **kwargs)

        return outputs

