
import torch
import loralib as lora

from torch import nn
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers.tokenization_utils import PreTrainedTokenizer

from src.models.loss import PairWiseLoss
from src.models.lora import convert_to_lora_recursively


class RewardModel(PreTrainedModel):
    supports_gradient_checkpointing = True

    def __init__(self, config, model, tokenizer):
        super().__init__(config)
        self.config = config
        self.model_type = config.model_type
        self.pad_id = tokenizer.pad_token_id
        self.transformer = model
        self.v_head = nn.Linear(config.hidden_size, 1, bias=False)
        self.loss_fn = PairWiseLoss()
        if config.lora_rank > 0:
            convert_to_lora_recursively(model, config.lora_rank, config.lora_alpha)
            lora.mark_only_lora_as_trainable(model, config.lora_train_bias)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, PreTrainedModel):
            module.gradient_checkpointing = value

    def reward(
            self,
            input_ids=None,
            attention_mask=None,
            position_ids=None
    ):
        transformer_outputs = self.transformer(input_ids, attention_mask=attention_mask, position_ids=position_ids)
        if self.model_type == "glm":
            hidden_states = transformer_outputs.mems[-1]
        else:
            hidden_states = transformer_outputs[0]
        rewards = self.v_head(hidden_states).squeeze(-1)

        # outputs = self.body(sequences, attention_mask=attention_mask)
        # last_hidden_states = outputs['last_hidden_state']
        # values = self.value_head(last_hidden_states)[:, :-1]

        rewards = rewards.mean(dim=-1)
        if len(rewards.shape) == 2:
            rewards = rewards.squeeze(1)    # ensure shape is (B)

        return rewards

    def forward(
            self,
            chosen_input_ids,
            chosen_attention_mask=None,
            chosen_position_ids=None,
            rejected_input_ids=None,
            rejected_attention_mask=None,
            rejected_position_ids=None,
            past_key_values=None,
            token_type_ids=None,
            head_mask=None,
            inputs_embeds=None,
            mc_token_ids=None,
            labels=None,
            return_dict=False,
            output_attentions=False,
            output_hidden_states=False,
    ):
        chosen_reward = self.reward(chosen_input_ids, attention_mask=chosen_attention_mask, position_ids=chosen_position_ids)
        if rejected_input_ids is not None:
            reject_reward = self.reward(rejected_input_ids, attention_mask=rejected_attention_mask, position_ids=rejected_position_ids)
            loss = self.loss_fn(chosen_reward, reject_reward)
        else:
            reject_reward = None
            loss = None

        # # Split the inputs and rewards into two parts, chosen and rejected
        # assert len(input_ids.shape) == 2
        # bs = input_ids.shape[0] // 2
        # chosen = input_ids[:bs]
        # rejected = input_ids[bs:]
        # chosen_rewards = rewards[:bs]
        # rejected_rewards = rewards[bs:]
        #
        # # Compute pairwise loss. Only backprop on the last value before padding
        # loss = 0
        # inference = False
        # for i in range(bs):
        #     if torch.all(torch.eq(chosen[i], rejected[i])).item():
        #         c_inds = (chosen[i] == self.pad_id).nonzero()
        #         c_ind = c_inds[0].item() if len(c_inds) > 0 else chosen.shape[1]
        #         chosen_end_scores.append(chosen_rewards[i, c_ind - 1])
        #         inference = True
        #         continue
        #
        #     # Check if there is any padding otherwise take length of sequence
        #     c_inds = (chosen[i] == self.pad_id).nonzero()
        #     c_ind = c_inds[0].item() if len(c_inds) > 0 else chosen.shape[1]
        #     r_inds = (rejected[i] == self.pad_id).nonzero()
        #     r_ind = r_inds[0].item() if len(r_inds) > 0 else rejected.shape[1]
        #     end_ind = max(c_ind, r_ind)
        #
        #     # Retrieve first index where trajectories diverge
        #     divergence_ind = (chosen[i] != rejected[i]).nonzero()[0]
        #     assert divergence_ind > 0
        #
        #     # Index into the correct rewards
        #     c_truncated_reward = chosen_rewards[i][divergence_ind:end_ind]
        #     r_truncated_reward = rejected_rewards[i][divergence_ind:end_ind]
        #
        #     # Append the last rewards to the list of end scores
        #     chosen_end_scores.append(c_truncated_reward[-1])
        #     rejected_end_scores.append(r_truncated_reward[-1])
        #
        #     # Compute loss
        #     loss += -torch.log(torch.sigmoid(c_truncated_reward - r_truncated_reward)).mean()
        # loss = loss / bs

        # if not inference:
        #     chosen_end_scores = torch.stack(chosen_end_scores)
        #     rejected_end_scores = torch.stack(rejected_end_scores)
        #
        # if inference:
        #     chosen_end_scores = torch.stack(chosen_end_scores)
        #     return {"chosen_end_scores": chosen_end_scores}

        return {
            "loss": loss,
            "chosen_reward": torch.sigmoid(chosen_reward),
            "reject_reward": torch.sigmoid(reject_reward) if reject_reward is not None else reject_reward,
        }


# class RewardModelWithLoRA(LoRAModule):
#     def __init__(self,
#                  config: PretrainedConfig,
#                  model: nn.Module,
#                  tokenizer: PreTrainedTokenizer) -> None:
#         super().__init__(lora_rank=config.lora_rank,
#                          lora_alpha=config.lora_alpha,
#                          lora_train_bias=config.lora_train_bias)
#         self.config = config
#         self.model_type = config.model_type
#         self.pad_id = tokenizer.pad_token_id
#
#         self.transformer = model
#         self.v_head = nn.Linear(config.hidden_size, 1, bias=False)
#         self.loss_fn = PairWiseLoss()
#         self.convert_to_lora()
#
#     def reward(
#             self,
#             input_ids=None,
#             attention_mask=None,
#             position_ids=None
#     ):
#         transformer_outputs = self.transformer(input_ids, attention_mask=attention_mask, position_ids=position_ids)
#         if self.model_type == "glm":
#             hidden_states = transformer_outputs.mems[-1]
#         else:
#             hidden_states = transformer_outputs[0]
#         rewards = self.v_head(hidden_states).squeeze(-1)
#
#         rewards = rewards.mean(dim=-1)
#         if len(rewards.shape) == 2:
#             rewards = rewards.squeeze(1)    # ensure shape is (B)
#
#         return rewards
#
#     def forward(
#             self,
#             chosen_input_ids,
#             chosen_attention_mask=None,
#             chosen_position_ids=None,
#             rejected_input_ids=None,
#             rejected_attention_mask=None,
#             rejected_position_ids=None
#     ):
#         chosen_reward = self.reward(chosen_input_ids, attention_mask=chosen_attention_mask, position_ids=chosen_position_ids)
#         if rejected_input_ids is not None and rejected_attention_mask is not None:
#             reject_reward = self.reward(rejected_input_ids, attention_mask=rejected_attention_mask, position_ids=rejected_position_ids)
#             loss = self.loss_fn(chosen_reward, reject_reward)
#         else:
#             reject_reward = None
#             loss = None
#
#         return {
#             "loss": loss,
#             "chosen_reward": torch.sigmoid(chosen_reward),
#             "reject_reward": torch.sigmoid(reject_reward) if reject_reward is not None else reject_reward,
#         }
