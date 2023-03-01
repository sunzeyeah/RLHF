
import torch

from torch import nn
from src.models.loss import PairWiseLoss


class GPTRewardModel(nn.Module):
    def __init__(self, model, tokenizer):
        super().__init__()

        self.pad_id = tokenizer.pad_token_id

        self.config = model.config
        # `gpt-neo(x)` models use `hidden_size` attribute names instead of `n_embd``
        self.config.n_embd = self.config.hidden_size if hasattr(self.config, "hidden_size") else self.config.n_embd
        self.transformer = model.transformer
        self.v_head = nn.Linear(self.config.n_embd, 1, bias=False)
        self.loss_fn = PairWiseLoss()

    def reward(
            self,
            input_ids=None,
            attention_mask=None
    ):
        transformer_outputs = self.transformer(input_ids, attention_mask=attention_mask)
        hidden_states = transformer_outputs[0]
        rewards = self.v_head(hidden_states).squeeze(-1)

        # outputs = self.body(sequences, attention_mask=attention_mask)
        # last_hidden_states = outputs['last_hidden_state']
        # values = self.value_head(last_hidden_states)[:, :-1]

        rewards = rewards.mean(dim=-1).squeeze(1)    # ensure shape is (B)

        return rewards

    def forward(
            self,
            chosen_input_ids=None,
            chosen_attention_mask=None,
            rejected_input_ids=None,
            rejected_attention_mask=None,
            past_key_values=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            mc_token_ids=None,
            labels=None,
            return_dict=False,
            output_attentions=False,
            output_hidden_states=False,
    ):
        chosen_reward = self.reward(chosen_input_ids, attention_mask=chosen_attention_mask)
        reject_reward = self.reward(rejected_input_ids, attention_mask=rejected_attention_mask)
        loss = self.loss_fn(chosen_reward, reject_reward)

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
            "chosen_reward": chosen_reward,
            "reject_reward": reject_reward,
        }
