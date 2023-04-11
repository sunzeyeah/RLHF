import sys
import torch
import torch.nn.functional as F

from functools import reduce
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
from torchtyping import TensorType

from src.utils.modeling_utils import (
    flatten_dict,
    get_tensor_stats,
    whiten,
)

# specifies a dictionary of method configs
_METHODS: Dict[str, Any] = {}  # registry


def register_method(name):
    """Decorator used register a method config
    Args:
        name: Name of the method
    """

    def register_class(cls, name):
        _METHODS[name] = cls
        setattr(sys.modules[__name__], name, cls)
        return cls

    if isinstance(name, str):
        name = name.lower()
        return lambda c: register_class(c, name)

    cls = name
    name = cls.__name__
    register_class(cls, name.lower())

    return cls


@dataclass
@register_method
class MethodConfig:
    """
    Config for a certain RL method.

    :param name: Name of the method
    :type name: str
    """

    name: str

    @classmethod
    def from_dict(cls, config: Dict[str, Any]):
        return cls(**config)


def get_method(name: str) -> MethodConfig:
    """
    Return constructor for specified method config
    """
    name = name.lower()
    if name in _METHODS:
        return _METHODS[name]
    else:
        raise Exception("Error: Trying to access a method that has not been registered")


# PPO Configs
@dataclass
@register_method
class PPOConfig(MethodConfig):
    """
    Config for PPO method

    :param ppo_epochs: Number of updates per batch
    :type ppo_epochs: int

    :param num_rollouts: Number  of experiences to observe before learning
    :type num_rollouts: int

    :param init_kl_coef: Initial value for KL coefficient
    :type init_kl_coef: float

    :param target: Target value for KL coefficient
    :type target: float

    :param horizon: Number of steps for KL coefficient to reach target
    :type horizon: int

    :param gamma: Discount factor
    :type gamma: float

    :param lam: GAE lambda
    :type lam: float

    :param cliprange: Clipping range for PPO policy loss (1 - cliprange, 1 + cliprange)
    :type cliprange: float

    :param cliprange_value: Clipping range for predicted values
                            (observed values - cliprange_value, observed values + cliprange_value)
    :type cliprange_value: float

    :param vf_coef: Value loss scale w.r.t policy loss
    :type vf_coef: float

    :param gen_kwargs: Additioanl kwargs for the generation
    :type gen_kwargs: Dict[str, Any]

    :param gen_experience_kwargs: if this is not None, then the experience is generated using this
    :type gen_experience_kwargs: Dict[str, Any]
    """

    ppo_epochs: int
    num_rollouts: int
    chunk_size: int
    init_kl_coef: float
    target: float
    horizon: int
    gamma: float
    lam: float
    cliprange: float
    cliprange_value: float
    vf_coef: float
    scale_reward: Optional[str]
    ref_mean: Optional[float]
    ref_std: Optional[float]
    cliprange_reward: float
    gen_kwargs: dict
    gen_experience_kwargs: Optional[dict] = None

    def get_advantages_and_returns(
            self,
            values: TensorType["batch_size", "response_size"],
            rewards: TensorType["batch_size", "response_size"],
            response_length: int,
            use_whitening: Optional[bool] = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Function that computes advantages and returns from rewards and values.
        Calculated as in the original PPO paper: https://arxiv.org/abs/1707.06347
        Note that rewards may include a KL divergence loss term.

        Advantages looks like this:
        Adv1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
              - V1 + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Returns looks like this:
        Ret1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
                   + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Args:
            values: Tensor of shape (batch_size, response_size)
            rewards: Tensor of shape (batch_size, response_size)
            response_length: Length of the response sequence
            use_whitening: Whether to use whitening (ie. normalize advantages) or not
        """
        lastgaelam = 0
        advantages_reversed = []
        for t in reversed(range(response_length)):
            nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
            delta = rewards[:, t] + self.gamma * nextvalues - values[:, t]
            lastgaelam = delta + self.gamma * self.lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        if use_whitening:
            advantages = whiten(advantages)
        return advantages.detach(), returns

    def loss(
            self,
            logprobs: TensorType["batch_size", "response_size"],
            values: TensorType["batch_size", "response_size"],
            old_logprobs: TensorType["batch_size", "response_size"],
            old_values: TensorType["batch_size", "response_size"],
            advantages: TensorType["batch_size", "response_size"],
            returns: TensorType["batch_size", "response_size"],
            mask: TensorType["batch_size", "response_size"],
    ):
        """PPO objective function.
        References:
        - https://stable-baselines.readthedocs.io/en/master/modules/ppo2.html
        """
        print(f"[ppo loss] values shape: {values.shape}, old_values shape: {old_values.shape}")
        values_clipped = torch.clamp(
            values,
            old_values - self.cliprange_value,
            old_values + self.cliprange_value,
            )
        n = mask.sum()

        vf_loss1 = (values - returns) ** 2
        vf_loss2 = (values_clipped - returns) ** 2
        vf_loss = 0.5 * torch.sum(torch.max(vf_loss1, vf_loss2) * mask) / n
        vf_clipfrac = torch.sum((vf_loss2 > vf_loss1).float() * mask) / n

        log_ratio = (logprobs - old_logprobs) * mask
        ratio = torch.exp(log_ratio)
        # Unbiased KL-div estimates (`k3`). Ref: http://joschu.net/blog/kl-approx.html
        with torch.no_grad():
            approx_kl = torch.mean((ratio - 1) - log_ratio)

        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(
            ratio,
            1.0 - self.cliprange,
            1.0 + self.cliprange,
            )
        pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2) * mask) / n
        pg_clipfrac = torch.sum((pg_loss2 > pg_loss1).float() * mask) / n

        loss = pg_loss + self.vf_coef * vf_loss

        stats = dict(
            losses=dict(
                total_loss=loss.item(),
                policy_loss=pg_loss.item(),
                value_loss=vf_loss.item(),
            ),
            values=dict(
                get_tensor_stats(values, mask, n),
                values_error=torch.sum(((values - returns) * mask) ** 2) / n,
                clipfrac=vf_clipfrac,
            ),
            old_values=get_tensor_stats(old_values, mask, n),
            returns=get_tensor_stats(returns, mask, n),
            policy=dict(approx_kl=approx_kl.item(), clipfrac=pg_clipfrac.item()),
            ratio=(ratio * mask).sum() / n,
            padding_percentage=n / mask.numel(),
        )

        return loss, flatten_dict(stats)


@dataclass
@register_method
class SFTConfig(MethodConfig):
    """
    Config for SFT training

    :param gen_kwargs: kwargs for generation
    :type gen_kwargs: Dict[str, Any]
    """

    gen_kwargs: dict


@dataclass
@register_method
class ILQLConfig(MethodConfig):
    tau: float
    gamma: float
    cql_scale: float
    awac_scale: float
    alpha: float
    beta: float
    steps_for_target_q_sync: float
    two_qs: bool
    gen_kwargs: dict

    def loss(self, outputs, labels):
        logits, (qs, target_qs, vs) = outputs
        terminal_mask = labels.dones[:, :-1]
        n_nonterminal = max(1, terminal_mask.sum())
        # check type of labels
        if isinstance(labels, ILQLBatch):
            actions = labels.input_ids[:, 1:].gather(dim=1, index=labels.actions_ixs).unsqueeze(-1)
        else:
            actions = labels.decoder_input_ids[:, 1:].unsqueeze(-1)
        nactions = actions.shape[1]
        bsize, _, dsize = logits.shape

        Q = [q.gather(-1, actions).squeeze(-1) for q in qs]
        targetQs = [q.gather(-1, actions).squeeze(-1).detach() for q in target_qs]
        targetQ = reduce(torch.minimum, targetQs)

        # values of current states
        V = vs[:, :-1].squeeze()
        # values of next states
        Vnext = vs[:, 1:].squeeze() * labels.dones[:, 1:]
        # target to fit Q
        Q_ = labels.rewards + self.gamma * Vnext.detach()

        loss_qs = [((Qi - Q_) * terminal_mask).pow(2).sum() / n_nonterminal for Qi in Q]
        loss_q = sum(loss_qs)

        targetQ = targetQ.detach()

        loss_v = (
                         (
                                 (targetQ >= V).int() * self.tau * (targetQ - V).pow(2)
                                 + (targetQ < V).int() * (1 - self.tau) * (targetQ - V).pow(2)
                         )
                         * terminal_mask
                 ).sum() / n_nonterminal

        def cql_loss(q):
            loss = F.cross_entropy(q.reshape(-1, dsize), actions.reshape(-1), reduction="none")
            loss = loss.reshape(bsize, nactions) * terminal_mask
            loss = loss.sum() / n_nonterminal
            return loss

        loss_cql = sum(cql_loss(q) for q in qs)

        # select logits from continuations
        action_logits = batched_index_select(logits, labels.actions_ixs, dim=1)
        cross_entropy = F.cross_entropy(
            action_logits.reshape(-1, dsize),
            actions.reshape(-1),
            reduction="none",
        ).reshape(bsize, nactions)

        with torch.no_grad():
            awac_weight = torch.exp(self.beta * (targetQ - V))

        loss_awac = torch.sum(cross_entropy * awac_weight * terminal_mask) / n_nonterminal
        loss = loss_q + loss_v + self.cql_scale * loss_cql + self.awac_scale * loss_awac

        stats = dict(
            losses=dict(
                loss=loss.item(),
                loss_q=loss_q.item(),
                loss_v=loss_v.item(),
                loss_cql=loss_cql.item(),
                loss_awac=loss_awac.item(),
            ),
            values=get_tensor_stats(V, terminal_mask, n_nonterminal),
            qvalues={str(ix): get_tensor_stats(Q[ix], terminal_mask, n_nonterminal) for ix in range(len(Q))},
            awac_weight=get_tensor_stats(awac_weight, terminal_mask, n_nonterminal),
        )

        return loss, flatten_dict(stats)
