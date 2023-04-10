
from dataclasses import dataclass
from typing import Iterable

from torchtyping import TensorType


@dataclass
class PromptElement:
    """
    Dataclass for a single prompt, containing its string and tokenized form.

    :param text: The prompt text.
    :type text: str

    :param tokens: The prompt tokens. Should be a long tensor
    :type tokens: torch.Tensor
    """

    text: str
    tokens: TensorType["num_tokens"]


@dataclass
class PromptBatch:
    """
    Batched PromptElement

    :param text: An iterable of prompt texts.
    :type text: Iterable[str]

    :param tokens: A long tensor batch of prompt tokens.
    :type tokens: torch.Tensor
    """

    text: Iterable[str]
    tokens: TensorType["batch_size", "num_tokens"]


@dataclass
class AccelerateRLElement:
    """
    Dataclass for RL elements, containing output tokens and rewards for each token.

    :param tokens: The output tokens. Should be a long tensor
    :type tokens: torch.Tensor

    :param rewards: The rewards for each token. Should be a float tensor of same size as tokens.
    :type rewards: torch.Tensor
    """

    output_tokens: TensorType["output_size"]
    rewards: TensorType["output_size"]


@dataclass
class AccelerateRLBatchElement:
    """
    Batched accelerate RL element

    :param tokens: Batches of long tensors of output tokens.
    :type tokens: torch.Tensor

    :param rewards: Batches of float tensors of rewards for each output token.
    :type rewards: torch.Tensor
    """

    output_tokens: TensorType["batch_size", "output_size"]
    rewards: TensorType["batch_size", "output_size"]


@dataclass
class PPORLElement:
    """
    :param query_tensor: The query tensor i.e. the prompt tokens.
                         Should be a long tensor.
    :type query_tensor: torch.Tensor

    :param response_tensor: The response tensor i.e. the output tokens.
                            Should be a long tensor.
    :type response_tensor: torch.Tensor

    :param logprobs: The log probabilities over the response tokens generated
                    by the policy network (i.e. the autoregressive model).
                    Should be a float tensor of same size as tokens.
    :type logprobs: torch.Tensor

    :param values: The values for each token generated from the value network or value head.
                    Should be a float tensor of same size as tokens.
    :type values: torch.Tensor

    :param rewards: The rewards for each token outputted in response.
                    Should be a float tensor of same size as tokens.
    :type rewards: torch.Tensor
    """

    query_tensor: TensorType["query_size"]
    response_tensor: TensorType["response_size"]
    attention_mask: TensorType["query_size"]
    position_ids: TensorType["query_size"]
    logprobs: TensorType["response_size"]
    values: TensorType["response_size"]
    rewards: TensorType["response_size"]


@dataclass
class PPORLBatch:
    """
    A batched version of the PPORLElement. See PPORLElement for more details on individual fields.

    :param query_tensors: A batch of query tensors. Should be a long tensor.
    :type query_tensors: torch.Tensor

    :param response_tensors: A batch of response tensors. Should be a long tensor.
    :type response_tensors: torch.Tensor

    :param logprobs: A batch of log probabilities from policy
    :type logprobs: torch.Tensor

    :param values: A batch of values from value network
    :type values: torch.Tensor

    :param rewards: A batch of rewards
    :type rewards: torch.Tensor
    """

    query_tensors: TensorType["batch_size", "query_size"]
    response_tensors: TensorType["batch_size", "response_size"]
    attention_mask: TensorType["batch_size", "query_size"]
    position_ids: TensorType["batch_size", "query_size"]
    logprobs: TensorType["batch_size", "response_size"]
    values: TensorType["batch_size", "response_size"]
    rewards: TensorType["batch_size", "response_size"]