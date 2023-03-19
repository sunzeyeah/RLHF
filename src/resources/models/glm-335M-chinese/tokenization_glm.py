import os
from typing import Optional, Tuple, List, Union
from shutil import copyfile
import torch
import numpy as np

from typing import Dict
from transformers import PreTrainedTokenizer, RobertaTokenizer, GPT2Tokenizer, BertTokenizer
from transformers.utils import logging
from transformers.tokenization_utils_base import BatchEncoding
from transformers.models.auto.tokenization_auto import get_tokenizer_config
# from transformers.utils import torch_required
from transformers.utils.generic import _is_torch_device
from transformers.utils import PaddingStrategy
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, TruncationStrategy, TensorType, EncodedInput
import sentencepiece as spm

logger = logging.get_logger(__name__)


class GLMBatchEncoding(BatchEncoding):
    # @torch_required
    def to(self, device: Union[str, "torch.device"]) -> "BatchEncoding":
        """
        Send all values to device by calling `v.to(device)` (PyTorch only).
        Args:
            device (`str` or `torch.device`): The device to put the tensors on.
        Returns:
            [`BatchEncoding`]: The same instance after modification.
        """

        # This check catches things like APEX blindly calling "to" on all inputs to a module
        # Otherwise it passes the casts down and casts the LongTensor containing the token idxs
        # into a HalfTensor
        if isinstance(device, str) or _is_torch_device(device) or isinstance(device, int):
            self.data = {k: v.to(device=device) if torch.is_tensor(v) else v for k, v in self.data.items()}
        else:
            logger.warning(f"Attempting to cast a BatchEncoding to type {str(device)}. This is not supported.")
        return self


class GLMTokenizerMixin(PreTrainedTokenizerBase):

    model_input_names: List[str] = ["input_ids", "position_ids", "attention_mask", "labels"]

    @property
    def ignore_index(self) -> int:
        return -100

    @property
    def sop_token(self) -> Optional[str]:
        return "<|startofpiece|>"

    @property
    def sop_token_id(self) -> Optional[int]:
        """
        `Optional[int]`: Id of the start token in the vocabulary, used when training a model with autoregressive blank filling.
        """
        return self.convert_tokens_to_ids(self.sop_token)

    @property
    def eop_token(self) -> Optional[str]:
        return "<|endofpiece|>"

    @property
    def eop_token_id(self) -> Optional[int]:
        """
        `Optional[int]`: Id of the end token in the vocabulary, used when training a model with autoregressive blank filling.
        """
        return self.convert_tokens_to_ids(self.eop_token)

    @property
    def gmask_token_id(self) -> int:
        return self.convert_tokens_to_ids("[gMASK]")

    @property
    def smask_token_id(self) -> int:
        return self.convert_tokens_to_ids("[sMASK]")

    @property
    def mask_token_ids(self):
        return [self.mask_token_id, self.smask_token_id, self.gmask_token_id]

    def _build_input_for_multiple_choice(self, context, choices):
        context_id = context["input_ids"]
        if torch.is_tensor(context_id):
            context_id = context_id.tolist()

        division = len(context_id)
        mask_position = context_id.index(self.mask_token_id)

        token = torch.tensor(context_id, dtype=torch.long)
        attention_mask = [context["attention_mask"].expand(division, -1)]
        position_id = torch.arange(division, dtype=torch.long)
        block_position_id = torch.zeros(division, dtype=torch.long)

        choice_ids, choice_indices = [], []

        for choice_str in choices:
            choice = torch.tensor(self(choice_str, add_special_tokens=False, padding=False)['input_ids'],
                                  dtype=torch.long)
            choice_ids.append(choice)
            choice_indices.append(torch.arange(len(token), len(token) + len(choice), dtype=torch.long))
            attention_mask.append(torch.tril(torch.ones((len(choice), len(choice)), dtype=torch.long)))

            token = torch.cat((token, torch.tensor([self.sop_token_id], dtype=torch.long), choice[:-1]))
            position_id = torch.cat((position_id, torch.tensor([mask_position] * len(choice), dtype=torch.long)))
            block_position_id = torch.cat((block_position_id, torch.arange(1, 1 + len(choice), dtype=torch.long)))

        attention_mask = torch.block_diag(*attention_mask)
        attention_mask[division:, :division] = context["attention_mask"].unsqueeze(0)

        return {
            "input_ids": token,
            "position_ids": torch.stack((position_id, block_position_id)),
            "attention_mask": attention_mask,
            "choice_ids": choice_ids,
            "choice_indices": choice_indices
        }

    def _pad_batch(self, tokens, position_ids, attention_mask, max_seq_length):
        pad_length = max_seq_length - len(tokens)
        attention_mask = torch.nn.functional.pad(
            attention_mask,
            (0, pad_length, 0, pad_length),
            mode="constant",
            value=0,
        )
        tokens = torch.cat((tokens, torch.zeros(pad_length, dtype=torch.long)))
        position_ids = torch.cat((position_ids, position_ids[..., -1:].expand(-1, pad_length)), dim=-1)
        return tokens, position_ids, attention_mask

    def _collate(self, samples):
        TILE = 1
        length_to_pad = (max(map(lambda spl: len(spl["input_ids"]), samples)) + TILE - 1) // TILE * TILE

        token_batch, position_id_batch, attention_mask_batch = [], [], []
        choices_batch, choice_target_ids_batch = [], []

        for sample in samples:
            token, position_id, attention_mask = self._pad_batch(
                sample["input_ids"], sample["position_ids"], sample["attention_mask"], length_to_pad
            )
            token_batch.append(token)
            position_id_batch.append(position_id)
            attention_mask_batch.append(attention_mask)
            choices_batch.append(sample["choice_ids"])
            choice_target_ids_batch.append(sample["choice_indices"])
        return {
            "input_ids": torch.stack(token_batch),
            "position_ids": torch.stack(position_id_batch),
            "attention_mask": torch.stack(attention_mask_batch).unsqueeze(1),
            "choice_ids": choices_batch,
            "choice_indices": choice_target_ids_batch,
        }

    def build_inputs_for_multiple_choice(self, model_input: BatchEncoding, choices, max_length=None):
        samples = [{key: value[i] for key, value in model_input.items()} for i in range(len(model_input["input_ids"]))]
        samples = [self._build_input_for_multiple_choice(sample, choice) for sample, choice in
                   zip(samples, choices)]
        inputs = self._collate(samples)
        return GLMBatchEncoding(inputs)

    def build_inputs_for_generation(self, model_input: BatchEncoding, max_gen_length=512, targets=None):
        mask_ids = self.mask_token_ids
        input_ids = model_input.input_ids
        batch_size, seq_length = input_ids.shape[:2]
        position_id, block_position_id = list(range(seq_length)), [0 for _ in range(seq_length)]
        position_ids, block_position_ids = [], []
        labels = None
        if targets is not None:
            is_batched = isinstance(targets, (list, tuple))
            targets = self(targets, add_special_tokens=False, padding=False).input_ids
            if not is_batched:
                targets = [targets]
            targets = [(target + [self.eop_token_id])[:max_gen_length] for target in targets]
            max_gen_length = max(map(len, targets))
            targets = [[self.sop_token_id] + target + [-100] * (max_gen_length - len(target)) for target in targets]
            assert len(targets) == len(input_ids)
            targets = torch.tensor(targets, dtype=input_ids.dtype, device=input_ids.device)
            labels = torch.cat((input_ids.new_full((batch_size, seq_length), -100), targets[:, 1:]), dim=1)
        for i in range(batch_size):
            mask_positions = []
            for mask_id in mask_ids:
                mask_positions += (input_ids[i] == mask_id).nonzero(as_tuple=True)[0].tolist()
            if not mask_positions:
                raise ValueError("Cannot find mask token in the input")
            mask_positions.sort()
            mask_pos = mask_positions[0]
            position_ids.append(position_id + [mask_pos] * max_gen_length)
            block_position_ids.append(block_position_id + list(range(1, max_gen_length + 1)))
        position_ids = torch.tensor(position_ids, dtype=input_ids.dtype, device=input_ids.device)
        block_position_ids = torch.tensor(block_position_ids, dtype=input_ids.dtype, device=input_ids.device)
        position_ids = torch.stack((position_ids, block_position_ids), dim=1)
        attention_mask = model_input.attention_mask
        attention_mask = attention_mask.unsqueeze(1).expand(-1, seq_length + max_gen_length, -1)
        generation_attention_mask = torch.cat([attention_mask.new_zeros((seq_length, max_gen_length)),
                                               torch.tril(attention_mask.new_ones((max_gen_length, max_gen_length)))],
                                              dim=0).unsqueeze(0).expand(batch_size, -1, -1)
        attention_mask = torch.cat((attention_mask, generation_attention_mask), dim=2)
        attention_mask = attention_mask.unsqueeze(1)
        if targets is None:
            input_ids = torch.cat((input_ids, input_ids.new_full((batch_size, 1), self.sop_token_id)), dim=-1)
        else:
            input_ids = torch.cat((input_ids, targets[:, :-1]), dim=1)
        batch = {"input_ids": input_ids, "position_ids": position_ids}
        if labels is None:
            batch["generation_attention_mask"] = attention_mask
        else:
            batch["attention_mask"] = attention_mask
            batch["labels"] = labels
        return BatchEncoding(batch)

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        """
        Returns the number of added tokens when encoding a sequence with special tokens.

        <Tip>

        This encodes a dummy input and checks the number of added tokens, and is therefore not efficient. Do not put
        this inside your training loop.

        </Tip>

        Args:
            pair (`bool`, *optional*, defaults to `False`):
                Whether the number of added tokens should be computed in the case of a sequence pair or a single
                sequence.

        Returns:
            `int`: Number of special tokens added to sequences.
        """
        token_ids_0 = []
        token_ids_1 = []
        t1, t2 = self.build_inputs_with_special_tokens(token_ids_0, token_ids_1 if pair else None)
        return len(t1)

    def prepare_for_model(
            self,
            ids: List[int],
            pair_ids: Optional[List[int]] = None,
            add_special_tokens: bool = True,
            padding: Union[bool, str, PaddingStrategy] = False,
            truncation: Union[bool, str, TruncationStrategy] = None,
            max_length: Optional[int] = None,
            stride: int = 0,
            pad_to_multiple_of: Optional[int] = None,
            return_tensors: Optional[Union[str, TensorType]] = None,
            return_token_type_ids: Optional[bool] = None,
            return_attention_mask: Optional[bool] = None,
            return_overflowing_tokens: bool = False,
            return_special_tokens_mask: bool = False,
            return_offsets_mapping: bool = False,
            return_length: bool = False,
            verbose: bool = True,
            prepend_batch_axis: bool = False,
            **kwargs
    ) -> BatchEncoding:
        """
        Prepares a sequence of input id, or a pair of sequences of inputs ids so that it can be used by the model. It
        adds special tokens, truncates sequences if overflowing while taking into account the special tokens and
        manages a moving window (with user defined stride) for overflowing tokens. Please Note, for *pair_ids*
        different than `None` and *truncation_strategy = longest_first* or `True`, it is not possible to return
        overflowing tokens. Such a combination of arguments will raise an error.

        Args:
            ids (`List[int]`):
                Tokenized input ids of the first sequence. Can be obtained from a string by chaining the `tokenize` and
                `convert_tokens_to_ids` methods.
            pair_ids (`List[int]`, *optional*):
                Tokenized input ids of the second sequence. Can be obtained from a string by chaining the `tokenize`
                and `convert_tokens_to_ids` methods.
        """

        # # if self.random_position and position_ids[-1] < self.max_seq_length - 1:
        # #     position_bias = self.max_seq_length - position_ids[-1]
        # #     position_bias = rng.randrange(0, position_bias)
        # #     position_ids = position_ids + position_bias
        # # if self.encoder_decoder or not self.shuffle_blocks:
        # #     block_spans.sort(key=lambda x: x[0])
        # # else:
        # #     rng.shuffle(block_spans)
        # # if self.sentinel_token:
        # #     block_spans = [(start, end, idx) for idx, (start, end) in enumerate(block_spans)]
        # # else:
        # #     block_spans = [(start, end, 0) for start, end in block_spans]
        # target_tokens, target_position_ids, target_block_position_ids, targets = [], [], [], []
        # # for start, end, idx in block_spans:
        # sop_token = 'sop' if idx == 0 else f"sop{idx}"
        # target_tokens.append([self.tokenizer.get_command(sop_token).Id])
        # span_tokens = copy.deepcopy(tokens[start: end])
        # if self.block_mask_prob > 0.0 and task == 'bert':
        #     for sub_idx in range(len(span_tokens)):
        #         if random.random() < self.block_mask_prob:
        #             span_tokens[sub_idx] = self.tokenizer.get_command('dBLOCK').Id
        # target_tokens.append(span_tokens)
        # targets.append(tokens[start: end])
        # targets.append([self.tokenizer.get_command('eop').Id])
        # if not self.sentinel_token:
        #     target_position_id = position_ids[start: end]
        #     target_position_ids.append(target_position_id)
        #     target_position_ids.append([target_position_id[0]])
        # else:
        #     target_position_ids.append([self.max_seq_length] * (end - start + 1))
        # if self.block_position_encoding:
        #     target_block_position_ids.append(np.arange(1, end - start + 2, dtype=np.long))
        # else:
        #     target_block_position_ids.append([1] * (end - start + 1))
        #
        # block_spans.sort(key=lambda x: x[0])
        # source_tokens, source_position_ids, local_spans = [], [], []
        # last, current_length = 0, 0
        # for start, end, idx in block_spans:
        #     if task == 'generation':
        #         mask_id = self.generation_mask
        #     elif task == 'gap_sentence':
        #         mask_id = self.gap_sentence_mask
        #     else:
        #         mask_token = 'MASK' if idx == 0 else f'MASK{idx}'
        #         mask_id = self.tokenizer.get_command(mask_token).Id
        #     local_spans.append((current_length, current_length + start - last))
        #     source_tokens.append(tokens[last: start])
        #     source_tokens.append([mask_id])
        #     source_position_ids.append(position_ids[last: start])
        #     source_position_ids.append([position_ids[start]])
        #     current_length += start - last + 1
        #     last = end
        # if last < len(tokens):
        #     local_spans.append((current_length, current_length + len(tokens) - last))
        #     source_tokens.append(tokens[last:])
        #     source_position_ids.append(position_ids[last:])
        # source_length = sum(map(len, source_tokens))
        # if attention_mask is not None:
        #     assert source_length == attention_mask
        # if target_tokens and self.eod_token in np.concatenate(target_tokens).tolist():
        #     print("Found EOS in target", self.tokenizer.DecodeIds(tokens))
        #     raise RuntimeError
        # if self.encoder_decoder:
        #     target_tokens = target_tokens + [self.tokenizer.get_command('eop').Id]
        #     loss_masks = np.ones(len(target_tokens), dtype=np.long)
        #     return source_tokens, target_tokens, loss_masks
        # else:
        #     tokens = np.concatenate(source_tokens + target_tokens)
        #     if task == 'bert' and self.context_mask_ratio > 0:
        #         mask_candidates = set()
        #         for start, end in local_spans:
        #             if start != 0:
        #                 local_end = min(end, start + self.context_mask_range)
        #                 mask_candidates.update(range(start, local_end))
        #             if end != 0:
        #                 local_start = max(start, end - self.context_mask_range)
        #                 mask_candidates.update(range(local_start, end))
        #         mask_pos = rng.sample(mask_candidates, int(self.context_mask_ratio * text_length))
        #         for pos in mask_pos:
        #             tokens[pos] = self.tokenizer.get_command('dBLOCK').Id
        #     targets = np.concatenate(source_tokens + targets)
        #     loss_masks = np.ones(len(tokens), dtype=np.long)
        #     loss_masks[:source_length] = 0
        #     position_ids = np.concatenate(source_position_ids + target_position_ids)
        #     block_position_ids = np.concatenate(
        #         [np.zeros(source_length, dtype=np.long)] + target_block_position_ids)
        #     position_ids = np.stack([position_ids, block_position_ids], axis=0)
        #     if attention_mask is not None:
        #         return tokens, targets, loss_masks, position_ids
        #     else:
        #         return tokens, targets, loss_masks, position_ids, source_length

        # Backward compatibility for 'truncation_strategy', 'pad_to_max_length'
        padding_strategy, truncation_strategy, max_length, kwargs = self._get_padding_truncation_strategies(
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            verbose=verbose,
            **kwargs,
        )

        pair = bool(pair_ids is not None)
        len_ids = len(ids)
        len_pair_ids = len(pair_ids) if pair else 0

        if return_token_type_ids and not add_special_tokens:
            raise ValueError(
                "Asking to return token_type_ids while setting add_special_tokens to False "
                "results in an undefined behavior. Please set add_special_tokens to True or "
                "set return_token_type_ids to None."
            )

        if (
                return_overflowing_tokens
                and truncation == TruncationStrategy.LONGEST_FIRST
                and pair_ids is not None
        ):
            raise ValueError(
                "Not possible to return overflowing tokens for pair of sequences with the "
                "`longest_first`. Please select another truncation strategy than `longest_first`, "
                "for instance `only_second` or `only_first`."
            )

        # Load from model defaults
        if return_token_type_ids is None:
            return_token_type_ids = "position_ids" in self.model_input_names
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names

        encoded_inputs = {}

        # Compute the total size of the returned encodings
        total_len = len_ids + len_pair_ids + (self.num_special_tokens_to_add(pair=pair) if add_special_tokens else 0)

        # Truncation: Handle max sequence length
        overflowing_tokens = []
        if truncation_strategy != TruncationStrategy.DO_NOT_TRUNCATE and max_length and total_len > max_length:
            ids, pair_ids, overflowing_tokens = self.truncate_sequences(
                ids,
                pair_ids=pair_ids,
                num_tokens_to_remove=total_len - max_length,
                truncation_strategy=truncation_strategy,
                stride=stride,
            )

        if return_overflowing_tokens:
            encoded_inputs["overflowing_tokens"] = overflowing_tokens
            encoded_inputs["num_truncated_tokens"] = total_len - max_length

        # Add special tokens
        if add_special_tokens:
            sequence, labels = self.build_inputs_with_special_tokens(ids, pair_ids)
            assert sequence[0] == self.cls_token_id
            text_length = len(sequence)
            eos_position = sequence.index(self.eos_token_id)
            mask_position = sequence.index(self.mask_token_id)
            # sop_position = sequence.index(self.sop_token_id)
            position_ids = np.array(list(range(eos_position+1)) + [mask_position]*(text_length-eos_position-1), dtype=np.int64)
            block_position_ids = np.array([0] * (eos_position+1) + list(range(text_length-eos_position-1)), dtype=np.int64)
            token_type_ids = np.stack([position_ids, block_position_ids], axis=0)
            # token_type_ids = self.create_token_type_ids_from_sequences(ids, pair_ids)
        else:
            sequence = ids + pair_ids if pair else ids
            token_type_ids = [0] * len(ids) + ([0] * len(pair_ids) if pair else [])
            if pair:
                labels = [self.ignore_index]*len(ids) + pair_ids
            else:
                labels = None

        # Build output dictionary
        encoded_inputs["input_ids"] = sequence
        if labels is not None:
            encoded_inputs["labels"] = labels
        if return_token_type_ids:
            encoded_inputs["position_ids"] = token_type_ids
        if return_special_tokens_mask:
            if add_special_tokens:
                encoded_inputs["special_tokens_mask"] = self.get_special_tokens_mask(ids, pair_ids)
            else:
                encoded_inputs["special_tokens_mask"] = [0] * len(sequence)

        # Check lengths
        # self._eventual_warn_about_too_long_sequence(encoded_inputs["input_ids"], max_length, verbose)

        # Padding
        if padding_strategy != PaddingStrategy.DO_NOT_PAD or return_attention_mask:
            encoded_inputs = self.pad(
                encoded_inputs,
                max_length=max_length,
                padding=padding_strategy.value,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
            )

        if return_length:
            encoded_inputs["length"] = len(encoded_inputs["input_ids"])

        batch_outputs = BatchEncoding(
            encoded_inputs, tensor_type=return_tensors, prepend_batch_axis=prepend_batch_axis
        )

        return batch_outputs

    def _pad(
            self,
            encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding],
            max_length: Optional[int] = None,
            padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
            pad_to_multiple_of: Optional[int] = None,
            return_attention_mask: Optional[bool] = None,
    ) -> dict:
        """
        Pad encoded inputs (on left/right and up to predefined length or max length in the batch)

        Args:
            encoded_inputs:
                Dictionary of tokenized inputs (`List[int]`) or batch of tokenized inputs (`List[List[int]]`).
            max_length: maximum length of the returned list and optionally padding length (see below).
                Will truncate by taking into account the special tokens.
            padding_strategy: PaddingStrategy to use for padding.

                - PaddingStrategy.LONGEST Pad to the longest sequence in the batch
                - PaddingStrategy.MAX_LENGTH: Pad to the max length (default)
                - PaddingStrategy.DO_NOT_PAD: Do not pad
                The tokenizer padding sides are defined in self.padding_side:

                    - 'left': pads on the left of the sequences
                    - 'right': pads on the right of the sequences
            pad_to_multiple_of: (optional) Integer if set will pad the sequence to a multiple of the provided value.
                This is especially useful to enable the use of Tensor Core on NVIDIA hardware with compute capability
                `>= 7.5` (Volta).
            return_attention_mask:
                (optional) Set to False to avoid returning attention mask (default: set to model specifics)
        """
        # Load from model defaults
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names

        required_input = encoded_inputs[self.model_input_names[0]]
        required_target = encoded_inputs[self.model_input_names[3]]

        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = len(required_input)

        if max_length is not None and pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
        #
        needs_to_be_padded = padding_strategy != PaddingStrategy.DO_NOT_PAD and len(required_input) != max_length

        # Initialize attention mask if not present.
        if return_attention_mask and "attention_mask" not in encoded_inputs:
            # encoded_inputs["attention_mask"] = [1] * len(required_input)
            # sop_position = encoded_inputs['input_ids'].index(self.sop_token_id)
            # encoded_inputs["attention_mask"] = [sop_position]
            input_length = required_input.index(self.sop_token_id)
            target_length = len(required_input) - input_length
            attention_mask = np.ones((len(required_input), input_length), dtype=np.int64)
            generation_attention_mask = np.concatenate([np.zeros((input_length, target_length), dtype=np.int64),
                                                   np.tril(np.ones((target_length, target_length), dtype=np.int64))],
                                                  axis=0)#.unsqueeze(0).expand(-1, -1)
            attention_mask = np.concatenate((attention_mask, generation_attention_mask), axis=1)
            attention_mask = np.expand_dims(attention_mask, 0)
            encoded_inputs["attention_mask"] = attention_mask

        if needs_to_be_padded:
            difference = max_length - len(required_input)

            if self.padding_side == "right":
                if return_attention_mask:
                    # encoded_inputs["attention_mask"] = encoded_inputs["attention_mask"] + [0] * difference
                    attention_mask = encoded_inputs["attention_mask"][0]
                    attention_mask = np.concatenate((attention_mask, np.zeros((len(required_input), difference), dtype=np.int64)), axis=1)
                    # pad_attention_mask = attention_mask.new_zeros((max_length, difference))
                    attention_mask = np.concatenate((attention_mask, np.zeros((difference, max_length), dtype=np.int64)), axis=0)
                    attention_mask = np.expand_dims(attention_mask, 0)
                    encoded_inputs["attention_mask"] = attention_mask
                if "position_ids" in encoded_inputs:
                    encoded_inputs["position_ids"][:, -difference:] = self.pad_token_type_id
                    # encoded_inputs["token_type_ids"] = (
                    #         encoded_inputs["token_type_ids"] + [self.pad_token_type_id] * difference
                    # )
                if "special_tokens_mask" in encoded_inputs:
                    encoded_inputs["special_tokens_mask"] = encoded_inputs["special_tokens_mask"] + [1] * difference
                encoded_inputs[self.model_input_names[0]] = required_input + [self.pad_token_id] * difference
                encoded_inputs[self.model_input_names[3]] = required_target + [self.ignore_index] * difference
            elif self.padding_side == "left":
                if return_attention_mask:
                    # encoded_inputs["attention_mask"] = [0] * difference + encoded_inputs["attention_mask"]
                    attention_mask = encoded_inputs["attention_mask"][0]
                    attention_mask = np.concatenate((np.zeros((len(required_input), difference), dtype=np.int64), attention_mask), axis=1)
                    # pad_attention_mask = attention_mask.new_zeros((max_length, difference))
                    attention_mask = np.concatenate((np.zeros((difference, max_length), dtype=np.int64), attention_mask), axis=0)
                    attention_mask = np.expand_dims(attention_mask, 0)
                    encoded_inputs["attention_mask"] = attention_mask
                if "position_ids" in encoded_inputs:
                    encoded_inputs["position_ids"][:, :-difference] = self.pad_token_type_id
                    # encoded_inputs["token_type_ids"] = [self.pad_token_type_id] * difference + encoded_inputs[
                    #     "token_type_ids"
                    # ]
                if "special_tokens_mask" in encoded_inputs:
                    encoded_inputs["special_tokens_mask"] = [1] * difference + encoded_inputs["special_tokens_mask"]
                encoded_inputs[self.model_input_names[0]] = [self.pad_token_id] * difference + required_input
                encoded_inputs[self.model_input_names[3]] = [self.ignore_index] * difference + required_target
            else:
                raise ValueError("Invalid padding strategy:" + str(self.padding_side))

        return encoded_inputs


class GLMRobertaTokenizer(RobertaTokenizer, GLMTokenizerMixin):
    model_input_names = ["input_ids", "position_ids", "attention_mask"]
    truncation_side: str = "left"

    @property
    def gmask_token_id(self) -> int:
        raise NotImplementedError("The model doesn't support gMASK")

    @property
    def smask_token_id(self) -> int:
        raise NotImplementedError("The model doesn't support sMASK")

    @property
    def mask_token_ids(self):
        return [self.mask_token_id]


class GLMChineseTokenizer(GLMTokenizerMixin, PreTrainedTokenizer):
    vocab_files_names = {"vocab_file": "cog-pretrain.model"}
    truncation_side: str = "left"

    def __init__(self, vocab_file, **kwargs):
        super().__init__(**kwargs)
        self.vocab_file = vocab_file
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(vocab_file)

    @property
    def vocab_size(self):
        return len(self.sp_model)

    def get_vocab(self):
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text, **kwargs):
        return self.sp_model.encode(text, out_type=str)

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.sp_model.PieceToId(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.sp_model.IdToPiece(index)

    def convert_tokens_to_string(self, tokens):
        return self.sp_model.decode(tokens)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + self.vocab_files_names["vocab_file"]
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        return (out_vocab_file,)

    def build_inputs_with_special_tokens(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A BERT sequence has the following format:
        - single sequence: ``[CLS] X [SEP]``
        - pair of sequences: ``[CLS] A [SEP] B [SEP]``
        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.
        Returns:
            :obj:`List[int]`: List of `input IDs <../glossary.html#input-ids>`__ with the appropriate special tokens.
        """
        # assert token_ids_1 is None
        cls = [self.cls_token_id]
        eos = [self.eos_token_id]
        eop = [self.eop_token_id]
        mask = [self.mask_token_id]
        sop = [self.sop_token_id]
        token_ids_0 = cls + token_ids_0 + mask + eos
        if token_ids_1 is None:
            return token_ids_0 + sop, None
        else:

            return token_ids_0 + sop + token_ids_1, [self.ignore_index]*len(token_ids_0) + token_ids_1 + eop


class GLMGPT2Tokenizer(GPT2Tokenizer, GLMTokenizerMixin):
    model_input_names = ["input_ids", "position_ids", "attention_mask"]
    truncation_side: str = "left"

    def build_inputs_with_special_tokens(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A BERT sequence has the following format:
        - single sequence: ``[CLS] X [SEP]``
        - pair of sequences: ``[CLS] A [SEP] B [SEP]``
        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.
        Returns:
            :obj:`List[int]`: List of `input IDs <../glossary.html#input-ids>`__ with the appropriate special tokens.
        """
        assert token_ids_1 is None
        cls = [self.cls_token_id]
        eos = [self.eos_token_id]
        return cls + token_ids_0 + eos


class GLMBertTokenizer(BertTokenizer, GLMTokenizerMixin):
    model_input_names = ["input_ids", "position_ids", "attention_mask"]
    truncation_side: str = "left"

    @property
    def gmask_token_id(self) -> int:
        raise NotImplementedError("The model doesn't support gMASK")

    @property
    def smask_token_id(self) -> int:
        raise NotImplementedError("The model doesn't support sMASK")

    @property
    def mask_token_ids(self):
        return [self.mask_token_id]


class GLMTokenizer:
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        tokenizer_config = get_tokenizer_config(pretrained_model_name_or_path, **kwargs)
        config_tokenizer_class = tokenizer_config.get("tokenizer_class")
        if config_tokenizer_class == "GLMRobertaTokenizer":
            tokenizer_class = GLMRobertaTokenizer
        elif config_tokenizer_class == "GLMChineseTokenizer":
            tokenizer_class = GLMChineseTokenizer
        elif config_tokenizer_class == "GLMGPT2Tokenizer":
            tokenizer_class = GLMGPT2Tokenizer
        elif config_tokenizer_class == "GLMBertTokenizer":
            tokenizer_class = GLMBertTokenizer
        else:
            raise NotImplementedError("Not implemented tokenizer type:", config_tokenizer_class)
        return tokenizer_class.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)