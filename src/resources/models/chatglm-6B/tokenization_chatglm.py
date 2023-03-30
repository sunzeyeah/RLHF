"""Tokenization classes for ChatGLM."""
import sys
import unicodedata
from typing import List, Optional, Union
from functools import lru_cache
import os
import collections
import torch

from transformers.tokenization_utils import PreTrainedTokenizer
from icetk.text_tokenizer import TextTokenizer
import icetk.sentencepiece_model_pb2 as sp_model
from transformers.utils import logging
from transformers.tokenization_utils_base import BatchEncoding


logger = logging.get_logger(__name__)

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "THUDM/chatglm-6b": 2048,
}


class SPTokenizer:
    def __init__(
        self,
        vocab_file,
        max_blank_length=80,
        byte_fallback=True,
    ):
        assert vocab_file is not None
        self.vocab_file = vocab_file
        self.special_tokens = ["[MASK]", "[gMASK]", "[sMASK]", "<unused_0>", "<sop>", "<eop>", "<ENC>", "<dBLOCK>"]
        self.max_blank_length = max_blank_length
        self.byte_fallback = byte_fallback
        self.text_tokenizer = self._build_text_tokenizer(encode_special_tokens=False)
        self.special_text_tokenizer = self._build_text_tokenizer(encode_special_tokens=True)

    @staticmethod
    def _configure_tokenizer(
        text_tokenizer: TextTokenizer,
        special_tokens: List[str],
        max_blank_length: int,
        byte_fallback: bool,
        encode_special_tokens=False,
    ):
        # special token
        special_token_type = 4 if encode_special_tokens else 3  # 3 - CONTROL, 4 - USER_DEFINE
        for token in special_tokens:
            text_tokenizer.proto.pieces.append(
                sp_model.ModelProto.SentencePiece(piece=token, score=0.0, type=special_token_type)
            )
        # whitespaces
        for token in [SPTokenizer.get_tab_token()] + [
            SPTokenizer.get_blank_token(i) for i in range(2, max_blank_length + 1)
        ]:
            text_tokenizer.proto.pieces.append(sp_model.ModelProto.SentencePiece(piece=token, score=0.0, type=4))
        # byte fallback
        if byte_fallback:
            text_tokenizer.proto.trainer_spec.byte_fallback = True
            for i in range(256):
                text_tokenizer.proto.pieces.append(
                    sp_model.ModelProto.SentencePiece(piece="<0x{:02X}>".format(i), score=0.0, type=6)
                )
        text_tokenizer.refresh()

    def _build_text_tokenizer(self, encode_special_tokens=False):
        tokenizer = TextTokenizer(self.vocab_file)
        self._configure_tokenizer(
            tokenizer, self.special_tokens, self.max_blank_length, self.byte_fallback, encode_special_tokens
        )
        return tokenizer

    def _get_text_tokenizer(self, encode_special_tokens=False):
        if encode_special_tokens:
            return self.special_text_tokenizer
        else:
            return self.text_tokenizer

    @staticmethod
    def get_blank_token(length: int):
        assert length >= 2
        return f"<|blank_{length}|>"

    @staticmethod
    def get_tab_token():
        return f"<|tab|>"

    @property
    def num_image_tokens(self):
        return 20000

    @property
    def num_text_tokens(self):
        return self.text_tokenizer.num_tokens

    @property
    def num_tokens(self):
        return self.num_image_tokens + self.num_text_tokens

    @staticmethod
    def _encode_whitespaces(text: str, max_len: int = 80):
        text = text.replace("\t", SPTokenizer.get_tab_token())
        for i in range(max_len, 1, -1):
            text = text.replace(" " * i, SPTokenizer.get_blank_token(i))
        return text

    def _preprocess(self, text: str, linebreak=True, whitespaces=True):
        if linebreak:
            text = text.replace("\n", "<n>")
        if whitespaces:
            text = self._encode_whitespaces(text, max_len=self.max_blank_length)
        return text

    def encode(
        self, text: str, linebreak=True, whitespaces=True, special_tokens=False, add_dummy_prefix=True
    ) -> List[int]:
        """
        @param text: Text to encode.
        @param linebreak: Whether to encode newline (\n) in text.
        @param whitespaces: Whether to encode multiple whitespaces or tab in text, useful for source code encoding.
        @param special_tokens: Whether to encode special token ([MASK], [gMASK], etc.) in text.
        @param add_dummy_prefix: Whether to add dummy blank space in the beginning.
        """
        text = self._preprocess(text, linebreak, whitespaces)
        if not add_dummy_prefix:
            text = "<n>" + text
        tmp = self._get_text_tokenizer(encode_special_tokens=special_tokens).encode(text)
        tokens = [x + self.num_image_tokens for x in tmp]
        return tokens if add_dummy_prefix else tokens[2:]

    def decode(self, text_ids: List[int], special_tokens=False) -> str:
        ids = [int(_id) - self.num_image_tokens for _id in text_ids]
        text = self._get_text_tokenizer(encode_special_tokens=special_tokens).decode(ids)
        text = text.replace("<n>", "\n")
        text = text.replace(SPTokenizer.get_tab_token(), "\t")
        for i in range(2, self.max_blank_length + 1):
            text = text.replace(self.get_blank_token(i), " " * i)
        return text

    def tokenize(
        self, text: str, linebreak=True, whitespaces=True, special_tokens=False, add_dummy_prefix=True
    ) -> List[str]:
        """
        @param text: Text to encode.
        @param linebreak: Whether to encode newline (\n) in text.
        @param whitespaces: Whether to encode multiple whitespaces or tab in text, useful for source code encoding.
        @param special_tokens: Whether to encode special token ([MASK], [gMASK], etc.) in text.
        @param add_dummy_prefix: Whether to add dummy blank space in the beginning.
        """
        text = self._preprocess(text, linebreak, whitespaces)
        if not add_dummy_prefix:
            text = "<n>" + text
        tokens = self._get_text_tokenizer(encode_special_tokens=special_tokens).tokenize(text)
        return tokens if add_dummy_prefix else tokens[2:]

    def __getitem__(self, x: Union[int, str]):
        if isinstance(x, int):
            if x < self.num_image_tokens:
                return "<image_{}>".format(x)
            else:
                return self.text_tokenizer.convert_id_to_token(x - self.num_image_tokens)
        elif isinstance(x, str):
            if x.startswith("<image_") and x.endswith(">") and x[7:-1].isdigit():
                return int(x[7:-1])
            else:
                return self.text_tokenizer.convert_token_to_id(x) + self.num_image_tokens
        else:
            raise ValueError("The key should be str or int.")


class ChatGLMTokenizer(PreTrainedTokenizer):
    """
    Construct a ChatGLM tokenizer. Based on byte-level Byte-Pair-Encoding.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
    """

    vocab_files_names = {"vocab_file": "ice_text.model"}
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids"]

    def __init__(
            self,
            vocab_file,
            do_lower_case=False,
            remove_space=False,
            bos_token='sop',
            eos_token='eos',
            eop_token='eop',
            mask_token='[MASK]',
            gmask_token='[gMASK]',
            padding_side="right",
            **kwargs
    ) -> None:
        super().__init__(
            do_lower_case=do_lower_case,
            remove_space=remove_space,
            padding_side=padding_side,
            **kwargs
        )

        self.do_lower_case = do_lower_case
        self.remove_space = remove_space
        self.vocab_file = vocab_file

        self.bos_token = bos_token
        self.eos_token = eos_token
        self.eop_token = eop_token
        self.mask_token = mask_token
        self.gMASK_token = gmask_token

        self.sp_tokenizer = SPTokenizer(vocab_file)

        """ Initialisation """

    @property
    def eop_token_id(self) -> Optional[int]:
        """
        `Optional[int]`: Id of the end of sentence token in the vocabulary. Returns `None` if the token has not been
        set.
        """
        if self.eop_token is None:
            return None
        return self.convert_tokens_to_ids(self.eop_token)

    @property
    def vocab_size(self):
        """ Returns vocab size """
        return self.sp_tokenizer.num_tokens

    def get_vocab(self):
        """ Returns vocab as a dict """
        vocab = {self._convert_id_to_token(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def preprocess_text(self, inputs):
        if self.remove_space:
            outputs = " ".join(inputs.strip().split())
        else:
            outputs = inputs

        if self.do_lower_case:
            outputs = outputs.lower()

        return outputs

    def _tokenize(self, text, **kwargs):
        """ Returns a tokenized string. """
        text = self.preprocess_text(text)

        seq = self.sp_tokenizer.tokenize(text)

        return seq

    def decode(
            self,
            token_ids: Union[List[int], List[List[int]]],
            skip_special_tokens: bool = False,
            clean_up_tokenization_spaces: bool = True,
            spaces_between_special_tokens: bool = True,
            **kwargs
    ) -> str:
        if isinstance(token_ids[0], list):
            tokens = []
            for single_token_ids in token_ids:
                if self.pad_token_id in single_token_ids:  # remove pad
                    single_token_ids = list(filter((self.pad_token_id).__ne__, single_token_ids))
                tokens.append(self.sp_tokenizer.decode(single_token_ids))
            return (tokens)
        else:
            if self.pad_token_id in token_ids:  # remove pad
                token_ids = list(filter((self.pad_token_id).__ne__, token_ids))
            return self.sp_tokenizer.decode(token_ids)

    def _convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        return self.sp_tokenizer[token]

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.sp_tokenizer[index]

    def save_vocabulary(self, save_directory, filename_prefix=None):
        """
        Save the vocabulary and special tokens file to a directory.

        Args:
            save_directory (`str`):
                The directory in which to save the vocabulary.
            filename_prefix (`str`, *optional*):
                An optional prefix to add to the named of the saved files.

        Returns:
            `Tuple(str)`: Paths to the files saved.
        """
        if os.path.isdir(save_directory):
            vocab_file = os.path.join(
                save_directory, VOCAB_FILES_NAMES["vocab_file"]
            )
        else:
            vocab_file = save_directory

        with open(self.vocab_file, 'rb') as fin:
            proto_str = fin.read()

        with open(vocab_file, "wb") as writer:
            writer.write(proto_str)

        return (vocab_file,)

    def build_inputs_with_special_tokens(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A BERT sequence has the following format:

        - single sequence: `[CLS] X [SEP]`
        - pair of sequences: `[CLS] A [SEP] B [SEP]`

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        mask_id = self.sp_tokenizer[self.mask_token]
        gmask_id = self.sp_tokenizer[self.gMASK_token]
        bos_id = self.sp_tokenizer[self.bos_token]
        eos_id = self.sp_tokenizer[self.eos_token]

        if token_ids_1 is not None:
            token_ids_0 += token_ids_1

        if mask_id not in token_ids_0 and gmask_id not in token_ids_0 and len(token_ids_0) > 0:
            token_ids_0 += [gmask_id]

        # if token_ids_0[-1] != mask_id and token_ids_0[-1] != gmask_id:
        token_ids_0 += [eos_id]

        return token_ids_0

    def build_inputs_for_generation(self, model_input: BatchEncoding, max_gen_length=512, targets=None, padding=False):
        mask_ids = [self.sp_tokenizer[self.mask_token],
                    self.sp_tokenizer[self.gMASK_token]]
        bos_id = self.sp_tokenizer[self.bos_token]
        pad_id = self.sp_tokenizer[self.pad_token]
        # eos_id = self.sp_tokenizer[self.eos_token]

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
            assert len(targets) == len(input_ids)
            targets = [target[:max_gen_length] for target in targets]
            if not padding:
                max_gen_length = max(map(len, targets))
            targets = [[bos_id] + target for target in targets]
            labels = [target[1:] for target in targets]
            targets = [target + [pad_id] * (max_gen_length + 1 - len(target)) for target in targets]
            labels = [label + [pad_id] * (max_gen_length - len(label)) for label in labels]
            targets = torch.tensor(targets, dtype=input_ids.dtype, device=input_ids.device)
            labels = torch.tensor(labels, dtype=input_ids.dtype, device=input_ids.device)
            labels = torch.cat((input_ids.new_full((batch_size, seq_length), pad_id), labels), dim=1)
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
        attention_mask = (attention_mask < 0.5).bool()
        if targets is None:
            input_ids = torch.cat((input_ids, input_ids.new_full((batch_size, 1), bos_id)), dim=-1)
        else:
            input_ids = torch.cat((input_ids, targets[:, :-1]), dim=1)
        batch = {"input_ids": input_ids, "position_ids": position_ids}
        if labels is None:
            batch["attention_mask"] = attention_mask
        else:
            batch["attention_mask"] = attention_mask
            batch["labels"] = labels

        return BatchEncoding(batch)
