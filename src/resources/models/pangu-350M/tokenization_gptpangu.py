
import torch
import sentencepiece
import jieba
import numpy as np

from transformers.tokenization_utils import PreTrainedTokenizer


class GPTPanguTokenizer(PreTrainedTokenizer):
    # Ref: https://git.openi.org.cn/PCL-Platform.Intelligence/PanGu-Alpha/src/branch/master/tokenization_jieba.py
    vocab_files_names = {
        "model_file": "vocab.model"
    }

    def __init__(
            self,
            model_file,
            **kwargs
    ):
        super().__init__()

        self.sp = sentencepiece.SentencePieceProcessor()
        self.sp.Load(model_file=model_file)
        self.translator = str.maketrans(" \n", "\u2582\u2583")

        # special token ids
        # self.eos_token_id = self.sp.piece_to_id("<eot>")

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
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
        if self.bos_token_id is not None:
            if token_ids_1 is None:
                return [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
            bos = [self.bos_token_id]
            sep = [self.sep_token_id]
            eos = [self.eos_token_id]
            return bos + token_ids_0 + sep + token_ids_1 + eos
        else:
            if token_ids_1 is None:
                return token_ids_0 + [self.eos_token_id]
            sep = [self.sep_token_id]
            eos = [self.eos_token_id]
            return token_ids_0 + sep + token_ids_1 + eos

    def tokenize(self, text, **kwargs):
        """ Tokenize a string. """
        seg_list = [x.translate(self.translator) for x in jieba.cut(text, cut_all=False)]
        return seg_list

    def convert_tokens_to_ids(self, tokens):
        if tokens is None:
            return None

        if isinstance(tokens, str):
            return self._convert_token_to_id_with_added_voc(tokens)
            
        new_seg = " ".join(tokens)
        return self.sp.encode(new_seg)
        # return tokens

    def _convert_token_to_id(self, token):
        return self.sp.piece_to_id(token)

    def _convert_id_to_token(self, index):
        return self.sp.id_to_piece(index)

    def convert_ids_to_tokens(self, ids):
        return self.decode(ids)

    def decode(self, tokens, **kwargs):
        if isinstance(tokens, torch.Tensor) or isinstance(tokens, np.ndarray):
            tokens = tokens.tolist()

        text = self.sp.decode(tokens)
        if isinstance(text, list):
            text = text[0]
        text = text.replace(' ', '').replace('\u2582', ' ').replace('\u2583', '\n')
        return text
