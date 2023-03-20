
import math
import os
import random
from numbers import Number

import numpy as np
import torch


def set_seed(seed_val=42):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)


def print_rank_0(*message):
    """
    Print only once from the main rank
    """
    if os.environ.get("RANK", "0") == "0":
        print(*message)


def significant(x: Number, ndigits=2) -> Number:
    """
    Cut the number up to its `ndigits` after the most significant
    """
    if isinstance(x, torch.Tensor):
        x = x.item()

    if not isinstance(x, Number) or math.isnan(x) or x == 0:
        return x

    return round(x, ndigits - int(math.floor(math.log10(abs(x)))))

#
# def set_seed(seed: int):
#     """
#     Sets seeds across package dependencies for reproducibility.
#     """
#     seed += int(os.environ.get("RANK", 0))
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)


