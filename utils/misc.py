import random
import numpy as np
import torch
import os

"""Sets the seed for reproducibility.
    Args:
        seed (int): The seed value.
    """
# Ensure deterministic algorithms are used where available
# Note: deterministic algorithms can sometimes be slower
# Set PYTHONHASHSEED environment variable (best set before Python starts)
# os.environ['PYTHONHASHSEED'] = str(seed) 