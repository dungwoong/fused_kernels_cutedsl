import cutlass
from cutlass import cute, pipeline
from cck.runtime import shared, mma, pipeline as my_pipeline
import torch


class Kernel:
    """
    We will focus the specific case where new_tokens is 16, causing
    something like triton/trinity to not optimize well
    """
    def __init__(
            self,
            dim: int, # 128
            new_tokens: int, # 16
            n_split: int, 
        ):
        pass
    
