from dataclasses import dataclass
from cutlass.pipeline import PipelineTmaAsync as PipelineTmaAsyncOg, PipelineState
from typing import Optional
from cutlass import Boolean, Int32, const_expr
from cutlass.cutlass_dsl import if_generate

@dataclass(frozen=True)
class PipelineTmaAsync(PipelineTmaAsyncOg):
    """
    Override producer_acquire to take in extra_tx_count parameter.
    """

    @staticmethod
    def create(*args, **kwargs):
        obj = PipelineTmaAsyncOg.create(*args, **kwargs)
        # Can't assign to __class__ directly since the dataclass is frozen
        # obj.__class__ = PipelineTmaAsync
        object.__setattr__(obj, "__class__", PipelineTmaAsync)
        return obj

    def producer_acquire(
        self,
        state: PipelineState,
        try_acquire_token: Optional[Boolean] = None,
        extra_tx_count: int = 0,
    ):
        """
        TMA producer commit conditionally waits on buffer empty and sets the transaction barrier for leader threadblocks.
        """
        if_generate(
            try_acquire_token is None or try_acquire_token == 0,
            lambda: self.sync_object_empty.wait(state.index, state.phase),
        )
        if const_expr(extra_tx_count == 0):
            self.sync_object_full.arrive(state.index, self.producer_mask)
        else:
            tx_count = self.sync_object_full.tx_count + extra_tx_count
            self.sync_object_full.arrive_and_expect_tx(state.index, tx_count)