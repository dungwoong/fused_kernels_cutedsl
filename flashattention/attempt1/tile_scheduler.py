from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Tuple, override
from functools import lru_cache

import cutlass
from cute_dsl_utils import ArgumentsBase, ParamsBase
from fast_math import FastDivmod
from cutlass import Int32, Boolean
from cutlass import cute, pipeline
from cutlass._mlir import ir


"""
Simple tile scheduler, just rasterize stuff and no persistent schedule

Clusters are always (m, n). Cluster coords are also M, N but rasterization dictates how we map 1D --> 2D
"""

@lru_cache
def get_max_active_clusters(cluster_size):
    return cutlass.utils.HardwareInfo().get_max_active_clusters(cluster_size=cluster_size)

@dataclass
class TileSchedulerArguments(ParamsBase):
    num_block: Int32
    num_head: Int32
    num_batch: Int32
    cluster_shape_mn: Tuple[int, int]

class SingleTileScheduler:
    @dataclass
    class Params(ParamsBase):
        num_block: Int32
        num_head: Int32
        num_batch: Int32
        cluster_shape_mn: cutlass.Constexpr[Tuple[int, int]] = (1, 1)

        @staticmethod
        def create(args: TileSchedulerArguments, *, loc=None, ip=None) -> "SingleTileScheduler.Params":
            return SingleTileScheduler.Params(
                args.num_block,
                args.num_head,
                args.num_batch,
                args.cluster_shape_mn,
            )
    
    def __init__(self, params: Params, blk_coord: cute.Coord, *, loc=None, ip=None):
        self.params = params
        self._blk_coord = blk_coord
        self._is_first_block = True
        self._loc = loc
        self._ip = ip
    
    @staticmethod
    def to_underlying_arguments(args: TileSchedulerArguments, *, loc=None, ip=None):
        return SingleTileScheduler.Params.create(args, loc=loc, ip=ip)

    @staticmethod
    def create(params: Params, *, loc=None, ip=None) -> "SingleTileScheduler":
        blk_coord = cute.arch.block_idx()
        return SingleTileScheduler(params, blk_coord, loc=loc, ip=ip)
    
    @staticmethod
    def get_grid_shape(params: Params, *, loc=None, ip=None) -> Tuple[Int32, Int32, Int32]:
        return (
            cute.round_up(params.num_block, params.cluster_shape_mn[0]),
            params.num_head,
            params.num_batch,
        )
    
    def get_current_work(self, *, loc=None, ip=None) -> cutlass.utils.WorkTileInfo:
        block_idx, head_idx, batch_idx = self._blk_coord
        return cutlass.utils.WorkTileInfo((block_idx, head_idx, batch_idx), self._is_first_block)
    
    def initial_work_tile_info(self, *, loc=None, ip=None):
        return self.get_current_work(loc=loc, ip=ip)

    def advance_to_next_work(self, *, loc=None, ip=None):
        self._is_first_block = False
    
    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [self.params, self._blk_coord]:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip([self.params, self._blk_coord], self._values_pos):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return SingleTileScheduler(*(tuple(obj_list)), loc=self._loc)