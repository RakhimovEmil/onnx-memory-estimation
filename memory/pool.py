# reproduced from dict/mt/libs/nn/ynmt/draft/onnx/applier_memory.cpp L189
from math import ceil, floor
from typing import Dict, List, Optional, Tuple

from containers import MutableTensorInfo
from memory import DeviceMemory, DevicePointer


class Pool:
    class DynamicBitmap:
        def __init__(self, max_limit: int, intervals: Optional[List[Tuple[int, int]]] = None):
            self.max_limit = max_limit
            self.intervals: List[Tuple[int, int]] = intervals if intervals is not None else list()  # ordered


        def add_interval(self, interval: Tuple[int, int]):
            interval_pruned = (max(0, interval[0]), min(self.max_limit, interval[1]))
            if interval_pruned[1] < interval_pruned[0]:
                return

            lb, ub = self.get_lower_bound(interval_pruned[0]), self.get_upper_bound(interval_pruned[1])

            if lb > ub:
                raise RuntimeError(f'Lower bound {lb} is higher than upper bound {ub}, bounds = {self.intervals}')

            interval_new = (
                (min(self.intervals[lb][0], interval_pruned[0]), max(self.intervals[ub - 1][1], interval_pruned[1]))
                if lb < ub
                else interval_pruned
            )

            self.intervals = self.intervals[:lb] + [interval_new] + self.intervals[ub:]
            idx = lb    # index of the new interval

            if idx < len(self.intervals) - 1:   # glue upper end
                if self.intervals[idx + 1][0] - self.intervals[idx][1] <= 0:
                    raise RuntimeError(f'New interval {self.intervals[idx]} overlaps with upper bound {self.intervals[idx + 1]}, bounds = {self.intervals}')
                if self.intervals[idx + 1][0] - self.intervals[idx][1] == 1:
                    self.intervals = self.intervals[:idx] + [(self.intervals[idx][0], self.intervals[idx + 1][1])] + self.intervals[idx + 2:]

            if idx > 0:     # glue lower end
                if self.intervals[idx][0] - self.intervals[idx - 1][1] <= 0:
                    raise RuntimeError(f'New interval {self.intervals[idx]} overlaps with lower bound {self.intervals[idx - 1]}, bounds = {self.intervals}')
                if self.intervals[idx][0] - self.intervals[idx - 1][1] == 1:
                    self.intervals = self.intervals[:idx - 1] + [(self.intervals[idx - 1][0], self.intervals[idx][1])] + self.intervals[idx + 1:]


        def get_lower_bound(self, point: int) -> int:
            l, r = 0, len(self.intervals)
            while l < r:
                m = ceil((l + r) / 2)
                rv_for_m = self.intervals[m - 1][1] if m > 0 else -1
                if point <= rv_for_m:
                    r = m - 1
                else:
                    l = m

            return l


        def get_upper_bound(self, point: int) -> int:
            l, r = 0, len(self.intervals)
            while l < r:
                m = floor((l + r) / 2)
                lv_for_m = self.intervals[m][0] if m < len(self.intervals) else float('inf')
                if point >= lv_for_m:
                    l = m + 1
                else:
                    r = m

            return l


        def flip(self):
            flipped_intervals = [
                (self.intervals[idx][1] + 1, self.intervals[idx + 1][0] - 1)
                for idx in range(len(self.intervals) - 1)
            ]

            if self.intervals[0][0] > 0:
                flipped_intervals.insert(0, (0, self.intervals[0][0] - 1))

            if self.intervals[-1][1] < self.max_limit:
                flipped_intervals.insert(len(flipped_intervals), (self.intervals[-1][1] + 1, self.max_limit))

            self.intervals = flipped_intervals

    class PointerInfo:
        def __init__(self, info: MutableTensorInfo, ptr: DevicePointer, size: int, max_tensor_lifetime: int):
            self.info = info
            self.ptr = ptr
            self.size = size
            self.times = Pool.DynamicBitmap(
                max_limit=max_tensor_lifetime,
                intervals=[(info.lifetime_begin, info.lifetime_end)]
            )


        def lifetime_intersects(self, other: MutableTensorInfo):
            candidate_idx = self.times.get_lower_bound(other.lifetime_begin)
            return candidate_idx < len(self.times.intervals) and self.times.intervals[candidate_idx][1] >= other.lifetime_begin and other.lifetime_end >= self.times.intervals[candidate_idx][0]

    def __init__(self, max_tensor_lifetime: int, device_memory: DeviceMemory):
        self.max_tensor_lifetime = max_tensor_lifetime
        self.device_memory = device_memory
        self.pool: List[Pool.PointerInfo] = list()  # ordered by size and lifetime
        self.ptr_to_idx: Dict[DevicePointer, int] = dict()
        self.ptr_to_storage_block_offset: Dict[DevicePointer, int] = dict()
        self.number_of_storage_blocks = 0

    def try_reserve(self, required: MutableTensorInfo) -> Optional[DevicePointer]:
        for candidate in self.pool:     # binary search could help
            if candidate.size < required.estimated_size or candidate.lifetime_intersects(required):
                continue

            candidate.times = Pool.DynamicBitmap(
                max_limit=self.max_tensor_lifetime,
                intervals=[(required.lifetime_begin, required.lifetime_end)]
            )   # maybe can just update intervals
            candidate_ptr = candidate.ptr

            aligned_required_size = self.device_memory.align_size(required.estimated_size, 'UINT8')
            if aligned_required_size < candidate.size:
                remainder = candidate.size - aligned_required_size
                remainder_info = MutableTensorInfo('ignored', required.lifetime_begin, required.lifetime_end)
                remainder_info.set_estimated_parameters(0, 0, remainder)

                # differs from original dict/mt/libs/nn/ynmt/draft/onnx/applier_memory.cpp L240
                # because python does not support pointer arithmetics
                remainder_ptr = DevicePointer(candidate.ptr.type, candidate.ptr.segment, candidate.ptr.offset + aligned_required_size)
                self.__put_virtual_block(remainder_ptr, remainder_info, candidate.ptr, aligned_required_size)
            
            return candidate_ptr

        return None


    def put(self, ptr: DevicePointer, info: MutableTensorInfo):
        val = Pool.PointerInfo(info, ptr, info.estimated_size, self.max_tensor_lifetime)

        idx = 0     # binary search could help
        for candidate in self.pool:
            if candidate.size <= val.size:
                idx += 1
            else:
                break

        self.pool.insert(idx, val)  # preserving order by size (check again)
        self.ptr_to_storage_block_offset[ptr] = 0
        self.ptr_to_idx[ptr] = self.number_of_storage_blocks
        self.number_of_storage_blocks += 1


    def get_storage_block_idx(self, ptr: DevicePointer) -> int:
        return self.ptr_to_idx[ptr]


    def get_storage_block_offset(self, ptr: DevicePointer) -> int:
        return self.ptr_to_storage_block_offset[ptr]


    def __put_virtual_block(self, ptr: DevicePointer, info: MutableTensorInfo, parent_ptr: DevicePointer, virtual_block_offset: int):
        val = Pool.PointerInfo(info, ptr, info.estimated_size, self.max_tensor_lifetime)
        val.times.flip()

        idx = 0
        for candidate in self.pool:
            if candidate.size < val.size:
                idx += 1
            else:
                break

        self.pool.insert(idx, val)  # preserving order by size (check again)
        self.ptr_to_storage_block_offset[ptr] = self.get_storage_block_offset(parent_ptr) + virtual_block_offset
        self.ptr_to_idx[ptr] = self.get_storage_block_idx(parent_ptr)

def search_in_pool(info: MutableTensorInfo, pool: Pool, total_memory_estimated):
    ptr = pool.try_reserve(info)
    if ptr is None:
        stub_ptr = DevicePointer('UINT8', 0, total_memory_estimated)
        size = info.estimated_size
        pool.put(stub_ptr, info)
        return size
    return 0
