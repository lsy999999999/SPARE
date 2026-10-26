from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(slots=True)
class AllocationAttempt:
    success: bool
    physical_delta_tokens: int = 0
    reason: str = ""
    physical_tokens_after: int = 0


class KVAllocator:
    def __init__(self, capacity_tokens: int, block_size_tokens: int):
        self.capacity_tokens = max(1, int(capacity_tokens))
        self.block_size_tokens = max(1, int(block_size_tokens))

    def clone(self) -> "KVAllocator":
        return copy.deepcopy(self)

    def mode_name(self) -> str:
        raise NotImplementedError

    def current_physical_tokens(self) -> int:
        raise NotImplementedError

    def estimate_growth(self, request_id: int, current_logical_tokens: int, delta_tokens: int) -> AllocationAttempt:
        preview = self.clone()
        return preview.apply_growth(request_id, current_logical_tokens, delta_tokens)

    def apply_growth(self, request_id: int, current_logical_tokens: int, delta_tokens: int) -> AllocationAttempt:
        raise NotImplementedError

    def release(self, request_id: int) -> None:
        raise NotImplementedError

    def request_physical_tokens(self, request_id: int) -> int:
        raise NotImplementedError

    def quantize_tokens(self, logical_tokens: int) -> int:
        raise NotImplementedError

    def metrics(self, logical_tokens_by_request: Dict[int, int]) -> Dict[str, float]:
        raise NotImplementedError


class TokenKVAllocator(KVAllocator):
    def __init__(self, capacity_tokens: int, block_size_tokens: int):
        super().__init__(capacity_tokens, block_size_tokens)
        self.allocations: Dict[int, int] = {}

    def mode_name(self) -> str:
        return "token"

    def current_physical_tokens(self) -> int:
        return sum(self.allocations.values())

    def apply_growth(self, request_id: int, current_logical_tokens: int, delta_tokens: int) -> AllocationAttempt:
        if delta_tokens <= 0:
            return AllocationAttempt(True, 0, physical_tokens_after=self.current_physical_tokens())
        self.allocations[request_id] = int(current_logical_tokens + delta_tokens)
        return AllocationAttempt(
            True,
            physical_delta_tokens=int(delta_tokens),
            physical_tokens_after=self.current_physical_tokens(),
        )

    def release(self, request_id: int) -> None:
        self.allocations.pop(request_id, None)

    def request_physical_tokens(self, request_id: int) -> int:
        return int(self.allocations.get(request_id, 0))

    def quantize_tokens(self, logical_tokens: int) -> int:
        return max(0, int(logical_tokens))

    def metrics(self, logical_tokens_by_request: Dict[int, int]) -> Dict[str, float]:
        logical_total = float(sum(logical_tokens_by_request.values()))
        physical_total = float(self.current_physical_tokens())
        return {
            "allocator_internal_fragmentation_tokens": max(0.0, physical_total - logical_total),
            "allocator_external_fragmentation_tokens": 0.0,
            "allocator_largest_free_run_tokens": float(max(0, self.capacity_tokens - int(physical_total))),
            "allocator_live_requests": float(len(self.allocations)),
        }


class BlockKVAllocator(KVAllocator):
    def __init__(self, capacity_tokens: int, block_size_tokens: int):
        super().__init__(capacity_tokens, block_size_tokens)
        self.capacity_blocks = max(1, self.capacity_tokens // self.block_size_tokens)
        self.allocations: Dict[int, Tuple[int, int]] = {}
        self.free_runs: List[Tuple[int, int]] = [(0, self.capacity_blocks)]

    def mode_name(self) -> str:
        return "block"

    def current_physical_tokens(self) -> int:
        return sum(length for _, length in self.allocations.values()) * self.block_size_tokens

    def quantize_tokens(self, logical_tokens: int) -> int:
        if logical_tokens <= 0:
            return 0
        blocks = int(math.ceil(logical_tokens / self.block_size_tokens))
        return blocks * self.block_size_tokens

    def _required_blocks(self, logical_tokens: int) -> int:
        return int(math.ceil(max(0, logical_tokens) / self.block_size_tokens))

    def _find_fit(self, required_blocks: int) -> Optional[int]:
        for index, (_, run_length) in enumerate(self.free_runs):
            if run_length >= required_blocks:
                return index
        return None

    def _allocate_from_run(self, request_id: int, required_blocks: int) -> bool:
        run_index = self._find_fit(required_blocks)
        if run_index is None:
            return False
        start, run_length = self.free_runs.pop(run_index)
        self.allocations[request_id] = (start, required_blocks)
        remaining = run_length - required_blocks
        if remaining > 0:
            self.free_runs.insert(run_index, (start + required_blocks, remaining))
        return True

    def _merge_free_runs(self) -> None:
        merged: List[Tuple[int, int]] = []
        for start, length in sorted(self.free_runs):
            if not merged:
                merged.append((start, length))
                continue
            last_start, last_length = merged[-1]
            if last_start + last_length == start:
                merged[-1] = (last_start, last_length + length)
            else:
                merged.append((start, length))
        self.free_runs = merged

    def release(self, request_id: int) -> None:
        allocation = self.allocations.pop(request_id, None)
        if allocation is None:
            return
        self.free_runs.append(allocation)
        self._merge_free_runs()

    def request_physical_tokens(self, request_id: int) -> int:
        return int(self.allocations.get(request_id, (0, 0))[1] * self.block_size_tokens)

    def apply_growth(self, request_id: int, current_logical_tokens: int, delta_tokens: int) -> AllocationAttempt:
        if delta_tokens <= 0:
            return AllocationAttempt(True, 0, physical_tokens_after=self.current_physical_tokens())
        snapshot_free_runs = copy.deepcopy(self.free_runs)
        snapshot_allocations = dict(self.allocations)
        old_blocks = self.allocations.get(request_id, (0, 0))[1]
        new_logical_tokens = int(current_logical_tokens + delta_tokens)
        required_blocks = self._required_blocks(new_logical_tokens)
        if required_blocks <= old_blocks:
            return AllocationAttempt(True, 0, physical_tokens_after=self.current_physical_tokens())

        existing = self.allocations.pop(request_id, None)
        if existing is not None:
            self.free_runs.append(existing)
            self._merge_free_runs()
        allocated = self._allocate_from_run(request_id, required_blocks)
        if not allocated:
            self.free_runs = snapshot_free_runs
            self.allocations = snapshot_allocations
            return AllocationAttempt(
                False,
                0,
                reason="contiguous_block_unavailable",
                physical_tokens_after=self.current_physical_tokens(),
            )

        physical_delta = (required_blocks - old_blocks) * self.block_size_tokens
        return AllocationAttempt(
            True,
            physical_delta_tokens=physical_delta,
            physical_tokens_after=self.current_physical_tokens(),
        )

    def metrics(self, logical_tokens_by_request: Dict[int, int]) -> Dict[str, float]:
        logical_total = float(sum(logical_tokens_by_request.values()))
        physical_total = float(self.current_physical_tokens())
        free_blocks = sum(length for _, length in self.free_runs)
        largest_free_run = max((length for _, length in self.free_runs), default=0)
        external_fragment_blocks = max(0, free_blocks - largest_free_run)
        return {
            "allocator_internal_fragmentation_tokens": max(0.0, physical_total - logical_total),
            "allocator_external_fragmentation_tokens": float(external_fragment_blocks * self.block_size_tokens),
            "allocator_largest_free_run_tokens": float(largest_free_run * self.block_size_tokens),
            "allocator_live_requests": float(len(self.allocations)),
        }


class PagedKVAllocator(KVAllocator):
    def __init__(self, capacity_tokens: int, block_size_tokens: int):
        super().__init__(capacity_tokens, block_size_tokens)
        self.capacity_blocks = max(1, self.capacity_tokens // self.block_size_tokens)
        self.allocations: Dict[int, List[int]] = {}
        self.free_blocks: List[int] = list(range(self.capacity_blocks))

    def mode_name(self) -> str:
        return "paged"

    def current_physical_tokens(self) -> int:
        return (self.capacity_blocks - len(self.free_blocks)) * self.block_size_tokens

    def quantize_tokens(self, logical_tokens: int) -> int:
        if logical_tokens <= 0:
            return 0
        blocks = int(math.ceil(logical_tokens / self.block_size_tokens))
        return blocks * self.block_size_tokens

    def _required_blocks(self, logical_tokens: int) -> int:
        return int(math.ceil(max(0, logical_tokens) / self.block_size_tokens))

    def release(self, request_id: int) -> None:
        blocks = self.allocations.pop(request_id, [])
        if blocks:
            self.free_blocks.extend(blocks)

    def estimate_growth(self, request_id: int, current_logical_tokens: int, delta_tokens: int) -> AllocationAttempt:
        if delta_tokens <= 0:
            return AllocationAttempt(True, 0, physical_tokens_after=self.current_physical_tokens())
        current_blocks = len(self.allocations.get(request_id, []))
        required_blocks = self._required_blocks(int(current_logical_tokens + delta_tokens))
        additional_blocks = max(0, required_blocks - current_blocks)
        if len(self.free_blocks) < additional_blocks:
            return AllocationAttempt(
                False,
                0,
                reason="paged_block_unavailable",
                physical_tokens_after=self.current_physical_tokens(),
            )
        return AllocationAttempt(
            True,
            physical_delta_tokens=additional_blocks * self.block_size_tokens,
            physical_tokens_after=self.current_physical_tokens() + additional_blocks * self.block_size_tokens,
        )

    def request_physical_tokens(self, request_id: int) -> int:
        return int(len(self.allocations.get(request_id, [])) * self.block_size_tokens)

    def apply_growth(self, request_id: int, current_logical_tokens: int, delta_tokens: int) -> AllocationAttempt:
        if delta_tokens <= 0:
            return AllocationAttempt(True, 0, physical_tokens_after=self.current_physical_tokens())
        current_blocks = len(self.allocations.get(request_id, []))
        new_logical_tokens = int(current_logical_tokens + delta_tokens)
        required_blocks = self._required_blocks(new_logical_tokens)
        additional_blocks = required_blocks - current_blocks
        if additional_blocks <= 0:
            return AllocationAttempt(True, 0, physical_tokens_after=self.current_physical_tokens())
        if len(self.free_blocks) < additional_blocks:
            return AllocationAttempt(
                False,
                0,
                reason="paged_block_unavailable",
                physical_tokens_after=self.current_physical_tokens(),
            )
        allocated = self.allocations.setdefault(request_id, [])
        for _ in range(additional_blocks):
            allocated.append(self.free_blocks.pop())
        return AllocationAttempt(
            True,
            physical_delta_tokens=additional_blocks * self.block_size_tokens,
            physical_tokens_after=self.current_physical_tokens(),
        )

    def metrics(self, logical_tokens_by_request: Dict[int, int]) -> Dict[str, float]:
        logical_total = float(sum(logical_tokens_by_request.values()))
        physical_total = float(self.current_physical_tokens())
        free_blocks = float(len(self.free_blocks))
        return {
            "allocator_internal_fragmentation_tokens": max(0.0, physical_total - logical_total),
            "allocator_external_fragmentation_tokens": 0.0,
            "allocator_largest_free_run_tokens": free_blocks * self.block_size_tokens,
            "allocator_live_requests": float(len(self.allocations)),
        }


class QueryScopedKVAllocator(KVAllocator):
    def __init__(self, mode: str, capacity_tokens: int, block_size_tokens: int):
        super().__init__(capacity_tokens, block_size_tokens)
        normalized = mode.strip().lower()
        if normalized not in {"token", "block", "paged"}:
            raise ValueError(f"unsupported query-scoped kv allocator mode: {mode}")
        self.mode = normalized
        self.allocations: Dict[int, int] = {}

    def mode_name(self) -> str:
        return self.mode

    def current_physical_tokens(self) -> int:
        return sum(self.allocations.values())

    def request_physical_tokens(self, request_id: int) -> int:
        return int(self.allocations.get(request_id, 0))

    def quantize_tokens(self, logical_tokens: int) -> int:
        if self.mode == "token":
            return max(0, int(logical_tokens))
        if logical_tokens <= 0:
            return 0
        blocks = int(math.ceil(logical_tokens / self.block_size_tokens))
        return blocks * self.block_size_tokens

    def apply_growth(self, request_id: int, current_logical_tokens: int, delta_tokens: int) -> AllocationAttempt:
        if delta_tokens <= 0:
            return AllocationAttempt(True, 0, physical_tokens_after=self.current_physical_tokens())
        current_physical = self.request_physical_tokens(request_id)
        new_physical = self.quantize_tokens(int(current_logical_tokens + delta_tokens))
        physical_delta = max(0, new_physical - current_physical)
        if self.current_physical_tokens() + physical_delta > self.capacity_tokens:
            return AllocationAttempt(
                False,
                0,
                reason="query_scoped_capacity_exceeded",
                physical_tokens_after=self.current_physical_tokens(),
            )
        self.allocations[request_id] = new_physical
        return AllocationAttempt(
            True,
            physical_delta_tokens=physical_delta,
            physical_tokens_after=self.current_physical_tokens(),
        )

    def release(self, request_id: int) -> None:
        self.allocations.pop(request_id, None)

    def metrics(self, logical_tokens_by_request: Dict[int, int]) -> Dict[str, float]:
        logical_total = float(sum(logical_tokens_by_request.values()))
        physical_total = float(self.current_physical_tokens())
        return {
            "allocator_internal_fragmentation_tokens": max(0.0, physical_total - logical_total),
            "allocator_external_fragmentation_tokens": 0.0,
            "allocator_largest_free_run_tokens": float(max(0, self.capacity_tokens - int(physical_total))),
            "allocator_live_requests": float(len(self.allocations)),
        }


def build_kv_allocator(
    mode: str,
    capacity_tokens: int,
    block_size_tokens: int,
    scope: str = "model",
) -> KVAllocator:
    normalized_scope = scope.strip().lower()
    if normalized_scope == "query":
        return QueryScopedKVAllocator(mode, capacity_tokens, block_size_tokens)
    normalized = mode.strip().lower()
    if normalized == "token":
        return TokenKVAllocator(capacity_tokens, block_size_tokens)
    if normalized == "block":
        return BlockKVAllocator(capacity_tokens, block_size_tokens)
    if normalized == "paged":
        return PagedKVAllocator(capacity_tokens, block_size_tokens)
    raise ValueError(f"unsupported kv allocator mode: {mode}")
