from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.enable_chunked_prefill = config.enable_chunked_prefill
        self.prefill_chunk_size = config.prefill_chunk_size
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], list[int]]:
        if self.enable_chunked_prefill:
            return self.schedule_chunked()
        else:
            return self.schedule_legacy()

    def schedule_chunked(self) -> tuple[list[Sequence], list[int]]:
        scheduled_seqs = []
        chunk_sizes = []
        num_seqs = 0
        num_batched_tokens = 0

        # First pass: existing running sequences, each processed at most once
        for _ in range(len(self.running)):
            if num_seqs >= self.max_num_seqs or not self.running:
                break
            seq = self.running.popleft()
            if seq.is_prefill:                                      # handle inflight prefill
                remaining = seq.num_prompt_tokens - seq.num_computed_tokens
                chunk = min(self.prefill_chunk_size, remaining, self.max_num_batched_tokens - num_batched_tokens)
                if chunk > 0:
                    num_seqs += 1
                    chunk_sizes.append(chunk)
                    num_batched_tokens += chunk
                    scheduled_seqs.append(seq)
                self.running.append(seq)                            # put back in running
            else:                                                   # decode path
                while not self.block_manager.can_append(seq):
                    if self.running:
                        self.preempt(self.running.pop())
                    else:
                        self.preempt(seq)
                        break
                else:
                    num_seqs += 1
                    chunk_sizes.append(1)
                    self.block_manager.may_append(seq)
                    scheduled_seqs.append(seq)
                    self.running.append(seq)                        # put back in running

        # Second pass: admit new sequences from waiting
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            current_start = seq.num_computed_tokens
            if current_start < len(seq):
                end = min(current_start + self.prefill_chunk_size, len(seq))
                if num_batched_tokens + (end - current_start) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                    break
                chunk_sizes.append(end - current_start)
                self.block_manager.allocate(seq)
                num_batched_tokens += end - current_start
                num_seqs += 1
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
                scheduled_seqs.append(seq)

        assert scheduled_seqs
        return scheduled_seqs, chunk_sizes

    def schedule_legacy(self) -> tuple[list[Sequence], list[int]]:
        # prefill
        scheduled_seqs = []
        chunk_sizes = []
        num_seqs = 0
        num_batched_tokens = 0
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break
            num_seqs += 1
            self.block_manager.allocate(seq)
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            seq.num_computed_tokens = seq.num_cached_tokens
            chunk_sizes.append(len(seq) - seq.num_cached_tokens)
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
        if scheduled_seqs:
            return scheduled_seqs, chunk_sizes

        # decode
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                num_seqs += 1
                chunk_sizes.append(1)
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, chunk_sizes

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        seq.num_computed_tokens = 0
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], chunk_sizes: list[int], token_ids: list[int]):
        sample_idx = 0
        for seq, chunk_size in zip(seqs, chunk_sizes):
            seq.num_computed_tokens += chunk_size
            if seq.is_prefill:
                continue  # mid-chunk, no token yet
            token_id = token_ids[sample_idx]
            sample_idx += 1
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
