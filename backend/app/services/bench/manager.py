"""In-memory async benchmark job manager.

Single-job semantics: at most one benchmark runs at a time. The job state
lives in a dict keyed by job id. No persistence in v0.1.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from app.schemas import BenchHistoryEntry, BenchJobStatus, BenchJobView
from app.services.bench import arc, fc, mmlu
from app.services.bench.history import history_store
from app.services.bench.registry import BENCHMARKS
from app.services.gpu_info import gpu_snapshot
from app.services.model_registry import list_models
from app.services.vllm_runner import runner


@dataclass
class BenchJob:
    id: str
    benchmark: str
    limit: Optional[int]
    temperature: float
    max_tokens: int
    status: BenchJobStatus = "queued"
    model_path: Optional[str] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    examples_done: int = 0
    examples_total: int = 0
    correct: int = 0
    score: Optional[float] = None
    error: Optional[str] = None
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    task: Optional[asyncio.Task] = None
    # Performance / hardware metrics
    total_tokens_generated: int = 0
    avg_throughput_tok_s: Optional[float] = None
    model_size_bytes: Optional[int] = None
    model_quantization: Optional[str] = None
    gpu_name: Optional[str] = None
    gpu_memory_total_mb: Optional[int] = None
    vram_used_peak_mb: Optional[int] = None
    vram_used_avg_mb: Optional[float] = None
    gpu_temp_avg_c: Optional[float] = None
    gpu_temp_peak_c: Optional[int] = None
    gpu_util_avg_pct: Optional[float] = None

    def view(self) -> BenchJobView:
        return BenchJobView(
            id=self.id,
            benchmark=self.benchmark,  # type: ignore[arg-type]
            status=self.status,
            model_path=self.model_path,
            started_at=self.started_at,
            finished_at=self.finished_at,
            examples_done=self.examples_done,
            examples_total=self.examples_total,
            correct=self.correct,
            score=self.score,
            error=self.error,
            limit=self.limit,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            total_tokens_generated=self.total_tokens_generated,
            avg_throughput_tok_s=self.avg_throughput_tok_s,
            model_size_bytes=self.model_size_bytes,
            model_quantization=self.model_quantization,
            gpu_name=self.gpu_name,
            gpu_memory_total_mb=self.gpu_memory_total_mb,
            vram_used_peak_mb=self.vram_used_peak_mb,
            vram_used_avg_mb=self.vram_used_avg_mb,
            gpu_temp_avg_c=self.gpu_temp_avg_c,
            gpu_temp_peak_c=self.gpu_temp_peak_c,
            gpu_util_avg_pct=self.gpu_util_avg_pct,
        )


def _lookup_model_meta(path: Optional[str]) -> tuple[Optional[int], Optional[str]]:
    if not path:
        return None, None
    try:
        target = Path(path).resolve()
    except Exception:
        return None, None
    try:
        for entry in list_models():
            try:
                if Path(entry.path).resolve() == target:
                    return entry.size_bytes, entry.quantization
            except Exception:
                continue
    except Exception:
        pass
    return None, None


class _GPUSampler:
    """Background asyncio sampler — accumulates VRAM/util/temp into running stats."""

    def __init__(self, interval: float = 1.0) -> None:
        self.interval = interval
        self.samples = 0
        self.used_mb_sum = 0
        self.used_mb_peak = 0
        self.util_sum = 0
        self.temp_sum = 0
        self.temp_count = 0
        self.temp_peak: Optional[int] = None
        self.gpu_name: Optional[str] = None
        self.total_mb: Optional[int] = None
        self._stop = asyncio.Event()
        self._task: Optional[asyncio.Task] = None

    def _take(self) -> None:
        snap = gpu_snapshot(0)
        if not snap:
            return
        self.samples += 1
        self.gpu_name = snap.get("name") or self.gpu_name
        if self.total_mb is None and snap.get("total_mb") is not None:
            self.total_mb = snap["total_mb"]
        used = int(snap.get("used_mb") or 0)
        self.used_mb_sum += used
        if used > self.used_mb_peak:
            self.used_mb_peak = used
        self.util_sum += int(snap.get("util_pct") or 0)
        temp = snap.get("temp_c")
        if temp is not None:
            self.temp_sum += int(temp)
            self.temp_count += 1
            if self.temp_peak is None or int(temp) > self.temp_peak:
                self.temp_peak = int(temp)

    async def _loop(self) -> None:
        # Take an initial sample immediately so very short jobs still get data.
        self._take()
        while not self._stop.is_set():
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.interval)
            except asyncio.TimeoutError:
                self._take()

    def start(self) -> None:
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task is None:
            return
        self._stop.set()
        try:
            await self._task
        except Exception:
            pass

    def apply_to_job(self, job: BenchJob) -> None:
        job.gpu_name = self.gpu_name
        job.gpu_memory_total_mb = self.total_mb
        if self.samples > 0:
            job.vram_used_peak_mb = self.used_mb_peak
            job.vram_used_avg_mb = self.used_mb_sum / self.samples
            job.gpu_util_avg_pct = self.util_sum / self.samples
        if self.temp_count > 0:
            job.gpu_temp_avg_c = self.temp_sum / self.temp_count
            job.gpu_temp_peak_c = self.temp_peak


class BenchManager:
    def __init__(self) -> None:
        self._jobs: dict[str, BenchJob] = {}
        self._current_id: Optional[str] = None

    def list_jobs(self) -> list[BenchJobView]:
        return [j.view() for j in self._jobs.values()]

    def get(self, job_id: str) -> Optional[BenchJob]:
        return self._jobs.get(job_id)

    @property
    def current_id(self) -> Optional[str]:
        return self._current_id

    async def submit(
        self,
        *,
        benchmark: str,
        limit: Optional[int],
        temperature: float,
        max_tokens: int,
    ) -> BenchJob:
        if self._current_id is not None:
            current = self._jobs.get(self._current_id)
            if current and current.status in ("queued", "running"):
                raise RuntimeError("Another benchmark is already running.")
        if not runner.loaded:
            raise RuntimeError("No model loaded. Load a model first.")
        if not runner.acquire_bench():
            raise RuntimeError("Engine is busy with another benchmark.")

        size, quant = _lookup_model_meta(runner.loaded_path)
        job = BenchJob(
            id=str(uuid.uuid4()),
            benchmark=benchmark,
            limit=limit,
            temperature=temperature,
            max_tokens=max_tokens,
            model_path=runner.loaded_path,
            model_size_bytes=size,
            model_quantization=quant or runner.loaded_quant,
        )
        self._jobs[job.id] = job
        self._current_id = job.id
        job.task = asyncio.create_task(self._run(job))
        return job

    def cancel(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if not job or job.status not in ("queued", "running"):
            return False
        job.cancel_event.set()
        return True

    async def _run(self, job: BenchJob) -> None:
        job.status = "running"
        job.started_at = time.time()

        async def on_progress(done: int, total: int, correct: int, tokens: int = 0) -> None:
            job.examples_done = done
            job.examples_total = total
            job.correct = correct
            job.total_tokens_generated = tokens
            # Periodic intermediate score so the UI can display a live number.
            if done > 0:
                job.score = correct / done
            if job.started_at and tokens > 0:
                elapsed = max(time.time() - job.started_at, 0.001)
                job.avg_throughput_tok_s = tokens / elapsed

        def should_cancel() -> bool:
            return job.cancel_event.is_set()

        sampler = _GPUSampler(interval=1.0)
        sampler.start()

        try:
            if job.benchmark == arc.NAME:
                async for _ in arc.run(
                    limit=job.limit,
                    temperature=job.temperature,
                    max_tokens=job.max_tokens,
                    on_progress=on_progress,
                    should_cancel=should_cancel,
                ):
                    pass
            elif job.benchmark == mmlu.NAME:
                async for _ in mmlu.run(
                    limit=job.limit,
                    temperature=job.temperature,
                    max_tokens=job.max_tokens,
                    on_progress=on_progress,
                    should_cancel=should_cancel,
                ):
                    pass
            elif job.benchmark == fc.NAME:
                async for _ in fc.run(
                    limit=job.limit,
                    temperature=job.temperature,
                    max_tokens=job.max_tokens,
                    on_progress=on_progress,
                    should_cancel=should_cancel,
                ):
                    pass
            else:
                raise ValueError(f"Unknown benchmark: {job.benchmark}")

            if job.cancel_event.is_set():
                job.status = "cancelled"
            else:
                job.status = "completed"
                if job.examples_done > 0:
                    job.score = job.correct / job.examples_done
        except Exception as e:  # pragma: no cover - defensive
            job.status = "failed"
            job.error = f"{type(e).__name__}: {e}"
        finally:
            job.finished_at = time.time()
            await sampler.stop()
            sampler.apply_to_job(job)
            if job.started_at and job.total_tokens_generated > 0:
                elapsed = max(job.finished_at - job.started_at, 0.001)
                job.avg_throughput_tok_s = job.total_tokens_generated / elapsed
            try:
                self._persist_job(job)
            except Exception:
                pass
            runner.release_bench()
            if self._current_id == job.id:
                self._current_id = None

    def _persist_job(self, job: BenchJob) -> None:
        info = BENCHMARKS.get(job.benchmark)
        duration_seconds = None
        if job.started_at is not None and job.finished_at is not None:
            duration_seconds = max(job.finished_at - job.started_at, 0.0)

        history_store.append(
            BenchHistoryEntry(
                id=job.id,
                benchmark=job.benchmark,  # type: ignore[arg-type]
                benchmark_name=info.name if info else job.benchmark,
                status=job.status,
                model_path=job.model_path,
                started_at=job.started_at,
                finished_at=job.finished_at,
                duration_seconds=duration_seconds,
                examples_done=job.examples_done,
                examples_total=job.examples_total,
                correct=job.correct,
                score=job.score,
                error=job.error,
                limit=job.limit,
                temperature=job.temperature,
                max_tokens=job.max_tokens,
                subject_count=info.subject_count if info else None,
                total_tokens_generated=job.total_tokens_generated,
                avg_throughput_tok_s=job.avg_throughput_tok_s,
                model_size_bytes=job.model_size_bytes,
                model_quantization=job.model_quantization,
                gpu_name=job.gpu_name,
                gpu_memory_total_mb=job.gpu_memory_total_mb,
                vram_used_peak_mb=job.vram_used_peak_mb,
                vram_used_avg_mb=job.vram_used_avg_mb,
                gpu_temp_avg_c=job.gpu_temp_avg_c,
                gpu_temp_peak_c=job.gpu_temp_peak_c,
                gpu_util_avg_pct=job.gpu_util_avg_pct,
            )
        )


bench_manager = BenchManager()
