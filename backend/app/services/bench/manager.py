"""In-memory async benchmark job manager.

Single-job semantics: at most one benchmark runs at a time. The job state
lives in a dict keyed by job id. No persistence in v0.1.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from app.schemas import BenchHistoryEntry, BenchJobStatus, BenchJobView
from app.services.bench import arc, fc, mmlu
from app.services.bench.history import history_store
from app.services.bench.registry import BENCHMARKS
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
        )


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

        job = BenchJob(
            id=str(uuid.uuid4()),
            benchmark=benchmark,
            limit=limit,
            temperature=temperature,
            max_tokens=max_tokens,
            model_path=runner.loaded_path,
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

        async def on_progress(done: int, total: int, correct: int) -> None:
            job.examples_done = done
            job.examples_total = total
            job.correct = correct
            # Periodic intermediate score so the UI can display a live number.
            if done > 0:
                job.score = correct / done

        def should_cancel() -> bool:
            return job.cancel_event.is_set()

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
            )
        )


bench_manager = BenchManager()
