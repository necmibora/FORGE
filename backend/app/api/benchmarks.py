import time
import uuid

from fastapi import APIRouter, HTTPException

from app.schemas import (
    BenchHistoryEntry,
    BenchJobView,
    BenchmarkList,
    RunBenchmarkRequest,
)
from app.services.bench.history import history_store
from app.services.bench.manager import bench_manager
from app.services.bench.registry import BENCHMARKS
from app.services.vllm_runner import runner

router = APIRouter(prefix="/benchmarks", tags=["benchmarks"])


@router.get("", response_model=BenchmarkList)
def list_benchmarks() -> BenchmarkList:
    return BenchmarkList(benchmarks=list(BENCHMARKS.values()))


@router.post("/run", response_model=BenchJobView)
async def run_benchmark(req: RunBenchmarkRequest) -> BenchJobView:
    if req.benchmark not in BENCHMARKS:
        raise HTTPException(404, f"Unknown benchmark: {req.benchmark}")
    try:
        job = await bench_manager.submit(
            benchmark=req.benchmark,
            limit=req.limit,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
        )
    except RuntimeError as e:
        message = str(e)
        if "No model loaded" in message or not runner.inference_available:
            return BenchJobView(
                id=str(uuid.uuid4()),
                benchmark=req.benchmark,
                status="failed",
                model_path=None,
                started_at=time.time(),
                finished_at=time.time(),
                examples_done=0,
                examples_total=0,
                correct=0,
                score=None,
                error=message,
                limit=req.limit,
                temperature=req.temperature,
                max_tokens=req.max_tokens,
            )
        raise HTTPException(409, message)


@router.get("/jobs", response_model=list[BenchJobView])
def list_jobs() -> list[BenchJobView]:
    return bench_manager.list_jobs()


@router.get("/history", response_model=list[BenchHistoryEntry])
def list_history() -> list[BenchHistoryEntry]:
    return history_store.list()


@router.get("/jobs/{job_id}", response_model=BenchJobView)
def get_job(job_id: str) -> BenchJobView:
    job = bench_manager.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job.view()


@router.post("/jobs/{job_id}/cancel", response_model=BenchJobView)
def cancel_job(job_id: str) -> BenchJobView:
    job = bench_manager.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    bench_manager.cancel(job_id)
    return job.view()
