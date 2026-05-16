from typing import Literal, Optional
from pydantic import BaseModel, Field


class GPUInfo(BaseModel):
    index: int
    name: str
    total_memory_mb: int
    free_memory_mb: int
    used_memory_mb: int
    utilization_pct: int
    temperature_c: Optional[int] = None


class SystemInfo(BaseModel):
    hostname: str
    slurm_job_id: Optional[str] = None
    slurm_node: Optional[str] = None
    cpu_count: int
    ram_total_mb: int
    ram_available_mb: int
    gpus: list[GPUInfo]


class ModelEntry(BaseModel):
    name: str
    path: str
    kind: Literal["base", "quant"]
    size_bytes: int
    quantization: Optional[str] = None  # e.g. "awq"


class ModelList(BaseModel):
    models: list[ModelEntry]


class LoadModelRequest(BaseModel):
    path: str
    dtype: str = "auto"
    max_model_len: Optional[int] = None
    gpu_memory_utilization: float = 0.90
    quantization: Optional[str] = None  # e.g. "awq"


class LoadedModelStatus(BaseModel):
    loaded: bool
    path: Optional[str] = None
    quantization: Optional[str] = None
    inference_available: bool = True
    inference_message: Optional[str] = None


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    stream: bool = True


# ─── Benchmark ────────────────────────────────────────────────────────────

BenchmarkId = Literal["arc_easy", "mmlu", "bfcl_simple"]
BenchJobStatus = Literal["queued", "running", "completed", "failed", "cancelled"]
ToolCallMismatchCategory = Literal[
    "correct",
    "no_tool_call",
    "malformed_tool_call",
    "wrong_tool_name",
    "missing_argument",
    "wrong_argument_value",
]


class BenchmarkInfo(BaseModel):
    id: BenchmarkId
    name: str
    description: str
    total_examples: int
    kind: Literal["mcq", "generation", "function_calling"] = "mcq"
    subject_count: Optional[int] = None


class BenchmarkList(BaseModel):
    benchmarks: list[BenchmarkInfo]


class RunBenchmarkRequest(BaseModel):
    benchmark: BenchmarkId
    limit: Optional[int] = Field(default=None, ge=1)
    temperature: float = 0.0
    max_tokens: int = 4
    seed: Optional[int] = None


class BenchJobView(BaseModel):
    id: str
    benchmark: BenchmarkId
    status: BenchJobStatus
    model_path: Optional[str] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    examples_done: int = 0
    examples_total: int = 0
    correct: int = 0
    score: Optional[float] = None  # accuracy in [0,1]
    error: Optional[str] = None
    limit: Optional[int] = None
    temperature: float
    max_tokens: int
    # Performance / hardware metrics (populated as the job runs)
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
    # Function-calling benchmark diagnostics.
    tool_call_mismatches: dict[ToolCallMismatchCategory, int] = Field(default_factory=dict)


class DeleteHistoryRequest(BaseModel):
    ids: list[str]


class DeleteHistoryResponse(BaseModel):
    deleted: int


class BenchHistoryEntry(BaseModel):
    id: str
    benchmark: BenchmarkId
    benchmark_name: str
    status: BenchJobStatus
    model_path: Optional[str] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    duration_seconds: Optional[float] = None
    examples_done: int = 0
    examples_total: int = 0
    correct: int = 0
    score: Optional[float] = None
    error: Optional[str] = None
    limit: Optional[int] = None
    temperature: float
    max_tokens: int
    subject_count: Optional[int] = None
    # Performance / hardware metrics (None for entries written before tracking landed)
    total_tokens_generated: Optional[int] = None
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
    # Function-calling benchmark diagnostics.
    tool_call_mismatches: dict[ToolCallMismatchCategory, int] = Field(default_factory=dict)
