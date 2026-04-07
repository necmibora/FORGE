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


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    stream: bool = True
