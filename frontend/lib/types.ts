export type GPUInfo = {
  index: number;
  name: string;
  total_memory_mb: number;
  free_memory_mb: number;
  used_memory_mb: number;
  utilization_pct: number;
  temperature_c: number | null;
};

export type SystemInfo = {
  hostname: string;
  slurm_job_id: string | null;
  slurm_node: string | null;
  cpu_count: number;
  ram_total_mb: number;
  ram_available_mb: number;
  gpus: GPUInfo[];
};

export type ModelEntry = {
  name: string;
  path: string;
  kind: "base" | "quant";
  size_bytes: number;
  quantization: string | null;
};

export type ModelList = { models: ModelEntry[] };

export type LoadedModelStatus = {
  loaded: boolean;
  path: string | null;
  quantization: string | null;
};

export type LoadModelRequest = {
  path: string;
  dtype?: string;
  max_model_len?: number | null;
  gpu_memory_utilization?: number;
  quantization?: string | null;
};

export type ChatMessage = {
  role: "system" | "user" | "assistant";
  content: string;
};

export type ChatUsage = {
  prompt_tokens: number;
  completion_tokens: number;
  /** Wall-clock from request start to stream end (ms). */
  total_ms: number;
  /** From first delta to stream end (ms); excludes TTFT. */
  generation_ms: number;
  /** completion_tokens / (generation_ms / 1000). */
  throughput: number;
};
