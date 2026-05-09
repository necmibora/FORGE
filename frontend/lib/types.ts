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
  inference_available: boolean;
  inference_message: string | null;
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

// ── Benchmarks ────────────────────────────────────────────────────────────

export type BenchmarkId = "arc_easy" | "mmlu" | "bfcl_simple";

export type BenchmarkInfo = {
  id: BenchmarkId;
  name: string;
  description: string;
  total_examples: number;
  kind: "mcq" | "generation" | "function_calling";
  subject_count: number | null;
};

export type BenchmarkList = { benchmarks: BenchmarkInfo[] };

export type BenchJobStatus =
  | "queued"
  | "running"
  | "completed"
  | "failed"
  | "cancelled";

export type BenchPerfMetrics = {
  total_tokens_generated?: number | null;
  avg_throughput_tok_s?: number | null;
  model_size_bytes?: number | null;
  model_quantization?: string | null;
  gpu_name?: string | null;
  gpu_memory_total_mb?: number | null;
  vram_used_peak_mb?: number | null;
  vram_used_avg_mb?: number | null;
  gpu_temp_avg_c?: number | null;
  gpu_temp_peak_c?: number | null;
  gpu_util_avg_pct?: number | null;
};

export type BenchJob = {
  id: string;
  benchmark: BenchmarkId;
  status: BenchJobStatus;
  model_path: string | null;
  started_at: number | null;
  finished_at: number | null;
  examples_done: number;
  examples_total: number;
  correct: number;
  score: number | null;
  error: string | null;
  limit: number | null;
  temperature: number;
  max_tokens: number;
} & BenchPerfMetrics;

export type BenchHistoryEntry = {
  id: string;
  benchmark: BenchmarkId;
  benchmark_name: string;
  status: BenchJobStatus;
  model_path: string | null;
  started_at: number | null;
  finished_at: number | null;
  duration_seconds: number | null;
  examples_done: number;
  examples_total: number;
  correct: number;
  score: number | null;
  error: string | null;
  limit: number | null;
  temperature: number;
  max_tokens: number;
  subject_count: number | null;
} & BenchPerfMetrics;

export type RunBenchmarkRequest = {
  benchmark: BenchmarkId;
  limit?: number | null;
  temperature?: number;
  max_tokens?: number;
  seed?: number | null;
};
