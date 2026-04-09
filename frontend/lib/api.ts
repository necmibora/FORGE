import type {
  SystemInfo,
  ModelList,
  LoadedModelStatus,
  LoadModelRequest,
  ChatMessage,
  ChatUsage,
  BenchmarkList,
  BenchJob,
  BenchHistoryEntry,
  RunBenchmarkRequest,
} from "./types";

export const API_BASE =
  process.env.NEXT_PUBLIC_FORGE_API ?? "http://localhost:8000";

async function j<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      "content-type": "application/json",
      ...(init?.headers ?? {}),
    },
    cache: "no-store",
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? JSON.stringify(body);
    } catch { }
    throw new Error(`${res.status} ${detail}`);
  }
  return res.json();
}

export const api = {
  system: () => j<SystemInfo>("/system"),
  models: () => j<ModelList>("/models"),
  loaded: () => j<LoadedModelStatus>("/models/loaded"),
  load: (req: LoadModelRequest) =>
    j<LoadedModelStatus>("/models/load", {
      method: "POST",
      body: JSON.stringify(req),
    }),
  unload: () =>
    j<LoadedModelStatus>("/models/unload", { method: "POST" }),
  benchmarks: () => j<BenchmarkList>("/benchmarks"),
  runBenchmark: (req: RunBenchmarkRequest) =>
    j<BenchJob>("/benchmarks/run", {
      method: "POST",
      body: JSON.stringify(req),
    }),
  benchHistory: () => j<BenchHistoryEntry[]>("/benchmarks/history"),
  benchJob: (id: string) => j<BenchJob>(`/benchmarks/jobs/${id}`),
  benchJobs: () => j<BenchJob[]>("/benchmarks/jobs"),
  cancelBenchJob: (id: string) =>
    j<BenchJob>(`/benchmarks/jobs/${id}/cancel`, { method: "POST" }),
};

/** Stream /chat SSE; calls onDelta for each token chunk and onUsage when the
 * server emits its final usage payload. Also measures wall-clock timing on
 * the client and folds it into the usage object. */
export async function streamChat(
  messages: ChatMessage[],
  opts: { max_tokens: number; temperature: number; top_p?: number },
  onDelta: (text: string) => void,
  signal: AbortSignal,
  onUsage?: (usage: ChatUsage) => void,
): Promise<void> {
  const tStart = performance.now();
  let tFirstDelta: number | null = null;
  let serverUsage: { prompt_tokens: number; completion_tokens: number } | null = null;

  const res = await fetch(`${API_BASE}/chat`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      messages,
      max_tokens: opts.max_tokens,
      temperature: opts.temperature,
      top_p: opts.top_p ?? 0.95,
      stream: true,
    }),
    signal,
  });
  if (!res.ok || !res.body) {
    let detail = res.statusText;
    try {
      const b = await res.json();
      detail = b.detail ?? detail;
    } catch { }
    throw new Error(`${res.status} ${detail}`);
  }

  const finalize = () => {
    if (!onUsage || !serverUsage) return;
    const tEnd = performance.now();
    const total_ms = tEnd - tStart;
    const generation_ms = tFirstDelta != null ? tEnd - tFirstDelta : total_ms;
    const sec = Math.max(generation_ms / 1000, 0.001);
    onUsage({
      prompt_tokens: serverUsage.prompt_tokens,
      completion_tokens: serverUsage.completion_tokens,
      total_ms,
      generation_ms,
      throughput: serverUsage.completion_tokens / sec,
    });
  };

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    // SSE events separated by \n\n
    let idx: number;
    while ((idx = buffer.indexOf("\n\n")) >= 0) {
      const raw = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);
      for (const line of raw.split("\n")) {
        if (!line.startsWith("data:")) continue;
        const data = line.slice(5).trim();
        if (!data) continue;
        if (data === "[DONE]") {
          finalize();
          return;
        }
        try {
          const obj = JSON.parse(data);
          if (typeof obj.delta === "string") {
            if (tFirstDelta == null) tFirstDelta = performance.now();
            onDelta(obj.delta);
          } else if (obj.usage) {
            serverUsage = {
              prompt_tokens: obj.usage.prompt_tokens ?? 0,
              completion_tokens: obj.usage.completion_tokens ?? 0,
            };
          } else if (obj.error) {
            throw new Error(obj.error);
          }
        } catch (e) {
          // swallow JSON parse errors on keepalives
        }
      }
    }
  }
  finalize();
}
