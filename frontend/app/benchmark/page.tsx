"use client";

import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { api } from "@/lib/api";
import type {
  BenchHistoryEntry,
  BenchJob,
  BenchmarkId,
  BenchmarkInfo,
} from "@/lib/types";

const STATUS_LABELS: Record<BenchJob["status"], string> = {
  queued: "Sirada",
  running: "Calisiyor",
  completed: "Tamamlandi",
  failed: "Hata",
  cancelled: "Iptal edildi",
};

function formatPercent(score: number | null) {
  return score == null ? "-" : `${(score * 100).toFixed(1)}%`;
}

function formatDuration(seconds: number | null) {
  return seconds == null ? "-" : `${seconds.toFixed(1)} s`;
}

function formatTimestamp(ts: number | null) {
  if (ts == null) return "-";
  return new Date(ts * 1000).toLocaleString("tr-TR", {
    dateStyle: "short",
    timeStyle: "short",
  });
}

function modelLabel(path: string | null) {
  if (!path) return "Model yok";
  const normalized = path.replace(/\\/g, "/");
  return normalized.split("/").filter(Boolean).pop() ?? path;
}

function historyConditions(item: BenchHistoryEntry) {
  const requested = item.limit ?? item.examples_total;
  const sample = `${requested}/${item.examples_total || "?"} ex`;
  const subjectText = item.subject_count ? ` - ${item.subject_count} subject` : "";
  return `${sample}${subjectText} - temp ${item.temperature} - max ${item.max_tokens}`;
}

export default function BenchmarkPage() {
  const qc = useQueryClient();

  const benchmarks = useQuery({
    queryKey: ["benchmarks"],
    queryFn: api.benchmarks,
  });
  const loaded = useQuery({
    queryKey: ["loaded"],
    queryFn: api.loaded,
    refetchInterval: 5000,
  });
  const history = useQuery({
    queryKey: ["benchHistory"],
    queryFn: api.benchHistory,
    refetchInterval: 5000,
  });

  const [selectedId, setSelectedId] = useState<BenchmarkId>("arc_easy");
  const [limit, setLimit] = useState<number>(50);
  const [useFull, setUseFull] = useState(false);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [historySyncedJobId, setHistorySyncedJobId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const selected: BenchmarkInfo | undefined = benchmarks.data?.benchmarks.find(
    (b) => b.id === selectedId,
  );

  useEffect(() => {
    if (!selected) return;
    if (useFull) return;
    if (limit > selected.total_examples && selected.total_examples > 0) {
      setLimit(selected.total_examples);
    }
  }, [limit, selected, useFull]);

  const job = useQuery({
    queryKey: ["benchJob", activeJobId],
    queryFn: () => api.benchJob(activeJobId as string),
    enabled: !!activeJobId,
    refetchInterval: (q) => {
      const data = q.state.data as BenchJob | undefined;
      if (!data) return 1000;
      if (data.status === "running" || data.status === "queued") return 1000;
      return false;
    },
  });

  useEffect(() => {
    const data = job.data;
    if (!data) return;
    const isTerminal =
      data.status === "completed" ||
      data.status === "failed" ||
      data.status === "cancelled";
    if (!isTerminal) return;
    if (historySyncedJobId === data.id) return;
    setHistorySyncedJobId(data.id);
    qc.invalidateQueries({ queryKey: ["benchHistory"] });
  }, [historySyncedJobId, job.data, qc]);

  const run = useMutation({
    mutationFn: () => {
      setError(null);
      return api.runBenchmark({
        benchmark: selectedId,
        limit: useFull ? null : limit,
        temperature: 0,
        max_tokens: 4,
      });
    },
    onSuccess: (j) => {
      setActiveJobId(j.id);
      setHistorySyncedJobId(null);
      qc.invalidateQueries({ queryKey: ["benchJob", j.id] });
    },
    onError: (e: Error) => setError(e.message),
  });

  const cancel = useMutation({
    mutationFn: () => api.cancelBenchJob(activeJobId as string),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["benchJob", activeJobId] });
      qc.invalidateQueries({ queryKey: ["benchHistory"] });
    },
    onError: (e: Error) => setError(e.message),
  });

  const j = job.data;
  const activeBenchmark = j
    ? benchmarks.data?.benchmarks.find((b) => b.id === j.benchmark)
    : undefined;
  const isLive = j?.status === "running" || j?.status === "queued";
  const progressPct =
    j && j.examples_total > 0
      ? Math.round((j.examples_done / j.examples_total) * 100)
      : 0;
  const elapsed =
    j?.started_at != null ? (j.finished_at ?? Date.now() / 1000) - j.started_at : null;

  return (
    <div className="grid gap-4 md:grid-cols-[1fr_320px]">
      <div className="space-y-4">
        <div>
          <h1 className="text-xl font-semibold">Benchmark</h1>
          <p className="text-sm text-forge-muted">
            Yuklu model uzerinde benchmark kos ve gecmis sonuclari incele.
          </p>
        </div>

        {!loaded.data?.loaded && (
          <div className="card border-yellow-900 text-sm text-yellow-400">
            Yuklu model yok. Once Models sayfasindan bir model yukleyin.
          </div>
        )}

        {error && (
          <div className="card border-red-900 text-sm text-red-400">{error}</div>
        )}

        <div className="space-y-2">
          <div className="label">Mevcut benchmarklar</div>
          {benchmarks.isLoading && (
            <div className="text-sm text-forge-muted">Yukleniyor...</div>
          )}
          {benchmarks.data?.benchmarks.map((b) => {
            const active = b.id === selectedId;
            return (
              <button
                key={b.id}
                onClick={() => setSelectedId(b.id)}
                className={`card w-full text-left transition ${
                  active ? "border-forge-accent" : "hover:border-forge-text/30"
                }`}
              >
                <div className="flex flex-wrap items-center gap-2">
                  <span className="font-medium">{b.name}</span>
                  <span className="badge">{b.kind}</span>
                  <span className="badge">{b.total_examples} ex</span>
                  {b.subject_count != null && (
                    <span className="badge">{b.subject_count} subject</span>
                  )}
                </div>
                <div className="mt-1 text-xs text-forge-muted">{b.description}</div>
              </button>
            );
          })}
        </div>

        {j ? (
          <div className="card space-y-3">
            <div className="flex items-center justify-between gap-3">
              <div className="flex flex-wrap items-center gap-2">
                <span className="font-medium">{activeBenchmark?.name ?? j.benchmark}</span>
                <span
                  className={`badge ${
                    j.status === "completed"
                      ? "border-forge-accent text-forge-accent"
                      : j.status === "failed"
                        ? "border-red-700 text-red-400"
                        : ""
                  }`}
                >
                  {STATUS_LABELS[j.status]}
                </span>
              </div>
              {isLive && (
                <button
                  className="btn"
                  onClick={() => cancel.mutate()}
                  disabled={cancel.isPending}
                >
                  {cancel.isPending ? "Iptal ediliyor..." : "Iptal"}
                </button>
              )}
            </div>

            <div>
              <div className="flex justify-between text-xs text-forge-muted">
                <span>
                  {j.examples_done} / {j.examples_total}
                </span>
                <span>{progressPct}%</span>
              </div>
              <div className="mt-1 h-2 overflow-hidden rounded bg-forge-border">
                <div
                  className="h-full bg-forge-accent transition-all"
                  style={{ width: `${progressPct}%` }}
                />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3 text-xs font-mono">
              <div>
                <div className="text-forge-muted">Dogru</div>
                <div className="text-base">{j.correct}</div>
              </div>
              <div>
                <div className="text-forge-muted">Skor</div>
                <div className="text-base">{formatPercent(j.score)}</div>
              </div>
              <div>
                <div className="text-forge-muted">Gecen sure</div>
                <div>{formatDuration(elapsed)}</div>
              </div>
              <div>
                <div className="text-forge-muted">Orneklem</div>
                <div>{j.limit ?? "tumu"}</div>
              </div>
              <div>
                <div className="text-forge-muted">Temperature</div>
                <div>{j.temperature}</div>
              </div>
              <div>
                <div className="text-forge-muted">Max tokens</div>
                <div>{j.max_tokens}</div>
              </div>
            </div>

            {j.model_path && (
              <div className="break-all font-mono text-[11px] text-forge-muted">
                {j.model_path}
              </div>
            )}

            {j.error && (
              <div className="rounded border border-red-900 p-2 text-xs text-red-400">
                {j.error}
              </div>
            )}
          </div>
        ) : (
          <div className="card text-sm text-forge-muted">
            Henuz calistirilan bir benchmark yok.
          </div>
        )}

        <section className="space-y-3">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-lg font-semibold">History</h2>
              <p className="text-sm text-forge-muted">
                Son benchmark kosulari ve kosul ozeti.
              </p>
            </div>
            {history.isFetching && (
              <span className="text-xs text-forge-muted">Guncelleniyor...</span>
            )}
          </div>

          {history.isLoading && (
            <div className="card text-sm text-forge-muted">History yukleniyor...</div>
          )}

          {history.data?.length === 0 && (
            <div className="card text-sm text-forge-muted">
              Henuz kaydedilmis benchmark sonucu yok.
            </div>
          )}

          <div className="space-y-3">
            {history.data?.map((item) => (
              <div key={item.id} className="card space-y-3">
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div className="space-y-1">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="font-medium">{item.benchmark_name}</span>
                      <span className="badge">{STATUS_LABELS[item.status]}</span>
                      <span className="badge">{formatPercent(item.score)}</span>
                    </div>
                    <div className="text-xs text-forge-muted">
                      {historyConditions(item)}
                    </div>
                  </div>
                  <div className="text-right text-xs text-forge-muted">
                    <div>{formatTimestamp(item.finished_at ?? item.started_at)}</div>
                    <div>{formatDuration(item.duration_seconds)}</div>
                  </div>
                </div>

                <div className="grid gap-3 text-xs md:grid-cols-4">
                  <div>
                    <div className="text-forge-muted">Model</div>
                    <div className="font-mono text-forge-text">
                      {modelLabel(item.model_path)}
                    </div>
                  </div>
                  <div>
                    <div className="text-forge-muted">Dogru / Toplam</div>
                    <div className="font-mono text-forge-text">
                      {item.correct} / {item.examples_done}
                    </div>
                  </div>
                  <div>
                    <div className="text-forge-muted">Benchmark</div>
                    <div className="font-mono text-forge-text">{item.benchmark}</div>
                  </div>
                  <div>
                    <div className="text-forge-muted">Job ID</div>
                    <div className="truncate font-mono text-forge-text">{item.id}</div>
                  </div>
                </div>

                {item.error && (
                  <div className="rounded border border-red-900 p-2 text-xs text-red-400">
                    {item.error}
                  </div>
                )}
              </div>
            ))}
          </div>
        </section>
      </div>

      <aside className="card self-start space-y-4">
        <div>
          <label className="label">Orneklem boyutu</label>
          <div className="flex items-center justify-between text-xs">
            <span className="text-forge-muted">Limit</span>
            <span className="font-mono">{useFull ? "tumu" : limit}</span>
          </div>
          <input
            type="range"
            min={10}
            max={Math.max(selected?.total_examples ?? 100, 100)}
            step={10}
            value={limit}
            disabled={useFull}
            onChange={(e) => setLimit(parseInt(e.target.value, 10))}
            className="w-full accent-[#e30613] disabled:opacity-50"
          />
          <label className="mt-2 flex cursor-pointer items-center gap-2 text-xs">
            <input
              type="checkbox"
              checked={useFull}
              onChange={(e) => setUseFull(e.target.checked)}
              className="accent-[#e30613]"
            />
            <span className="text-forge-muted">
              Tum dataseti kos
              {selected?.total_examples ? ` (${selected.total_examples} ex)` : ""}
            </span>
          </label>
          {selected?.subject_count != null && (
            <div className="mt-2 text-[11px] text-forge-muted">
              Bu benchmark {selected.subject_count} subject icerir.
            </div>
          )}
        </div>

        <button
          className="btn btn-accent w-full justify-center"
          disabled={!loaded.data?.loaded || run.isPending || isLive}
          onClick={() => run.mutate()}
        >
          {run.isPending ? "Baslatiliyor..." : isLive ? "Calisiyor..." : "Calistir"}
        </button>

        <div className="border-t border-forge-border pt-3 text-[11px] leading-relaxed text-forge-muted">
          Benchmark calisirken chat gecici olarak duraklatilir. vLLM engine tek is
          tutulur ki performans metrikleri bozulmasin.
        </div>
      </aside>
    </div>
  );
}
