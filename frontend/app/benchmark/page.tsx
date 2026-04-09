"use client";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { api } from "@/lib/api";
import type { BenchJob, BenchmarkInfo, BenchmarkId } from "@/lib/types";

const STATUS_LABELS: Record<BenchJob["status"], string> = {
  queued: "Sırada",
  running: "Çalışıyor",
  completed: "Tamamlandı",
  failed: "Hata",
  cancelled: "İptal edildi",
};

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

  const [selectedId, setSelectedId] = useState<BenchmarkId>("arc_easy");
  const [limit, setLimit] = useState<number>(50);
  const [useFull, setUseFull] = useState(false);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

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
      qc.invalidateQueries({ queryKey: ["benchJob", j.id] });
    },
    onError: (e: Error) => setError(e.message),
  });

  const cancel = useMutation({
    mutationFn: () => api.cancelBenchJob(activeJobId as string),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["benchJob", activeJobId] }),
    onError: (e: Error) => setError(e.message),
  });

  const selected: BenchmarkInfo | undefined = benchmarks.data?.benchmarks.find(
    (b) => b.id === selectedId,
  );

  const j = job.data;
  const isLive = j?.status === "running" || j?.status === "queued";
  const progressPct =
    j && j.examples_total > 0
      ? Math.round((j.examples_done / j.examples_total) * 100)
      : 0;
  const elapsed =
    j?.started_at != null
      ? ((j.finished_at ?? Date.now() / 1000) - j.started_at)
      : null;

  return (
    <div className="grid md:grid-cols-[1fr_320px] gap-4">
      {/* Main panel */}
      <div className="space-y-4">
        <div>
          <h1 className="text-xl font-semibold">Benchmark</h1>
          <p className="text-forge-muted text-sm">
            Yüklü model üzerinde değerlendirme suite'leri çalıştır.
          </p>
        </div>

        {!loaded.data?.loaded && (
          <div className="card border-yellow-900 text-yellow-400 text-sm">
            Yüklü model yok — Models sayfasından bir model yükleyin.
          </div>
        )}

        {error && (
          <div className="card border-red-900 text-red-400 text-sm">{error}</div>
        )}

        <div className="space-y-2">
          <div className="label">Mevcut benchmark'lar</div>
          {benchmarks.isLoading && (
            <div className="text-forge-muted text-sm">Yükleniyor…</div>
          )}
          {benchmarks.data?.benchmarks.map((b) => {
            const active = b.id === selectedId;
            return (
              <button
                key={b.id}
                onClick={() => setSelectedId(b.id)}
                className={`card w-full text-left transition ${
                  active
                    ? "border-forge-accent"
                    : "hover:border-forge-text/30"
                }`}
              >
                <div className="flex items-center gap-2">
                  <span className="font-medium">{b.name}</span>
                  <span className="badge">{b.kind}</span>
                  <span className="badge">{b.total_examples} ex</span>
                </div>
                <div className="text-xs text-forge-muted mt-1">
                  {b.description}
                </div>
              </button>
            );
          })}
        </div>

        {/* Job status */}
        {j ? (
          <div className="card space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="font-medium">{selected?.name}</span>
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
                  {cancel.isPending ? "İptal ediliyor…" : "İptal"}
                </button>
              )}
            </div>

            {/* Progress bar */}
            <div>
              <div className="flex justify-between text-xs text-forge-muted">
                <span>
                  {j.examples_done} / {j.examples_total}
                </span>
                <span>{progressPct}%</span>
              </div>
              <div className="h-2 bg-forge-border rounded mt-1 overflow-hidden">
                <div
                  className="h-full bg-forge-accent transition-all"
                  style={{ width: `${progressPct}%` }}
                />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3 text-xs font-mono">
              <div>
                <div className="text-forge-muted">Doğru</div>
                <div className="text-base">{j.correct}</div>
              </div>
              <div>
                <div className="text-forge-muted">Skor</div>
                <div className="text-base">
                  {j.score != null ? `${(j.score * 100).toFixed(1)}%` : "—"}
                </div>
              </div>
              <div>
                <div className="text-forge-muted">Geçen süre</div>
                <div>{elapsed != null ? `${elapsed.toFixed(1)} s` : "—"}</div>
              </div>
              <div>
                <div className="text-forge-muted">Limit</div>
                <div>{j.limit ?? "tümü"}</div>
              </div>
            </div>

            {j.model_path && (
              <div className="text-[11px] text-forge-muted font-mono break-all">
                {j.model_path}
              </div>
            )}

            {j.error && (
              <div className="text-xs text-red-400 border border-red-900 rounded p-2">
                {j.error}
              </div>
            )}
          </div>
        ) : (
          <div className="card text-forge-muted text-sm">
            Henüz çalıştırılan bir benchmark yok.
          </div>
        )}
      </div>

      {/* Sidebar */}
      <aside className="card space-y-4 self-start">
        <div>
          <label className="label">Örneklem boyutu</label>
          <div className="flex items-center justify-between text-xs">
            <span className="text-forge-muted">Limit</span>
            <span className="font-mono">{useFull ? "tümü" : limit}</span>
          </div>
          <input
            type="range"
            min={10}
            max={Math.max(selected?.total_examples ?? 100, 100)}
            step={10}
            value={limit}
            disabled={useFull}
            onChange={(e) => setLimit(parseInt(e.target.value))}
            className="w-full accent-[#e30613] disabled:opacity-50"
          />
          <label className="flex items-center gap-2 mt-2 text-xs cursor-pointer">
            <input
              type="checkbox"
              checked={useFull}
              onChange={(e) => setUseFull(e.target.checked)}
              className="accent-[#e30613]"
            />
            <span className="text-forge-muted">
              Tüm dataset'i çalıştır ({selected?.total_examples ?? "?"} ex)
            </span>
          </label>
        </div>

        <button
          className="btn btn-accent w-full justify-center"
          disabled={!loaded.data?.loaded || run.isPending || isLive}
          onClick={() => run.mutate()}
        >
          {run.isPending
            ? "Başlatılıyor…"
            : isLive
              ? "Çalışıyor…"
              : "Çalıştır"}
        </button>

        <div className="text-[11px] text-forge-muted leading-relaxed border-t border-forge-border pt-3">
          Benchmark çalışırken chat geçici olarak duraklatılır (409). vLLM
          engine tek-iş tutulur ki perf metrikleri saf olsun.
        </div>
      </aside>
    </div>
  );
}
