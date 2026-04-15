"use client";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { useState } from "react";
import type { ModelEntry } from "@/lib/types";

function gb(bytes: number) {
  return `${(bytes / 1024 ** 3).toFixed(1)} GB`;
}

export default function ModelsPage() {
  const qc = useQueryClient();
  const models = useQuery({ queryKey: ["models"], queryFn: api.models });
  const loaded = useQuery({
    queryKey: ["loaded"],
    queryFn: api.loaded,
    refetchInterval: 3000,
  });
  const [pending, setPending] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [maxModelLen, setMaxModelLen] = useState<number>(32768);
  const [gpuUtil, setGpuUtil] = useState<number>(0.9);

  const load = useMutation({
    mutationFn: (m: ModelEntry) => {
      setError(null);
      setPending(m.path);
      return api.load({
        path: m.path,
        quantization: m.quantization ?? null,
        max_model_len: maxModelLen,
        gpu_memory_utilization: gpuUtil,
      });
    },
    onSettled: () => {
      setPending(null);
      qc.invalidateQueries({ queryKey: ["loaded"] });
    },
    onError: (e: Error) => setError(e.message),
  });

  const unload = useMutation({
    mutationFn: () => api.unload(),
    onSettled: () => qc.invalidateQueries({ queryKey: ["loaded"] }),
    onError: (e: Error) => setError(e.message),
  });

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-semibold">Models</h1>
          <p className="text-forge-muted text-sm">
            Base and quantized models on the cluster.
          </p>
        </div>
        {loaded.data?.loaded && (
          <button
            className="btn"
            disabled={unload.isPending}
            onClick={() => unload.mutate()}
          >
            {unload.isPending ? "Unloading…" : "Unload"}
          </button>
        )}
      </div>

      {error && (
        <div className="card border-red-900 text-red-400 text-sm">{error}</div>
      )}

      <div className="card flex flex-wrap items-center gap-4 text-sm">
        <label className="flex items-center gap-2">
          <span className="text-forge-muted">Max Context Length</span>
          <input
            type="number"
            className="input w-28"
            value={maxModelLen}
            min={1024}
            step={1024}
            onChange={(e) => setMaxModelLen(Number(e.target.value))}
          />
        </label>
        <label className="flex items-center gap-2">
          <span className="text-forge-muted">GPU Memory %</span>
          <input
            type="number"
            className="input w-20"
            value={gpuUtil}
            min={0.5}
            max={0.99}
            step={0.05}
            onChange={(e) => setGpuUtil(Number(e.target.value))}
          />
        </label>
      </div>

      {models.isLoading && <p className="text-forge-muted">Loading…</p>}
      {models.isError && (
        <div className="card border-red-900 text-red-400">
          {(models.error as Error).message}
        </div>
      )}

      <div className="grid gap-3">
        {models.data?.models.map((m) => {
          const isLoaded = loaded.data?.loaded && loaded.data.path === m.path;
          const isBusy = load.isPending && pending === m.path;
          return (
            <div
              key={m.path}
              className={`card flex items-center gap-4 ${
                isLoaded ? "border-forge-accent" : ""
              }`}
            >
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <span className="font-medium truncate">{m.name}</span>
                  <span className="badge">{m.kind}</span>
                  {m.quantization && <span className="badge">{m.quantization}</span>}
                  {isLoaded && (
                    <span className="badge border-forge-accent text-forge-accent">
                      LOADED
                    </span>
                  )}
                </div>
                <div className="text-xs text-forge-muted font-mono truncate">
                  {m.path}
                </div>
              </div>
              <div className="text-xs text-forge-muted w-20 text-right">
                {gb(m.size_bytes)}
              </div>
              <button
                className="btn btn-accent"
                disabled={isBusy || isLoaded}
                onClick={() => load.mutate(m)}
              >
                {isBusy ? "Loading…" : isLoaded ? "Active" : "Load"}
              </button>
            </div>
          );
        })}
        {models.data?.models.length === 0 && (
          <div className="card text-forge-muted text-sm">
            No models found. Check FORGE_MODELS_DIR / FORGE_QUANT_DIR env vars.
          </div>
        )}
      </div>
    </div>
  );
}
