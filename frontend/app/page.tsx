"use client";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";

function mb(m: number) {
  return `${(m / 1024).toFixed(1)} GB`;
}

export default function Dashboard() {
  const sys = useQuery({
    queryKey: ["system"],
    queryFn: api.system,
    refetchInterval: 3000,
  });
  const loaded = useQuery({
    queryKey: ["loaded"],
    queryFn: api.loaded,
    refetchInterval: 3000,
  });

  if (sys.isLoading) return <p className="text-forge-muted">Yükleniyor…</p>;
  if (sys.isError)
    return (
      <div className="card border-red-900 text-red-400">
        Backend'e ulaşılamadı: {(sys.error as Error).message}
      </div>
    );
  const s = sys.data!;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold">Dashboard</h1>
        <p className="text-forge-muted text-sm">
          Compute node durumu ve yüklü model.
        </p>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <div className="card">
          <div className="label">Host</div>
          <div className="font-mono text-sm">{s.hostname}</div>
        </div>
        <div className="card">
          <div className="label">Slurm Job</div>
          <div className="font-mono text-sm">
            {s.slurm_job_id ?? "—"} {s.slurm_node ? `@${s.slurm_node}` : ""}
          </div>
        </div>
        <div className="card">
          <div className="label">CPU</div>
          <div className="font-mono text-sm">{s.cpu_count} çekirdek</div>
        </div>
        <div className="card">
          <div className="label">RAM</div>
          <div className="font-mono text-sm">
            {mb(s.ram_total_mb - s.ram_available_mb)} / {mb(s.ram_total_mb)}
          </div>
        </div>
      </div>

      <div>
        <h2 className="text-sm uppercase tracking-wider text-forge-muted mb-2">
          GPUs
        </h2>
        {s.gpus.length === 0 ? (
          <div className="card text-forge-muted text-sm">GPU bulunamadı.</div>
        ) : (
          <div className="grid gap-3 md:grid-cols-2">
            {s.gpus.map((g) => {
              const usedPct = (g.used_memory_mb / g.total_memory_mb) * 100;
              return (
                <div key={g.index} className="card">
                  <div className="flex justify-between items-baseline">
                    <div className="font-medium">
                      GPU {g.index} · {g.name}
                    </div>
                    <div className="text-xs text-forge-muted">
                      {g.temperature_c ?? "—"}°C
                    </div>
                  </div>
                  <div className="mt-3 text-xs text-forge-muted flex justify-between">
                    <span>VRAM</span>
                    <span>
                      {mb(g.used_memory_mb)} / {mb(g.total_memory_mb)}
                    </span>
                  </div>
                  <div className="h-1.5 bg-forge-border rounded mt-1 overflow-hidden">
                    <div
                      className="h-full bg-forge-accent"
                      style={{ width: `${usedPct}%` }}
                    />
                  </div>
                  <div className="mt-3 text-xs text-forge-muted flex justify-between">
                    <span>Util</span>
                    <span>{g.utilization_pct}%</span>
                  </div>
                  <div className="h-1.5 bg-forge-border rounded mt-1 overflow-hidden">
                    <div
                      className="h-full bg-forge-text/70"
                      style={{ width: `${g.utilization_pct}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      <div>
        <h2 className="text-sm uppercase tracking-wider text-forge-muted mb-2">
          Loaded Model
        </h2>
        <div className="card">
          {loaded.data?.loaded ? (
            <div className="space-y-1">
              <div className="font-mono text-sm">{loaded.data.path}</div>
              {loaded.data.quantization && (
                <span className="badge">{loaded.data.quantization}</span>
              )}
            </div>
          ) : (
            <div className="text-forge-muted text-sm">
              Yüklü model yok. Models sayfasından bir model yükleyin.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
