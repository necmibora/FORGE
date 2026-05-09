"use client";

import { Fragment, Suspense, useMemo } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { useQuery } from "@tanstack/react-query";

import { api } from "@/lib/api";
import type { BenchHistoryEntry } from "@/lib/types";

function modelLabel(path: string | null | undefined) {
  if (!path) return "—";
  const normalized = path.replace(/\\/g, "/");
  return normalized.split("/").filter(Boolean).pop() ?? path;
}

function fmtPercent(v: number | null | undefined) {
  return v == null ? "—" : `${(v * 100).toFixed(1)}%`;
}
function fmtNum(v: number | null | undefined, digits = 1) {
  return v == null ? "—" : v.toFixed(digits);
}
function fmtInt(v: number | null | undefined) {
  return v == null ? "—" : Math.round(v).toLocaleString();
}
function fmtSeconds(v: number | null | undefined) {
  if (v == null) return "—";
  if (v < 60) return `${v.toFixed(1)} s`;
  const m = Math.floor(v / 60);
  const s = Math.round(v - m * 60);
  return `${m}m ${s}s`;
}
function fmtBytes(v: number | null | undefined) {
  if (!v) return "—";
  const gb = v / (1024 ** 3);
  if (gb >= 1) return `${gb.toFixed(2)} GB`;
  const mb = v / (1024 ** 2);
  return `${mb.toFixed(1)} MB`;
}
function fmtMb(v: number | null | undefined) {
  if (v == null) return "—";
  if (v >= 1024) return `${(v / 1024).toFixed(2)} GB`;
  return `${Math.round(v).toLocaleString()} MB`;
}
function fmtTemp(v: number | null | undefined) {
  return v == null ? "—" : `${typeof v === "number" ? v.toFixed(1) : v}°C`;
}
function fmtTimestamp(ts: number | null | undefined) {
  if (ts == null) return "—";
  return new Date(ts * 1000).toLocaleString("en-US", {
    dateStyle: "short",
    timeStyle: "short",
  });
}
function fmtConditions(item: BenchHistoryEntry) {
  const requested = item.limit ?? item.examples_total;
  const sample = `${requested}/${item.examples_total || "?"} ex`;
  return `${sample} • temp ${item.temperature} • max ${item.max_tokens}`;
}

type RowDef = {
  key: string;
  label: string;
  group: "Run" | "Score" | "Performance" | "Model" | "GPU";
  render: (item: BenchHistoryEntry) => string;
};

const ROWS: RowDef[] = [
  { key: "benchmark", label: "Benchmark", group: "Run", render: (i) => i.benchmark_name },
  { key: "status", label: "Status", group: "Run", render: (i) => i.status },
  { key: "started", label: "Started", group: "Run", render: (i) => fmtTimestamp(i.started_at) },
  { key: "duration", label: "Duration", group: "Run", render: (i) => fmtSeconds(i.duration_seconds) },
  { key: "conditions", label: "Conditions", group: "Run", render: (i) => fmtConditions(i) },

  { key: "score", label: "Accuracy", group: "Score", render: (i) => fmtPercent(i.score) },
  { key: "correct", label: "Correct / Total", group: "Score", render: (i) => `${i.correct} / ${i.examples_done}` },
  { key: "wrong", label: "Wrong", group: "Score", render: (i) => String(Math.max(i.examples_done - i.correct, 0)) },

  { key: "tokens", label: "Total tokens", group: "Performance", render: (i) => fmtInt(i.total_tokens_generated) },
  { key: "throughput", label: "Avg tok/s", group: "Performance", render: (i) => fmtNum(i.avg_throughput_tok_s, 1) },

  { key: "model", label: "Model", group: "Model", render: (i) => modelLabel(i.model_path) },
  { key: "quant", label: "Quantization", group: "Model", render: (i) => i.model_quantization ?? "—" },
  { key: "size", label: "Size on disk", group: "Model", render: (i) => fmtBytes(i.model_size_bytes) },

  { key: "gpu", label: "GPU", group: "GPU", render: (i) => i.gpu_name ?? "—" },
  { key: "vram_total", label: "VRAM total", group: "GPU", render: (i) => fmtMb(i.gpu_memory_total_mb) },
  { key: "vram_avg", label: "VRAM avg used", group: "GPU", render: (i) => fmtMb(i.vram_used_avg_mb) },
  { key: "vram_peak", label: "VRAM peak", group: "GPU", render: (i) => fmtMb(i.vram_used_peak_mb) },
  { key: "temp_avg", label: "Avg temp", group: "GPU", render: (i) => fmtTemp(i.gpu_temp_avg_c) },
  { key: "temp_peak", label: "Peak temp", group: "GPU", render: (i) => fmtTemp(i.gpu_temp_peak_c) },
  { key: "util", label: "Avg util", group: "GPU", render: (i) => i.gpu_util_avg_pct == null ? "—" : `${i.gpu_util_avg_pct.toFixed(0)}%` },
];

export default function ComparePage() {
  return (
    <Suspense fallback={<div className="card text-sm text-forge-muted">Loading…</div>}>
      <CompareView />
    </Suspense>
  );
}

function CompareView() {
  const search = useSearchParams();
  const idsParam = search.get("ids") ?? "";
  const ids = useMemo(
    () => idsParam.split(",").map((s) => s.trim()).filter(Boolean),
    [idsParam],
  );

  const history = useQuery({
    queryKey: ["benchHistory"],
    queryFn: api.benchHistory,
  });

  const items: BenchHistoryEntry[] = useMemo(() => {
    const all = history.data ?? [];
    const byId = new Map(all.map((h) => [h.id, h]));
    return ids.map((id) => byId.get(id)).filter((x): x is BenchHistoryEntry => !!x);
  }, [history.data, ids]);

  const missing = ids.length - items.length;

  const groups = useMemo(() => {
    const order: RowDef["group"][] = ["Run", "Score", "Performance", "Model", "GPU"];
    return order.map((g) => ({ name: g, rows: ROWS.filter((r) => r.group === g) }));
  }, []);

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold">Compare benchmarks</h1>
          <p className="text-sm text-forge-muted">
            Side-by-side comparison of {items.length} run{items.length === 1 ? "" : "s"}.
            {missing > 0 && (
              <span className="ml-2 text-yellow-400">
                ({missing} not found in history)
              </span>
            )}
          </p>
        </div>
        <Link href="/benchmark" className="btn">
          ← Back to benchmark
        </Link>
      </div>

      {history.isLoading && (
        <div className="card text-sm text-forge-muted">Loading history…</div>
      )}

      {!history.isLoading && items.length === 0 && (
        <div className="card text-sm text-forge-muted">
          No matching history entries. Pick runs from the benchmark page first.
        </div>
      )}

      {items.length > 0 && (
        <div className="card overflow-x-auto p-0">
          <table className="w-full border-collapse text-sm">
            <thead>
              <tr className="border-b border-forge-border bg-[#11151b]">
                <th className="sticky left-0 z-10 bg-[#11151b] px-4 py-3 text-left text-xs font-medium uppercase tracking-wider text-forge-muted">
                  Metric
                </th>
                {items.map((it, idx) => (
                  <th
                    key={it.id}
                    className="px-4 py-3 text-left text-xs font-medium uppercase tracking-wider text-forge-muted"
                  >
                    Run {idx + 1}
                    <div className="mt-0.5 truncate font-mono text-[10px] normal-case tracking-normal text-forge-muted/70">
                      {it.id.slice(0, 8)}
                    </div>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {groups.map((g) => (
                <Fragment key={g.name}>
                  <tr className="border-b border-forge-border/60">
                    <td
                      colSpan={items.length + 1}
                      className="bg-[#0e1217] px-4 py-2 text-[11px] uppercase tracking-wider text-forge-muted"
                    >
                      {g.name}
                    </td>
                  </tr>
                  {g.rows.map((row) => (
                    <tr
                      key={row.key}
                      className="border-b border-forge-border/40 last:border-b-0"
                    >
                      <td className="sticky left-0 z-10 bg-forge-panel px-4 py-2 text-forge-muted">
                        {row.label}
                      </td>
                      {items.map((it) => (
                        <td
                          key={it.id}
                          className="break-words px-4 py-2 font-mono text-forge-text"
                        >
                          {row.render(it)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </Fragment>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
