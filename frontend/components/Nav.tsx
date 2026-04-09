"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import { api, API_BASE } from "@/lib/api";

const links = [
  { href: "/", label: "Dashboard" },
  { href: "/models", label: "Models" },
  { href: "/chat", label: "Chat" },
];

export function Nav() {
  const pathname = usePathname();
  const { data: loaded } = useQuery({
    queryKey: ["loaded"],
    queryFn: api.loaded,
    refetchInterval: 5000,
  });

  return (
    <header className="border-b border-forge-border bg-forge-panel">
      <div className="max-w-6xl mx-auto px-6 h-14 flex items-center gap-6">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-forge-accent" />
          <span className="font-semibold tracking-wide">FORGE</span>
          <span className="text-forge-muted text-xs">v0.1</span>
        </div>
        <nav className="flex gap-1">
          {links.map((l) => {
            const active = pathname === l.href;
            return (
              <Link
                key={l.href}
                href={l.href}
                className={`px-3 py-1.5 text-sm rounded-md ${
                  active
                    ? "bg-[#1a1f27] text-forge-text"
                    : "text-forge-muted hover:text-forge-text"
                }`}
              >
                {l.label}
              </Link>
            );
          })}
        </nav>
        <div className="ml-auto flex items-center gap-3 text-xs text-forge-muted">
          <span>{API_BASE}</span>
          {loaded?.loaded ? (
            <span className="badge border-forge-accent/60 text-forge-accent">
              {loaded.path?.split("/").pop()}
            </span>
          ) : (
            <span className="badge">no model</span>
          )}
        </div>
      </div>
    </header>
  );
}
