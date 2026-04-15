"use client";
import { useQuery } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import { api, streamChat } from "@/lib/api";
import type { ChatMessage, ChatUsage } from "@/lib/types";

const LS_KEY = "forge.chat.history";
const LS_SYS = "forge.chat.system";

export default function ChatPage() {
  const loaded = useQuery({
    queryKey: ["loaded"],
    queryFn: api.loaded,
    refetchInterval: 5000,
  });

  const [system, setSystem] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [temperature, setTemperature] = useState(0.7);
  const [maxTokens, setMaxTokens] = useState(512);
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastUsage, setLastUsage] = useState<ChatUsage | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  // Hydrate from localStorage
  useEffect(() => {
    try {
      const h = localStorage.getItem(LS_KEY);
      if (h) setMessages(JSON.parse(h));
      const s = localStorage.getItem(LS_SYS);
      if (s) setSystem(s);
    } catch {}
  }, []);
  useEffect(() => {
    localStorage.setItem(LS_KEY, JSON.stringify(messages));
  }, [messages]);
  useEffect(() => {
    localStorage.setItem(LS_SYS, system);
  }, [system]);
  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight });
  }, [messages, streaming]);

  const canChat = !!loaded.data?.loaded && !streaming;

  async function send() {
    if (!input.trim() || !canChat) return;
    setError(null);
    const userMsg: ChatMessage = { role: "user", content: input.trim() };
    const next = [...messages, userMsg];
    setMessages(next);
    setInput("");

    const payload: ChatMessage[] = system.trim()
      ? [{ role: "system", content: system.trim() }, ...next]
      : next;

    setMessages((m) => [...m, { role: "assistant", content: "" }]);
    setStreaming(true);
    const ac = new AbortController();
    abortRef.current = ac;

    try {
      await streamChat(
        payload,
        { max_tokens: maxTokens, temperature },
        (delta) => {
          setMessages((m) => {
            const copy = [...m];
            const last = copy[copy.length - 1];
            copy[copy.length - 1] = { ...last, content: last.content + delta };
            return copy;
          });
        },
        ac.signal,
        (usage) => setLastUsage(usage),
      );
    } catch (e: unknown) {
      if ((e as Error).name !== "AbortError") {
        setError((e as Error).message);
      }
    } finally {
      setStreaming(false);
      abortRef.current = null;
    }
  }

  function stop() {
    abortRef.current?.abort();
  }

  function clearHistory() {
    setMessages([]);
  }

  return (
    <div className="grid md:grid-cols-[1fr_280px] gap-4 h-[calc(100vh-8rem)]">
      <div className="flex flex-col card p-0 overflow-hidden">
        <div className="px-4 py-3 border-b border-forge-border flex items-center justify-between">
          <div className="text-sm">
            {loaded.data?.loaded ? (
              <span className="font-mono">
                {loaded.data.path?.split("/").pop()}
              </span>
            ) : (
              <span className="text-forge-muted">
                No model loaded — load one from the Models page.
              </span>
            )}
          </div>
          <button className="btn" onClick={clearHistory} disabled={streaming}>
            Clear
          </button>
        </div>

        <div ref={scrollRef} className="flex-1 overflow-y-auto p-4 space-y-4">
          {messages.length === 0 && (
            <div className="text-forge-muted text-sm">
              Type below to start a conversation.
            </div>
          )}
          {messages.map((m, i) => (
            <div key={i} className="space-y-1">
              <div className="text-[10px] uppercase tracking-wider text-forge-muted">
                {m.role}
              </div>
              <div
                className={`whitespace-pre-wrap text-sm leading-relaxed ${
                  m.role === "user" ? "text-forge-text" : "text-forge-text/90"
                }`}
              >
                {m.content}
                {streaming && i === messages.length - 1 && (
                  <span className="inline-block w-1.5 h-3.5 bg-forge-accent ml-0.5 align-middle animate-pulse" />
                )}
              </div>
            </div>
          ))}
          {error && (
            <div className="text-red-400 text-sm border border-red-900 rounded p-2">
              {error}
            </div>
          )}
        </div>

        <div className="p-3 border-t border-forge-border">
          <div className="flex gap-2">
            <textarea
              className="input resize-none font-mono"
              rows={2}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder={
                canChat ? "Your message… (Enter to send, Shift+Enter for newline)" : "Load a model before chatting."
              }
              disabled={!loaded.data?.loaded}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  send();
                }
              }}
            />
            {streaming ? (
              <button className="btn" onClick={stop}>
                Stop
              </button>
            ) : (
              <button
                className="btn btn-accent"
                onClick={send}
                disabled={!canChat || !input.trim()}
              >
                Send
              </button>
            )}
          </div>
        </div>
      </div>

      <aside className="card space-y-4 overflow-y-auto">
        <div>
          <label className="label">System Prompt</label>
          <textarea
            className="input resize-none font-mono text-xs"
            rows={6}
            value={system}
            onChange={(e) => setSystem(e.target.value)}
            placeholder="Optional. Guides model behavior."
          />
        </div>
        <div>
          <label className="label">Temperature · {temperature.toFixed(2)}</label>
          <input
            type="range"
            min={0}
            max={2}
            step={0.05}
            value={temperature}
            onChange={(e) => setTemperature(parseFloat(e.target.value))}
            className="w-full accent-[#e30613]"
          />
        </div>
        <div>
          <label className="label">Max Tokens · {maxTokens}</label>
          <input
            type="range"
            min={32}
            max={4096}
            step={32}
            value={maxTokens}
            onChange={(e) => setMaxTokens(parseInt(e.target.value))}
            className="w-full accent-[#e30613]"
          />
        </div>

        <div className="border-t border-forge-border pt-4">
          <label className="label">Performance · last message</label>
          {lastUsage ? (
            <div className="space-y-1.5 text-xs font-mono">
              <div className="flex justify-between">
                <span className="text-forge-muted">Throughput</span>
                <span className="text-forge-text">
                  {lastUsage.throughput.toFixed(1)} tok/s
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-forge-muted">Total</span>
                <span className="text-forge-text">
                  {(lastUsage.total_ms / 1000).toFixed(2)} s
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-forge-muted">Prompt tok</span>
                <span className="text-forge-text">{lastUsage.prompt_tokens}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-forge-muted">Completion tok</span>
                <span className="text-forge-text">{lastUsage.completion_tokens}</span>
              </div>
            </div>
          ) : (
            <div className="text-forge-muted text-xs">
              {streaming ? "Measuring…" : "No messages sent yet."}
            </div>
          )}
        </div>
      </aside>
    </div>
  );
}
