"use client";
// The weakness register as a list of expandable rows, with filters. The data comes from docs/WEAKNESSES.md.
import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

export type Entry = { id: string; title: string; body: string; area: string; severity: string; status: string };
export type Group = { name: string; entries: Entry[] };

const bucket = (status: string) =>
  /^(Fixed|Avoided)/.test(status) ? "fixed" : /^(Partly|Fixes measured|Use )/.test(status) ? "progress" : "open";
const TONE = { fixed: "border-ok text-ok", progress: "border-accent text-accent", open: "border-warn text-warn" };
const SEV = (s: string) => (/^High/.test(s) ? "bg-warn" : /^Medium/.test(s) ? "bg-accent" : "bg-ghost");

const md = {
  p: ({ children }: { children?: React.ReactNode }) => <p className="text-[14px] leading-relaxed my-2">{children}</p>,
  ul: ({ children }: { children?: React.ReactNode }) => <ul className="list-disc pl-5 my-2 flex flex-col gap-1.5 text-[14px] leading-relaxed">{children}</ul>,
  li: ({ children }: { children?: React.ReactNode }) => <li className="marker:text-muted">{children}</li>,
  strong: ({ children }: { children?: React.ReactNode }) => <strong className="font-semibold">{children}</strong>,
  em: ({ children }: { children?: React.ReactNode }) => <em className="not-italic text-muted">{children}</em>,
  code: ({ children }: { children?: React.ReactNode }) => <code className="font-mono text-[12px] px-1 py-0.5 rounded bg-track">{children}</code>,
};

export function WeaknessList({ groups, intro }: { groups: Group[]; intro: string }) {
  const [filter, setFilter] = useState<"all" | "open" | "progress" | "fixed">("all");
  const [open, setOpen] = useState<Record<string, boolean>>({});
  const all = groups.flatMap((g) => g.entries);
  const count = (b: string) => all.filter((e) => bucket(e.status) === b).length;
  const shown = (e: Entry) => filter === "all" || bucket(e.status) === filter;
  const setAll = (v: boolean) => setOpen(Object.fromEntries(all.map((e) => [e.id, v])));

  return (
    <div className="flex flex-col gap-6">
      <div className="text-muted"><ReactMarkdown remarkPlugins={[remarkGfm]} components={md}>{intro}</ReactMarkdown></div>

      <div className="flex items-center gap-3 flex-wrap sticky top-16 z-10 bg-bg/95 backdrop-blur py-3 border-b border-line">
        <div className="flex border border-line rounded-lg p-[3px] gap-0.5">
          {([["all", `All · ${all.length}`], ["open", `Open · ${count("open")}`], ["progress", `In progress · ${count("progress")}`], ["fixed", `Fixed · ${count("fixed")}`]] as const).map(([k, l]) => (
            <button key={k} onClick={() => setFilter(k)} aria-pressed={filter === k}
              className={`h-8 px-3 rounded-[5px] text-[13px] ${filter === k ? "bg-track text-fg" : "text-muted hover:text-fg"}`}>{l}</button>
          ))}
        </div>
        <div className="flex-1" />
        <span className="text-xs text-muted flex items-center gap-3">
          <span className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-warn" />high</span>
          <span className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-accent" />medium</span>
          <span className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-ghost" />low</span>
        </span>
        <button onClick={() => setAll(true)} className="h-8 px-3 rounded-md border border-line text-[13px] hover:bg-track">Expand all</button>
        <button onClick={() => setAll(false)} className="h-8 px-3 rounded-md border border-line text-[13px] hover:bg-track">Collapse all</button>
      </div>

      {groups.map((g) => {
        const entries = g.entries.filter(shown);
        if (!entries.length) return null;
        return (
          <section key={g.name} className="flex flex-col gap-2">
            <h2 className="text-xs font-medium text-muted uppercase tracking-wider">{g.name}</h2>
            <div className="rounded-xl border border-line bg-card divide-y divide-line">
              {entries.map((e) => {
                const isOpen = !!open[e.id];
                return (
                  <div key={e.id} id={e.id}>
                    <button onClick={() => setOpen((o) => ({ ...o, [e.id]: !o[e.id] }))} aria-expanded={isOpen}
                      className="w-full text-left px-4 py-3 flex items-center gap-3 hover:bg-track/50 transition-colors">
                      <svg width="14" height="14" viewBox="0 0 24 24" aria-hidden="true" className={`shrink-0 text-muted transition-transform ${isOpen ? "rotate-90" : ""}`}
                        fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M9 6l6 6-6 6" /></svg>
                      <span className="font-mono text-xs text-muted w-8 shrink-0">{e.id}</span>
                      <span className={`w-2 h-2 rounded-full shrink-0 ${SEV(e.severity)}`} title={`Severity: ${e.severity}`} />
                      <span className="text-[14px] font-medium flex-1 min-w-0">{e.title}</span>
                      <span className="hidden md:inline text-xs text-muted">{e.severity}</span>
                      <span className={`text-xs px-2 py-0.5 rounded-full border whitespace-nowrap ${TONE[bucket(e.status)]}`}>{e.status}</span>
                    </button>
                    {isOpen && (
                      <div className="px-4 pb-4 pl-[3.25rem]">
                        <ReactMarkdown remarkPlugins={[remarkGfm]} components={md}>{e.body}</ReactMarkdown>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </section>
        );
      })}
    </div>
  );
}
