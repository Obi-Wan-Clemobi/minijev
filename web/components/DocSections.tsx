"use client";
// A markdown document as expandable sections.
import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

const md = {
  p: ({ children }: { children?: React.ReactNode }) => <p className="text-[14px] leading-relaxed my-2">{children}</p>,
  ul: ({ children }: { children?: React.ReactNode }) => <ul className="list-disc pl-5 my-2 flex flex-col gap-1.5 text-[14px] leading-relaxed">{children}</ul>,
  ol: ({ children }: { children?: React.ReactNode }) => <ol className="list-decimal pl-5 my-2 flex flex-col gap-1.5 text-[14px] leading-relaxed">{children}</ol>,
  li: ({ children }: { children?: React.ReactNode }) => <li className="marker:text-muted">{children}</li>,
  strong: ({ children }: { children?: React.ReactNode }) => <strong className="font-semibold">{children}</strong>,
  em: ({ children }: { children?: React.ReactNode }) => <em className="not-italic text-muted">{children}</em>,
  code: ({ children }: { children?: React.ReactNode }) => <code className="font-mono text-[12px] px-1 py-0.5 rounded bg-track">{children}</code>,
  pre: ({ children }: { children?: React.ReactNode }) => <pre className="my-2 p-3 rounded-lg bg-surface border border-line overflow-x-auto text-[12px] [&_code]:bg-transparent [&_code]:p-0">{children}</pre>,
  table: ({ children }: { children?: React.ReactNode }) => <div className="overflow-x-auto my-3 rounded-lg border border-line"><table className="w-full text-[13px]">{children}</table></div>,
  th: ({ children }: { children?: React.ReactNode }) => <th className="text-left font-medium text-muted px-3 py-2 bg-surface border-b border-line">{children}</th>,
  td: ({ children }: { children?: React.ReactNode }) => <td className="px-3 py-2 border-b border-line align-top">{children}</td>,
};

export function DocSections({ intro, sections, open: initial = [] }: { intro: string; sections: { title: string; body: string }[]; open?: string[] }) {
  const [open, setOpen] = useState<Record<string, boolean>>(Object.fromEntries(initial.map((t) => [t, true])));
  return (
    <div className="flex flex-col gap-3">
      <div className="text-muted"><ReactMarkdown remarkPlugins={[remarkGfm]} components={md}>{intro}</ReactMarkdown></div>
      <div className="rounded-xl border border-line bg-card divide-y divide-line">
        {sections.map((s) => (
          <div key={s.title}>
            <button onClick={() => setOpen((o) => ({ ...o, [s.title]: !o[s.title] }))} aria-expanded={!!open[s.title]}
              className="w-full text-left px-4 py-3 flex items-center gap-3 hover:bg-track/50 transition-colors">
              <svg width="14" height="14" viewBox="0 0 24 24" aria-hidden="true" className={`shrink-0 text-muted transition-transform ${open[s.title] ? "rotate-90" : ""}`}
                fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M9 6l6 6-6 6" /></svg>
              <span className="text-[14px] font-medium">{s.title}</span>
            </button>
            {open[s.title] && <div className="px-4 pb-4 pl-[3.25rem]"><ReactMarkdown remarkPlugins={[remarkGfm]} components={md}>{s.body}</ReactMarkdown></div>}
          </div>
        ))}
      </div>
    </div>
  );
}
