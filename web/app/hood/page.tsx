"use client";
import { useEffect, useState } from "react";
import { Icon } from "@/components/Icon";
import { Tip } from "@/components/Tip";
import { RequestEditor, useReady } from "@/components/RequestEditor";
import { api, type TreeResponse } from "@/lib/api";
import { useStore } from "@/lib/store";

const show = (t: string) => t.replace(/\n/g, "↵").replace(/ /g, "·");

export default function Hood() {
  const { req, settings, response, run, running } = useStore();
  const ready = useReady();
  const [tree, setTree] = useState<TreeResponse | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [k, setK] = useState(0);
  const [showTemplate, setShowTemplate] = useState(false);

  // Tokenizing is cheap (no forward pass), so the layout follows your typing after a short pause.
  useEffect(() => {
    if (!ready) return;
    const t = setTimeout(() => {
      api.tree(req, settings).then((t) => { setErr(null); setTree(t); setK((x) => Math.min(x, t.branches.length - 1)); })
        .catch((e) => setErr(e.message));
    }, 350);
    return () => clearTimeout(t);
  }, [req, settings, ready]);

  return (
    <div className="flex-1 grid lg:grid-cols-[minmax(400px,480px)_minmax(0,1fr)]">
      <aside className="lg:border-r border-line px-4 md:px-6 py-6 lg:sticky lg:top-16 lg:self-start lg:max-h-[calc(100vh-4rem)] lg:overflow-y-auto">
        <RequestEditor presets>
          <Tip k="hoodRun">
            <button onClick={run} disabled={!ready || running}
              className="h-10 px-4 rounded-lg bg-inv-bg text-inv-fg text-sm font-medium flex items-center justify-center gap-2 disabled:opacity-40">
              <Icon name="play" size={14} fill />{running ? "Running…" : "Run the model for label mass"}
            </button>
          </Tip>
        </RequestEditor>
      </aside>
      <div className="min-w-0">
        {!ready && <p className="px-4 md:px-8 pt-6 text-sm text-warn">Fix the request on the left to update the tree.</p>}
        {err && <p role="alert" className="px-4 md:px-8 pt-6 text-warn text-sm">{err}</p>}
        {!tree ? <p className="p-8 text-sm text-muted">Building the prefix tree…</p> : <Tree tree={tree} k={k} setK={setK} response={response} showTemplate={showTemplate} setShowTemplate={setShowTemplate} />}
      </div>
    </div>
  );
}

function Tree({ tree, k, setK, response, showTemplate, setShowTemplate }: {
  tree: TreeResponse; k: number; setK: (i: number) => void; response: ReturnType<typeof useStore>["response"];
  showTemplate: boolean; setShowTemplate: (v: boolean) => void;
}) {

  const B = tree.branches, P = tree.prefix.length, sel = B[Math.min(k, B.length - 1)];
  const [lo, hi] = tree.prefix.state_span;
  const rowH = Math.max(30, Math.min(64, 380 / B.length));
  const svgH = Math.max(240, B.length * rowH + 16);
  const S = 396 / tree.total;
  const segs = [{ start: 0, length: P }, ...B.map((b) => ({ start: b.start, length: b.length }))];
  const mass = (b: (typeof B)[number], i: number) => {
    const m = response?.debug.raw[b.question]?.mass;
    if (!m) return null;
    const j = B.slice(0, i).filter((x) => x.question === b.question).length;
    return m[j] ?? m[0];
  };

  return (
    <div className="px-4 md:px-8 py-8 flex flex-col gap-6">
      <div className="flex items-end gap-4 flex-wrap">
        <div className="flex flex-col gap-1.5">
          <h1 className="m-0 text-[32px] font-semibold tracking-tight">One request, one forward pass</h1>
          <p className="m-0 text-[15px] text-muted max-w-[820px] leading-normal">
            The state is prefilled once. Each question is a branch that sees only the state and itself; a pointwise Score or Choice gets one yes/no branch per item. Select a branch to trace it.
          </p>
        </div>
        <div className="flex-1" />
        <span className="font-mono text-xs text-muted">{P} + {B.map((b) => b.length).join(" + ")} = {tree.total} tokens</span>
      </div>

      <section className="rounded-xl border border-line bg-card p-6 flex flex-col gap-4">
        <div className="flex items-center gap-2">
          <h2 className="text-[17px] font-semibold m-0">What is a readout?</h2>
          <Tip k="hoodReadout" />
        </div>

        <div className="grid md:grid-cols-2 gap-4">
          <div className="rounded-lg border border-line bg-track p-4 flex flex-col gap-3">
            <div className="flex items-center gap-2">
              <span className="text-sm font-semibold text-warn">Generation</span>
              <span className="text-xs text-muted">(normal LLM)</span>
            </div>
            <div className="flex flex-col gap-2 font-mono text-xs">
              <div className="flex items-center gap-2">
                <span className="text-muted">pass 1 →</span>
                <span className="px-2 py-1 rounded bg-warn/20 text-warn">b</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-muted">pass 2 →</span>
                <span className="px-2 py-1 rounded bg-warn/20 text-warn">i</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-muted">pass 3 →</span>
                <span className="px-2 py-1 rounded bg-warn/20 text-warn">l</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-muted">pass 4 →</span>
                <span className="px-2 py-1 rounded bg-warn/20 text-warn">l</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-muted">pass 5 →</span>
                <span className="px-2 py-1 rounded bg-warn/20 text-warn">i</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-muted">pass 6 →</span>
                <span className="px-2 py-1 rounded bg-warn/20 text-warn">n</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-muted">pass 7 →</span>
                <span className="px-2 py-1 rounded bg-warn/20 text-warn">g</span>
              </div>
            </div>
            <p className="text-xs text-muted m-0 pt-2 border-t border-line">7 model passes to write "billing"</p>
          </div>

          <div className="rounded-lg border border-accent bg-soft p-4 flex flex-col gap-3">
            <div className="flex items-center gap-2">
              <span className="text-sm font-semibold text-accent">Readout</span>
              <span className="text-xs text-muted">(minijev)</span>
            </div>
            <div className="flex flex-col gap-2 font-mono text-xs">
              <div className="flex items-center gap-2">
                <span className="text-muted">pass 1 →</span>
                <div className="flex gap-1.5">
                  <span className="px-2 py-1 rounded bg-accent/20 text-accent">A: 0.02</span>
                  <span className="px-2 py-1 rounded bg-accent/20 text-accent">B: 0.04</span>
                  <span className="px-2 py-1 rounded bg-accent text-inv-fg font-semibold">C: 0.94</span>
                </div>
              </div>
            </div>
            <div className="flex flex-col gap-1.5 text-xs pt-2">
              <span className="text-fg">Answer: <span className="font-semibold">C (billing)</span> with 94% confidence</span>
              <span className="text-muted">All probabilities returned instantly</span>
            </div>
            <p className="text-xs text-muted m-0 pt-2 border-t border-line">1 model pass, probabilities extracted directly</p>
          </div>
        </div>

        <div className="flex flex-col gap-2 text-sm text-muted">
          <p className="m-0">Every language model already computes probabilities for all ~150,000 tokens at every position. Generation samples from those probabilities and loops. A readout uses them directly and stops.</p>
          <p className="m-0 font-medium text-fg">The answer IS the probabilities—no tokens are generated.</p>
        </div>
      </section>

      <div className="grid xl:grid-cols-[minmax(0,1.25fr)_minmax(0,1fr)] gap-6">
        <section className="rounded-xl border border-line bg-card p-6 flex gap-5 flex-wrap">
          <div className="flex-1 min-w-[300px] flex flex-col gap-3">
            <h2 className="text-[15px] font-semibold">Prefix tree</h2>
            <svg viewBox={`0 0 440 ${svgH}`} className="w-full max-w-[440px]" role="img" aria-label={`State with ${B.length} branches`}>
              {B.map((b, i) => {
                const y = 8 + i * rowH + rowH / 2 - 4;
                return <path key={i} className="anim" d={`M158 ${svgH / 2} C 205 ${svgH / 2}, 205 ${y}, 250 ${y}`} fill="none"
                  stroke={i === k ? "var(--accent)" : "var(--line)"} strokeWidth={i === k ? 2 : 1} />;
              })}
              <rect x="8" y={svgH / 2 - 30} width="150" height="60" rx="8" fill="var(--track)" stroke="var(--fg)" />
              <text x="83" y={svgH / 2 - 4} textAnchor="middle" fill="var(--fg)" fontSize="14" fontWeight="600">STATE</text>
              <text x="83" y={svgH / 2 + 15} textAnchor="middle" fill="var(--muted)" fontSize="12" fontFamily="var(--font-geist-mono)">{P} tok · pos 0–{P - 1}</text>
              {B.map((b, i) => {
                const y = 8 + i * rowH, h = rowH - 8;
                return (
                  <g key={i} onClick={() => setK(i)} className="cursor-pointer">
                    <rect x="250" y={y} width="185" height={h} rx="7" fill={i === k ? "var(--soft)" : "var(--card)"} stroke={i === k ? "var(--accent)" : "var(--line)"} />
                    <text x="262" y={y + h / 2 + 4} fill="var(--fg)" fontSize={rowH < 40 ? 11 : 13} fontFamily="var(--font-geist-mono)">
                      {b.label.length > 22 ? b.label.slice(0, 21) + "…" : b.label}
                    </text>
                  </g>
                );
              })}
            </svg>
          </div>
          <div className="w-full sm:w-[210px] flex flex-col gap-1.5 max-h-[460px] overflow-y-auto">
            <span className="text-xs text-muted uppercase tracking-wider flex items-center gap-1">Branches <Tip k="hoodBranch" /></span>
            {B.map((b, i) => {
              const m = mass(b, i);
              return (
                <button key={i} onClick={() => setK(i)} aria-pressed={i === k}
                  className={`flex flex-col items-start gap-0.5 min-h-11 px-2.5 py-1.5 rounded-md border text-left transition-colors ${i === k ? "border-accent bg-soft" : "border-line hover:bg-track"}`}>
                  <span className="font-mono text-xs truncate max-w-full">{b.label}</span>
                  <span className="text-[11px] text-muted">{b.length} tok · {b.pointwise ? "yes/no log-odds" : b.type === "noul" ? "P(Yes) vs P(No)" : "letter labels"}{m !== null ? ` · mass ${m.toFixed(4)}` : ""}</span>
                </button>
              );
            })}
          </div>
        </section>

        <section className="rounded-xl border border-line bg-card p-6 flex flex-col gap-3.5">
          <div className="flex items-baseline justify-between gap-3 flex-wrap">
            <h2 className="text-[15px] font-semibold flex items-center gap-1">Attention mask · {tree.total} × {tree.total}, to scale <Tip k="hoodMask" /></h2>
            <span className="text-xs text-muted">rows attend to columns</span>
          </div>
          <div className="flex gap-4 flex-wrap">
            <div className="relative w-[396px] h-[396px] max-w-full bg-track border border-line" role="img"
              aria-label={`Branch ${sel.label} attends to the state and to itself only`}>
              {segs.flatMap((r, ri) => segs.map((c, ci) => {
                if (!(ci === 0 || ri === ci)) return null;
                const hot = ri === 0 || ri === k + 1;
                return <div key={`${ri}-${ci}`} className="anim absolute bg-accent"
                  style={{ left: c.start * S, top: r.start * S, width: c.length * S, height: r.length * S,
                    clipPath: ri === ci ? "polygon(0 0, 0 100%, 100% 100%)" : undefined, opacity: hot ? 1 : 0.3 }} />;
              }))}
            </div>
            <div className="flex-1 min-w-[160px] flex flex-col gap-2.5 text-xs text-muted leading-normal">
              <span className="flex gap-2 items-center"><span className="w-3 h-3 bg-accent" />can attend</span>
              <span className="flex gap-2 items-center"><span className="w-3 h-3 bg-track border border-line" />masked</span>
              <span className="mt-2 text-fg font-mono">{sel.label} · rows {sel.start}–{sel.start + sel.length - 1}</span>
              <span>sees the state (columns 0–{P - 1}) and its own tokens, causally. It never sees another branch.</span>
              <span className="mt-2">The packed pass still computes the grey blocks, so its cost grows with the square of the total length.</span>
            </div>
          </div>
        </section>
      </div>

      <section className="rounded-xl border border-line bg-card p-6 flex flex-col gap-3">
        <div className="flex items-baseline justify-between gap-3 flex-wrap">
          <h2 className="text-[15px] font-semibold">Your state, as tokens · {lo === hi ? 0 : hi - lo} of {P} prefix tokens</h2>
          <label className="flex items-center gap-2 text-xs text-muted">
            <input type="checkbox" checked={showTemplate} onChange={(e) => setShowTemplate(e.target.checked)} className="accent-[var(--accent)]" />
            show the template tokens ({lo} before, {P - hi} after)
          </label>
          <Tip k="hoodTemplate" />
        </div>
        <div className="flex flex-wrap gap-1 font-mono text-[11px]" aria-label="State tokens with positions">
          {tree.prefix.tokens.map((t, i) => {
            const inState = i >= lo && i < hi;
            if (!inState && !showTemplate) return null;
            return (
              <span key={i} className={`inline-flex flex-col items-center rounded border px-1.5 py-0.5 ${inState ? "border-line" : "border-dashed border-line opacity-60"}`}>
                <span className="text-fg whitespace-pre">{show(t)}</span>
                <span className="text-muted text-[10px]">{i}</span>
              </span>
            );
          })}
        </div>
        <p className="text-xs text-muted leading-normal">
          Each chip is one token with its position. The model sees these once; every branch below attends to all of them. · marks a space and ↵ a newline.
          A token can join the end of your text with the blank line after it.
        </p>
      </section>

      <section className="rounded-xl border border-line bg-card p-6 flex flex-col gap-3">
        <div className="flex items-baseline justify-between gap-3 flex-wrap">
          <h2 className="text-[15px] font-semibold flex items-center gap-1">Packed sequence · position ids restart after the state <Tip k="hoodPositions" /></h2>
          <span className="text-xs text-muted font-mono">max position = {P} + longest branch − 1 = {tree.max_position}</span>
        </div>
        <div className="flex h-11 rounded-md overflow-hidden border border-line">
          <div className="bg-fg text-bg grid place-items-center text-[11px] font-mono border-r border-bg" style={{ width: `${(P / tree.total) * 100}%` }}>STATE</div>
          {B.map((b, i) => (
            <button key={i} onClick={() => setK(i)} aria-label={`Select ${b.label}`}
              className={`anim grid place-items-center text-[11px] font-mono overflow-hidden whitespace-nowrap border-r border-bg ${i === k ? "bg-accent text-inv-fg" : "bg-track text-muted"}`}
              style={{ width: `${(b.length / tree.total) * 100}%` }}>{b.length / tree.total > 0.06 ? b.label : ""}</button>
          ))}
        </div>
        <div className="flex flex-wrap gap-1 font-mono text-[11px] mt-2" aria-label={`Tokens of ${sel.label}`}>
          {sel.tokens.map((t, i) => (
            <span key={i} className="inline-flex flex-col items-center rounded border border-line px-1.5 py-0.5">
              <span className="text-fg whitespace-pre">{show(t)}</span>
              <span className="text-muted text-[10px]">{sel.positions[0] + i}</span>
            </span>
          ))}
        </div>
        <p className="text-xs text-muted">The readout happens at the last token of the branch (position {sel.positions[1]}): the start of the assistant turn, where the answer label would be generated.</p>
      </section>
    </div>
  );
}
