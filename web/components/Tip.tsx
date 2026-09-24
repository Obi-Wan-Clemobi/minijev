"use client";
// A tooltip for every control: opens on hover (after a short delay) or keyboard focus, closes on Escape,
// blur, mouse leave or scroll. It renders in a portal with fixed position, so scroll containers never clip it.
import { useEffect, useId, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { HELP, type Help, type HelpKey } from "@/lib/help";
import { Icon } from "./Icon";

const BADGE = {
  instant: ["Instant", "the answers change right away; no need to run the model"],
  run: ["Needs a Run", "press Run to see the effect"],
  display: ["Display only", "changes what you see, not the answers"],
} as const;

export function Tip({ k, children, className = "" }: { k: HelpKey; children?: React.ReactNode; className?: string }) {
  const h: Help = HELP[k];
  const id = useId();
  const ref = useRef<HTMLSpanElement>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [pos, setPos] = useState<{ left: number; top?: number; bottom?: number } | null>(null);

  const show = (delay: number) => {
    if (timer.current) clearTimeout(timer.current);
    timer.current = setTimeout(() => {
      const r = ref.current?.getBoundingClientRect();
      if (!r) return;
      const w = 320, left = Math.min(Math.max(8, r.left + r.width / 2 - w / 2), window.innerWidth - w - 8);
      setPos(r.bottom + 220 < window.innerHeight ? { left, top: r.bottom + 8 } : { left, bottom: window.innerHeight - r.top + 8 });
    }, delay);
  };
  const hide = () => { if (timer.current) clearTimeout(timer.current); setPos(null); };

  useEffect(() => {
    if (!pos) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && hide();
    window.addEventListener("keydown", onKey);
    window.addEventListener("scroll", hide, true);
    return () => { window.removeEventListener("keydown", onKey); window.removeEventListener("scroll", hide, true); };
  }, [pos]);
  useEffect(() => () => { if (timer.current) clearTimeout(timer.current); }, []);

  return (
    <span ref={ref} className={`inline-flex items-center ${className}`} aria-describedby={pos ? id : undefined}
      onMouseEnter={() => show(250)} onMouseLeave={hide} onFocus={() => show(0)} onBlur={hide}>
      {children ?? (
        <button type="button" aria-label={`About: ${h.t}`}
          className="w-5 h-5 -my-1 rounded-full grid place-items-center text-muted hover:text-fg focus-visible:text-fg transition-colors">
          <Icon name="info" size={13} />
        </button>
      )}
      {pos && createPortal(
        <div id={id} role="tooltip" style={{ position: "fixed", width: 320, zIndex: 50, ...pos }}
          className="rounded-lg border border-line bg-card text-fg shadow-xl p-3.5 flex flex-col gap-2 text-[13px] leading-snug pointer-events-none">
          <span className="font-semibold">{h.t}</span>
          {h.d.map((line, i) => <span key={i} className="text-muted">{line}</span>)}
          {h.e && (
            <span className="flex items-center gap-1.5 text-xs pt-1 border-t border-line">
              <span className={`px-1.5 py-0.5 rounded font-medium ${h.e === "instant" ? "bg-soft text-accent" : h.e === "run" ? "bg-track text-warn" : "bg-track text-muted"}`}>{BADGE[h.e][0]}</span>
              <span className="text-muted">{BADGE[h.e][1]}</span>
            </span>
          )}
        </div>,
        document.body,
      )}
    </span>
  );
}
