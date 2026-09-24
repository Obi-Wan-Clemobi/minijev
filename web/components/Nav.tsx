"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { useStore } from "@/lib/store";
import { Icon } from "./Icon";
import { Tip } from "./Tip";

const LINKS = [["/", "Playground"], ["/compare", "Compare"], ["/hood", "Under the hood"], ["/findings", "Findings"], ["/weaknesses", "Weaknesses"]];

export function Nav() {
  const path = usePathname();
  const { model, models, switchModel, switching } = useStore();
  // Use useState with effect to avoid hydration mismatch - server and client start with same value
  const [dark, setDark] = useState(true);  // Default matches layout.tsx

  useEffect(() => {
    // Set initial state from DOM after mount
    setDark(document.documentElement.classList.contains("dark"));

    // Listen for theme changes
    const handler = () => setDark(document.documentElement.classList.contains("dark"));
    window.addEventListener("mj-theme", handler);
    return () => window.removeEventListener("mj-theme", handler);
  }, []);

  const toggle = () => {
    const next = !dark;
    document.documentElement.classList.toggle("dark", next);
    try { localStorage.setItem("mj-theme", next ? "dark" : "light"); } catch { /* ignore */ }
    window.dispatchEvent(new Event("mj-theme"));
  };
  return (
    <header className="h-16 px-4 md:px-8 border-b border-line flex items-center gap-4 md:gap-8 sticky top-0 z-20 bg-bg/90 backdrop-blur">
      <Link href="/" className="flex items-center gap-2.5 shrink-0">
        <span className="w-[26px] h-[26px] rounded-md bg-fg text-bg grid place-items-center font-mono text-xs font-semibold">mj</span>
        <span className="font-semibold text-[15px] tracking-tight">minijev</span>
        <span className="hidden sm:inline text-xs text-muted border border-line rounded-full px-2 py-0.5">local</span>
      </Link>
      <nav className="flex gap-1 text-sm">
        {LINKS.map(([href, label]) => (
          <Link key={href} href={href}
            className={`px-3 py-2 rounded-md whitespace-nowrap transition-colors ${path === href ? "bg-track text-fg font-medium" : "text-muted hover:text-fg"}`}>
            {label}
          </Link>
        ))}
      </nav>
      <div className="flex-1" />
      <Tip k="model" className="hidden md:inline-flex">
      <label className="flex items-center gap-2 h-9 px-3 rounded-md border border-line bg-card text-[13px]">
        <span className={`w-2 h-2 rounded-full ${switching ? "bg-warn animate-pulse" : model ? "bg-ok" : "bg-ghost"}`} />
        <span className="sr-only">Model</span>
        <select value={model} disabled={switching || !models.length} onChange={(e) => switchModel(e.target.value)}
          className="bg-transparent font-mono outline-none">
          {!models.length && <option value="">API offline</option>}
          {models.map((m) => <option key={m} value={m} className="bg-card">{m.split("/")[1]}</option>)}
        </select>
        <span className="text-muted">{switching ? "· loading model…" : "· CPU"}</span>
      </label>
      </Tip>
      <Tip k="theme">
        <button onClick={toggle} aria-label={dark ? "Switch to light theme" : "Switch to dark theme"}
          className="w-9 h-9 rounded-md border border-line grid place-items-center hover:bg-track transition-colors">
          <Icon name={dark ? "sun" : "moon"} />
        </button>
      </Tip>
    </header>
  );
}
