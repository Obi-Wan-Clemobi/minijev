// Reads docs/WEAKNESSES.md on every request, so the page always shows the current register.
// Edit the markdown file to record a fix; reload the page to read it here.
import { readFile } from "node:fs/promises";
import path from "node:path";
import { connection } from "next/server";
import { WeaknessList, type Group } from "@/components/WeaknessList";

const DOC = path.join(process.cwd(), "..", "docs", "WEAKNESSES.md");

function parse(md: string): { title: string; intro: string; groups: Group[] } {
  const title = md.match(/^# (.*)$/m)?.[1] ?? "Weaknesses";
  // The summary table gives each weakness its area, severity and status.
  const meta: Record<string, { area: string; severity: string; status: string }> = {};
  for (const m of md.matchAll(/^\| (W\d+) \| [^|]+ \| ([^|]+) \| ([^|]+) \| ([^|]+) \|$/gm)) {
    meta[m[1]] = { area: m[2].trim(), severity: m[3].trim(), status: m[4].trim() };
  }
  const intro = md.slice(md.indexOf("\n", md.indexOf("# ")) + 1, md.indexOf("## Summary")).replace(/\n---\n/g, "\n").trim();
  const body = md.slice(md.indexOf("## Summary"));
  const groups: Group[] = [];
  for (const part of body.split(/^## /m).slice(1)) {
    const name = part.split("\n")[0].trim();
    if (name === "Summary") continue;
    const entries = part.split(/^### /m).slice(1).map((e) => {
      const head = e.split("\n")[0].trim();
      const id = head.match(/^(W\d+)\./)?.[1] ?? head;
      return { id, title: head.replace(/^W\d+\.\s*/, ""), body: e.slice(e.indexOf("\n") + 1).replace(/\n---\s*$/, "").trim(),
        ...(meta[id] ?? { area: name, severity: "", status: "" }) };
    });
    groups.push({ name, entries });
  }
  return { title, intro, groups };
}

export default async function Weaknesses() {
  await connection();
  let md: string;
  try { md = await readFile(DOC, "utf8"); }
  catch { return <div className="p-8 text-sm text-warn">Cannot read {DOC}. The page reads the register from the repository&apos;s docs folder.</div>; }
  const { title, intro, groups } = parse(md);
  return (
    <div className="px-4 md:px-8 py-8 max-w-[1100px] w-full mx-auto flex flex-col gap-4">
      <h1 className="m-0 text-[32px] font-semibold tracking-tight">{title.replace(/^minijev — /, "")}</h1>
      <WeaknessList groups={groups} intro={intro} />
      <p className="text-xs text-muted">Source: docs/WEAKNESSES.md. Edit it when you fix something; this page reads it on every load.</p>
    </div>
  );
}
