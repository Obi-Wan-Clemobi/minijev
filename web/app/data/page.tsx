// The Data page: live checks and charts (DataFacts), then docs/DATA.md, read on every request.
import { readFile } from "node:fs/promises";
import path from "node:path";
import { connection } from "next/server";
import { DataFacts } from "@/components/DataFacts";
import { DocSections } from "@/components/DocSections";

const DOC = path.join(process.cwd(), "..", "docs", "DATA.md");

export default async function DataPage() {
  await connection();
  let md = "";
  try { md = await readFile(DOC, "utf8"); } catch { /* shown below */ }
  const intro = md.slice(md.indexOf("\n", md.indexOf("# ")) + 1, md.indexOf("\n## ")).trim();
  const sections = md.split(/^## /m).slice(1).map((p) => ({ title: p.split("\n")[0].trim(), body: p.slice(p.indexOf("\n") + 1).trim() }));
  return (
    <div className="px-4 md:px-8 py-8 max-w-[1100px] w-full mx-auto flex flex-col gap-6">
      <div className="flex flex-col gap-2">
        <h1 className="m-0 text-[32px] font-semibold tracking-tight">Our data</h1>
        <p className="m-0 text-[15px] text-muted max-w-[860px] leading-relaxed">
          Where every number comes from: the datasets, how the sample was drawn, which split was used for what, and proof,
          re-checked now from the files.
        </p>
      </div>
      <DataFacts />
      {md ? <DocSections intro={intro} sections={sections} open={["1. Summary", "4. What each split is for"]} />
        : <p className="text-sm text-warn">Cannot read {DOC}.</p>}
      <p className="text-xs text-muted">Source: docs/DATA.md and poc/datasets/splits_v2.json. <code className="font-mono">uv run python data.py check</code> runs the same checks in a terminal.</p>
    </div>
  );
}
