# minijev playground (web)

A local web page for minijev: build a request, read the typed answers, turn the calibration dials, compare the
readout with generation, and look inside the packed forward pass. It calls the API in `../poc/server.py`.

Run both parts (two terminals):

1. `cd poc && uv run minijev serve --port 8000`. The first start downloads the model (~1 GB for 0.5B).
2. `cd web && npm install && npm run dev`, then open http://localhost:3000.

The API address comes from `NEXT_PUBLIC_API_URL` (default `http://localhost:8000`).

| Page | What it shows |
|---|---|
| Playground | The question editor (Noul, Choice options, Score levels, or raw JSON), the answers, and the calibration dials. The dials re-score the returned logits in the page (`lib/scoring.ts`), so they need no model run. |
| Compare | Everything uses the same model; only the way of getting the answer changes. 1: your request answered in minijev's JSON format, read out vs written freely vs written with the format enforced. 2: quality on questions with known answers (E11). 3: other ways to ask several questions, and the recorded 13-question speed results. |
| Under the hood | The prefix tree, the attention mask to scale, and the tokens of each branch with their position ids. |
| Findings | Charts from `poc/results/*.json`, including the option-order flaw and its fixes (E13). |
| Data | The data card (`docs/DATA.md`), every `data.py check` claim re-run on load, the split balance, and the held-out results (E14). |
| Weaknesses | The register in `docs/WEAKNESSES.md` as expandable rows with filters. The page reads the file on every load, so an edit shows at once. |

Checks: `npm test` compares `lib/scoring.ts` with the Python `answer()` on 60 fixture cases
(`poc/tests/scoring_fixture.py` writes them). `npm run lint` and `npx tsc --noEmit` check the code.
