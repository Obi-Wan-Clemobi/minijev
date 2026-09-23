# minijev — Project Writing Rules

The general writing rules (modified ASD-STE100) are in `~/.claude/CLAUDE.md`. They apply
to all written output in this repo: docs (`README.md`, `docs/*.md`), code comments, commit
messages, and Claude's chat replies. This file adds the project-specific rules.

## Evidence labels

Use the labels from RESEARCH.md:
- **Stated**: TypeSafe says it (docs, launch blog, FAQ, press quotes). Give the source.
- **Observed**: visible in real Jev outputs or third-party measurements.
- **Inferred**: our reasoning. It can be wrong.
- **Measured**: a result from our own POC runs.
- Say "unknown" when nobody outside TypeSafe knows.

## Terms

Normal ML terms are allowed (for example: logit, calibration, forward pass, tokenizer).
Define each term on first use in each document.

## Glossary

Add a term here when a document defines it. Use the term exactly as written. The
"Do not use" column lists synonyms that must not replace the term.

| Term | Meaning | Do not use |
|------|---------|------------|
| Jev | TypeSafe's product: text in, typed and calibrated probabilistic decisions out. | |
| minijev | This project: a small, local, open re-creation of the idea behind Jev. | |
| POC | The proof of concept in `poc/`. | |
| state | The input text of a request. All questions share it. A specific state can be named ("the GDPR article"). | document, text, context (for the input slot) |
| question | One typed item in a request: a Noul, a Choice, or a Score. | query, prompt (for one item) |
| branch | The tokens for one question (or one Score level) that follow the state. A branch sees only the state and itself. | fork, path |
| option | One allowed answer of a Choice. | answer (for the allowed set), class |
| level | One point on a Score scale, numbered 0 to n−1. | grade, bucket |
| label | The token the model is asked to produce for an option or answer (`A`, `B`, `Yes`, `No`). | letter (except when the label is a letter) |
| Noul | A yes/no question. Returns P(yes). | boolean question |
| Choice | A question with a fixed set of options. Returns one option and probabilities. | classification question |
| Score | A question on an ordered scale of levels. Returns the expected level and probabilities. | rating question |
| prefill | The model processes input tokens. All input tokens go through the model together. | read, reading (for input tokens) |
| generate | The model produces output tokens, one model pass per token. The phase is called decode. | write, writing (for output tokens) |
| readout | Take the probabilities of the label tokens at one position, from one forward pass. No tokens are generated. | read (for probabilities) |
| generation baseline | An LLM that generates its answer as text, which code then parses. | "the LLM way" |
| packed | The mode that runs the state and all branches as one sequence, with an attention mask between branches. | |
