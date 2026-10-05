# CLAUDE.md

Read `WORKORDER.md` first and follow its ground rules in every session.

- Work on the lowest unfinished task; write `reports/T<nn>.md` before stopping.
- Never read the Putra test split before Phase 2; log every test-set access.
- Pin `jev-1.13`; persist every request and response under `data/raw_responses/`; stop at `MAX_USD`.
- Every reported number must be traceable to a file under `results/` produced from raw responses.
- `data/` is never committed, except `data/raw_responses/` and `data/derived/` while the repository is private (decisions D29). Secrets live in `.env` (see `.env.example`).
- Bootstrap samples (Section 1.3b, `scripts/t00_bootstrap_sample.py`) are for building and debugging only; nothing computed on them appears in a report table, and their ids are excluded from all transfer samples.
