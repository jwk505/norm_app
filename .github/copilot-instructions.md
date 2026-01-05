Project quicknotes for AI coding agents

- Purpose: This repo is a small Streamlit-based normalization/ERP console + supporting batch engine. Frontend (Streamlit) lives at the repo root (`app.py`), feature pages under `features/`, and core helpers under `core/`.

- Quick start (dev):
  - Run Streamlit UI: `streamlit run app.py`
  - Background/batch runner: `python norm_engine.py` (reads `.env` directly)
  - DB config: place credentials in `.env` at project root (see `core/config.py` and `norm_engine.py`).

- Architecture (big picture):
  - UI: `app.py` mounts feature pages by calling `show_*_page()` functions exported from `features/*` (examples: `features/inventory.py`, `features/sales.py`, `features.customer.py`).
  - Core utilities: `core/` contains `config.py` (loads .env), `db.py` (mysql access wrappers), `schema.py` (column helpers), `ui.py` (table render helpers), and small init utilities.
  - Batch/engine: `norm_engine.py` performs normalization rules loading, applying rules, ingesting raw rows, mapping via alias tables, and reprocessing queues. Treat it as the canonical reference for normalization logic (rule fetch, `normalize()`, `apply_rules()`, `ingest()`, `process_new()`).

- Dataflow & integration points:
  - All DB access uses `mysql.connector` and raw SQL strings with `%s` parameters. Use `core.db.query_df(sql, params)` when a pandas DataFrame is expected (features rely on this pattern).
  - `core.db.exec_sql` / `exec_many` used for writes. Ensure `params` are passed as tuples (or None) to avoid SQL injection.
  - Normalization rules are stored in DB and loaded by `norm_engine.fetch_rules()`; rules are compiled into Python `re` objects and applied in `apply_rules()`.

- Project-specific conventions & patterns (copyable examples):
  - Feature pages export `show_<name>_page()` and are wired from `app.py` radio menu.
  - `features/*` build SQL strings directly, then call `query_df(...)`; e.g. `features/inventory.py` constructs a SELECT, then `df = query_df(sql, tuple(params) if params else None)`.
  - Column discovery: `core.schema.get_columns(table_name)` is used to adapt to optional columns (see `features/inventory.py` for maker/item heuristics).
  - DB credentials are required at runtime; many modules raise `RuntimeError` if env vars missing — prefer loading `.env` in tests or local runs.

- Dependencies to install (inferred):
  - `streamlit`, `pandas`, `mysql-connector-python`, `python-dotenv` (check your environment). No `requirements.txt` present — add one if you modify dependencies.

- Debugging & dev tips specific to this repo:
  - If DB env missing, `norm_engine.py` prints diagnostic env values then raises — check `.env` in repo root.
  - To reproduce UI behavior locally, run `streamlit run app.py` and open the UI; features expect live DB access for meaningful content.
  - For SQL examples and the canonical normalization behavior, inspect `norm_engine.py` (functions: `basic_normalize`, `apply_rules`, `normalize`, `ingest`, `process_new`, `reprocess_queue`).

- What NOT to change without discussion:
  - Normalization logic and rule storage format in DB — small changes can alter bulk mappings. If modifying regex handling, run `norm_engine.py` on a small sample first.
  - DB schema expectations (table/column names) used across features and engine (e.g., `inventory_snapshot`, `raw_incoming`, `normalize_rule`, `party_alias`, `item_alias`).

- Where to look for examples in-code:
  - UI wiring: `app.py`
  - Feature patterns: `features/inventory.py`, `features/sales.py`, `features/customer.py`
  - DB helpers: `core/db.py` and `core/config.py`
  - Normalization engine: `norm_engine.py`

If any area is unclear or you want the doc expanded with run/playback commands, tests, or a `requirements.txt`, tell me which part to expand.
