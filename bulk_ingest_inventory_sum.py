# bulk_ingest_inventory_sum.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from datetime import datetime

import mysql.connector
import pandas as pd


# =========================
# CONFIG
# =========================
XLSX_PATH = os.environ.get("XLSX_PATH", r"C:\norm_app\inventory.xlsx")
SHEET_NAME = os.environ.get("SHEET_NAME", "합산")
SCAN_MAX_ROWS = int(os.environ.get("SCAN_MAX_ROWS", "80"))

SOURCE_SYSTEM = os.environ.get("SOURCE_SYSTEM", "INV_SUM")
SNAPSHOT_TAG = os.environ.get("SNAPSHOT_TAG", datetime.now().strftime("%Y-%m-%d"))

# 0이면 전체
ROW_LIMIT = int(os.environ.get("ROW_LIMIT", "0"))

# 배치 크기 (회사 DB면 500~2000 추천)
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1000"))

DEFAULT_RAW_PARTY = None


# =========================
# .env loader
# =========================
def load_env(env_path: Path) -> None:
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        os.environ.setdefault(k, v)


ENV_PATH = Path(__file__).resolve().parent / ".env"
load_env(ENV_PATH)
print("ENV FILE:", str(ENV_PATH), "EXISTS:", ENV_PATH.exists())


def env_get(key: str, default: str | None = None) -> str | None:
    v = os.environ.get(key)
    if v is None:
        return default
    v = v.strip()
    if v == "":
        return default
    return v


DB_CFG = {
    "host": env_get("DB_HOST", "127.0.0.1"),
    "port": int(env_get("DB_PORT", "3306")),
    "user": env_get("DB_USER", "root"),
    "password": env_get("DB_PASS", ""),
    "database": env_get("DB_NAME", "normdb"),
}


def get_conn():
    host = DB_CFG["host"]
    port = DB_CFG["port"]
    user = DB_CFG["user"]
    db = DB_CFG["database"]
    pwd = DB_CFG["password"] or ""
    print(f"[DB] connecting host={host} port={port} user={user} db={db} pass={'SET' if pwd else 'EMPTY'}")
    conn = mysql.connector.connect(**DB_CFG)
    conn.autocommit = False  # ✅ 배치 커밋을 위해 끔
    return conn


# =========================
# helpers
# =========================
def _norm_key(x: str) -> str:
    return re.sub(r"\s+", "", str(x)).replace("\n", "").replace("\r", "").upper()


def _to_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def _to_decimal(x):
    if pd.isna(x) or x == "":
        return None
    try:
        s = str(x).replace(",", "").strip()
        if s == "":
            return None
        return float(s)
    except Exception:
        return None


def detect_header_row(xlsx_path: str, sheet_name: str, scan_max_rows: int) -> int:
    preview = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=None, nrows=scan_max_rows)
    tokens = ["품목", "품명", "품목명", "메이커", "MAKER", "수량", "단가", "평균단가", "금액", "재고"]
    token_keys = [_norm_key(t) for t in tokens]

    best_row = 0
    best_score = -1
    for r in range(len(preview)):
        row_vals = preview.iloc[r].tolist()
        keys = [_norm_key(v) for v in row_vals if pd.notna(v)]
        score = 0
        for tk in token_keys:
            if any(tk in k for k in keys):
                score += 1
        if score > best_score:
            best_score = score
            best_row = r

    print(f"[HEADER_DETECT] best_row={best_row}, best_score={best_score}")
    return best_row


def find_col(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    cols = list(df.columns)
    col_map = {_norm_key(c): c for c in cols}
    cand_keys = [_norm_key(x) for x in candidates]

    for ck in cand_keys:
        if ck in col_map:
            return col_map[ck]
    for c in cols:
        nk = _norm_key(c)
        for ck in cand_keys:
            if ck and ck in nk:
                return c

    if required:
        raise RuntimeError(f"[ERROR] 컬럼을 못 찾음. 후보={candidates}, 실제컬럼={cols}")
    return None


# =========================
# schema auto-migrate (minimal)
# =========================
def column_exists(conn, table: str, col: str) -> bool:
    cur = conn.cursor(dictionary=True)
    cur.execute(
        """
        SELECT COUNT(*) AS cnt
        FROM information_schema.COLUMNS
        WHERE TABLE_SCHEMA = DATABASE()
          AND TABLE_NAME = %s
          AND COLUMN_NAME = %s
        """,
        (table, col),
    )
    row = cur.fetchone()
    return (row["cnt"] or 0) > 0


def ensure_columns(conn):
    table = "inventory_snapshot"

    alters = []
    if not column_exists(conn, table, "snapshot_tag"):
        alters.append("ADD COLUMN snapshot_tag VARCHAR(32) NOT NULL DEFAULT ''")
    if not column_exists(conn, table, "maker"):
        alters.append("ADD COLUMN maker VARCHAR(64) NULL")
    if not column_exists(conn, table, "qty"):
        alters.append("ADD COLUMN qty DECIMAL(18,4) NULL")
    if not column_exists(conn, table, "unit_price"):
        alters.append("ADD COLUMN unit_price DECIMAL(18,4) NULL")
    if not column_exists(conn, table, "amount"):
        alters.append("ADD COLUMN amount DECIMAL(18,2) NULL")

    if alters:
        sql = f"ALTER TABLE {table} " + ", ".join(alters)
        print("[SCHEMA]", sql)
        cur = conn.cursor()
        cur.execute(sql)
        conn.commit()

    # 유니크키 (없으면 추가 시도)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            ALTER TABLE inventory_snapshot
            ADD UNIQUE KEY uq_inv_snap (source_system, snapshot_tag, raw_item)
            """
        )
        conn.commit()
        print("[SCHEMA] added UNIQUE uq_inv_snap (source_system,snapshot_tag,raw_item)")
    except Exception:
        conn.rollback()
        # 이미 있으면 무시


# =========================
# ingest
# =========================
def main():
    print("[CONFIG]", {
        "XLSX_PATH": XLSX_PATH,
        "SHEET_NAME": SHEET_NAME,
        "SCAN_MAX_ROWS": SCAN_MAX_ROWS,
        "SOURCE_SYSTEM": SOURCE_SYSTEM,
        "SNAPSHOT_TAG": SNAPSHOT_TAG,
        "ROW_LIMIT": ROW_LIMIT,
        "CHUNK_SIZE": CHUNK_SIZE,
    })

    if not Path(XLSX_PATH).exists():
        print(f"[ERROR] 엑셀 파일이 없음: {XLSX_PATH}")
        sys.exit(1)

    header_row = detect_header_row(XLSX_PATH, SHEET_NAME, SCAN_MAX_ROWS)

    df = pd.read_excel(XLSX_PATH, sheet_name=SHEET_NAME, header=header_row)
    # ✅ 컬럼명 strip / NaN 제거
    df.columns = [str(c).strip() if pd.notna(c) else "" for c in df.columns]

    item_col = find_col(df, ["품목명", "품명", "품목", "자재명", "ITEM", "ITEM명"], required=True)
    qty_col = find_col(df, ["수량", "재고수량", "QTY", "Qty"], required=True)  # ✅ 수량 컬럼
    maker_col = find_col(df, ["메이커", "maker", "제조사", "브랜드", "BRAND"], required=False)
    amount_col = find_col(df, ["금액", "재고금액", "AMOUNT"], required=False)
    unitprice_col = find_col(df, ["평균단가", "단가", "UNITPRICE", "UNIT_PRICE"], required=False)

    df["_raw_item"] = df[item_col].apply(_to_str)
    df = df[df["_raw_item"] != ""].copy()

    if ROW_LIMIT and ROW_LIMIT > 0:
        df = df.head(ROW_LIMIT).copy()

    print("[META]", {
        "xlsx": XLSX_PATH,
        "sheet": SHEET_NAME,
        "rows": int(len(df)),
        "header_row": int(header_row),
        "columns": list(df.columns),
        "item_col": item_col,
        "qty_col": qty_col,
        "maker_col": maker_col,
        "amount_col": amount_col,
        "unitprice_col": unitprice_col,
        "kept_rows": int(len(df)),
    })

    conn = get_conn()
    ensure_columns(conn)

    raw_sql = """
    INSERT INTO raw_incoming
      (source_system, raw_party, raw_item, status, created_at)
    VALUES
      (%s, %s, %s, 'NEW', NOW())
    """

    snap_sql = """
    INSERT INTO inventory_snapshot
      (source_system, snapshot_tag, raw_item, norm_item, maker, qty, unit_price, amount, created_at)
    VALUES
      (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
    ON DUPLICATE KEY UPDATE
      maker = VALUES(maker),
      qty = VALUES(qty),
      unit_price = VALUES(unit_price),
      amount = VALUES(amount)
    """

    total = len(df)
    raw_ins = 0
    snap_upsert = 0

    cur = conn.cursor()

    def flush(batch_raw, batch_snap):
        nonlocal raw_ins, snap_upsert
        if not batch_raw and not batch_snap:
            return
        if batch_raw:
            cur.executemany(raw_sql, batch_raw)
            raw_ins += len(batch_raw)
        if batch_snap:
            cur.executemany(snap_sql, batch_snap)
            snap_upsert += len(batch_snap)
        conn.commit()

    batch_raw = []
    batch_snap = []

    for i, r in enumerate(df.itertuples(index=False), start=1):
        raw_item = _to_str(getattr(r, item_col))
        maker = _to_str(getattr(r, maker_col)) if maker_col else ""
        qty = _to_decimal(getattr(r, qty_col))
        unit_price = _to_decimal(getattr(r, unitprice_col)) if unitprice_col else None
        amount = _to_decimal(getattr(r, amount_col)) if amount_col else None

        # raw_incoming
        batch_raw.append((SOURCE_SYSTEM, DEFAULT_RAW_PARTY, raw_item))

        # inventory_snapshot (norm_item은 일단 raw_item)
        batch_snap.append((SOURCE_SYSTEM, SNAPSHOT_TAG, raw_item, raw_item, maker, qty, unit_price, amount))

        if (i % CHUNK_SIZE) == 0:
            flush(batch_raw, batch_snap)
            batch_raw.clear()
            batch_snap.clear()
            print(f"[PROGRESS] {i}/{total} inserted raw={raw_ins} snap={snap_upsert}")

    # tail
    flush(batch_raw, batch_snap)
    print(f"[DONE] total={total}")
    print(f"OK: raw_incoming inserted={raw_ins} (status=NEW)")
    print(f"OK: inventory_snapshot upserted={snap_upsert} (snapshot_tag={SNAPSHOT_TAG})")


if __name__ == "__main__":
    main()
