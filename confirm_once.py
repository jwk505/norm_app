import os
import mysql.connector
from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).with_name(".env")
load_dotenv(dotenv_path=env_path)

def clean_env(v):
    return (v or "").strip().strip('"').strip("'")

DB_CFG = {
    "host": clean_env(os.getenv("MYSQL_HOST", "127.0.0.1")),
    "port": int(clean_env(os.getenv("MYSQL_PORT", "3306"))),
    "user": clean_env(os.getenv("MYSQL_USER", "root")),
    "password": clean_env(os.getenv("MYSQL_PASSWORD", "")),
    "database": clean_env(os.getenv("MYSQL_DB", "normdb")),
}

RAW_ID = 4          # 확정할 raw_incoming id
PARTY_ID = 1        # 서원베어링 party_master.id 로 바꿔
ITEM_ID = 1         # 6205-2RS item_master.id 로 바꿔
CONFIRMED_BY = "JK"

conn = mysql.connector.connect(**DB_CFG)
conn.autocommit = False
try:
    cur = conn.cursor(dictionary=True)
    cur.execute("""
        SELECT source_system, raw_party, norm_party
        FROM raw_incoming WHERE id=%s
    """, (RAW_ID,))
    r = cur.fetchone()
    if not r:
        raise RuntimeError("raw_id not found")
    if not r["norm_party"]:
        raise RuntimeError("norm_party empty (normalize 먼저)")

    cur.execute("""
        INSERT INTO party_alias (raw_text, norm_text, party_id, source_system, confidence, confirmed_by)
        VALUES (%s,%s,%s,%s,1.000,%s)
    """, (r["raw_party"], r["norm_party"], PARTY_ID, r["source_system"], CONFIRMED_BY))

    cur.execute("""
        SELECT source_system, raw_item, norm_item
        FROM raw_incoming WHERE id=%s
    """, (RAW_ID,))
    r2 = cur.fetchone()
    if not r2:
        raise RuntimeError("raw_id not found")
    if not r2["norm_item"]:
        raise RuntimeError("norm_item empty (normalize 먼저)")

    cur.execute("""
        INSERT INTO item_alias (raw_text, norm_text, item_id, source_system, confidence, confirmed_by)
        VALUES (%s,%s,%s,%s,1.000,%s)
    """, (r2["raw_item"], r2["norm_item"], ITEM_ID, r2["source_system"], CONFIRMED_BY))

    # raw_incoming도 매핑 완료 표시
    cur.execute("""
        UPDATE raw_incoming
        SET mapped_party_id=%s, mapped_item_id=%s, status='MAPPED'
        WHERE id=%s
    """, (PARTY_ID, ITEM_ID, RAW_ID))

    conn.commit()
    print("OK: confirmed RAW_ID =", RAW_ID)

finally:
    conn.close()
