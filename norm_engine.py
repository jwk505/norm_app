from pathlib import Path
import os
import re
from dotenv import load_dotenv
import mysql.connector   # ← 이 줄 추가


# === .env 강제 로드 (절대경로) ===
ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=True)

print("ENV FILE:", ENV_PATH, "EXISTS:", ENV_PATH.exists())
print("ENV HOST:", os.getenv("DB_HOST"))
print("ENV USER:", os.getenv("DB_USER"))
print("ENV PASS:", "SET" if os.getenv("DB_PASS") else "EMPTY")

# === 안전장치: env 안 읽히면 즉시 중단 ===
required = ["DB_HOST", "DB_USER", "DB_PASS", "DB_NAME"]
missing = [k for k in required if not os.getenv(k)]
if missing:
    raise RuntimeError(f"Missing env keys: {missing}. Check .env file.")


DB_CFG = {
    "host": os.getenv("MYSQL_HOST", "127.0.0.1"),
    "port": int(os.getenv("MYSQL_PORT", "3306")),
    "user": os.getenv("MYSQL_USER", "root"),
    "password": os.getenv("MYSQL_PASSWORD", ""),
    "database": os.getenv("MYSQL_DB", "normdb"),
    "autocommit": False,
}

# ---------- DB helpers ----------
def get_conn():
    import os
    import mysql.connector

    host = os.getenv("DB_HOST")
    user = os.getenv("DB_USER")
    passwd = os.getenv("DB_PASS")
    db = os.getenv("DB_NAME")
    port = int(os.getenv("DB_PORT", "3306"))

    # 필수값 체크
    missing = [k for k in ["DB_HOST", "DB_USER", "DB_PASS", "DB_NAME"] if not os.getenv(k)]
    if missing:
        raise RuntimeError(f"Missing env keys: {missing} (.env not loaded or wrong key names)")

    # 실제로 무엇으로 접속하는지 로그 (중요)
    print(f"[DB] connecting host={host} port={port} user={user} db={db} pass={'SET' if passwd else 'EMPTY'}")

    return mysql.connector.connect(
        host=host,
        user=user,
        password=passwd,
        database=db,
        port=port,
        autocommit=True,
    )


def fetch_rules(conn, target: str):
    """
    normalize_rule에서 (COMMON + target) 활성 규칙을 sort_order 순으로 읽어옴
    """
    sql = """
    SELECT target, rule_type, pattern, replacement
    FROM normalize_rule
    WHERE is_active = 1
      AND (target = 'COMMON' OR target = %s)
    ORDER BY sort_order ASC, id ASC
    """
    cur = conn.cursor(dictionary=True)
    cur.execute(sql, (target,))
    rows = cur.fetchall()
    cur.close()

    compiled = []
    for r in rows:
        # MySQL에 저장된 정규식은 Python regex로 처리
        compiled.append(
            (r["rule_type"], re.compile(r["pattern"]), r.get("replacement", "") or "")
        )
    return compiled

# ---------- Normalization ----------
def basic_normalize(text: str) -> str:
    if text is None:
        return ""
    s = text.strip().upper()
    s = re.sub(r"\s+", " ", s)
    return s

def apply_rules(text: str, compiled_rules) -> str:
    s = text
    for rule_type, pattern, replacement in compiled_rules:
        if rule_type == "REMOVE":
            s = pattern.sub("", s)
        else:  # REPLACE
            s = pattern.sub(replacement, s)
        s = re.sub(r"\s+", " ", s).strip()
    return s

def normalize(text: str, target: str, rules_cache: dict) -> str:
    s = basic_normalize(text)
    s = apply_rules(s, rules_cache[target])
    return s

# ---------- Mapping (exact match first) ----------
def exact_map_party(conn, norm_party: str):
    sql = """
    SELECT party_id
    FROM party_alias
    WHERE norm_text = %s
    ORDER BY confirmed_at DESC, id DESC
    LIMIT 1
    """
    cur = conn.cursor()
    cur.execute(sql, (norm_party,))
    row = cur.fetchone()
    cur.close()
    return row[0] if row else None

def exact_map_item(conn, norm_item: str):
    sql = """
    SELECT item_id
    FROM item_alias
    WHERE norm_text = %s
    ORDER BY confirmed_at DESC, id DESC
    LIMIT 1
    """
    cur = conn.cursor()
    cur.execute(sql, (norm_item,))
    row = cur.fetchone()
    cur.close()
    return row[0] if row else None

# ---------- Core actions ----------
def ingest(conn, source_system: str, raw_party: str | None, raw_item: str | None, rules_cache: dict):
    """
    raw_incoming에 원문 저장 + norm 생성 + alias 정확매칭 시도 + status 업데이트
    """
    norm_party = normalize(raw_party or "", "PARTY", rules_cache) if raw_party else None
    norm_item  = normalize(raw_item or "", "ITEM", rules_cache) if raw_item else None

    mapped_party_id = exact_map_party(conn, norm_party) if norm_party else None
    mapped_item_id  = exact_map_item(conn, norm_item) if norm_item else None

    if mapped_party_id and mapped_item_id:
        status = "MAPPED"
    else:
        # 하나라도 못 찾으면 사람이 봐야 하므로 NEED_REVIEW (또는 NEW로 둬도 됨)
        status = "NEED_REVIEW"

    sql = """
    INSERT INTO raw_incoming
      (source_system, raw_party, raw_item, norm_party, norm_item,
       mapped_party_id, mapped_item_id, status)
    VALUES
      (%s,%s,%s,%s,%s,%s,%s,%s)
    """
    cur = conn.cursor()
    cur.execute(sql, (
        source_system,
        raw_party,
        raw_item,
        norm_party,
        norm_item,
        mapped_party_id,
        mapped_item_id,
        status
    ))
    raw_id = cur.lastrowid
    cur.close()
    conn.commit()

    return {
        "raw_id": raw_id,
        "norm_party": norm_party,
        "norm_item": norm_item,
        "mapped_party_id": mapped_party_id,
        "mapped_item_id": mapped_item_id,
        "status": status
    }

def list_queue(conn, limit: int = 50):
    """
    미매핑/검토 대상 목록
    """
    sql = """
    SELECT id, source_system, raw_party, raw_item, norm_party, norm_item,
           mapped_party_id, mapped_item_id, status, created_at
    FROM raw_incoming
    WHERE status IN ('NEW','NEED_REVIEW')
    ORDER BY created_at DESC
    LIMIT %s
    """
    cur = conn.cursor(dictionary=True)
    cur.execute(sql, (limit,))
    rows = cur.fetchall()
    cur.close()
    return rows

def confirm_party(conn, raw_id: int, party_id: int, confirmed_by: str = "system"):
    cur = conn.cursor(dictionary=True)

    # raw_incoming 읽기
    cur.execute("SELECT * FROM raw_incoming WHERE id=%s", (raw_id,))
    r = cur.fetchone()
    if not r:
        raise ValueError(f"raw_incoming id={raw_id} not found")

    # 1) party_alias upsert (중복이면 업데이트)
    cur.execute(
        """
        INSERT INTO party_alias
          (raw_text, norm_text, party_id, source_system, confidence, confirmed_by, confirmed_at)
        VALUES
          (%s, %s, %s, %s, 1.000, %s, NOW())
        ON DUPLICATE KEY UPDATE
          norm_text     = VALUES(norm_text),
          party_id      = VALUES(party_id),
          source_system = VALUES(source_system),
          confidence    = GREATEST(confidence, VALUES(confidence)),
          confirmed_by  = VALUES(confirmed_by),
          confirmed_at  = NOW()
        """,
        (r["raw_party"], r["norm_party"], party_id, r["source_system"], confirmed_by),
    )

    # 2) raw_incoming 매핑 업데이트
    cur.execute(
        """
        UPDATE raw_incoming
        SET mapped_party_id=%s
        WHERE id=%s
        """,
        (party_id, raw_id),
    )

    conn.commit()
    cur.close()


def confirm_item(conn, raw_id: int, item_id: int, confirmed_by: str | None = None):
    """
    사람이 확정: item_alias에 저장 + raw_incoming 업데이트
    """
    cur = conn.cursor(dictionary=True)
    cur.execute("""
        SELECT source_system, raw_item, norm_item, mapped_party_id
        FROM raw_incoming WHERE id = %s
    """, (raw_id,))
    r = cur.fetchone()
    cur.close()
    if not r:
        raise ValueError("raw_id not found")
    if not r["raw_item"] or not r["norm_item"]:
        raise ValueError("raw_item/norm_item empty")

    cur = conn.cursor()
    cur.execute("""
        INSERT INTO item_alias (raw_text, norm_text, item_id, source_system, confidence, confirmed_by)
        VALUES (%s,%s,%s,%s,1.000,%s)
    """, (r["raw_item"], r["norm_item"], item_id, r["source_system"], confirmed_by))

    new_status = "MAPPED" if r["mapped_party_id"] is not None else "NEED_REVIEW"
    cur.execute("""
        UPDATE raw_incoming
        SET mapped_item_id = %s, status = %s
        WHERE id = %s
    """, (item_id, new_status, raw_id))
    cur.close()
    conn.commit()

def reprocess_queue(conn, limit: int = 500):
    """
    NEED_REVIEW 중 norm_*가 있는 행들을 alias로 다시 매핑해서 MAPPED 처리
    """
    cur = conn.cursor(dictionary=True)
    cur.execute("""
        SELECT id, norm_party, norm_item
        FROM raw_incoming
        WHERE status='NEED_REVIEW'
          AND norm_party IS NOT NULL AND norm_party <> ''
          AND norm_item  IS NOT NULL AND norm_item  <> ''
        ORDER BY created_at ASC
        LIMIT %s
    """, (limit,))
    rows = cur.fetchall()
    cur.close()

    updated = 0
    for r in rows:
        raw_id = r["id"]
        mp = exact_map_party(conn, r["norm_party"])
        mi = exact_map_item(conn, r["norm_item"])

        if mp and mi:
            cur2 = conn.cursor()
            cur2.execute("""
                UPDATE raw_incoming
                SET mapped_party_id=%s, mapped_item_id=%s, status='MAPPED'
                WHERE id=%s
            """, (mp, mi, raw_id))
            cur2.close()
            updated += 1

    conn.commit()
    return updated

def process_new(conn, rules_cache: dict, limit: int = 1000):
    """
    raw_incoming.status='NEW'를 가져와 정규화(norm_*) + alias 기반 자동매핑 후
    MAPPED 또는 NEED_REVIEW로 업데이트
    """
    cur = conn.cursor(dictionary=True)
    cur.execute("""
        SELECT id, source_system, raw_party, raw_item
        FROM raw_incoming
        WHERE status='NEW'
        ORDER BY id ASC
        LIMIT %s
    """, (limit,))
    rows = cur.fetchall()
    cur.close()

    mapped = 0
    need = 0

    for r in rows:
        raw_id = r["id"]
        src = r["source_system"]
        raw_party = r["raw_party"] or ""
        raw_item = r["raw_item"] or ""

        norm_party = normalize(raw_party, "PARTY", rules_cache) if raw_party else ""
        norm_item  = normalize(raw_item,  "ITEM",  rules_cache) if raw_item else ""

        mp = exact_map_party(conn, norm_party) if norm_party else None
        mi = exact_map_item(conn, norm_item)   if norm_item else None

        status = "MAPPED" if mi is not None else "NEED_REVIEW"
        # 재고장 데이터는 party가 없으니 item만 기준으로 매핑해도 OK
        cur2 = conn.cursor()
        cur2.execute("""
            UPDATE raw_incoming
            SET norm_party=%s, norm_item=%s,
                mapped_party_id=%s, mapped_item_id=%s,
                status=%s
            WHERE id=%s
        """, (norm_party, norm_item, mp, mi, status, raw_id))
        cur2.close()

        if status == "MAPPED":
            mapped += 1
        else:
            need += 1

    conn.commit()
    return {"processed": len(rows), "mapped": mapped, "need_review": need}


def renormalize_pending(conn, rules_cache: dict, limit: int = 500):
    """
    NEED_REVIEW 중 norm_party/norm_item이 비었거나 구형인 것들을 다시 정규화
    """
    cur = conn.cursor(dictionary=True)
    cur.execute("""
        SELECT id, raw_party, raw_item, norm_party, norm_item
        FROM raw_incoming
        WHERE status='NEED_REVIEW'
        ORDER BY created_at ASC
        LIMIT %s
    """, (limit,))
    rows = cur.fetchall()
    cur.close()

    updated = 0
    for r in rows:
        np = normalize(r["raw_party"] or "", "PARTY", rules_cache) if r["raw_party"] else None
        ni = normalize(r["raw_item"]  or "", "ITEM",  rules_cache) if r["raw_item"]  else None

        # 바뀐 경우에만 업데이트(불필요 UPDATE 줄이기)
        if (np or "") != (r["norm_party"] or "") or (ni or "") != (r["norm_item"] or ""):
            cur2 = conn.cursor()
            cur2.execute("""
                UPDATE raw_incoming
                SET norm_party=%s, norm_item=%s
                WHERE id=%s
            """, (np, ni, r["id"]))
            cur2.close()
            updated += 1

    conn.commit()
    return updated

# ---------- Demo runner ----------
def main():
    conn = get_conn()
    try:
        # 규칙 캐시 (속도/일관성)
        rules_cache = {
            "PARTY": fetch_rules(conn, "PARTY"),
            "ITEM": fetch_rules(conn, "ITEM"),
        }

        # 1) 샘플 ingest (원문 들어왔다고 가정)
        r1 = ingest(conn, "ERP1", "(주) 서원베어링", "6205 2RS", rules_cache)
        print("[INGEST]", r1)

        ren = renormalize_pending(conn, rules_cache, limit=500)
        print(f"[RENORMALIZE] updated={ren}")

        updated = reprocess_queue(conn, limit=500)
        print(f"[REPROCESS] updated={updated}")


        # NEW 배치 처리 (9167건이므로 여러 번 돌려도 됨)
        res = process_new(conn, rules_cache, limit=20000)
        print("[PROCESS_NEW]", res)

        updated = reprocess_queue(conn, limit=5000)
        print(f"[REPROCESS] updated={updated}")

        q = list_queue(conn, limit=20)
        print("\n[QUEUE]")
        for row in q:
            print(row)

        # 2) 큐 보기
        q = list_queue(conn, limit=20)
        print("\n[QUEUE]")
        for row in q:
            print(row)

        # 3) 사람이 확정(예시)
        # raw_id를 실제 출력된 값으로 바꾸고 실행해도 됨
         #confirm_party(conn, raw_id=r1["raw_id"], party_id=1, confirmed_by="jaewoo")
         #confirm_item(conn, raw_id=r1["raw_id"], item_id=1, confirmed_by="jaewoo")

        # ---- 임시: raw_id=4 확정 (한 번만 실행) ----
        if r1["raw_id"] == 4:
            confirm_party(conn, raw_id=4, party_id=1, confirmed_by="JK")
            confirm_item(conn, raw_id=4, item_id=1, confirmed_by="JK")




    finally:
        conn.close()

if __name__ == "__main__":
    main()
