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
