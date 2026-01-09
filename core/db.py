import pandas as pd
import mysql.connector
from typing import Tuple, List

from core.config import (
    DB_HOST, DB_USER, DB_PASS, DB_NAME, DB_PORT
)

def get_conn():
    if not DB_USER or not DB_NAME:
        raise RuntimeError("DB 설정이 올바르지 않습니다. (.env 확인)")
    return mysql.connector.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASS,
        database=DB_NAME,
        port=DB_PORT,
    )

def query_df(sql: str, params: Tuple = ()) -> pd.DataFrame:
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(sql, params)
        cols = [c[0] for c in cur.description] if cur.description else []
        rows = cur.fetchall() if cur.description else []
        return pd.DataFrame(rows, columns=cols)
    finally:
        conn.close()

def exec_sql(sql: str, params: Tuple = ()) -> int:
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(sql, params)
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()

def exec_many(sql: str, rows: List[Tuple], batch_size: int = 1000) -> int:
    if not rows:
        return 0
    if batch_size <= 0:
        batch_size = len(rows)
    conn = get_conn()
    try:
        cur = conn.cursor()
        total = 0
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            cur.executemany(sql, batch)
            conn.commit()
            total += len(batch)
        return total
    finally:
        conn.close()
