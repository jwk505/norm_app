# core/schema.py
from core.db import query_df

def get_columns(table_name: str, db_name: str | None = None) -> set[str]:
    """
    INFORMATION_SCHEMA에서 컬럼 목록 조회
    db_name이 None이면 현재 연결 DB 스키마 사용(대부분 core.db에서 DB_NAME 사용)
    """
    if db_name:
        df = query_df(
            """
            SELECT COLUMN_NAME
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA=%s AND TABLE_NAME=%s
            """,
            (db_name, table_name),
        )
    else:
        # DB_NAME을 모르니, current database() 사용
        df = query_df(
            """
            SELECT COLUMN_NAME
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME=%s
            """,
            (table_name,),
        )

    if df is None or df.empty:
        return set()
    return set(df["COLUMN_NAME"].astype(str).tolist())
