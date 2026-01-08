# features/upload.py
from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd
import streamlit as st

from core.db import exec_many, exec_sql, query_df
from core.schema import get_columns
from features.sales import (
    load_sales_all_sheets,
    detect_amount_col,
    detect_year_col,
    normalize_str_series,
    parse_year,
    pick_customer_col_for_sheet,
)


def _table_exists(table_name: str) -> bool:
    df = query_df(
        """
        SELECT COUNT(*) AS c
        FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME=%s
        """,
        (table_name,),
    )
    return (not df.empty) and int(df.iloc[0]["c"]) > 0


def _norm_key(x: str) -> str:
    return str(x).strip().replace(" ", "").replace("\n", "").replace("\r", "").upper()


def _detect_header_row(excel_source, sheet_name: str, scan_max_rows: int = 80) -> int:
    preview = pd.read_excel(
        excel_source, sheet_name=sheet_name, header=None, nrows=scan_max_rows, engine="openpyxl"
    )
    tokens = ["품목", "품명", "재고", "수량", "금액", "단가", "MAKER", "BRAND", "ITEM", "QTY"]
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

    return best_row


def _find_col(df: pd.DataFrame, candidates: list[str], required: bool = True) -> Optional[str]:
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
        raise RuntimeError(f"컬럼을 찾지 못했습니다. 후보={candidates}, 실제컬럼={cols}")
    return None


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


def _show_sales_upload():
    st.subheader("매출 업로드 (.xlsx)")
    needed = ["sales_raw", "customer_master", "customer_alias"]
    missing = [t for t in needed if not _table_exists(t)]
    if missing:
        st.error(f"DB에 필요한 테이블이 없습니다: {missing}")
        return

    up = st.file_uploader("매출 엑셀 업로드", type=["xlsx"], key="sales_upload")
    if up is None:
        st.info("엑셀 파일을 업로드하면 미리보기와 적재 옵션이 표시됩니다.")
        return

    raw = load_sales_all_sheets(up)
    year_col = detect_year_col(raw)
    amount_col = detect_amount_col(raw)
    if not year_col:
        st.error("엑셀에서 '년도/연도' 컬럼을 찾지 못했습니다.")
        return
    if not amount_col:
        st.error("엑셀에서 금액 컬럼을 찾지 못했습니다.")
        return

    delete_before = st.checkbox("적재 전 sales_raw 비우기(DELETE)", value=False, key="sales_delete")
    only_2020_2025 = st.checkbox("2020~2025만 적재", value=True, key="sales_range")

    df = raw.copy()
    df["year"] = df[year_col].apply(parse_year)
    df["amount"] = pd.to_numeric(df[amount_col], errors="coerce")
    df["customer_col"] = None
    df["customer_raw"] = pd.NA

    for idx, r in df.iterrows():
        sheet = str(r["_sheet"])
        cust_col = pick_customer_col_for_sheet(df, sheet)
        df.at[idx, "customer_col"] = cust_col or ""
        if cust_col and cust_col in df.columns:
            df.at[idx, "customer_raw"] = r.get(cust_col)

    df["customer_raw"] = normalize_str_series(df["customer_raw"])
    df = df[df["year"].notna() & df["amount"].notna() & df["customer_raw"].notna()]
    df["year"] = df["year"].astype(int)

    if only_2020_2025:
        df = df[df["year"].between(2020, 2025, inclusive="both")]

    st.caption("적재 데이터 미리보기 (상위 200행)")
    st.dataframe(
        df[["_sheet", "year", "customer_raw", "amount", "customer_col"]].head(200),
        use_container_width=True,
        hide_index=True,
    )

    k1, k2, k3 = st.columns(3)
    k1.metric("적재 건수", f"{len(df):,}")
    k2.metric("거래처 수", f"{df['customer_raw'].nunique():,}")
    k3.metric("금액 합계", f"{int(df['amount'].sum()):,}")

    if st.button("DB 적재 실행", key="sales_upload_btn"):
        if delete_before:
            deleted = exec_sql("DELETE FROM sales_raw")
            st.info(f"sales_raw 삭제: {deleted:,} rows")

        insert_sql = """
        INSERT INTO sales_raw (src_file, sheet_name, year, customer_raw, amount, customer_col)
        VALUES (%s, %s, %s, %s, %s, %s)
        """

        rows = []
        src_file_label = getattr(up, "name", "uploaded.xlsx")
        for _, r in df.iterrows():
            rows.append(
                (
                    src_file_label,
                    str(r["_sheet"]),
                    int(r["year"]),
                    str(r["customer_raw"]),
                    float(r["amount"]),
                    str(r["customer_col"]),
                )
            )

        inserted = exec_many(insert_sql, rows)
        st.success(f"sales_raw 적재 완료: {inserted:,} rows")

        exec_sql(
            """
            INSERT INTO customer_alias (alias_name, src_hint)
            SELECT DISTINCT customer_raw, customer_col
            FROM sales_raw
            WHERE customer_raw IS NOT NULL AND TRIM(customer_raw) <> ''
            ON DUPLICATE KEY UPDATE src_hint = VALUES(src_hint)
            """
        )
        st.info("customer_alias 자동 수집 완료.")

        st.cache_data.clear()
        st.rerun()


def _show_inventory_upload():
    st.subheader("재고 업로드 (.xlsx)")
    if not _table_exists("inventory_snapshot"):
        st.error("DB에 inventory_snapshot 테이블이 없습니다.")
        return

    up = st.file_uploader("재고 엑셀 업로드", type=["xlsx"], key="inv_upload")
    if up is None:
        st.info("엑셀 파일을 업로드하면 미리보기와 적재 옵션이 표시됩니다.")
        return

    try:
        xls = pd.ExcelFile(up, engine="openpyxl")
        sheet_name = st.selectbox("시트 선택", xls.sheet_names, index=0, key="inv_sheet")
    except Exception as e:
        st.error(f"엑셀 시트 정보를 읽지 못했습니다: {e}")
        return

    scan_max = st.number_input("헤더 탐색 행 수", min_value=10, max_value=200, value=80, step=10, key="inv_scan")
    source_system = st.text_input("source_system", value="INV_SUM", key="inv_source")
    snapshot_tag = st.text_input("snapshot_tag", value=datetime.now().strftime("%Y-%m-%d"), key="inv_snapshot")

    try:
        header_row = _detect_header_row(up, sheet_name, int(scan_max))
        df = pd.read_excel(up, sheet_name=sheet_name, header=header_row, engine="openpyxl")
    except Exception as e:
        st.error(f"엑셀을 읽지 못했습니다: {e}")
        return

    df.columns = [str(c).strip() if pd.notna(c) else "" for c in df.columns]

    try:
        item_col = _find_col(df, ["품목명", "품명", "품목", "상품명", "ITEM"], required=True)
        qty_col = _find_col(df, ["수량", "재고수량", "QTY", "Qty"], required=True)
        maker_col = _find_col(df, ["메이커", "maker", "제조사", "브랜드", "BRAND"], required=False)
        amount_col = _find_col(df, ["금액", "재고금액", "AMOUNT"], required=False)
        unitprice_col = _find_col(df, ["단가", "평균단가", "UNITPRICE", "UNIT_PRICE"], required=False)
    except RuntimeError as e:
        st.error(str(e))
        return

    df["_raw_item"] = df[item_col].astype("string").str.strip()
    df = df[df["_raw_item"] != ""].copy()

    st.caption("적재 데이터 미리보기 (상위 200행)")
    preview_cols = [c for c in [item_col, maker_col, qty_col, unitprice_col, amount_col] if c]
    st.dataframe(df[preview_cols].head(200), use_container_width=True, hide_index=True)

    k1, k2, k3 = st.columns(3)
    k1.metric("적재 건수", f"{len(df):,}")
    k2.metric("품목 수", f"{df['_raw_item'].nunique():,}")
    total_amt = df[amount_col].apply(_to_decimal).dropna().sum() if amount_col else 0
    k3.metric("금액 합계", f"{int(total_amt):,}")

    if st.button("DB 적재 실행", key="inv_upload_btn"):
        snap_cols = get_columns("inventory_snapshot")
        required = {"source_system", "snapshot_tag", "raw_item", "norm_item"}
        if not required.issubset(snap_cols):
            st.error("inventory_snapshot 테이블 컬럼이 부족합니다.")
            return

        insert_cols = ["source_system", "snapshot_tag", "raw_item", "norm_item"]
        update_cols = []
        if "maker" in snap_cols:
            insert_cols.append("maker")
            update_cols.append("maker")
        if "qty" in snap_cols:
            insert_cols.append("qty")
            update_cols.append("qty")
        if "unit_price" in snap_cols:
            insert_cols.append("unit_price")
            update_cols.append("unit_price")
        if "amount" in snap_cols:
            insert_cols.append("amount")
            update_cols.append("amount")
        if "created_at" in snap_cols:
            insert_cols.append("created_at")

        value_exprs = []
        for c in insert_cols:
            if c == "created_at":
                value_exprs.append("NOW()")
            else:
                value_exprs.append("%s")

        update_sql = ""
        if update_cols:
            update_sql = " ON DUPLICATE KEY UPDATE " + ", ".join([f"{c}=VALUES({c})" for c in update_cols])

        insert_sql = (
            f"INSERT INTO inventory_snapshot ({', '.join(insert_cols)}) "
            f"VALUES ({', '.join(value_exprs)})" + update_sql
        )

        rows = []
        for _, r in df.iterrows():
            raw_item = str(r[item_col]).strip()
            if not raw_item:
                continue
            row = [source_system, snapshot_tag, raw_item, raw_item]
            if "maker" in snap_cols:
                row.append(str(r[maker_col]).strip() if maker_col else "")
            if "qty" in snap_cols:
                row.append(_to_decimal(r[qty_col]))
            if "unit_price" in snap_cols:
                row.append(_to_decimal(r[unitprice_col]) if unitprice_col else None)
            if "amount" in snap_cols:
                row.append(_to_decimal(r[amount_col]) if amount_col else None)
            if "created_at" in snap_cols:
                pass
            rows.append(tuple(row))

        inserted = exec_many(insert_sql, rows)
        st.success(f"inventory_snapshot 적재 완료: {inserted:,} rows")

        if _table_exists("raw_incoming"):
            raw_cols = get_columns("raw_incoming")
            need_cols = {"source_system", "raw_item", "status", "created_at"}
            if need_cols.issubset(raw_cols):
                raw_insert_cols = ["source_system", "raw_item", "status", "created_at"]
                raw_values = ["%s", "%s", "'NEW'", "NOW()"]
                if "raw_party" in raw_cols:
                    raw_insert_cols.insert(1, "raw_party")
                    raw_values.insert(1, "%s")

                raw_sql = (
                    f"INSERT INTO raw_incoming ({', '.join(raw_insert_cols)}) "
                    f"VALUES ({', '.join(raw_values)})"
                )

                raw_rows = []
                for _, r in df.iterrows():
                    raw_item = str(r[item_col]).strip()
                    if not raw_item:
                        continue
                    if "raw_party" in raw_cols:
                        raw_rows.append((source_system, None, raw_item))
                    else:
                        raw_rows.append((source_system, raw_item))

                if raw_rows:
                    exec_many(raw_sql, raw_rows)
                    st.info("raw_incoming 적재 완료.")

        st.cache_data.clear()
        st.rerun()


def show_upload_page():
    st.header("데이터 업로드")
    tab1, tab2, tab3 = st.tabs(["매출", "재고", "재무제표"])

    with tab1:
        _show_sales_upload()

    with tab2:
        _show_inventory_upload()

    with tab3:
        st.subheader("재무제표 업로드 (.xlsx)")
        st.info("재무제표 업로드용 DB 테이블/포맷 확정이 필요합니다. 스키마를 알려주시면 연결하겠습니다.")
