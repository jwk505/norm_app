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
    detect_corp_col,
    find_sales_raw_corp_col,
    find_sales_raw_customer_cols,
    compute_row_hash,
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
        st.info("엑셀 파일을 업로드하면 적재 옵션이 표시됩니다.")
        return

    delete_before = st.checkbox("Delete sales_raw before load (DELETE)", value=False, key="sales_delete")
    only_2020_2025 = st.checkbox("Only load 2020-2025", value=True, key="sales_range")

    if st.button("Load into DB", key="sales_upload_btn"):
        status = st.status("엑셀 로딩 및 전처리 중...", expanded=True)
        with st.spinner("Loading Excel and preprocessing..."):
            raw = load_sales_all_sheets(up)
            year_col = detect_year_col(raw)
            amount_col = detect_amount_col(raw)
            if not year_col:
                st.error("Could not find a year column in the Excel file.")
                return
            if not amount_col:
                st.error("Could not find an amount column in the Excel file.")
                return

            df = raw.copy()
            df["year"] = df[year_col].apply(parse_year)
            df["amount"] = pd.to_numeric(df[amount_col], errors="coerce")
            df["customer_col"] = ""
            df["customer_raw"] = pd.NA
            df["corp"] = pd.NA

            corp_col = detect_corp_col(raw)
            if corp_col and corp_col in df.columns:
                df["corp"] = df[corp_col]

            sheet_to_cust = {}
            for sheet in df["_sheet"].astype(str).unique().tolist():
                cust_col = pick_customer_col_for_sheet(df, sheet)
                sheet_to_cust[sheet] = cust_col or ""

            for sheet, cust_col in sheet_to_cust.items():
                mask = df["_sheet"].astype(str) == sheet
                df.loc[mask, "customer_col"] = cust_col or ""
                if cust_col and cust_col in df.columns:
                    df.loc[mask, "customer_raw"] = df.loc[mask, cust_col]

            df["customer_raw"] = normalize_str_series(df["customer_raw"])
            df["corp"] = normalize_str_series(df["corp"])
            df = df[df["year"].notna() & df["amount"].notna() & df["customer_raw"].notna()]
            df["year"] = df["year"].astype(int)

            if only_2020_2025:
                df = df[df["year"].between(2020, 2025, inclusive="both")]
        status.write(f"전처리 완료: {len(df):,} rows")
        if delete_before:
            status.write("기존 데이터 삭제 중...")
            deleted = exec_sql("DELETE FROM sales_raw")
            status.write(f"sales_raw 삭제: {deleted:,} rows")
        else:
            status.write("기존 데이터 삭제: 건너뜀")

        sales_cols = get_columns("sales_raw")
        use_row_hash = "row_hash" in sales_cols
        corp_db_col = find_sales_raw_corp_col(sales_cols)
        customer_cols = find_sales_raw_customer_cols(sales_cols)
        if not customer_cols:
            status.update(label="고객 컬럼 없음", state="error", expanded=True)
            st.error("sales_raw에 고객 컬럼(customer_raw/customer_cd)이 없습니다.")
            return
        customer_alias_col = "customer_raw" if "customer_raw" in sales_cols else customer_cols[0]
        if corp_col and not corp_db_col:
            status.write("주의: sales_raw에 구분 컬럼이 없어 엑셀 구분 값을 저장하지 못합니다.")

        insert_cols = []
        if use_row_hash:
            insert_cols.append("row_hash")
        insert_cols += ["src_file", "sheet_name", "year"]
        insert_cols += customer_cols
        insert_cols += ["amount", "customer_col"]
        if corp_db_col:
            insert_cols.append(corp_db_col)

        update_sql = ""
        if use_row_hash:
            update_sql = " ON DUPLICATE KEY UPDATE row_hash = row_hash"
            if corp_db_col:
                update_sql = update_sql + f", {corp_db_col}=VALUES({corp_db_col})"

        placeholders = ", ".join(["%s"] * len(insert_cols))
        insert_sql = f"INSERT INTO sales_raw ({', '.join(insert_cols)}) VALUES ({placeholders}){update_sql}"

        rows = []
        src_file_label = getattr(up, "name", "uploaded.xlsx")
        for _, r in df.iterrows():
            customer_val = str(r["customer_raw"])
            base_hash = (
                src_file_label,
                str(r["_sheet"]),
                int(r["year"]),
                customer_val,
                float(r["amount"]),
                str(r["customer_col"]),
            )

            row = []
            if use_row_hash:
                row.append(compute_row_hash(*base_hash))
            row.extend([src_file_label, str(r["_sheet"]), int(r["year"])])
            row.extend([customer_val] * len(customer_cols))
            row.append(float(r["amount"]))
            row.append(str(r["customer_col"]))
            if corp_db_col:
                corp_val = r["corp"]
                corp_val = None if pd.isna(corp_val) else str(corp_val)
                row.append(corp_val)
            rows.append(tuple(row))

        progress = st.progress(0, text="DB 적재 준비...")
        status_text = st.empty()

        def _fmt_time(seconds: float | None) -> str:
            if seconds is None:
                return "계산중"
            seconds = max(0, int(seconds))
            m, s = divmod(seconds, 60)
            h, m = divmod(m, 60)
            if h > 0:
                return f"{h}h {m:02d}m"
            return f"{m}m {s:02d}s"

        def _progress_cb(done: int, total: int, elapsed: float) -> None:
            frac = (done / total) if total else 1.0
            remaining = (elapsed / done) * (total - done) if done else None
            progress.progress(min(frac, 1.0), text=f"DB 적재 {done:,}/{total:,} ({frac*100:.1f}%)")
            status_text.write(f"경과 {_fmt_time(elapsed)} | 남은시간 {_fmt_time(remaining)}")

        inserted = exec_many(insert_sql, rows, progress_cb=_progress_cb)
        progress.progress(1.0, text="DB 적재 완료")
        st.success(f"sales_raw 적재 완료: {inserted:,} rows")
        status.update(label="DB 적재 완료", state="complete", expanded=False)

        exec_sql(
            f"""
            INSERT INTO customer_alias (alias_name, src_hint)
            SELECT DISTINCT {customer_alias_col}, customer_col
            FROM sales_raw
            WHERE {customer_alias_col} IS NOT NULL AND TRIM({customer_alias_col}) <> ''
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
