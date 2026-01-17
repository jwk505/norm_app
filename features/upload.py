# features/upload.py
from __future__ import annotations

from datetime import datetime
import io
import os
import re
from typing import Optional

import pandas as pd
import streamlit as st

from core.db import exec_many, exec_sql, query_df
from core.schema import get_columns
from features.finance import ensure_finance_tables
from features.sales import (
    load_sales_all_sheets,
    detect_amount_col,
    detect_year_col,
    detect_corp_col,
    detect_channel_col,
    detect_item_code_col,
    detect_item_name_col,
    detect_maker_col,
    detect_unit_price_col,
    find_sales_raw_corp_col,
    find_sales_raw_channel_col,
    find_sales_raw_customer_cols,
    pick_sales_raw_item_code_col,
    pick_sales_raw_item_name_col,
    pick_sales_raw_maker_col,
    pick_sales_raw_channel_col,
    pick_sales_raw_unit_price_col,
    compute_row_hash,
    ensure_sales_raw_optional_columns,
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


def _otsu_threshold(img) -> int:
    hist = img.histogram()
    total = sum(hist)
    sum_total = 0
    for i in range(256):
        sum_total += i * hist[i]
    sum_b = 0
    w_b = 0
    max_var = 0.0
    threshold = 160
    for i in range(256):
        w_b += hist[i]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += i * hist[i]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        var_between = w_b * w_f * (m_b - m_f) ** 2
        if var_between > max_var:
            max_var = var_between
            threshold = i
    return threshold


def _ensure_tesseract_cmd():
    candidates = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                import pytesseract

                pytesseract.pytesseract.tesseract_cmd = p
            except Exception:
                pass
            break
    user_tessdata = os.path.join(os.path.expanduser("~"), ".tesseract", "tessdata")
    if os.path.isdir(user_tessdata):
        os.environ["TESSDATA_PREFIX"] = user_tessdata


def _ocr_score(text: str) -> float:
    if not text:
        return 0.0
    good = len(re.findall(r"[가-힣A-Za-z0-9]", text))
    return good / max(1, len(text))


def _map_statement_type(val: str) -> str:
    s = str(val).strip().lower()
    if "손익" in s or s in ("is", "pl", "p/l"):
        return "IS"
    if "대차" in s or "재무상태" in s or s in ("bs", "b/s"):
        return "BS"
    if "현금" in s or "cash" in s or s in ("cf", "c/f"):
        return "CF"
    return str(val).strip() or "UNKNOWN"


def _normalize_period(val) -> str | None:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    if isinstance(val, (datetime, pd.Timestamp)):
        return val.strftime("%Y-%m")
    s = str(val).strip()
    if not s:
        return None
    m = re.search(r"(20\d{2})(?:\D{0,3}(\d{1,2}))?", s)
    if m:
        y = m.group(1)
        mm = m.group(2)
        if mm:
            return f"{y}-{int(mm):02d}"
        return y
    if s.isdigit():
        return s
    return s


def _ingest_financial_rows(load_df: pd.DataFrame, source_file: str, sheet_name: str, delete_before: bool) -> None:
    if load_df.empty:
        st.warning("적재 대상 데이터가 없습니다.")
        return

    corp_names = load_df["_corp"].dropna().astype(str).unique().tolist()
    exec_many(
        """
        INSERT INTO financial_corp (corp_name)
        VALUES (%s)
        ON DUPLICATE KEY UPDATE corp_name=corp_name
        """,
        [(c,) for c in corp_names],
    )

    corp_map = query_df(
        "SELECT id, corp_name FROM financial_corp WHERE corp_name IN ({})".format(
            ",".join(["%s"] * len(corp_names))
        ),
        tuple(corp_names),
    )
    corp_id_map = {r.corp_name: int(r.id) for r in corp_map.itertuples(index=False)}

    if delete_before:
        exec_sql(
            """
            DELETE s
            FROM financial_statement s
            JOIN financial_corp c ON c.id = s.corp_id
            WHERE s.source_file = %s AND s.sheet_name = %s
            """,
            (source_file, sheet_name),
        )

    insert_sql = """
        INSERT INTO financial_statement
        (corp_id, period, statement_type, account_name, amount, source_file, sheet_name)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """
    rows = []
    for _, r in load_df.iterrows():
        corp_id = corp_id_map.get(str(r["_corp"]))
        if not corp_id:
            continue
        rows.append(
            (
                corp_id,
                str(r["_period"]),
                str(r["_stmt"]),
                str(r["_account"]),
                float(r["_amount"]),
                source_file,
                sheet_name,
            )
        )

    inserted = exec_many(insert_sql, rows)
    st.success(f"재무제표 적재 완료: {inserted:,} rows")
    st.cache_data.clear()
    st.rerun()


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
            channel_col = detect_channel_col(raw)
            item_code_col = detect_item_code_col(raw)
            item_name_col = detect_item_name_col(raw)
            maker_col = detect_maker_col(raw)
            unit_price_col = detect_unit_price_col(raw)
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
            df["channel"] = pd.NA
            df["item_code"] = df[item_code_col] if item_code_col else pd.NA
            df["item_name"] = df[item_name_col] if item_name_col else pd.NA
            df["maker"] = df[maker_col] if maker_col else pd.NA
            df["unit_price"] = df[unit_price_col] if unit_price_col else pd.NA

            corp_col = detect_corp_col(raw)
            if corp_col and corp_col in df.columns:
                df["corp"] = df[corp_col]
            if channel_col and channel_col in df.columns:
                df["channel"] = df[channel_col]

            sheet_to_cust = {}
            for sheet in df["_sheet"].astype(str).unique().tolist():
                cust_col = pick_customer_col_for_sheet(df, sheet)
                sheet_to_cust[sheet] = cust_col or ""

            for sheet, cust_col in sheet_to_cust.items():
                mask = df["_sheet"].astype(str) == sheet
                df.loc[mask, "customer_col"] = cust_col or ""
                if cust_col and cust_col in df.columns:
                    df.loc[mask, "customer_raw"] = df.loc[mask, cust_col]

            cust_map = pd.DataFrame(
                [{"sheet": k, "customer_col": v or ""} for k, v in sheet_to_cust.items()]
            )
            st.caption("시트별 고객 컬럼 감지 결과")
            st.dataframe(cust_map, use_container_width=True, hide_index=True)

            df["customer_raw"] = normalize_str_series(df["customer_raw"])
            df["corp"] = normalize_str_series(df["corp"])
            df["channel"] = normalize_str_series(df["channel"])
            df["item_code"] = normalize_str_series(df["item_code"])
            df["item_name"] = normalize_str_series(df["item_name"])
            df["maker"] = normalize_str_series(df["maker"])
            df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce")

            total_rows = len(df)
            mask_year = df["year"].notna()
            mask_amount = df["amount"].notna()
            mask_customer = df["customer_raw"].notna()
            drop_stats = pd.DataFrame(
                [
                    {"항목": "전체", "행수": total_rows},
                    {"항목": "연도 없음", "행수": int((~mask_year).sum())},
                    {"항목": "금액 없음", "행수": int((~mask_amount).sum())},
                    {"항목": "고객 없음", "행수": int((~mask_customer).sum())},
                    {
                        "항목": "적재 대상",
                        "행수": int((mask_year & mask_amount & mask_customer).sum()),
                    },
                ]
            )
            st.caption("전처리 제외 사유별 집계")
            st.dataframe(drop_stats, use_container_width=True, hide_index=True)

            df = df[mask_year & mask_amount & mask_customer]
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
        item_code_db_col = pick_sales_raw_item_code_col(sales_cols)
        item_name_db_col = pick_sales_raw_item_name_col(sales_cols)
        maker_db_col = pick_sales_raw_maker_col(sales_cols)
        unit_price_db_col = pick_sales_raw_unit_price_col(sales_cols)
        channel_db_col = pick_sales_raw_channel_col(sales_cols)
        to_add = {}
        if item_code_col and not item_code_db_col:
            to_add["item_code"] = "VARCHAR(64)"
        if item_name_col and not item_name_db_col:
            to_add["item_name"] = "VARCHAR(255)"
        if maker_col and not maker_db_col:
            to_add["maker"] = "VARCHAR(64)"
        if unit_price_col and not unit_price_db_col:
            to_add["unit_price"] = "DECIMAL(14,2)"
        if channel_col and not channel_db_col:
            to_add["channel"] = "VARCHAR(64)"
        if to_add:
            sales_cols = ensure_sales_raw_optional_columns(to_add)
            item_code_db_col = pick_sales_raw_item_code_col(sales_cols)
            item_name_db_col = pick_sales_raw_item_name_col(sales_cols)
            maker_db_col = pick_sales_raw_maker_col(sales_cols)
            unit_price_db_col = pick_sales_raw_unit_price_col(sales_cols)
            channel_db_col = pick_sales_raw_channel_col(sales_cols)
        use_row_hash = "row_hash" in sales_cols
        corp_db_col = find_sales_raw_corp_col(sales_cols)
        channel_db_col = channel_db_col or find_sales_raw_channel_col(sales_cols)
        customer_cols = find_sales_raw_customer_cols(sales_cols)
        if not customer_cols:
            status.update(label="고객 컬럼 없음", state="error", expanded=True)
            st.error("sales_raw에 고객 컬럼(customer_raw/customer_cd)이 없습니다.")
            return
        customer_alias_col = "customer_raw" if "customer_raw" in sales_cols else customer_cols[0]
        if corp_col and not corp_db_col:
            sales_cols = ensure_sales_raw_optional_columns({"corp": "VARCHAR(64)"})
            corp_db_col = find_sales_raw_corp_col(sales_cols)
        if channel_col and not channel_db_col:
            sales_cols = ensure_sales_raw_optional_columns({"channel": "VARCHAR(64)"})
            channel_db_col = find_sales_raw_channel_col(sales_cols)

        insert_cols = []
        if use_row_hash:
            insert_cols.append("row_hash")
        insert_cols += ["src_file", "sheet_name", "year"]
        insert_cols += customer_cols
        insert_cols += ["amount", "customer_col"]
        if corp_db_col:
            insert_cols.append(corp_db_col)
        if item_code_db_col:
            insert_cols.append(item_code_db_col)
        if item_name_db_col:
            insert_cols.append(item_name_db_col)
        if maker_db_col:
            insert_cols.append(maker_db_col)
        if unit_price_db_col:
            insert_cols.append(unit_price_db_col)
        if channel_db_col:
            insert_cols.append(channel_db_col)

        update_sql = ""
        if use_row_hash:
            update_sql = " ON DUPLICATE KEY UPDATE row_hash = row_hash"
            if corp_db_col:
                update_sql = update_sql + f", {corp_db_col}=VALUES({corp_db_col})"
            if item_code_db_col:
                update_sql = update_sql + f", {item_code_db_col}=VALUES({item_code_db_col})"
            if item_name_db_col:
                update_sql = update_sql + f", {item_name_db_col}=VALUES({item_name_db_col})"
            if maker_db_col:
                update_sql = update_sql + f", {maker_db_col}=VALUES({maker_db_col})"
            if unit_price_db_col:
                update_sql = update_sql + f", {unit_price_db_col}=VALUES({unit_price_db_col})"
            if channel_db_col:
                update_sql = update_sql + f", {channel_db_col}=VALUES({channel_db_col})"

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
            if item_code_db_col:
                row.append(None if pd.isna(r["item_code"]) else str(r["item_code"]))
            if item_name_db_col:
                row.append(None if pd.isna(r["item_name"]) else str(r["item_name"]))
            if maker_db_col:
                row.append(None if pd.isna(r["maker"]) else str(r["maker"]))
            if unit_price_db_col:
                row.append(None if pd.isna(r["unit_price"]) else float(r["unit_price"]))
            if channel_db_col:
                row.append(None if pd.isna(r["channel"]) else str(r["channel"]))
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


def _show_financial_upload():
    st.subheader("재무제표 업로드 (.xlsx)")
    ensure_finance_tables()

    template_df = pd.DataFrame(
        [
            {"법인": "ABC", "기간": "2024-12", "구분": "손익", "계정": "매출액", "금액": 1000000},
            {"법인": "ABC", "기간": "2024-12", "구분": "손익", "계정": "영업이익", "금액": 120000},
            {"법인": "ABC", "기간": "2024-12", "구분": "대차", "계정": "자산총계", "금액": 2500000},
        ]
    )
    buf = io.BytesIO()
    template_df.to_excel(buf, index=False, sheet_name="FINANCE", engine="openpyxl")
    st.download_button(
        "재무제표 업로드 양식 다운로드",
        data=buf.getvalue(),
        file_name="financial_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    up = st.file_uploader("재무제표 엑셀 업로드", type=["xlsx"], key="fin_upload")
    if up is None:
        st.info("엑셀 파일을 업로드하면 적재 옵션이 표시됩니다.")
    else:
        try:
            xls = pd.ExcelFile(up, engine="openpyxl")
            sheet_name = st.selectbox("시트 선택", xls.sheet_names, index=0, key="fin_sheet")
            df = pd.read_excel(up, sheet_name=sheet_name, header=0, engine="openpyxl")
        except Exception as e:
            st.error(f"엑셀을 읽지 못했습니다: {e}")
            df = None
        if df is not None:
            df.columns = [str(c).strip() if pd.notna(c) else "" for c in df.columns]
            try:
                corp_col = _find_col(df, ["법인", "법인명", "회사", "회사명", "기업", "corp", "entity"], required=True)
                period_col = _find_col(df, ["기간", "연도", "년도", "결산기", "기준기간", "period", "year"], required=True)
                stmt_col = _find_col(df, ["구분", "재무제표", "재무제표구분", "statement", "type"], required=True)
                account_col = _find_col(df, ["계정", "계정과목", "항목", "account", "account_name"], required=True)
                amount_col = _find_col(df, ["금액", "amount", "값", "금액(원)"], required=True)
            except RuntimeError as e:
                st.error(str(e))
                corp_col = period_col = stmt_col = account_col = amount_col = None
            if corp_col and period_col and stmt_col and account_col and amount_col:
                df["_corp"] = df[corp_col].astype("string").str.strip()
                df["_period"] = df[period_col].apply(_normalize_period)
                df["_stmt"] = df[stmt_col].apply(_map_statement_type)
                df["_account"] = df[account_col].astype("string").str.strip()
                df["_amount"] = pd.to_numeric(df[amount_col].apply(_to_decimal), errors="coerce")

                st.caption("적재 데이터 미리보기 (상위 200행)")
                st.dataframe(
                    df[[corp_col, period_col, stmt_col, account_col, amount_col]].head(200),
                    use_container_width=True,
                    hide_index=True,
                )

                total_rows = len(df)
                mask_corp = df["_corp"].notna() & (df["_corp"] != "")
                mask_period = df["_period"].notna() & (df["_period"] != "")
                mask_account = df["_account"].notna() & (df["_account"] != "")
                mask_amount = df["_amount"].notna()
                drop_stats = pd.DataFrame(
                    [
                        {"항목": "전체", "행수": total_rows},
                        {"항목": "법인 없음", "행수": int((~mask_corp).sum())},
                        {"항목": "기간 없음", "행수": int((~mask_period).sum())},
                        {"항목": "계정 없음", "행수": int((~mask_account).sum())},
                        {"항목": "금액 없음", "행수": int((~mask_amount).sum())},
                        {
                            "항목": "적재 대상",
                            "행수": int((mask_corp & mask_period & mask_account & mask_amount).sum()),
                        },
                    ]
                )
                st.caption("전처리 제외 사유별 집계")
                st.dataframe(drop_stats, use_container_width=True, hide_index=True)

                delete_before = st.checkbox("해당 시트 데이터 삭제 후 적재", value=False, key="fin_delete")

                if st.button("DB 적재 실행", key="fin_upload_btn"):
                    load_df = df[mask_corp & mask_period & mask_account & mask_amount].copy()
                    _ingest_financial_rows(
                        load_df,
                        source_file=getattr(up, "name", "uploaded.xlsx"),
                        sheet_name=str(sheet_name),
                        delete_before=delete_before,
                    )

    st.divider()
    st.subheader("재무제표 PDF 업로드 (실험)")
    pdf_up = st.file_uploader("재무제표 PDF 업로드", type=["pdf"], key="fin_pdf_upload")
    if pdf_up is None:
        st.info("PDF를 업로드하면 페이지 선택 및 추출 옵션이 표시됩니다.")
        return

    try:
        import pdfplumber
        import pytesseract
    except Exception as e:
        st.error(f"PDF/OCR 라이브러리 로드 실패: {e}")
        return

    _ensure_tesseract_cmd()

    with pdfplumber.open(pdf_up) as pdf_doc:
        page_cnt = len(pdf_doc.pages)
        if page_cnt == 0:
            st.error("PDF 페이지를 읽지 못했습니다.")
            return
        page_no = st.number_input("페이지 선택", min_value=1, max_value=page_cnt, value=1, step=1)
        page = pdf_doc.pages[int(page_no) - 1]

        mode = st.radio("추출 방식", ["표 인식", "OCR"], horizontal=True)

        if mode == "표 인식":
            status = st.status("표 인식 중...", expanded=False)
            table = page.extract_table()
            status.update(label="표 인식 완료", state="complete", expanded=False)
            if not table:
                st.warning("표를 찾지 못했습니다. OCR 모드를 사용해 보세요.")
                return
            header = table[0]
            rows = table[1:]
            pdf_df = pd.DataFrame(rows, columns=header)
            st.caption("추출 미리보기 (상위 200행)")
            st.dataframe(pdf_df.head(200), use_container_width=True, hide_index=True)

            st.caption("컬럼 매핑")
            cols = list(pdf_df.columns)
            corp_sel = st.selectbox("법인 컬럼", cols, index=0, key="pdf_corp_col")
            period_sel = st.selectbox("기간 컬럼", cols, index=0, key="pdf_period_col")
            stmt_sel = st.selectbox("구분 컬럼", cols, index=0, key="pdf_stmt_col")
            account_sel = st.selectbox("계정 컬럼", cols, index=0, key="pdf_account_col")
            amount_sel = st.selectbox("금액 컬럼", cols, index=0, key="pdf_amount_col")

            delete_before_pdf = st.checkbox("해당 페이지 데이터 삭제 후 적재", value=False, key="fin_pdf_delete")
            if st.button("PDF 적재 실행", key="fin_pdf_load"):
                load_df = pd.DataFrame()
                load_df["_corp"] = pdf_df[corp_sel].astype("string").str.strip()
                load_df["_period"] = pdf_df[period_sel].apply(_normalize_period)
                load_df["_stmt"] = pdf_df[stmt_sel].apply(_map_statement_type)
                load_df["_account"] = pdf_df[account_sel].astype("string").str.strip()
                load_df["_amount"] = pd.to_numeric(pdf_df[amount_sel].apply(_to_decimal), errors="coerce")
                load_df = load_df[
                    load_df["_corp"].notna()
                    & (load_df["_corp"] != "")
                    & load_df["_period"].notna()
                    & (load_df["_period"] != "")
                    & load_df["_account"].notna()
                    & (load_df["_account"] != "")
                    & load_df["_amount"].notna()
                ]
                _ingest_financial_rows(
                    load_df,
                    source_file=getattr(pdf_up, "name", "uploaded.pdf"),
                    sheet_name=f"page_{page_no}",
                    delete_before=delete_before_pdf,
                )
        else:
            status = st.status("OCR 처리 중...", expanded=False)
            image = page.to_image(resolution=350).original
            try:
                from PIL import Image, ImageEnhance, ImageFilter

                gray = image.convert("L")
                gray = ImageEnhance.Contrast(gray).enhance(2.0)
                gray = gray.filter(ImageFilter.SHARPEN)
                use_auto_thr = st.checkbox("자동 임계값(OTSU)", value=True, key="fin_ocr_auto_thr")
                thr = _otsu_threshold(gray) if use_auto_thr else st.slider(
                    "이진화 임계값", min_value=80, max_value=220, value=160, step=5, key="fin_ocr_thr"
                )
                bw = gray.point(lambda x: 0 if x < thr else 255, "1")
                tessdata_dir = os.path.join(os.path.expanduser("~"), ".tesseract", "tessdata")
                tessdata_arg = tessdata_dir.replace("\\", "/")
                psm_pick = st.selectbox("OCR 모드(PSM)", ["자동", "4", "6", "11"], index=0, key="fin_ocr_psm")
                psm_list = ["4", "6", "11"] if psm_pick == "자동" else [psm_pick]
                best_text = ""
                best_score = -1.0
                for psm in psm_list:
                    status.update(label=f"OCR 인식 중... (PSM {psm})", state="running", expanded=False)
                    text = pytesseract.image_to_string(
                        bw,
                        lang="kor+eng",
                        config=f"--oem 3 --psm {psm} --tessdata-dir {tessdata_arg}",
                    )
                    score = _ocr_score(text)
                    if score > best_score:
                        best_score = score
                        best_text = text
                ocr_text = best_text
                status.update(label="OCR 완료", state="complete", expanded=False)
            except Exception as e:
                status.update(label="OCR 실패", state="error", expanded=False)
                st.error(f"OCR 실패: {e}")
                return

            st.caption("OCR 텍스트 미리보기")
            if not ocr_text or not ocr_text.strip():
                st.warning("추출된 텍스트가 없습니다. 임계값/PSM을 조정하거나 다른 페이지로 시도해주세요.")
            st.text_area("추출 결과", value=ocr_text or "", height=240)

            c1, c2, c3 = st.columns(3)
            corp_val = c1.text_input("법인(고정값)", value="")
            period_val = c2.text_input("기간(고정값)", value="")
            stmt_val = c3.text_input("구분(고정값)", value="손익")

            lines = []
            for raw in ocr_text.splitlines():
                line = raw.strip()
                if not line:
                    continue
                m = re.search(r"(-?\d[\d,]*)\s*$", line)
                if not m:
                    continue
                amount_str = m.group(1)
                account = line[: m.start()].strip()
                if not account:
                    continue
                lines.append({"account": account, "amount": amount_str})
            ocr_df = pd.DataFrame(lines)
            st.caption("OCR 파싱 미리보기")
            st.dataframe(ocr_df.head(200), use_container_width=True, hide_index=True)

            delete_before_pdf = st.checkbox("해당 페이지 데이터 삭제 후 적재", value=False, key="fin_ocr_delete")
            if st.button("OCR 적재 실행", key="fin_ocr_load"):
                if not corp_val.strip() or not period_val.strip():
                    st.warning("법인/기간 고정값을 입력해주세요.")
                    return
                load_df = pd.DataFrame()
                load_df["_corp"] = corp_val.strip()
                load_df["_period"] = _normalize_period(period_val.strip())
                load_df["_stmt"] = _map_statement_type(stmt_val.strip())
                load_df["_account"] = ocr_df["account"].astype("string").str.strip()
                load_df["_amount"] = pd.to_numeric(ocr_df["amount"].apply(_to_decimal), errors="coerce")
                load_df = load_df[load_df["_amount"].notna()]
                _ingest_financial_rows(
                    load_df,
                    source_file=getattr(pdf_up, "name", "uploaded.pdf"),
                    sheet_name=f"page_{page_no}",
                    delete_before=delete_before_pdf,
                )


def show_upload_page():
    st.header("데이터 업로드")
    tab1, tab2, tab3 = st.tabs(["매출", "재고", "재무제표"])

    with tab1:
        _show_sales_upload()

    with tab2:
        _show_inventory_upload()

    with tab3:
        _show_financial_upload()
