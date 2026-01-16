# pages/sales.py
from typing import List, Optional, Tuple
import hashlib
import re

import pandas as pd
import streamlit as st

from core.db import exec_many, exec_sql, query_df
from core.schema import get_columns
from core.ui import render_table


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


def _norm_key(val: str) -> str:
    return re.sub(r"[\s_]+", "", str(val)).strip().lower()


def _find_header_row(
    df_no_header: pd.DataFrame, tokens: Optional[List[str]] = None, max_scan: int = 40
) -> Optional[int]:
    tokens = tokens or ["년도", "연도", "연월", "년", "year", "yr"]
    token_keys = [_norm_key(t) for t in tokens]

    n = min(max_scan, len(df_no_header))
    for i in range(n):
        row_vals = df_no_header.iloc[i].astype(str).tolist()
        for v in row_vals:
            vk = _norm_key(v)
            if any(tk and tk in vk for tk in token_keys):
                return i
    return None


def _find_col_by_tokens(columns, tokens: List[str]) -> Optional[str]:
    cols = list(columns)
    col_map = {_norm_key(c): c for c in cols}
    token_keys = [_norm_key(t) for t in tokens]

    for tk in token_keys:
        if tk in col_map:
            return col_map[tk]
    for c in cols:
        ck = _norm_key(c)
        if any(tk and tk in ck for tk in token_keys):
            return c
    return None


def _read_one_sheet_any_header(excel_source, sheet_name) -> pd.DataFrame:
    raw = pd.read_excel(excel_source, sheet_name=sheet_name, header=None, engine="openpyxl")
    header_row = _find_header_row(raw, max_scan=40)

    if header_row is None:
        df = raw.copy()
        df.columns = [f"Unnamed_{i}" for i in range(df.shape[1])]
    else:
        df = pd.read_excel(excel_source, sheet_name=sheet_name, header=header_row, engine="openpyxl")

    df["_sheet"] = str(sheet_name)
    df["_header_row"] = header_row
    return df


def compute_row_hash(*values) -> str:
    h = hashlib.md5()
    for v in values:
        s = "" if v is None else str(v)
        h.update(s.encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()


@st.cache_data(ttl=300)
def load_sales_all_sheets(upload_or_path) -> pd.DataFrame:
    xls = pd.ExcelFile(upload_or_path, engine="openpyxl")
    frames = [_read_one_sheet_any_header(upload_or_path, sh) for sh in xls.sheet_names]
    return pd.concat(frames, ignore_index=True, sort=False)


def parse_year(val) -> Optional[int]:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    if isinstance(val, (int, float)) and not pd.isna(val):
        y = int(val)
        if 2000 <= y <= 2100:
            return y
    s = str(val).strip()
    m = re.search(r"(20\d{2}|21\d{2})", s)
    if m:
        y = int(m.group(1))
        if 2000 <= y <= 2100:
            return y
    return None


def normalize_str_series(s: pd.Series) -> pd.Series:
    s = s.astype("string").str.strip()
    s = s.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    return s


def is_sheet_2425(sheet_name: str) -> bool:
    s = str(sheet_name)
    if "24-25" in s:
        return True
    if "24" in s and "25" in s:
        return True
    return False


def detect_year_col(df: pd.DataFrame) -> Optional[str]:
    tokens = ["년도", "연도", "연월", "년", "year", "yr"]
    return _find_col_by_tokens(df.columns, tokens)


def detect_amount_col(df: pd.DataFrame) -> Optional[str]:
    tokens = ["매출금액", "매출액", "공급가액", "금액", "매출", "amount", "sales", "amt"]
    return _find_col_by_tokens(df.columns, tokens)


def detect_corp_col(df: pd.DataFrame) -> Optional[str]:
    tokens = ["구분", "법인", "매출구분", "채널", "channel", "corp", "division"]
    return _find_col_by_tokens(df.columns, tokens)


def find_sales_raw_corp_col(sales_cols: set[str]) -> Optional[str]:
    tokens = ["구분", "법인", "매출구분", "채널", "sales_channel", "channel", "corp", "division"]
    return _find_col_by_tokens(list(sales_cols), tokens)


def find_sales_raw_customer_cols(sales_cols: set[str]) -> List[str]:
    candidates = [
        "customer_raw",
        "customer_cd",
        "customer",
        "customer_name",
        "customer_nm",
        "cust_name",
        "cust_nm",
    ]
    return [c for c in candidates if c in sales_cols]


def pick_customer_col_for_sheet(df: pd.DataFrame, sheet: str) -> Optional[str]:
    if is_sheet_2425(sheet):
        for candidates in [
            ["통합(실제상호)", "통합상호", "실제상호", "통합", "통합명", "realname", "real_name"],
        ]:
            col = _find_col_by_tokens(df.columns, candidates)
            if col:
                return col
        for candidates in [
            ["사업자등록번호", "사업자번호", "사업자등록", "사업자", "businessno", "bizno", "biz_no"],
            ["사용상호", "거래처명", "거래처", "업체명", "상호", "customer", "client", "account", "buyer"],
        ]:
            col = _find_col_by_tokens(df.columns, candidates)
            if col:
                return col
        return None

    for candidates in [
        ["통합(실제상호)", "통합상호", "실제상호", "통합", "통합명", "realname", "real_name"],
        ["거래처명", "거래처", "업체명", "상호", "사용상호", "customer", "client", "account", "buyer"],
        ["사업자등록번호", "사업자번호", "사업자", "businessno", "bizno", "biz_no"],
    ]:
        col = _find_col_by_tokens(df.columns, candidates)
        if col:
            return col
    return None


def show_sales_report_page():
    st.header("매출 리포트 (raw 기준)")

    st.subheader("DB 데이터 리포트 (기본)")
    if not _table_exists("sales_raw"):
        st.error("DB에 sales_raw 테이블이 없습니다. 초기화 메뉴를 처음 실행해주세요.")
        return

    has_alias = _table_exists("customer_alias")
    has_master = _table_exists("customer_master")

    sales_cols = get_columns("sales_raw")
    customer_cols = find_sales_raw_customer_cols(sales_cols)
    if not customer_cols:
        st.error("sales_raw에 고객 컬럼(customer_raw/customer_cd)이 없습니다.")
        return
    customer_src_col = customer_cols[0]
    customer_join_col = "customer_raw" if "customer_raw" in sales_cols else customer_src_col

    brand_expr = "'(UNKNOWN)'"
    brand_note = "(customer_master 메이커 컬럼 없음)"
    corp_expr = "'(UNKNOWN)'"
    corp_note = "(sales_raw 구분 컬럼 없음)"
    if has_master:
        master_cols = get_columns("customer_master")
        brand_col = next(
            (c for c in ["maker", "brand", "brand_name", "brand_nm", "maker_name", "manufacturer", "메이커", "브랜드", "제조사"] if c in master_cols),
            None,
        )
        if brand_col:
            brand_expr = f"IFNULL(NULLIF(TRIM(cm.{brand_col}),''), '(UNKNOWN)')"
            brand_note = f"(customer_master.{brand_col} 기준)"
        # corp prefers sales_raw when available to match Excel "구분"

    corp_col = find_sales_raw_corp_col(sales_cols)
    if corp_col:
        corp_expr = f"IFNULL(NULLIF(TRIM(sr.{corp_col}),''), '(UNKNOWN)')"
        corp_note = f"(sales_raw.{corp_col} 기준)"
    elif has_master:
        master_cols = get_columns("customer_master")
        cm_corp_col = next(
            (c for c in ["sales_channel", "channel", "corp", "corp_type", "corp_kind", "division", "구분", "법인"] if c in master_cols),
            None,
        )
        if cm_corp_col:
            corp_expr = f"IFNULL(NULLIF(TRIM(cm.{cm_corp_col}),''), '(UNKNOWN)')"
            corp_note = f"(customer_master.{cm_corp_col} 기준)"

    customer_expr = f"COALESCE(NULLIF(TRIM(sr.{customer_src_col}),''), cm.display_name)"

    if has_alias and has_master:
        base_df = query_df(
            f"""
            SELECT
              sr.year,
              {customer_expr} AS customer,
              {brand_expr} AS brand,
              {corp_expr} AS corp,
              SUM(sr.amount) AS amount
            FROM sales_raw sr
            LEFT JOIN customer_alias ca ON ca.alias_name = sr.{customer_join_col}
            LEFT JOIN customer_master cm ON cm.id = ca.customer_id
            GROUP BY sr.year, {customer_expr}, {brand_expr}, {corp_expr}
            ORDER BY sr.year DESC, amount DESC
            """
        )
        st.caption(f"연도/업체/구분/메이커 집계 {brand_note} {corp_note}")
    else:
        base_df = query_df(
            """
            SELECT
              sr.year,
              sr.{customer_join_col} AS customer,
              '(UNKNOWN)' AS brand,
              '(UNKNOWN)' AS corp,
              SUM(sr.amount) AS amount
            FROM sales_raw sr
            GROUP BY sr.year, sr.customer_raw
            ORDER BY sr.year DESC, amount DESC
            """
        )
        st.caption("연도/업체 집계 (alias/customer_master 미존재)")

    if base_df.empty:
        st.info("sales_raw 테이블에 데이터가 없습니다. 아래 '엑셀 업로드/재적재(옵션)'에서 데이터 적재를 진행해주세요.")
    else:
        base_df = base_df.copy()
        base_df["year"] = pd.to_numeric(base_df["year"], errors="coerce").astype("Int64")
        base_df["brand"] = base_df["brand"].fillna("(UNKNOWN)")
        base_df["corp"] = base_df["corp"].fillna("(UNKNOWN)")

        years = sorted([int(y) for y in base_df["year"].dropna().unique()], reverse=True)
        brands = sorted(base_df["brand"].dropna().astype(str).unique().tolist())
        corps = sorted(base_df["corp"].dropna().astype(str).unique().tolist())
        customers = sorted(base_df["customer"].dropna().astype(str).unique().tolist())

        c1, c2, c3 = st.columns(3)
        sel_years = c1.multiselect("연도 선택(다중)", years, default=years)
        sel_brands = c2.multiselect("메이커 선택", brands, default=brands)
        sel_corps = c3.multiselect("구분(법인) 선택", corps, default=corps)
        sel_customers = st.multiselect("업체 선택 (다중)", customers, default=customers)

        view = base_df
        if sel_years:
            view = view[view["year"].isin(sel_years)]
        if sel_brands:
            view = view[view["brand"].isin(sel_brands)]
        if sel_corps:
            view = view[view["corp"].isin(sel_corps)]
        if sel_customers:
            view = view[view["customer"].isin(sel_customers)]

        st.caption("메이커/구분/연도/업체 멀티선택 필터 적용")

        if view.empty:
            st.info("선택한 조건에 해당하는 데이터가 없습니다.")
        else:
            st.metric("행 수", f"{len(view):,}")
            st.metric("업체 수", f"{view['customer'].nunique():,}")
            st.metric("금액 합계", f"{int(view['amount'].sum()):,}")
            render_table(view.sort_values(["year", "amount"], ascending=[False, False]), number_cols=["amount"])
    st.divider()

    with st.expander("옵션: 매출 데이터 엑셀 업로드/재적재", expanded=False):
        st.subheader("매출 엑셀 업로드 → DB 적재")
        st.caption("엑셀 전체 시트를 읽어 sales_raw로 적재합니다. (24-25 시트는 '사용상호' 우선)")

        needed = ["sales_raw", "customer_master", "customer_alias"]
        missing = [t for t in needed if not _table_exists(t)]
        if missing:
            st.error(f"DB에 필수 테이블이 없습니다: {missing}")
            return

        up = st.file_uploader("매출 엑셀 업로드 (.xlsx)", type=["xlsx"])
        if up is None:
            st.info("엑셀 파일을 올리지 않으면 DB 데이터 그대로 사용합니다.")
            return

        delete_before = st.checkbox("Delete sales_raw before load (DELETE)", value=False)
        only_2020_2025 = st.checkbox("Only load 2020-2025", value=True)

        if st.button("Load into DB"):
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

            rows: List[Tuple] = []
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
