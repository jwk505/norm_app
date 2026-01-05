# pages/sales.py
from typing import List, Optional, Tuple

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


def _find_header_row(df_no_header: pd.DataFrame, needle: str = "년도", max_scan: int = 40) -> Optional[int]:
    n = min(max_scan, len(df_no_header))
    for i in range(n):
        row_vals = df_no_header.iloc[i].astype(str).tolist()
        if any(needle in v for v in row_vals):
            return i
    return None


def _read_one_sheet_any_header(excel_source, sheet_name) -> pd.DataFrame:
    raw = pd.read_excel(excel_source, sheet_name=sheet_name, header=None, engine="openpyxl")
    header_row = _find_header_row(raw, needle="년도", max_scan=40)

    if header_row is None:
        df = raw.copy()
        df.columns = [f"Unnamed_{i}" for i in range(df.shape[1])]
    else:
        df = pd.read_excel(excel_source, sheet_name=sheet_name, header=header_row, engine="openpyxl")

    df["_sheet"] = str(sheet_name)
    df["_header_row"] = header_row
    return df


@st.cache_data(ttl=300)
def load_sales_all_sheets(upload_or_path) -> pd.DataFrame:
    xls = pd.ExcelFile(upload_or_path, engine="openpyxl")
    frames = [_read_one_sheet_any_header(upload_or_path, sh) for sh in xls.sheet_names]
    return pd.concat(frames, ignore_index=True, sort=False)


def parse_year(val) -> Optional[int]:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    s = str(val).strip().replace("년", "").strip()
    try:
        y = int(float(s))
        if 2000 <= y <= 2100:
            return y
    except Exception:
        return None
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
    if "년도" in df.columns:
        return "년도"
    if "연도" in df.columns:
        return "연도"
    return None


def detect_amount_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["공급가액", "매출액", "매출", "금액", "공급가", "매출금액"]:
        if c in df.columns:
            return c
    return None


def pick_customer_col_for_sheet(df: pd.DataFrame, sheet: str) -> Optional[str]:
    if is_sheet_2425(sheet):
        if "사용상호" in df.columns:
            return "사용상호"
        if "거래처명" in df.columns:
            return "거래처명"
        if "거래처" in df.columns:
            return "거래처"
        return None

    if "거래처명" in df.columns:
        return "거래처명"
    if "거래처" in df.columns:
        return "거래처"
    if "사용상호" in df.columns:
        return "사용상호"
    return None


def show_sales_report_page():
    st.header("매출 리포트 (raw 기준)")

    st.subheader("DB 데이터 리포트 (기본)")
    if not _table_exists("sales_raw"):
        st.error("DB에 sales_raw 테이블이 없습니다. 초기화 메뉴를 처음 실행해주세요.")
        return

    has_alias = _table_exists("customer_alias")
    has_master = _table_exists("customer_master")

    brand_expr = "'(UNKNOWN)'"
    brand_note = "(customer_master에 브랜드 컬럼 없음)"
    if has_master:
        master_cols = get_columns("customer_master")
        brand_col = next((c for c in ["brand", "brand_name", "brand_nm", "브랜드"] if c in master_cols), None)
        if brand_col:
            brand_expr = f"IFNULL(NULLIF(TRIM(cm.{brand_col}),''), '(UNKNOWN)')"
            brand_note = f"(customer_master.{brand_col} 기준)"

    customer_expr = "COALESCE(cm.display_name, sr.customer_raw)"

    if has_alias and has_master:
        base_df = query_df(
            f"""
            SELECT
              sr.year,
              {customer_expr} AS customer,
              {brand_expr} AS brand,
              SUM(sr.amount) AS amount
            FROM sales_raw sr
            LEFT JOIN customer_alias ca ON ca.alias_name = sr.customer_raw
            LEFT JOIN customer_master cm ON cm.id = ca.customer_id
            GROUP BY sr.year, {customer_expr}, {brand_expr}
            ORDER BY sr.year DESC, amount DESC
            """
        )
        st.caption(f"연도/업체/브랜드 집계 {brand_note}")
    else:
        base_df = query_df(
            """
            SELECT
              sr.year,
              sr.customer_raw AS customer,
              '(UNKNOWN)' AS brand,
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

        years = sorted([int(y) for y in base_df["year"].dropna().unique()], reverse=True)
        brands = sorted(base_df["brand"].dropna().astype(str).unique().tolist())
        customers = sorted(base_df["customer"].dropna().astype(str).unique().tolist())

        c1, c2 = st.columns(2)
        sel_years = c1.multiselect("연도 선택(다중)", years, default=years)
        sel_brands = c2.multiselect("브랜드(메이커) 선택", brands, default=brands)
        sel_customers = st.multiselect("업체 선택 (다중)", customers, default=customers)

        view = base_df
        if sel_years:
            view = view[view["year"].isin(sel_years)]
        if sel_brands:
            view = view[view["brand"].isin(sel_brands)]
        if sel_customers:
            view = view[view["customer"].isin(sel_customers)]

        st.caption("브랜드 = 메이커, 연도/브랜드/업체 멀티선택 필터 적용")

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

        raw = load_sales_all_sheets(up)

        year_col = detect_year_col(raw)
        amount_col = detect_amount_col(raw)
        if not year_col:
            st.error("엑셀에서 '년도'/'연도' 컬럼을 찾지 못했습니다.")
            return
        if not amount_col:
            st.error("엑셀에서 금액 컬럼(공급가액/매출액/금액 등)을 찾지 못했습니다.")
            return

        delete_before = st.checkbox("적재 전 sales_raw 비우기(DELETE)", value=False)
        only_2020_2025 = st.checkbox("2020~2025만 적재", value=True)

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

        st.subheader("적재 대상 미리보기 (상위 200행)")
        st.dataframe(
            df[["_sheet", "year", "customer_raw", "amount", "customer_col"]].head(200),
            use_container_width=True,
            hide_index=True,
        )

        k1, k2, k3 = st.columns(3)
        k1.metric("적재 대상 행 수", f"{len(df):,}")
        k2.metric("거래처(원문) 수", f"{df['customer_raw'].nunique():,}")
        k3.metric("금액 합계", f"{int(df['amount'].sum()):,}")

        if st.button("↳ DB 적재 실행"):
            if delete_before:
                deleted = exec_sql("DELETE FROM sales_raw")
                st.info(f"sales_raw 삭제: {deleted:,} rows")

            insert_sql = """
            INSERT INTO sales_raw (src_file, sheet_name, year, customer_raw, amount, customer_col)
            VALUES (%s, %s, %s, %s, %s, %s)
            """

            rows: List[Tuple] = []
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
