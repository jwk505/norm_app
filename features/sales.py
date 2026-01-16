# pages/sales.py
from typing import List, Optional, Tuple
import hashlib
import re
import time

import pandas as pd
import streamlit as st

from core.db import exec_sql, query_df
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
    tokens = ["구분", "법인", "매출구분", "corp", "division"]
    return _find_col_by_tokens(df.columns, tokens)


def detect_channel_col(df: pd.DataFrame) -> Optional[str]:
    tokens = ["채널", "channel", "sales_channel"]
    return _find_col_by_tokens(df.columns, tokens)


def detect_item_code_col(df: pd.DataFrame) -> Optional[str]:
    tokens = [
        "품목코드",
        "상품코드",
        "제품코드",
        "item_code",
        "itemcode",
        "item_cd",
        "itemcd",
        "product_code",
        "sku",
    ]
    return _find_col_by_tokens(df.columns, tokens)


def detect_item_name_col(df: pd.DataFrame) -> Optional[str]:
    tokens = [
        "품목명",
        "품명",
        "상품명",
        "제품명",
        "item_name",
        "item",
        "product_name",
        "product",
    ]
    return _find_col_by_tokens(df.columns, tokens)


def detect_maker_col(df: pd.DataFrame) -> Optional[str]:
    tokens = [
        "메이커",
        "브랜드",
        "제조사",
        "maker",
        "brand",
        "manufacturer",
    ]
    return _find_col_by_tokens(df.columns, tokens)


def detect_unit_price_col(df: pd.DataFrame) -> Optional[str]:
    tokens = [
        "단가",
        "평균단가",
        "unit_price",
        "unitprice",
        "price",
    ]
    return _find_col_by_tokens(df.columns, tokens)


def find_sales_raw_corp_col(sales_cols: set[str]) -> Optional[str]:
    tokens = ["구분", "법인", "매출구분", "corp", "division"]
    return _find_col_by_tokens(list(sales_cols), tokens)


def find_sales_raw_channel_col(sales_cols: set[str]) -> Optional[str]:
    tokens = ["채널", "sales_channel", "channel"]
    return _find_col_by_tokens(list(sales_cols), tokens)


def pick_sales_raw_item_code_col(sales_cols: set[str]) -> Optional[str]:
    candidates = ["item_code", "item_cd", "itemcode", "sku", "product_code"]
    return next((c for c in candidates if c in sales_cols), None)


def pick_sales_raw_item_name_col(sales_cols: set[str]) -> Optional[str]:
    candidates = ["item_name", "item_nm", "item", "product_name", "prod_name", "product"]
    return next((c for c in candidates if c in sales_cols), None)


def pick_sales_raw_maker_col(sales_cols: set[str]) -> Optional[str]:
    candidates = ["maker", "brand", "brand_name", "brand_nm", "maker_name", "manufacturer"]
    return next((c for c in candidates if c in sales_cols), None)


def pick_sales_raw_channel_col(sales_cols: set[str]) -> Optional[str]:
    candidates = ["channel", "sales_channel"]
    return next((c for c in candidates if c in sales_cols), None)


def pick_sales_raw_unit_price_col(sales_cols: set[str]) -> Optional[str]:
    candidates = ["unit_price", "unitprice", "price", "avg_unit_price"]
    return next((c for c in candidates if c in sales_cols), None)


def ensure_sales_raw_optional_columns(optional_cols: dict[str, str]) -> set[str]:
    if not optional_cols:
        return get_columns("sales_raw")
    sales_cols = get_columns("sales_raw")
    missing = [c for c in optional_cols if c not in sales_cols]
    if not missing:
        return sales_cols
    alter = "ALTER TABLE sales_raw " + ", ".join(
        [f"ADD COLUMN `{c}` {optional_cols[c]} NULL" for c in missing]
    )
    exec_sql(alter)
    return get_columns("sales_raw")


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
    item_code_col = pick_sales_raw_item_code_col(sales_cols)
    item_name_col = pick_sales_raw_item_name_col(sales_cols)
    unit_price_col = pick_sales_raw_unit_price_col(sales_cols)
    channel_col = pick_sales_raw_channel_col(sales_cols)

    brand_expr = "'(UNKNOWN)'"
    brand_note = "(sales_raw 메이커/브랜드 컬럼 없음)"
    channel_expr = "'(UNKNOWN)'"
    channel_note = "(sales_raw 채널 컬럼 없음)"
    corp_expr = "'(UNKNOWN)'"
    corp_note = "(sales_raw 법인(구분) 컬럼 없음)"
    maker_db_col = pick_sales_raw_maker_col(sales_cols)
    if maker_db_col:
        brand_expr = f"IFNULL(NULLIF(TRIM(sr.{maker_db_col}),''), '(UNKNOWN)')"
        brand_note = f"(sales_raw.{maker_db_col} 기준)"
    elif has_master:
        master_cols = get_columns("customer_master")
        brand_col = next(
            (c for c in ["maker", "brand", "brand_name", "brand_nm", "maker_name", "manufacturer", "메이커", "브랜드", "제조사"] if c in master_cols),
            None,
        )
        if brand_col:
            brand_expr = f"IFNULL(NULLIF(TRIM(cm.{brand_col}),''), '(UNKNOWN)')"
            brand_note = f"(customer_master.{brand_col} 기준)"
        # corp prefers sales_raw when available to match Excel "구분"

    channel_db_col = find_sales_raw_channel_col(sales_cols)
    if channel_db_col:
        channel_expr = f"IFNULL(NULLIF(TRIM(sr.{channel_db_col}),''), '(UNKNOWN)')"
        channel_note = f"(sales_raw.{channel_db_col} 기준)"
    elif has_master:
        master_cols = get_columns("customer_master")
        cm_channel_col = next((c for c in ["sales_channel", "channel"] if c in master_cols), None)
        if cm_channel_col:
            channel_expr = f"IFNULL(NULLIF(TRIM(cm.{cm_channel_col}),''), '(UNKNOWN)')"
            channel_note = f"(customer_master.{cm_channel_col} 기준)"

    corp_col = find_sales_raw_corp_col(sales_cols)
    if corp_col:
        corp_expr = f"IFNULL(NULLIF(TRIM(sr.{corp_col}),''), '(UNKNOWN)')"
        corp_note = f"(sales_raw.{corp_col} 기준)"
    elif has_master:
        master_cols = get_columns("customer_master")
        cm_corp_col = next(
            (c for c in ["corp", "corp_type", "corp_kind", "division", "구분", "법인"] if c in master_cols),
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
              {channel_expr} AS channel,
              {corp_expr} AS corp,
              SUM(sr.amount) AS amount
            FROM sales_raw sr
            LEFT JOIN customer_alias ca ON ca.alias_name = sr.{customer_join_col}
            LEFT JOIN customer_master cm ON cm.id = ca.customer_id
            GROUP BY sr.year, {customer_expr}, {brand_expr}, {channel_expr}, {corp_expr}
            ORDER BY sr.year DESC, amount DESC
            """
        )
        st.caption(f"연도/업체/채널/법인/메이커 집계 {brand_note} {channel_note} {corp_note}")
    else:
        base_df = query_df(
            """
            SELECT
              sr.year,
              sr.{customer_join_col} AS customer,
              '(UNKNOWN)' AS brand,
              '(UNKNOWN)' AS channel,
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
        base_df["channel"] = base_df["channel"].fillna("(UNKNOWN)")
        base_df["corp"] = base_df["corp"].fillna("(UNKNOWN)")

        years = sorted([int(y) for y in base_df["year"].dropna().unique()], reverse=True)
        brands = sorted(base_df["brand"].dropna().astype(str).unique().tolist())
        channels = sorted(base_df["channel"].dropna().astype(str).unique().tolist())
        corps = sorted(base_df["corp"].dropna().astype(str).unique().tolist())
        customers = sorted(base_df["customer"].dropna().astype(str).unique().tolist())

        col_labels = {
            "year": "연도",
            "customer": "업체",
            "brand": "브랜드",
            "channel": "채널",
            "corp": "법인",
            "item_code": "품목코드",
            "item_name": "품목명",
            "unit_price": "단가",
            "amount": "금액",
        }
        base_columns = ["year", "customer", "brand", "channel", "corp"]
        item_columns = []
        if item_code_col or item_name_col:
            item_columns = ["item_code", "item_name", "unit_price"]

        with st.expander("필터", expanded=False):
            with st.form("sales_pivot_form"):
                c1, c2, c3, c4 = st.columns(4)
                sel_years = c1.multiselect("연도 선택(다중)", years, default=years)
                sel_brands = c2.multiselect("메이커 선택", brands, default=brands)
                sel_channels = c3.multiselect("채널 선택", channels, default=channels)
                sel_corps = c4.multiselect("법인(구분) 선택", corps, default=corps)
                sel_customers = st.multiselect("업체 선택 (다중)", customers, default=customers)
                st.caption("메이커/채널/법인/연도/업체 멀티선택 필터 적용")

                source_options = ["기본"] + (["품목 포함"] if item_columns else [])
                source = st.selectbox("피벗 데이터", source_options, index=0, key="sales_pivot_source")
                source_columns = base_columns + (item_columns if source == "품목 포함" else [])
                label_map = {c: col_labels.get(c, c) for c in source_columns}
                reverse_map = {v: k for k, v in label_map.items()}

                row_defaults = ["year"]
                if "brand" in source_columns:
                    row_defaults.append("brand")
                row_cols_label = st.multiselect(
                    "행(그룹) 선택",
                    [label_map[c] for c in source_columns],
                    default=[label_map[c] for c in row_defaults if c in source_columns],
                    key="sales_pivot_rows",
                )
                col_options = ["(없음)"] + [label_map[c] for c in source_columns]
                col_pick = st.selectbox("열(컬럼) 선택", col_options, index=0, key="sales_pivot_cols")
                item_q = st.text_input("품목 검색(부분일치)", value="", key="sales_item_search")
                apply_clicked = st.form_submit_button("적용")

        if "sales_pivot_cached" not in st.session_state:
            st.session_state.sales_pivot_cached = None

        if apply_clicked:
            status = st.empty()
            start_time = time.monotonic()

            def _elapsed_str() -> str:
                elapsed = max(0, int(time.monotonic() - start_time))
                m, s = divmod(elapsed, 60)
                h, m = divmod(m, 60)
                if h > 0:
                    return f"{h}h {m:02d}m {s:02d}s"
                return f"{m:02d}m {s:02d}s"

            def _status(msg: str) -> None:
                status.write(f"{msg} | 경과 {_elapsed_str()}")

            _status("필터 적용 중...")
            view = base_df
            if sel_years:
                view = view[view["year"].isin(sel_years)]
            if sel_brands:
                view = view[view["brand"].isin(sel_brands)]
            if sel_channels:
                view = view[view["channel"].isin(sel_channels)]
            if sel_corps:
                view = view[view["corp"].isin(sel_corps)]
            if sel_customers:
                view = view[view["customer"].isin(sel_customers)]
            _status("필터 적용 완료")

            summary = {
                "row_count": int(len(view)),
                "customer_count": int(view["customer"].nunique()) if "customer" in view.columns else 0,
                "amount_sum": int(view["amount"].sum()) if "amount" in view.columns else 0,
            }

            if view.empty:
                st.session_state.sales_pivot_cached = {
                    "info": "선택한 조건에 해당하는 데이터가 없습니다.",
                    "df": None,
                    "num_cols": None,
                    "summary": summary,
                }
            else:
                _status("피벗 데이터 준비 중...")
                pivot_df = view
                if source == "품목 포함":
                    item_code_expr = f"NULLIF(TRIM(sr.{item_code_col}), '')" if item_code_col else "NULL"
                    item_name_expr = f"NULLIF(TRIM(sr.{item_name_col}), '')" if item_name_col else "NULL"
                    unit_price_expr = f"sr.{unit_price_col}" if unit_price_col else "NULL"

                    if has_alias and has_master:
                        item_df = query_df(
                            f"""
                            SELECT
                              sr.year,
                              {customer_expr} AS customer,
                              {brand_expr} AS brand,
                              {channel_expr} AS channel,
                              {corp_expr} AS corp,
                              {item_code_expr} AS item_code,
                              {item_name_expr} AS item_name,
                              {unit_price_expr} AS unit_price,
                              SUM(sr.amount) AS amount
                            FROM sales_raw sr
                            LEFT JOIN customer_alias ca ON ca.alias_name = sr.{customer_join_col}
                            LEFT JOIN customer_master cm ON cm.id = ca.customer_id
                            GROUP BY sr.year, {customer_expr}, {brand_expr}, {channel_expr}, {corp_expr},
                                     {item_code_expr}, {item_name_expr}, {unit_price_expr}
                            """
                        )
                    else:
                        item_df = query_df(
                            f"""
                            SELECT
                              sr.year,
                              sr.{customer_join_col} AS customer,
                              '(UNKNOWN)' AS brand,
                              '(UNKNOWN)' AS channel,
                              '(UNKNOWN)' AS corp,
                              {item_code_expr} AS item_code,
                              {item_name_expr} AS item_name,
                              {unit_price_expr} AS unit_price,
                              SUM(sr.amount) AS amount
                            FROM sales_raw sr
                            GROUP BY sr.year, sr.{customer_join_col},
                                     {item_code_expr}, {item_name_expr}, {unit_price_expr}
                            """
                        )

                    if item_df.empty:
                        st.session_state.sales_pivot_cached = {
                            "info": "품목 기준 데이터를 찾지 못했습니다. (품목 컬럼 확인 필요)",
                            "df": None,
                            "num_cols": None,
                            "summary": summary,
                        }
                        _status("피벗 데이터 준비 완료")
                    else:
                        item_df = item_df.copy()
                        item_df["year"] = pd.to_numeric(item_df["year"], errors="coerce").astype("Int64")
                        item_df["brand"] = item_df["brand"].fillna("(UNKNOWN)")
                        item_df["channel"] = item_df["channel"].fillna("(UNKNOWN)")
                        item_df["corp"] = item_df["corp"].fillna("(UNKNOWN)")
                        item_df["customer"] = item_df["customer"].fillna("(UNKNOWN)")

                        pivot_df = item_df
                        if sel_years:
                            pivot_df = pivot_df[pivot_df["year"].isin(sel_years)]
                        if sel_brands:
                            pivot_df = pivot_df[pivot_df["brand"].isin(sel_brands)]
                        if sel_channels:
                            pivot_df = pivot_df[pivot_df["channel"].isin(sel_channels)]
                        if sel_corps:
                            pivot_df = pivot_df[pivot_df["corp"].isin(sel_corps)]
                        if sel_customers:
                            pivot_df = pivot_df[pivot_df["customer"].isin(sel_customers)]

                        if item_q.strip():
                            mask = False
                            if "item_code" in pivot_df.columns:
                                mask = mask | pivot_df["item_code"].astype(str).str.contains(
                                    item_q, case=False, na=False
                                )
                            if "item_name" in pivot_df.columns:
                                mask = mask | pivot_df["item_name"].astype(str).str.contains(
                                    item_q, case=False, na=False
                                )
                            pivot_df = pivot_df[mask]
                        _status("피벗 데이터 준비 완료")

                if pivot_df is None or pivot_df.empty:
                    st.session_state.sales_pivot_cached = {
                        "info": "피벗에 사용할 데이터가 없습니다.",
                        "df": None,
                        "num_cols": None,
                        "summary": summary,
                    }
                else:
                    row_cols = [reverse_map[c] for c in row_cols_label if c in reverse_map]
                    col_col = None if col_pick == "(없음)" else reverse_map.get(col_pick)
                    if not row_cols:
                        st.session_state.sales_pivot_cached = {
                            "info": "행(그룹) 컬럼을 최소 1개 선택해주세요.",
                            "df": None,
                            "num_cols": None,
                            "summary": summary,
                        }
                    elif col_col and col_col in row_cols:
                        st.session_state.sales_pivot_cached = {
                            "info": "열 컬럼은 행 컬럼과 겹치지 않게 선택해주세요.",
                            "df": None,
                            "num_cols": None,
                            "summary": summary,
                        }
                    else:
                        _status("피벗 계산 중...")
                        pivot = pd.pivot_table(
                            pivot_df,
                            index=row_cols,
                            columns=col_col,
                            values="amount",
                            aggfunc="sum",
                            fill_value=0,
                        ).reset_index()
                        _status("완료")
                        num_cols = [c for c in pivot.columns if c not in row_cols]
                        pivot_display = pivot.rename(columns={c: col_labels.get(c, c) for c in pivot.columns})
                        num_cols_display = [col_labels.get(c, c) for c in num_cols]
                        if num_cols_display:
                            total_row = {c: "" for c in pivot_display.columns}
                            label_col = pivot_display.columns[0]
                            total_row[label_col] = "총계"
                            for c in num_cols_display:
                                if c in pivot_display.columns:
                                    total_row[c] = pivot_display[c].sum()
                            pivot_display = pd.concat(
                                [pivot_display, pd.DataFrame([total_row])], ignore_index=True
                            )
                        st.session_state.sales_pivot_cached = {
                            "info": None,
                            "df": pivot_display,
                            "num_cols": num_cols_display,
                            "summary": summary,
                        }

        cached = st.session_state.sales_pivot_cached
        if cached is None:
            st.info("필터/피벗을 설정한 뒤 '적용'을 눌러주세요.")
        else:
            if cached.get("summary"):
                s = cached["summary"]
                st.metric("행 수", f"{s['row_count']:,}")
                st.metric("업체 수", f"{s['customer_count']:,}")
                st.metric("금액 합계", f"{s['amount_sum']:,}")
            if cached["info"]:
                st.info(cached["info"])
            elif cached["df"] is not None:
                render_table(cached["df"], number_cols=cached["num_cols"], total_label="총계")
                export_df = cached["df"].copy()
                if "연도" in export_df.columns:
                    export_df["연도"] = pd.to_numeric(export_df["연도"], errors="coerce").apply(
                        lambda v: "" if pd.isna(v) else str(int(v))
                    )
                for c in cached["num_cols"] or []:
                    if c in export_df.columns:
                        export_df[c] = pd.to_numeric(export_df[c], errors="coerce").apply(
                            lambda v: "" if pd.isna(v) else f"{int(v):,}"
                        )
                csv_bytes = export_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    "엑셀 다운로드(CSV)",
                    data=csv_bytes,
                    file_name="sales_pivot.csv",
                    mime="text/csv",
                )
