# inventory_report_app.py
# ============================================================
# Norm ERP Console (운영용 단일 파일, 최종 통합본)
#
# 메뉴
# 1) 재고 / 품목 매핑 (inventory_snapshot + item_master)
# 2) 매출 엑셀 → DB 적재 (sales_raw)  *기본: 파일 업로드*
# 3) 거래처 정규화 (customer_alias → customer_master)
# 4) 거래처 전략 리포트 (정규화 기준 TOP/성장/감소 + 확장뷰로 raw alias 표시)
#
# 실행:
#   python -m streamlit run inventory_report_app.py
#
# .env (C:\norm_app\.env) 예시:
#   DB_HOST=127.0.0.1
#   DB_USER=normuser
#   DB_PASS=비밀번호
#   DB_NAME=normdb
#   DB_PORT=3306
# ============================================================

import os
from pathlib import Path
from typing import Optional, Tuple, Set, List

import pandas as pd
import streamlit as st
import mysql.connector
from dotenv import load_dotenv


# =============================
# ENV / PATH
# =============================
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=True)

DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS") or ""
DB_NAME = os.getenv("DB_NAME", "normdb")
DB_PORT = int(os.getenv("DB_PORT", "3306"))

DEFAULT_SALES_XLSX = BASE_DIR / "20-25년_전체매출.xlsx"


# =============================
# DB Helpers
# =============================
def get_conn():
    if not DB_USER:
        raise RuntimeError("DB_USER가 비어있습니다. .env 파일을 확인하세요.")
    return mysql.connector.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASS,
        database=DB_NAME,
        port=DB_PORT,
    )


@st.cache_data(ttl=60)
def query_df(sql: str, params: Tuple = ()) -> pd.DataFrame:
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(sql, params)
        cols = [c[0] for c in cur.description] if cur.description else []
        rows = cur.fetchall() if cur.description else []
        cur.close()
        return pd.DataFrame(rows, columns=cols)
    finally:
        conn.close()


def exec_sql(sql: str, params: Tuple = ()) -> int:
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(sql, params)
        conn.commit()
        rc = cur.rowcount
        cur.close()
        return int(rc)
    finally:
        conn.close()


def exec_many(sql: str, rows: List[Tuple]) -> int:
    if not rows:
        return 0
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.executemany(sql, rows)
        conn.commit()
        cur.close()
        return int(len(rows))
    finally:
        conn.close()


@st.cache_data(ttl=300)
def get_columns(table_name: str) -> Set[str]:
    df = query_df(
        """
        SELECT COLUMN_NAME
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s
        """,
        (DB_NAME, table_name),
    )
    if df.empty:
        return set()
    return set(df["COLUMN_NAME"].astype(str).tolist())


@st.cache_data(ttl=300)
def table_exists(table_name: str) -> bool:
    df = query_df(
        """
        SELECT COUNT(*) AS c
        FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_SCHEMA=%s AND TABLE_NAME=%s
        """,
        (DB_NAME, table_name),
    )
    return (not df.empty) and int(df.iloc[0]["c"]) > 0


# =============================
# Formatting / common helpers
# =============================
def style_numbers(df: pd.DataFrame, num_cols: Optional[List] = None):
    """Pandas Styler: None/NaN 안전 + 숫자 콤마 + 오른쪽 정렬"""
    if df is None or df.empty:
        return df

    if num_cols is None:
        candidates = [
            "qty", "stock_value", "line_cnt",
            "amount", "TOTAL", "GROWTH_23_25", "GROWTH_24_25",
            2020, 2021, 2022, 2023, 2024, 2025,
        ]
        num_cols = [c for c in candidates if c in df.columns]

    def fmt(x):
        try:
            if x is None or (isinstance(x, float) and pd.isna(x)) or pd.isna(x):
                return ""
            return f"{float(x):,.0f}"
        except Exception:
            return x

    fmt_map = {c: fmt for c in num_cols if c in df.columns}
    sty = df.style.format(fmt_map)

    right_cols = [c for c in num_cols if c in df.columns]
    if right_cols:
        sty = sty.set_properties(subset=right_cols, **{"text-align": "right"})
    return sty


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


# =============================
# Inventory helpers
# =============================
def pick_maker_expr(item_cols: Set[str], snap_cols: Set[str]) -> Tuple[str, str]:
    candidates = ["maker", "brand", "make", "mfg", "manufacturer"]
    s_col = next((c for c in candidates if c in snap_cols), None)
    m_col = next((c for c in candidates if c in item_cols), None)

    if s_col and m_col:
        return (
            f"s.{s_col}/m.{m_col}",
            f"IFNULL(NULLIF(TRIM(s.{s_col}),''), IFNULL(NULLIF(TRIM(m.{m_col}),''), '(UNKNOWN)'))",
        )
    if s_col:
        return (f"s.{s_col}", f"IFNULL(NULLIF(TRIM(s.{s_col}),''), '(UNKNOWN)')")
    if m_col:
        return (f"m.{m_col}", f"IFNULL(NULLIF(TRIM(m.{m_col}),''), '(UNKNOWN)')")
    return ("(none)", "'(UNKNOWN)'")


# =============================
# Sales Excel multi-sheet loader (header auto-detect)
# =============================
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
    # 24-25 시트: 사용상호 우선
    if is_sheet_2425(sheet):
        if "사용상호" in df.columns:
            return "사용상호"
        if "거래처명" in df.columns:
            return "거래처명"
        if "거래처" in df.columns:
            return "거래처"
        return None

    # 그 외: 거래처명 우선
    if "거래처명" in df.columns:
        return "거래처명"
    if "거래처" in df.columns:
        return "거래처"
    if "사용상호" in df.columns:
        return "사용상호"
    return None


# =============================
# Page 1: Inventory / Item Mapping
# =============================
def show_inventory_page():
    st.subheader("재고 / 품목 매핑 (inventory_snapshot)")
    if not table_exists("inventory_snapshot"):
        st.error("DB에 inventory_snapshot 테이블이 없습니다.")
        return
    if not table_exists("item_master"):
        st.error("DB에 item_master 테이블이 없습니다.")
        return

    item_cols = get_columns("item_master")
    snap_cols = get_columns("inventory_snapshot")
    maker_label, maker_col_expr = pick_maker_expr(item_cols, snap_cols)

    # source_system 목록
    src_df = query_df("SELECT DISTINCT source_system FROM inventory_snapshot ORDER BY source_system")
    src_options = src_df["source_system"].dropna().astype(str).tolist() if not src_df.empty else ["INV_SUM"]
    default_src = "INV_SUM" if "INV_SUM" in src_options else (src_options[0] if src_options else "INV_SUM")

    with st.expander("연결정보/스키마(디버그)", expanded=False):
        st.write("DB_HOST =", DB_HOST)
        st.write("DB_USER =", DB_USER)
        st.write("DB_PASS SET =", bool(DB_PASS.strip()))
        st.write("DB_NAME =", DB_NAME)
        st.write("DB_PORT =", DB_PORT)
        st.write("ENV_PATH =", str(ENV_PATH))
        st.write("Maker expr =", maker_label, "=>", maker_col_expr)

    st.sidebar.header("재고 필터")
    source_system = st.sidebar.selectbox(
        "Source System",
        src_options,
        index=src_options.index(default_src) if default_src in src_options else 0,
        key="inv_source_system",
    )
    view_mode = st.sidebar.radio("보기 모드", ["매핑된 품목", "미매핑 품목"], index=0, key="inv_view_mode")
    enable_mapping_ui = st.sidebar.checkbox("미매핑 매핑 UI 표시", value=True, key="inv_enable_mapping_ui")
    show_all = st.sidebar.checkbox("전체 보기 (LIMIT 해제)", value=False, key="inv_show_all")
    top_n = st.sidebar.slider("TOP N", 10, 500, 100, 10, disabled=show_all, key="inv_topn")
    min_stock_value = st.sidebar.number_input("최소 재고금액(원)", min_value=0.0, value=0.0, step=10000.0, key="inv_min_stock_value")
    search_item = st.sidebar.text_input("품목 검색(부분일치)", value="", key="inv_search_item")
    only_outliers = st.sidebar.checkbox("이상치(0수량/음수수량)만", value=False, key="inv_only_outliers")
    show_line_items = st.sidebar.checkbox("선택 그룹 상세 라인 보기", value=True, key="inv_show_line_items")

    limit_sql = "" if show_all else f"LIMIT {int(top_n)}"

    # maker 목록
    maker_list_df = query_df(
        f"""
        SELECT DISTINCT {maker_col_expr} AS maker
        FROM inventory_snapshot s
        LEFT JOIN item_master m ON m.id = s.mapped_item_id
        WHERE s.source_system=%s
        ORDER BY maker
        """,
        (source_system,),
    )
    maker_options = ["(ALL)"] + (maker_list_df["maker"].dropna().astype(str).tolist() if not maker_list_df.empty else [])
    maker = st.sidebar.selectbox("Maker", maker_options, index=0, key="inv_maker")

    # WHERE + params
    where: List[str] = ["s.source_system=%s"]
    params: List = [source_system]

    if maker != "(ALL)":
        where.append(f"{maker_col_expr} = %s")
        params.append(maker)

    if float(min_stock_value) > 0:
        where.append("s.stock_value >= %s")
        params.append(float(min_stock_value))

    if only_outliers:
        where.append("(s.qty <= 0)")

    if search_item.strip():
        like = f"%{search_item.strip()}%"
        where.append("(s.raw_item LIKE %s OR s.norm_item LIKE %s)")
        params.extend([like, like])
        if view_mode == "매핑된 품목" and "display_name" in item_cols:
            where[-1] = "(s.raw_item LIKE %s OR s.norm_item LIKE %s OR m.display_name LIKE %s)"
            params.append(like)

    if view_mode == "매핑된 품목":
        where.append("(s.mapped_item_id IS NOT NULL AND s.mapped_item_id <> 0)")
    else:
        where.append("(s.mapped_item_id IS NULL OR s.mapped_item_id = 0)")

    where_sql = " AND ".join(where)

    # Summary SQL
    if view_mode == "매핑된 품목":
        item_name_expr = "m.display_name" if "display_name" in item_cols else "CAST(s.norm_item AS CHAR)"
        item_id_expr = "s.mapped_item_id"
        group_expr = "s.mapped_item_id, item, maker"
    else:
        item_name_expr = "CAST(s.norm_item AS CHAR)"
        item_id_expr = "0"
        group_expr = "item, maker"

    summary_sql = f"""
    SELECT
      {item_id_expr} AS item_id,
      {item_name_expr} AS item,
      {maker_col_expr} AS maker,
      COALESCE(SUM(s.qty),0) AS qty,
      COALESCE(SUM(s.stock_value),0) AS stock_value,
      COUNT(*) AS line_cnt
    FROM inventory_snapshot s
    LEFT JOIN item_master m ON m.id = s.mapped_item_id
    WHERE {where_sql}
    GROUP BY {group_expr}
    ORDER BY stock_value DESC
    {limit_sql}
    """
    summary_df = query_df(summary_sql, tuple(params))

    total_qty = float(summary_df["qty"].sum()) if (summary_df is not None and not summary_df.empty and "qty" in summary_df.columns) else 0.0
    total_value = float(summary_df["stock_value"].sum()) if (summary_df is not None and not summary_df.empty and "stock_value" in summary_df.columns) else 0.0

    k1, k2, k3 = st.columns(3)
    k1.metric("표시 그룹 수", f"{len(summary_df):,}")
    k2.metric("표시 재고수량 합계", f"{int(total_qty):,}")
    k3.metric("표시 재고금액 합계(원)", f"{int(total_value):,}")

    st.subheader("요약")
    st.dataframe(style_numbers(summary_df, num_cols=["qty", "stock_value", "line_cnt"]), use_container_width=True, hide_index=True)

    # Unmapped mapping UI
    if enable_mapping_ui and view_mode == "미매핑 품목":
        st.divider()
        st.header("미매핑 품목 매핑 (MVP)")

        unmapped_sql = f"""
        SELECT
          CAST(s.norm_item AS CHAR) AS norm_item,
          {maker_col_expr} AS maker,
          COALESCE(SUM(s.qty),0) AS qty,
          COALESCE(SUM(s.stock_value),0) AS stock_value,
          COUNT(*) AS line_cnt
        FROM inventory_snapshot s
        LEFT JOIN item_master m ON m.id = s.mapped_item_id
        WHERE s.source_system=%s
          AND (s.mapped_item_id IS NULL OR s.mapped_item_id=0)
          AND (s.norm_item IS NOT NULL AND TRIM(s.norm_item) <> '')
        GROUP BY norm_item, maker
        ORDER BY stock_value DESC
        LIMIT 300
        """
        unmapped_df = query_df(unmapped_sql, (source_system,))

        if unmapped_df is None or unmapped_df.empty:
            st.info("미매핑 데이터가 없습니다.")
        else:
            left, right = st.columns([1, 2])

            with left:
                st.subheader("미매핑 TOP (norm_item)")
                st.dataframe(style_numbers(unmapped_df, num_cols=["qty", "stock_value", "line_cnt"]), use_container_width=True, hide_index=True)

                options = [
                    f"{r.norm_item} | {r.maker} | {int((r.stock_value or 0)):,}원 ({int((r.line_cnt or 0)):,} lines)"
                    for r in unmapped_df.itertuples(index=False)
                ]
                picked = st.selectbox("매핑할 norm_item 선택", options, index=0, key="inv_pick_norm_item")
                selected_norm_item = picked.split(" | ")[0].strip()

            with right:
                st.subheader("선택 norm_item 라인 샘플")
                sample_sql = """
                SELECT
                  s.id, s.raw_item, s.norm_item, s.qty, s.stock_value, s.created_at
                FROM inventory_snapshot s
                WHERE s.source_system=%s
                  AND s.norm_item=%s
                  AND (s.mapped_item_id IS NULL OR s.mapped_item_id=0)
                ORDER BY s.stock_value DESC, s.qty DESC
                LIMIT 50
                """
                sample_df = query_df(sample_sql, (source_system, selected_norm_item))
                st.dataframe(style_numbers(sample_df, num_cols=["qty", "stock_value"]), use_container_width=True, hide_index=True)

                st.markdown("### 기준 품목(item_master) 검색")
                q = st.text_input("검색어(기준품목명 일부)", value="", key="inv_item_search_q")

                name_col = "display_name" if "display_name" in item_cols else ("name" if "name" in item_cols else None)
                if not name_col:
                    st.warning("item_master에 display_name/name 컬럼이 없습니다. 검색 컬럼을 스키마에 맞게 수정해야 합니다.")
                else:
                    like = f"%{q.strip()}%" if q.strip() else "%"

                    maker_col_in_master = next((c for c in ["maker", "brand", "make", "mfg", "manufacturer"] if c in item_cols), None)

                    if maker_col_in_master:
                        item_search_sql = f"""
                        SELECT id, {name_col} AS item_name, {maker_col_in_master} AS maker
                        FROM item_master
                        WHERE {name_col} LIKE %s
                        ORDER BY {name_col}
                        LIMIT 50
                        """
                    else:
                        item_search_sql = f"""
                        SELECT id, {name_col} AS item_name
                        FROM item_master
                        WHERE {name_col} LIKE %s
                        ORDER BY {name_col}
                        LIMIT 50
                        """

                    cand_df = query_df(item_search_sql, (like,))
                    st.dataframe(cand_df, use_container_width=True, hide_index=True)

                    if cand_df is None or cand_df.empty:
                        st.info("검색 결과가 없습니다.")
                    else:
                        if "maker" in cand_df.columns:
                            cand_opts = [f"{int(r.id)} | {r.item_name} | {r.maker}" for r in cand_df.itertuples(index=False)]
                        else:
                            cand_opts = [f"{int(r.id)} | {r.item_name}" for r in cand_df.itertuples(index=False)]

                        picked_item = st.selectbox("매핑할 기준 품목 선택", cand_opts, index=0, key="inv_pick_item_master")
                        selected_item_id = int(picked_item.split("|")[0].strip())

                        st.markdown("### 매핑 적용")
                        st.caption("동일 norm_item의 미매핑 라인 전체에 mapped_item_id를 일괄 UPDATE 합니다.")

                        if st.button("✅ 매핑(UPDATE)", key="inv_do_mapping"):
                            upd_sql = """
                            UPDATE inventory_snapshot
                            SET mapped_item_id=%s
                            WHERE source_system=%s
                              AND norm_item=%s
                              AND (mapped_item_id IS NULL OR mapped_item_id=0)
                            """
                            affected = exec_sql(upd_sql, (selected_item_id, source_system, selected_norm_item))
                            st.success(f"매핑 완료: '{selected_norm_item}' → item_id={selected_item_id} (rows={affected:,})")
                            st.cache_data.clear()
                            st.rerun()

    # Detail lines
    if show_line_items and summary_df is not None and not summary_df.empty:
        st.divider()
        st.subheader("상세(라인)")

        if view_mode == "매핑된 품목":
            options = [f"{row.item} (id={int(row.item_id)})" for row in summary_df.itertuples(index=False)]
            picked = st.selectbox("상세로 볼 품목 선택", options, index=0, key="inv_detail_pick_mapped")
            selected_item_id = int(picked.split("id=")[-1].rstrip(")"))

            detail_sql = """
            SELECT
              s.id, s.source_system, s.raw_item, s.norm_item, s.qty, s.stock_value, s.created_at
            FROM inventory_snapshot s
            WHERE s.source_system=%s
              AND s.mapped_item_id = %s
            ORDER BY s.stock_value DESC, s.qty DESC
            LIMIT 1000
            """
            detail_df = query_df(detail_sql, (source_system, selected_item_id))
            st.dataframe(style_numbers(detail_df, num_cols=["qty", "stock_value"]), use_container_width=True, hide_index=True)

            csv = detail_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "선택 품목 상세 CSV 다운로드",
                data=csv,
                file_name=f"inv_{source_system}_item_{selected_item_id}_detail.csv",
                mime="text/csv",
                key="inv_dl_detail_mapped",
            )

    st.divider()
    st.subheader("다운로드")
    sum_csv = summary_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "현재 필터 기준 요약 CSV 다운로드",
        data=sum_csv,
        file_name=f"inv_{source_system}_{'mapped' if view_mode=='매핑된 품목' else 'unmapped'}_summary.csv",
        mime="text/csv",
        key="inv_dl_summary",
    )


# =============================
# Page 2: Sales Excel -> DB Import (default: upload)
# =============================
def show_sales_import_page():
    st.subheader("매출 엑셀 → DB 적재 (sales_raw)")
    st.caption("기본은 파일 업로드. 24-25 시트는 '사용상호', 그 외는 '거래처명'을 우선 사용합니다.")

    needed = ["sales_raw", "customer_master", "customer_alias"]
    missing = [t for t in needed if not table_exists(t)]
    if missing:
        st.error(f"DB에 필수 테이블이 없습니다: {missing}")
        st.stop()

    src_mode = st.radio(
        "데이터 소스",
        ["파일 업로드", "기본 파일 경로"],
        index=0,
        horizontal=True,
    )

    if src_mode == "파일 업로드":
        up = st.file_uploader("매출 엑셀 업로드 (.xlsx)", type=["xlsx"])
        if up is None:
            st.info("엑셀 파일을 업로드하세요.")
            return
        raw = load_sales_all_sheets(up)
        src_file_label = getattr(up, "name", "uploaded.xlsx")
    else:
        if not DEFAULT_SALES_XLSX.exists():
            st.error(f"기본 파일이 없습니다: {DEFAULT_SALES_XLSX}")
            return
        raw = load_sales_all_sheets(str(DEFAULT_SALES_XLSX))
        src_file_label = DEFAULT_SALES_XLSX.name

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
    st.dataframe(df[["_sheet", "year", "customer_raw", "amount", "customer_col"]].head(200), use_container_width=True, hide_index=True)

    k1, k2, k3 = st.columns(3)
    k1.metric("적재 대상 행 수", f"{len(df):,}")
    k2.metric("거래처(원문) 수", f"{df['customer_raw'].nunique():,}")
    k3.metric("금액 합계", f"{int(df['amount'].sum()):,}")

    if st.button("📥 DB 적재 실행"):
        if delete_before:
            deleted = exec_sql("DELETE FROM sales_raw")
            st.info(f"sales_raw 삭제: {deleted:,} rows")

        insert_sql = """
        INSERT INTO sales_raw (src_file, sheet_name, year, customer_raw, amount, customer_col)
        VALUES (%s, %s, %s, %s, %s, %s)
        """

        rows: List[Tuple] = []
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


# =============================
# Page 3: Customer Normalization
# =============================
def show_customer_normalize_page():
    st.subheader("거래처 정규화 (alias → master)")
    needed = ["sales_raw", "customer_master", "customer_alias"]
    missing = [t for t in needed if not table_exists(t)]
    if missing:
        st.error(f"DB에 필수 테이블이 없습니다: {missing}")
        return

    unmapped_df = query_df(
        """
        SELECT
          ca.alias_name,
          ca.src_hint,
          COALESCE(SUM(sr.amount),0) AS total_sales,
          COUNT(sr.id) AS line_cnt
        FROM customer_alias ca
        LEFT JOIN sales_raw sr ON sr.customer_raw = ca.alias_name
        WHERE ca.customer_id IS NULL
        GROUP BY ca.alias_name, ca.src_hint
        ORDER BY total_sales DESC
        LIMIT 300
        """
    )

    mapped_cnt = query_df("SELECT COUNT(*) AS c FROM customer_alias WHERE customer_id IS NOT NULL").iloc[0]["c"]
    all_cnt = query_df("SELECT COUNT(*) AS c FROM customer_alias").iloc[0]["c"]
    progress = 0.0 if int(all_cnt) == 0 else float(mapped_cnt) / float(all_cnt)

    k1, k2, k3 = st.columns(3)
    k1.metric("미매핑 alias 수", f"{len(unmapped_df):,}")
    k2.metric("매핑 완료 alias 수", f"{int(mapped_cnt):,}")
    k3.metric("진행률", f"{progress*100:,.1f}%")

    st.subheader("미매핑 alias TOP")
    st.dataframe(style_numbers(unmapped_df, num_cols=["total_sales", "line_cnt"]), use_container_width=True, hide_index=True)

    if unmapped_df is None or unmapped_df.empty:
        st.info("미매핑 alias가 없습니다.")
        return

    options = [
        f"{r.alias_name} | {r.src_hint} | {int((r.total_sales or 0)):,}원"
        for r in unmapped_df.itertuples(index=False)
    ]
    picked = st.selectbox("정규화할 alias 선택", options, index=0)
    alias_name = picked.split(" | ")[0].strip()

    left, right = st.columns([1, 1])

    with left:
        st.subheader("기존 기준 거래처 선택")
        q = st.text_input("검색어(대표 거래처명 일부)", value="", key="cust_search_q")
        like = f"%{q.strip()}%" if q.strip() else "%"

        master_df = query_df(
            """
            SELECT id, display_name, erp_customer_code, is_active
            FROM customer_master
            WHERE display_name LIKE %s
            ORDER BY display_name
            LIMIT 100
            """,
            (like,),
        )
        st.dataframe(master_df, use_container_width=True, hide_index=True)

        if master_df is not None and not master_df.empty:
            opts = [f"{int(r.id)} | {r.display_name}" for r in master_df.itertuples(index=False)]
            pick_master = st.selectbox("선택", opts, index=0, key="cust_pick_master")
            master_id = int(pick_master.split("|")[0].strip())

            if st.button("🔗 선택 거래처로 매핑(UPDATE)", key="cust_do_map"):
                rc = exec_sql(
                    "UPDATE customer_alias SET customer_id=%s WHERE alias_name=%s",
                    (master_id, alias_name),
                )
                st.success(f"매핑 완료: {alias_name} → customer_id={master_id} (rows={rc:,})")
                st.cache_data.clear()
                st.rerun()

    with right:
        st.subheader("신규 거래처 생성 + 매핑")
        new_name = st.text_input("대표 거래처명(display_name)", value=alias_name, key="cust_new_name")
        erp_code = st.text_input("ERP 거래처 코드(선택)", value="", key="cust_new_erp")
        is_active = st.checkbox("활성", value=True, key="cust_new_active")

        if st.button("➕ 생성 + 매핑", key="cust_create_map"):
            exec_sql(
                """
                INSERT INTO customer_master (display_name, erp_customer_code, is_active)
                VALUES (%s, %s, %s)
                ON DUPLICATE KEY UPDATE
                  erp_customer_code = COALESCE(NULLIF(VALUES(erp_customer_code),''), erp_customer_code),
                  is_active = VALUES(is_active)
                """,
                (new_name, erp_code.strip(), 1 if is_active else 0),
            )
            mid_df = query_df("SELECT id FROM customer_master WHERE display_name=%s", (new_name,))
            master_id = int(mid_df.iloc[0]["id"])
            exec_sql("UPDATE customer_alias SET customer_id=%s WHERE alias_name=%s", (master_id, alias_name))
            st.success(f"생성/갱신 + 매핑 완료: {alias_name} → {new_name} (id={master_id})")
            st.cache_data.clear()
            st.rerun()


# =============================
# Strategy helpers (expander alias view)
# =============================
@st.cache_data(ttl=120)
def get_alias_list_by_customer_id(customer_id: int) -> List[str]:
    df = query_df(
        """
        SELECT alias_name
        FROM customer_alias
        WHERE customer_id=%s
        ORDER BY alias_name
        """,
        (customer_id,),
    )
    if df is None or df.empty:
        return []
    return df["alias_name"].astype(str).tolist()


# =============================
# Page 4: Strategy report + expanders showing raw aliases
# =============================
def show_strategy_page():
    st.subheader("거래처 전략 리포트 (정규화 기준)")
    st.caption("대표 거래처 기준 집계 + 아래에서 펼치면 raw 거래처명(alias) 리스트를 확인할 수 있습니다.")

    needed = ["sales_raw", "customer_master", "customer_alias"]
    missing = [t for t in needed if not table_exists(t)]
    if missing:
        st.error(f"DB에 필수 테이블이 없습니다: {missing}")
        return

    df = query_df(
        """
        SELECT
          cm.id AS customer_id,
          cm.display_name AS customer,
          sr.year,
          SUM(sr.amount) AS amount
        FROM sales_raw sr
        JOIN customer_alias ca ON ca.alias_name = sr.customer_raw
        JOIN customer_master cm ON cm.id = ca.customer_id
        WHERE sr.year BETWEEN 2020 AND 2025
        GROUP BY cm.id, cm.display_name, sr.year
        """
    )

    if df is None or df.empty:
        st.info("정규화된 매출 데이터가 없습니다. (거래처 정규화에서 매핑을 먼저 진행하세요)")
        return

    pivot = (
        df.pivot_table(index=["customer_id", "customer"], columns="year", values="amount", aggfunc="sum", fill_value=0)
        .reset_index()
    )

    for y in [2020, 2021, 2022, 2023, 2024, 2025]:
        if y not in pivot.columns:
            pivot[y] = 0

    pivot["TOTAL"] = pivot[[2020, 2021, 2022, 2023, 2024, 2025]].sum(axis=1)
    pivot["GROWTH_23_25"] = pivot[2025] - pivot[2023]
    pivot["GROWTH_24_25"] = pivot[2025] - pivot[2024]

    top_n = st.sidebar.slider("TOP N", 10, 300, 50, 10, key="str_topn")

    k1, k2, k3 = st.columns(3)
    k1.metric("정규화 거래처 수", f"{pivot['customer_id'].nunique():,}")
    k2.metric("총 매출(20~25)", f"{int(pivot['TOTAL'].sum()):,}")
    k3.metric("25년 매출", f"{int(pivot[2025].sum()):,}")

    # ---- TOP table
    st.subheader("TOP 거래처 (2020~2025 누적)")
    top = pivot.sort_values("TOTAL", ascending=False).head(int(top_n))
    st.dataframe(style_numbers(top, num_cols=[2020, 2021, 2022, 2023, 2024, 2025, "TOTAL"]), use_container_width=True, hide_index=True)

    st.markdown("### TOP 거래처별 raw 거래처명(alias) 보기 (확장)")
    for r in top.itertuples(index=False):
        cid = int(r.customer_id)
        cname = str(r.customer)
        total = float(r.TOTAL)
        with st.expander(f"{cname}  |  TOTAL {int(total):,}원  |  customer_id={cid}"):
            aliases = get_alias_list_by_customer_id(cid)
            if not aliases:
                st.info("연결된 alias가 없습니다.")
            else:
                st.write("**raw 거래처명(alias) 목록**")
                st.dataframe(pd.DataFrame({"alias_name": aliases}), use_container_width=True, hide_index=True)

    # ---- Growth / Decrease tables (선택)
    st.subheader("성장 거래처 (2023→2025)")
    grow = pivot.sort_values("GROWTH_23_25", ascending=False).head(int(top_n))
    st.dataframe(style_numbers(grow, num_cols=[2023, 2024, 2025, "GROWTH_23_25"]), use_container_width=True, hide_index=True)

    st.subheader("감소 거래처 (2023→2025)")
    dec = pivot.sort_values("GROWTH_23_25", ascending=True).head(int(top_n))
    st.dataframe(style_numbers(dec, num_cols=[2023, 2024, 2025, "GROWTH_23_25"]), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("다운로드")
    csv = pivot.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "정규화 거래처×연도 Pivot CSV 다운로드",
        data=csv,
        file_name="normalized_customer_year_sales_pivot_2020_2025.csv",
        mime="text/csv",
    )


# =============================
# Main
# =============================
st.set_page_config(page_title="Norm ERP Console", layout="wide")
st.title("Norm ERP Console")

st.sidebar.header("메뉴")
menu = st.sidebar.radio(
    "선택",
    [
        "재고 / 품목 매핑",
        "매출 엑셀 → DB 적재",
        "거래처 정규화",
        "거래처 전략 리포트",
    ],
    index=0,
)

if st.sidebar.button("🔄 전체 캐시 비우기"):
    st.cache_data.clear()
    st.rerun()

try:
    if menu == "재고 / 품목 매핑":
        show_inventory_page()
    elif menu == "매출 엑셀 → DB 적재":
        show_sales_import_page()
    elif menu == "거래처 정규화":
        show_customer_normalize_page()
    else:
        show_strategy_page()
except Exception as e:
    st.error("오류가 발생했습니다. 아래 내용을 확인하세요.")
    st.exception(e)
