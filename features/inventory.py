# pages/inventory.py
import streamlit as st
from core.db import query_df
from core.ui import render_table
from core.schema import get_columns

def show_inventory_page():
    st.header("📦 재고 리포트")

    cols = get_columns("inventory_snapshot")

    # source_system 있으면 INV_SUM만 보기 토글 제공
    has_source = "source_system" in cols
    source_filter = ""
    params = []
    if has_source:
        source = st.selectbox("Source System", ["INV_SUM", "(ALL)"], index=0)
        if source != "(ALL)":
            source_filter = "WHERE s.source_system=%s"
            params.append(source)

    # maker 컬럼 후보
    maker_col = None
    for c in ["maker", "brand", "make", "mfg", "manufacturer"]:
        if c in cols:
            maker_col = c
            break
    maker_expr = f"IFNULL(NULLIF(TRIM(s.{maker_col}),''),'(UNKNOWN)')" if maker_col else "'(UNKNOWN)'"

    # item 컬럼 후보
    item_candidates = []
    if "norm_item" in cols:
        item_candidates.append(("정규화 품목(norm_item)", "norm_item"))
    if "raw_item" in cols:
        item_candidates.append(("원본 품목(raw_item)", "raw_item"))
    if not item_candidates:
        st.error("inventory_snapshot 테이블에서 norm_item/raw_item 컬럼을 찾지 못했습니다.")
        return

    label_to_col = {lbl: col for lbl, col in item_candidates}
    basis = st.radio("품목 기준", [lbl for lbl, _ in item_candidates], horizontal=True)
    item_col = label_to_col[basis]

    # qty / stock_value 존재 확인
    qty_col = "qty" if "qty" in cols else None
    value_col = "stock_value" if "stock_value" in cols else None
    if not qty_col or not value_col:
        st.error("inventory_snapshot 테이블에 qty 또는 stock_value 컬럼이 없습니다.")
        return

    sql = f"""
    SELECT
      {maker_expr} AS maker,
      CAST(s.{item_col} AS CHAR) AS item,
      SUM(s.{qty_col}) AS qty,
      SUM(s.{value_col}) AS stock_value
    FROM inventory_snapshot s
    {source_filter}
    GROUP BY maker, item
    ORDER BY stock_value DESC
    LIMIT 200
    """
    df = query_df(sql, tuple(params) if params else None)

    render_table(df, number_cols=["qty", "stock_value"])
