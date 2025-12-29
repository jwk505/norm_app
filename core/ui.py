# core/ui.py
import pandas as pd
import streamlit as st

def fmt_int(v) -> str:
    try:
        if v is None:
            return "0"
        if isinstance(v, float) and pd.isna(v):
            return "0"
        return f"{int(v):,}"
    except Exception:
        return "0"

def render_table(df: pd.DataFrame, number_cols=None, hide_index=True, key=None):
    """
    ✅ 표 안 숫자 3자리 콤마 '확실히' 표시 (Pandas Styler)
    - 데이터는 숫자 유지
    - 화면에서만 {:,.0f}
    """
    if df is None or df.empty:
        st.info("표시할 데이터가 없습니다.")
        return

    number_cols = number_cols or []
    view = df.copy()

    for c in number_cols:
        if c in view.columns:
            view[c] = pd.to_numeric(view[c], errors="coerce").fillna(0)

    fmt = {c: "{:,.0f}" for c in number_cols if c in view.columns}
    sty = view.style.format(fmt, na_rep="")

    if number_cols:
        sty = sty.set_properties(subset=[c for c in number_cols if c in view.columns], **{"text-align": "right"})

    st.dataframe(sty, use_container_width=True, hide_index=hide_index, key=key)

def safe_int(v, default=None):
    try:
        return int(v)
    except Exception:
        return default
