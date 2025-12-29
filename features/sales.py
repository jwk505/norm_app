# pages/sales.py
import streamlit as st
from core.db import query_df
from core.ui import render_table

def show_sales_report_page():
    st.header("📈 매출 리포트 (raw 기준)")

    df = query_df(
        """
        SELECT
          year,
          customer_raw,
          SUM(amount) AS amount
        FROM sales_raw
        GROUP BY year, customer_raw
        ORDER BY amount DESC
        LIMIT 300
        """
    )

    render_table(df, number_cols=["amount"])
