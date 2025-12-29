# app.py
import streamlit as st

# ✅ pages/ 대신 features/ 로 import
from features.customer import show_customer_normalize_page
from features.inventory import show_inventory_page
from features.sales import show_sales_report_page


st.set_page_config(page_title="Norm ERP Console", layout="wide")

st.sidebar.title("메뉴")

menu = st.sidebar.radio(
    "이동",
    [
        "거래처 정규화",
        "재고 리포트",
        "매출 리포트",
        "초기화/점검(init)",
    ],
    index=0,
)

if menu == "거래처 정규화":
    show_customer_normalize_page()

elif menu == "재고 리포트":
    show_inventory_page()

elif menu == "매출 리포트":
    show_sales_report_page()


