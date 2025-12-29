import streamlit as st
from features.customer import show_customer_normalize_page
from features.inventory import show_inventory_report

st.set_page_config(page_title="SKB 경영 대시보드", layout="wide")

menu = st.sidebar.radio(
    "메뉴",
    ["거래처 정규화", "재고 리포트"]
)

if menu == "거래처 정규화":
    show_customer_normalize_page()
elif menu == "재고 리포트":
    show_inventory_report()
