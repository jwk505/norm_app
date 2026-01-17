# app.py
import streamlit as st

from features.customer import show_customer_normalize_page
from features.finance import show_finance_page
from features.inventory import show_inventory_page
from features.sales import show_sales_report_page
from features.upload import show_upload_page


st.set_page_config(page_title="Norm ERP Console", layout="wide")

st.sidebar.title("메뉴")

menu = st.sidebar.radio(
    "이동",
    [
        "거래처 정규화",
        "재고 리포트",
        "매출 리포트",
        "재무제표 분석",
        "데이터 업로드",
        "초기화(init)",
    ],
    index=0,
)

if menu == "거래처 정규화":
    show_customer_normalize_page()
elif menu == "재고 리포트":
    show_inventory_page()
elif menu == "매출 리포트":
    show_sales_report_page()
elif menu == "재무제표 분석":
    show_finance_page()
elif menu == "데이터 업로드":
    show_upload_page()
else:
    st.info("준비 중입니다.")
