# pages/customer.py
import streamlit as st
import pandas as pd

from core.db import query_df, exec_sql
from core.ui import render_table, fmt_int, safe_int
from core.utils import normalize_customer_name_strict, similarity


# ================================
# 채널 정의
# ================================
CHANNEL_OPTIONS = ["(미지정)", "시판", "직납", "수출"]
CHANNEL_TO_DB = {
    "(미지정)": None,
    "시판": "시판",
    "직납": "직납",
    "수출": "수출",
}


# ================================
# 대표 거래처 로드
# ================================
@st.cache_data(ttl=300)
def get_active_customers():
    df = query_df(
        """
        SELECT id, display_name, sales_channel
        FROM customer_master
        WHERE is_active = 1
        ORDER BY display_name
        """
    )

    if df is None or df.empty:
        return pd.DataFrame(columns=["id", "display_name", "sales_channel"])

    df = df.copy()
    df["id"] = df["id"].apply(lambda x: safe_int(x))
    df["display_name"] = df["display_name"].astype(str)
    return df


def channel_to_ui(v):
    if v in ("시판", "직납", "수출"):
        return v
    return "(미지정)"


def update_customer_channel(customer_id: int, channel_ui: str):
    exec_sql(
        "UPDATE customer_master SET sales_channel=%s WHERE id=%s",
        (CHANNEL_TO_DB.get(channel_ui), int(customer_id)),
    )


# ================================
# 메인 페이지
# ================================
def show_customer_normalize_page():
    st.header("🏷️ 거래처 정규화")

    # ----------------------------
    # 📊 KPI 대시보드
    # ----------------------------
    kpi_alias = query_df(
        """
        SELECT
          COUNT(*) AS total_alias,
          SUM(CASE WHEN customer_id IS NOT NULL THEN 1 ELSE 0 END) AS mapped_alias,
          SUM(CASE WHEN customer_id IS NULL THEN 1 ELSE 0 END) AS unmapped_alias
        FROM customer_alias
        """
    )

    kpi_master = query_df(
        """
        SELECT COUNT(*) AS total_master
        FROM customer_master
        WHERE is_active = 1
        """
    )

    total_alias = int(kpi_alias.iloc[0]["total_alias"]) if (kpi_alias is not None and not kpi_alias.empty) else 0
    mapped_alias = int(kpi_alias.iloc[0]["mapped_alias"]) if (kpi_alias is not None and not kpi_alias.empty) else 0
    unmapped_alias = int(kpi_alias.iloc[0]["unmapped_alias"]) if (kpi_alias is not None and not kpi_alias.empty) else 0
    total_master = int(kpi_master.iloc[0]["total_master"]) if (kpi_master is not None and not kpi_master.empty) else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("전체 거래처(alias)", f"{total_alias:,}")
    c2.metric("매핑 완료", f"{mapped_alias:,}")
    c3.metric("미매핑", f"{unmapped_alias:,}")
    c4.metric("대표 거래처", f"{total_master:,}")

    st.divider()

    # ----------------------------
    # 대표 거래처 로드
    # ----------------------------
    customers = get_active_customers()
    if customers is None or customers.empty:
        st.warning("대표 거래처가 없습니다.")
        return

    # ----------------------------
    # 1) 미매핑 alias
    # ----------------------------
    st.subheader("미매핑 거래처(alias)")

    unmapped = query_df(
        """
        SELECT
          ca.alias_name,
          COALESCE(SUM(sr.amount),0) AS total_sales
        FROM customer_alias ca
        LEFT JOIN sales_raw sr ON sr.customer_raw = ca.alias_name
        WHERE ca.customer_id IS NULL
        GROUP BY ca.alias_name
        ORDER BY total_sales DESC
        LIMIT 300
        """
    )

    render_table(unmapped, number_cols=["total_sales"])

    if unmapped is None or unmapped.empty:
        st.info("미매핑 alias가 없습니다.")
        # 미매핑이 없어도 아래 확장뷰는 보고 싶을 수 있으니 return 하지 않고 진행
        alias_name = ""
    else:
        alias_name = st.selectbox(
            "정규화할 alias 선택",
            unmapped["alias_name"].astype(str).tolist(),
        )

   # ----------------------------
   # 2) 🤖 자동 추천 (유사도)
   # ----------------------------
    st.subheader("🤖 자동 추천 (유사도 기반, 강화)")

    if not alias_name:
      st.info("미매핑 alias가 없어서 추천/매핑 작업은 생략됩니다.")
    else:
      import re

    alias_key = normalize_customer_name_strict(alias_name)

    scored = []
    for r in customers.itertuples(index=False):
        master_key = normalize_customer_name_strict(r.display_name)

        base = similarity(alias_key, master_key)

        bonus = 0.0
        if alias_key and master_key:
            if alias_key in master_key or master_key in alias_key:
                bonus += 0.12

        def tokens(s: str):
            return set(re.findall(r"[0-9a-z가-힣]{2,}", s.lower()))

        a_tokens = tokens(alias_name)
        m_tokens = tokens(r.display_name)
        if a_tokens & m_tokens:
            bonus += min(0.06, 0.02 * len(a_tokens & m_tokens))

        scored.append((base + bonus, int(r.id), r.display_name, base, bonus))

    scored.sort(reverse=True, key=lambda x: x[0])

    top_n = st.slider("추천 후보 개수", 5, 30, 12, key="rec_top_n")
    threshold = st.slider("강조 임계치", 0.4, 0.95, 0.68, 0.01, key="rec_threshold")

    shown = scored[:top_n]

    rec_df = pd.DataFrame(
        [{
            "대표ID": cid,
            "대표명": name,
            "점수": round(final, 3),
            "base": round(base, 3),
            "bonus": round(bonus, 3),
            "추천": "✅" if final >= threshold else ""
        } for (final, cid, name, base, bonus) in shown]
    )
    render_table(rec_df, number_cols=["대표ID", "점수", "base", "bonus"])

    for (final, cid, name, base, bonus) in shown:
        col1, col2 = st.columns([4, 1])
        col1.write(f"**{name}** | {final:.3f}")
        if col2.button("이 후보로 매핑", key=f"auto_map_{alias_name}_{cid}"):
            exec_sql(
                "UPDATE customer_alias SET customer_id=%s WHERE alias_name=%s",
                (cid, alias_name),
            )
            st.cache_data.clear()
            st.rerun()

    # ----------------------------
    # 3) 수동 매핑 (⭐ for 루프 밖!)
    # ----------------------------
    st.subheader("🔗 수동 매핑")

    q = st.text_input("대표 거래처 검색", key="customer_search_master")
    view = customers[customers["display_name"].str.contains(q, case=False, na=False)] if q else customers

    opt = st.selectbox(
        "대표 거래처 선택",
        [f"{int(r.id)} | {r.display_name}" for r in view.itertuples(index=False)],
        key="manual_pick",
    )
    target_id = int(opt.split("|")[0])

    if st.button("선택 거래처로 매핑", key="manual_map_btn"):
        exec_sql(
            "UPDATE customer_alias SET customer_id=%s WHERE alias_name=%s",
            (target_id, alias_name),
        )
        st.cache_data.clear()
        st.rerun()

    # ----------------------------
    # 4) 신규 생성 + 매핑
    # ----------------------------
    st.subheader("➕ 신규 생성 + 매핑")

    new_name = st.text_input("대표 거래처명", value=alias_name, key="new_master_name")
    new_channel = st.selectbox("매출 구분", CHANNEL_OPTIONS, key="new_master_channel")

    if st.button("생성(없으면) + 매핑", key="create_and_map"):
        exist = query_df(
            "SELECT id FROM customer_master WHERE display_name=%s AND is_active=1 LIMIT 1",
            (new_name,),
        )

        if not exist.empty:
            mid = int(exist.iloc[0]["id"])
        else:
            exec_sql(
                "INSERT INTO customer_master (display_name, is_active, sales_channel) VALUES (%s,1,%s)",
                (new_name, CHANNEL_TO_DB.get(new_channel)),
            )
            mid = int(query_df(
                "SELECT id FROM customer_master WHERE display_name=%s ORDER BY id DESC LIMIT 1",
                (new_name,),
            ).iloc[0]["id"])

        exec_sql(
            "UPDATE customer_alias SET customer_id=%s WHERE alias_name=%s",
            (mid, alias_name),
        )
        st.cache_data.clear()
        st.rerun()


    # ==================================================
    # 5) 매핑된 대표 거래처 확장뷰 + 채널 수정  (⭐ 반드시 함수 안!)
    # ==================================================
    st.divider()

    if "show_mapped_view" not in st.session_state:
        st.session_state.show_mapped_view = True

    st.session_state.show_mapped_view = st.checkbox(
        "📂 매핑된 대표 거래처 확장뷰 보기",
        value=st.session_state.show_mapped_view,
        key="show_mapped_view_ck",
    )

    if st.session_state.show_mapped_view:
        st.subheader("📂 매핑된 대표 거래처 (확장뷰)")

        mapped = query_df(
            """
            SELECT
              cm.id,
              cm.display_name,
              cm.sales_channel,
              COUNT(ca.alias_name) AS alias_cnt,
              COALESCE(SUM(sr.amount),0) AS total_sales
            FROM customer_master cm
            LEFT JOIN customer_alias ca ON ca.customer_id = cm.id
            LEFT JOIN sales_raw sr ON sr.customer_raw = ca.alias_name
            WHERE cm.is_active = 1
            GROUP BY cm.id, cm.display_name, cm.sales_channel
            ORDER BY total_sales DESC, cm.display_name
            """
        )

        if mapped is None or mapped.empty:
            st.info("대표 거래처가 없습니다.")
        else:
            for r in mapped.itertuples(index=False):
                cid = int(r.id)
                channel_ui = channel_to_ui(r.sales_channel)

                title = (
                    f"{r.display_name} | "
                    f"채널 {channel_ui} | "
                    f"alias {int(r.alias_cnt):,}개 | "
                    f"매출 {fmt_int(r.total_sales)}"
                )

                with st.expander(title, expanded=False):  # ✅ 기본은 닫힘
                    alias_df = query_df(
                        """
                        SELECT
                          ca.alias_name,
                          COALESCE(SUM(sr.amount),0) AS total_sales
                        FROM customer_alias ca
                        LEFT JOIN sales_raw sr ON sr.customer_raw = ca.alias_name
                        WHERE ca.customer_id = %s
                        GROUP BY ca.alias_name
                        ORDER BY total_sales DESC
                        """,
                        (cid,),
                    )

                    render_table(alias_df, number_cols=["total_sales"])

                    new_channel2 = st.selectbox(
                        "매출 구분(채널)",
                        CHANNEL_OPTIONS,
                        index=CHANNEL_OPTIONS.index(channel_ui),
                        key=f"channel_edit_{cid}",
                    )

                    if st.button("💾 채널 저장", key=f"save_channel_{cid}"):
                        update_customer_channel(cid, new_channel2)
                        st.cache_data.clear()
                        st.success("채널 저장 완료")
                        st.rerun()
