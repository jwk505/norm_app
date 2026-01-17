# features/finance.py
from __future__ import annotations

import pandas as pd
import streamlit as st
import altair as alt

from core.db import exec_many, exec_sql, query_df
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


def ensure_finance_tables() -> None:
    exec_sql(
        """
        CREATE TABLE IF NOT EXISTS financial_corp (
          id INT AUTO_INCREMENT PRIMARY KEY,
          corp_name VARCHAR(100) NOT NULL,
          is_active TINYINT DEFAULT 1,
          created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
          UNIQUE KEY uq_fin_corp_name (corp_name)
        )
        """
    )
    exec_sql(
        """
        CREATE TABLE IF NOT EXISTS financial_ownership (
          id INT AUTO_INCREMENT PRIMARY KEY,
          parent_corp_id INT NOT NULL,
          child_corp_id INT NOT NULL,
          ownership_pct DECIMAL(5,2) NOT NULL,
          created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
          UNIQUE KEY uq_fin_owner (parent_corp_id, child_corp_id)
        )
        """
    )
    exec_sql(
        """
        CREATE TABLE IF NOT EXISTS financial_statement (
          id INT AUTO_INCREMENT PRIMARY KEY,
          corp_id INT NOT NULL,
          period VARCHAR(16) NOT NULL,
          statement_type VARCHAR(8) NOT NULL,
          account_name VARCHAR(128) NOT NULL,
          amount DECIMAL(18,2) NOT NULL,
          source_file VARCHAR(255) NULL,
          sheet_name VARCHAR(128) NULL,
          created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
          INDEX idx_fin_stmt_corp_period (corp_id, period),
          INDEX idx_fin_stmt_period (period),
          INDEX idx_fin_stmt_type (statement_type)
        )
        """
    )
    exec_sql(
        """
        CREATE TABLE IF NOT EXISTS financial_account_map (
          id INT AUTO_INCREMENT PRIMARY KEY,
          corp_id INT NULL,
          statement_type VARCHAR(8) NOT NULL,
          major_account VARCHAR(64) NOT NULL,
          detail_account VARCHAR(128) NOT NULL,
          created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
          UNIQUE KEY uq_fin_acc_map (corp_id, statement_type, detail_account)
        )
        """
    )


def _load_corps() -> pd.DataFrame:
    return query_df(
        """
        SELECT id, corp_name, is_active
        FROM financial_corp
        ORDER BY corp_name
        """
    )


def _load_ownership() -> pd.DataFrame:
    return query_df(
        """
        SELECT
          o.id,
          p.corp_name AS parent,
          c.corp_name AS child,
          o.ownership_pct
        FROM financial_ownership o
        JOIN financial_corp p ON p.id = o.parent_corp_id
        JOIN financial_corp c ON c.id = o.child_corp_id
        ORDER BY p.corp_name, c.corp_name
        """
    )


def _load_account_map() -> pd.DataFrame:
    return query_df(
        """
        SELECT
          m.id,
          m.statement_type,
          m.major_account,
          m.detail_account,
          c.corp_name
        FROM financial_account_map m
        LEFT JOIN financial_corp c ON c.id = m.corp_id
        ORDER BY m.statement_type, m.major_account, m.detail_account
        """
    )


def _stmt_label(stmt: str) -> str:
    m = {"IS": "손익", "BS": "대차", "CF": "현금흐름"}
    return m.get(str(stmt).strip().upper(), str(stmt))


def _major_accounts_map() -> dict[str, list[str]]:
    return {
        "IS": ["매출액", "매출원가", "매출총이익", "판매비와관리비", "영업이익", "영업외수익", "영업외비용", "법인세비용", "당기순이익"],
        "BS": ["자산총계", "유동자산", "비유동자산", "부채총계", "유동부채", "비유동부채", "자본총계"],
        "CF": ["영업활동현금흐름", "투자활동현금흐름", "재무활동현금흐름"],
    }


def _computed_major_accounts() -> dict[str, dict[str, list[str]]]:
    return {
        "IS": {
            "매출총이익": ["매출액", "매출원가"],
            "영업이익": ["매출총이익", "판매비와관리비"],
            "당기순이익": ["영업이익", "영업외수익", "영업외비용", "법인세비용"],
        }
    }


def _apply_account_map(detail_df: pd.DataFrame, map_df: pd.DataFrame) -> pd.DataFrame:
    detail_df = detail_df.copy()
    detail_df["major_account"] = ""
    if map_df is None or map_df.empty:
        return detail_df
    map_df = map_df.copy()
    map_df["statement_type"] = map_df["statement_type"].apply(lambda x: str(x).strip().upper())

    global_map = {}
    corp_map = {}
    for r in map_df.itertuples(index=False):
        key = (r.statement_type, str(r.detail_account))
        if r.corp_name:
            corp_map[(str(r.corp_name),) + key] = str(r.major_account)
        else:
            global_map[key] = str(r.major_account)

    def _pick_major(row):
        stmt = str(row.get("statement_type", "")).strip().upper()
        detail = str(row.get("account_name", "")).strip()
        corp = str(row.get("corp_name", "")).strip()
        if corp:
            m = corp_map.get((corp, stmt, detail))
            if m:
                return m
        return global_map.get((stmt, detail), "")

    detail_df["major_account"] = detail_df.apply(_pick_major, axis=1)
    return detail_df


def _sum_account(detail_df: pd.DataFrame, stmt: str, name: str, keywords: list[str] | None = None) -> float:
    sub = detail_df[detail_df["statement_type"] == stmt].copy()
    if sub.empty:
        return 0.0
    mask = sub["major_account"].astype(str) == name
    if not mask.any():
        mask = sub["account_name"].astype(str).str.contains(name, case=False, na=False)
    if keywords:
        kw_mask = False
        for k in keywords:
            kw_mask = kw_mask | sub["account_name"].astype(str).str.contains(k, case=False, na=False)
        mask = mask | kw_mask
    return float(pd.to_numeric(sub.loc[mask, "amount"], errors="coerce").fillna(0).sum())


def _compute_kpis(detail_df: pd.DataFrame, periods: list[str]) -> None:
    if detail_df is None or detail_df.empty or not periods:
        return
    map_df = _load_account_map()
    df = _apply_account_map(detail_df, map_df)

    def _kpi_for(df_p: pd.DataFrame) -> dict:
        sales = _sum_account(df_p, "IS", "매출액")
        cogs = _sum_account(
            df_p,
            "IS",
            "매출원가",
            keywords=["매출원가", "상품매출원가", "제품매출원가", "판매원가"],
        )
        sga = _sum_account(df_p, "IS", "판매비와관리비")
        non_op_inc = _sum_account(df_p, "IS", "영업외수익")
        non_op_exp = _sum_account(df_p, "IS", "영업외비용")
        tax = _sum_account(df_p, "IS", "법인세비용")
        interest_exp = _sum_account(df_p, "IS", "이자비용")
        labor = _sum_account(df_p, "IS", "인건비", keywords=["급여", "인건비", "급료"])

        inv_begin = _sum_account(df_p, "BS", "기초재고", keywords=["기초재고", "기초상품재고액", "기초상품재고"])
        inv_end = _sum_account(df_p, "BS", "기말재고", keywords=["기말재고", "기말상품재고액", "기말상품재고"])
        if not inv_begin and not inv_end:
            inv_begin = _sum_account(
                df_p, "IS", "기초재고", keywords=["기초재고", "기초상품재고액", "기초상품재고"]
            )
            inv_end = _sum_account(
                df_p, "IS", "기말재고", keywords=["기말재고", "기말상품재고액", "기말상품재고"]
            )
        inv_begin = abs(inv_begin)
        inv_end = abs(inv_end)

        inventory = _sum_account(df_p, "BS", "재고자산")
        avg_inventory = (inv_begin + inv_end) / 2.0 if inv_begin and inv_end else inventory

        op_profit = sales - cogs - sga
        net_profit = op_profit + non_op_inc - non_op_exp - tax

        inv_turn = (cogs / avg_inventory) if avg_inventory else 0.0
        inv_days = (avg_inventory / cogs * 365) if cogs and avg_inventory else 0.0
        ar = _sum_account(df_p, "BS", "매출채권")
        ap = _sum_account(df_p, "BS", "매입채무")
        ar_days = (ar / sales * 365) if sales else 0.0
        ap_days = (ap / cogs * 365) if cogs else 0.0
        op_cycle = inv_days + ar_days
        sga_ratio = (sga / sales) if sales else 0.0
        labor_ratio = (labor / sales) if sales else 0.0
        interest_cov = (op_profit / interest_exp) if interest_exp else 0.0
        gross_margin = ((sales - cogs) / sales) if sales else 0.0
        op_margin = (op_profit / sales) if sales else 0.0
        net_margin = (net_profit / sales) if sales else 0.0

        return {
            "매출액": sales,
            "매출이익률(%)": gross_margin * 100,
            "영업이익률(%)": op_margin * 100,
            "당기순이익률(%)": net_margin * 100,
            "재고회전율(x)": inv_turn,
            "재고일": inv_days,
            "매출채권일": ar_days,
            "매입채무일": ap_days,
            "영업주기": op_cycle,
            "판관비율(%)": sga_ratio * 100,
            "인건비율(%)": labor_ratio * 100,
            "이자보상배율(x)": interest_cov,
        }

    if len(periods) > 1:
        st.subheader("기간 비교")
        rows = []
        for p in periods:
            df_p = df[df["period"].astype(str) == str(p)]
            k = _kpi_for(df_p)
            k["기간"] = str(p)
            rows.append(k)
        comp_df = pd.DataFrame(rows)
        cols = ["기간"] + [c for c in comp_df.columns if c != "기간"]
        comp_df = comp_df[cols]
        comp_df = comp_df.sort_values("기간", ascending=True).reset_index(drop=True)
        num_cols = [c for c in comp_df.columns if c != "기간"]
        for c in num_cols:
            if "(%)" in c or "(x)" in c:
                comp_df[c] = pd.to_numeric(comp_df[c], errors="coerce").round(2)
        for c in num_cols:
            comp_df[c] = pd.to_numeric(comp_df[c], errors="coerce")
            if "(%)" in c:
                comp_df[c] = comp_df[c].round(2)

        display_cols = ["기간"] + num_cols
        comp_df = comp_df[display_cols]
        amount_cols = [c for c in num_cols if c in ["매출액"]]
        other_num_cols = [c for c in num_cols if c not in amount_cols]
        format_map = {c: "{:,.1f}" for c in other_num_cols}
        render_table(comp_df, number_cols=amount_cols, number_cols_format=format_map)

        st.subheader("기간 비교 그래프")
        metric_options = [c for c in num_cols]
        default_metrics = [c for c in ["매출액", "영업이익률(%)", "당기순이익률(%)"] if c in metric_options]
        sel_metrics = st.multiselect("그래프 지표 선택", metric_options, default=default_metrics)
        if sel_metrics:
            chart_df = comp_df[["기간"] + sel_metrics].copy()
            long_df = chart_df.melt(id_vars=["기간"], var_name="지표", value_name="값")
            chart = (
                alt.Chart(long_df)
                .mark_line(point=True)
                .encode(
                    x=alt.X("기간:N", axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("값:Q"),
                    color="지표:N",
                )
            )
            st.altair_chart(chart, use_container_width=True)

    period = st.selectbox("지표 기준 기간", periods, index=0, key="fin_kpi_period")
    df = df[df["period"].astype(str) == str(period)]

    sales = _sum_account(df, "IS", "매출액")
    cogs_raw = _sum_account(
        df,
        "IS",
        "매출원가",
        keywords=["매출원가", "상품매출원가", "제품매출원가", "판매원가"],
    )
    sga = _sum_account(df, "IS", "판매비와관리비")
    non_op_inc = _sum_account(df, "IS", "영업외수익")
    non_op_exp = _sum_account(df, "IS", "영업외비용")
    tax = _sum_account(df, "IS", "법인세비용")
    interest_exp = _sum_account(df, "IS", "이자비용")
    labor = _sum_account(df, "IS", "인건비", keywords=["급여", "인건비", "급료"])

    inv_begin = _sum_account(df, "BS", "기초재고", keywords=["기초재고", "기초상품재고액", "기초상품재고"])
    inv_end = _sum_account(df, "BS", "기말재고", keywords=["기말재고", "기말상품재고액", "기말상품재고"])
    if not inv_begin and not inv_end:
        inv_begin = _sum_account(df, "IS", "기초재고", keywords=["기초재고", "기초상품재고액", "기초상품재고"])
        inv_end = _sum_account(df, "IS", "기말재고", keywords=["기말재고", "기말상품재고액", "기말상품재고"])
    inv_begin = abs(inv_begin)
    inv_end = abs(inv_end)
    purchases = _sum_account(df, "IS", "당기상품매입액", keywords=["당기상품매입", "매입액"])
    inventory = _sum_account(df, "BS", "재고자산")
    avg_inventory = 0.0
    inv_basis = "기말재고"
    cogs = cogs_raw
    cogs_basis = "매출원가(계정)"
    op_profit = sales - cogs - sga
    net_profit = op_profit + non_op_inc - non_op_exp - tax

    if inv_begin and inv_end:
        avg_inventory = (inv_begin + inv_end) / 2.0
        inv_basis = "평균재고(기초+기말/2)"
    elif inventory:
        avg_inventory = inventory
        inv_basis = "재고자산(기말)"
    ar = _sum_account(df, "BS", "매출채권")
    ap = _sum_account(df, "BS", "매입채무")

    inv_turn = (cogs / avg_inventory) if avg_inventory else 0.0
    inv_days = (avg_inventory / cogs * 365) if cogs and avg_inventory else 0.0
    ar_days = (ar / sales * 365) if sales else 0.0
    ap_days = (ap / cogs * 365) if cogs else 0.0
    op_cycle = inv_days + ar_days
    sga_ratio = (sga / sales) if sales else 0.0
    labor_ratio = (labor / sales) if sales else 0.0
    interest_cov = (op_profit / interest_exp) if interest_exp else 0.0
    gross_margin = ((sales - cogs) / sales) if sales else 0.0
    op_margin = (op_profit / sales) if sales else 0.0
    net_margin = (net_profit / sales) if sales else 0.0

    st.subheader("경영 지표", help=(
        "계산식 요약\n"
        "- 재고회전율 = 매출원가 / 평균재고\n"
        "- 재고일 = 평균재고 / 매출원가 × 365\n"
        "- 매출채권일 = 매출채권 / 매출액 × 365\n"
        "- 매입채무일 = 매입채무 / 매출원가 × 365\n"
        "- 영업주기 = 재고일 + 매출채권일\n"
        "- 판관비율 = 판매비와관리비 / 매출액\n"
        "- 인건비율 = 인건비 / 매출액\n"
        "- 이자보상배율 = 영업이익 / 이자비용\n"
        "- 매출이익률 = (매출액-매출원가) / 매출액\n"
        "- 영업이익률 = 영업이익 / 매출액\n"
        "- 당기순이익률 = 당기순이익 / 매출액\n"
    ))
    if not sales:
        st.info("매출액이 0이라 비율 지표 계산이 제한됩니다. 매출액 계정 매핑을 확인해주세요.")
    if not cogs or not avg_inventory:
        st.info("매출원가/재고 값이 없으면 재고회전율이 0으로 표시됩니다. 계정 매핑을 확인해주세요.")

    with st.expander("목표/권장 수치 설정", expanded=False):
        t1, t2, t3, t4 = st.columns(4)
        target_gross = t1.number_input("매출이익률 목표(%)", min_value=0.0, max_value=100.0, value=25.0, step=1.0)
        target_op = t2.number_input("영업이익률 목표(%)", min_value=0.0, max_value=100.0, value=8.0, step=0.5)
        target_net = t3.number_input("당기순이익률 목표(%)", min_value=0.0, max_value=100.0, value=5.0, step=0.5)
        target_cycle = t4.number_input("영업주기 목표(일)", min_value=0.0, max_value=3650.0, value=90.0, step=5.0)
        t5, t6, t7, t8 = st.columns(4)
        target_sga = t5.number_input("판관비율 목표(%)", min_value=0.0, max_value=100.0, value=20.0, step=1.0)
        target_labor = t6.number_input("인건비율 목표(%)", min_value=0.0, max_value=100.0, value=12.0, step=1.0)
        target_inv_turn = t7.number_input("재고회전율 목표(x)", min_value=0.0, max_value=100.0, value=6.0, step=0.5)
        target_inv_days = t8.number_input("재고일 목표(일)", min_value=0.0, max_value=3650.0, value=60.0, step=5.0)
    c1, c2, c3, c4 = st.columns(4)
    def _status_badge(value: float, target: float, higher_is_better: bool) -> str:
        if target == 0:
            return "⚪"
        delta = (value - target) / target
        if higher_is_better:
            if delta >= 0.1:
                return "🟢"
            if delta <= -0.1:
                return "🔴"
        else:
            if delta <= -0.1:
                return "🟢"
            if delta >= 0.1:
                return "🔴"
        return "🟡"

    c1.metric(
        f"재고회전율 {_status_badge(inv_turn, target_inv_turn, True)}",
        f"{inv_turn:,.2f}x",
        delta=f"{(inv_turn - target_inv_turn):,.2f}x",
        help=f"매출원가 기준: {cogs_basis}",
    )
    c2.metric(
        f"재고일 {_status_badge(inv_days, target_inv_days, False)}",
        f"{inv_days:,.0f}일",
        delta=f"{(inv_days - target_inv_days):,.0f}일",
        help=f"재고 기준: {inv_basis}",
    )
    c3.metric(
        f"매출채권일 {_status_badge(ar_days, 90.0, False)}",
        f"{ar_days:,.0f}일",
        help="매출채권 / 매출액 × 365",
    )
    c4.metric(
        f"매입채무일 {_status_badge(ap_days, 60.0, False)}",
        f"{ap_days:,.0f}일",
        help="매입채무 / 매출원가 × 365",
    )

    c5, c6, c7, c8 = st.columns(4)
    c5.metric(
        f"영업주기 {_status_badge(op_cycle, target_cycle, False)}",
        f"{op_cycle:,.0f}일",
        delta=f"{(op_cycle - target_cycle):,.0f}일",
        help="재고일 + 매출채권일",
    )
    c6.metric(
        f"판관비율 {_status_badge(sga_ratio*100, target_sga, False)}",
        f"{sga_ratio*100:,.1f}%",
        delta=f"{(sga_ratio*100 - target_sga):,.1f}%",
        help="판매비와관리비 / 매출액",
    )
    c7.metric(
        f"인건비율 {_status_badge(labor_ratio*100, target_labor, False)}",
        f"{labor_ratio*100:,.1f}%",
        delta=f"{(labor_ratio*100 - target_labor):,.1f}%",
        help="인건비 / 매출액",
    )
    c8.metric(
        f"이자보상배율 {_status_badge(interest_cov, 3.0, True)}",
        f"{interest_cov:,.2f}x",
        help="영업이익 / 이자비용",
    )

    c9, c10, c11, c12 = st.columns(4)
    c9.metric(
        f"매출이익률 {_status_badge(gross_margin*100, target_gross, True)}",
        f"{gross_margin*100:,.1f}%",
        delta=f"{(gross_margin*100 - target_gross):,.1f}%",
        help="(매출액-매출원가) / 매출액",
    )
    c10.metric(
        f"영업이익률 {_status_badge(op_margin*100, target_op, True)}",
        f"{op_margin*100:,.1f}%",
        delta=f"{(op_margin*100 - target_op):,.1f}%",
        help="영업이익 / 매출액",
    )
    c11.metric(
        f"당기순이익률 {_status_badge(net_margin*100, target_net, True)}",
        f"{net_margin*100:,.1f}%",
        delta=f"{(net_margin*100 - target_net):,.1f}%",
        help="당기순이익 / 매출액",
    )
    c12.metric("영업주기 목표", f"{target_cycle:,.0f}일")

    with st.expander("계산 근거", expanded=False):
        basis_rows = [
            {"항목": "매출액", "금액": sales},
            {"항목": "매출원가", "금액": cogs_raw},
            {"항목": "기초재고", "금액": inv_begin},
            {"항목": "기말재고", "금액": inv_end},
            {"항목": "평균재고", "금액": avg_inventory},
        ]
        basis_df = pd.DataFrame(basis_rows)
        render_table(basis_df, number_cols=["금액"])


def _render_major_accounts(detail_df: pd.DataFrame) -> None:
    if detail_df is None or detail_df.empty:
        return

    majors = _major_accounts_map()
    map_df = _load_account_map()
    detail_df = _apply_account_map(detail_df, map_df)
    computed = _computed_major_accounts()
    summary_rows = []
    for stmt, major_list in majors.items():
        sub = detail_df[detail_df["statement_type"] == stmt].copy()
        sub["statement_type"] = sub["statement_type"].apply(_stmt_label)
        with st.expander(f"{_stmt_label(stmt)} 주요계정", expanded=True):
            matched_idx = set()
            major_sum = {}

            def _mask_for_major(name: str):
                mask = sub["major_account"].astype(str) == name
                if not mask.any():
                    mask = sub["account_name"].astype(str).str.contains(name, case=False, na=False)
                return mask

            def _sum_for(name: str) -> float:
                mask = _mask_for_major(name)
                return float(pd.to_numeric(sub.loc[mask, "amount"], errors="coerce").fillna(0).sum())

            def _calc_major(name: str) -> float:
                if stmt in computed and name in computed[stmt]:
                    parts = computed[stmt][name]
                    if name == "매출총이익":
                        return _sum_for("매출액") - _sum_for("매출원가")
                    if name == "영업이익":
                        return _calc_major("매출총이익") - _sum_for("판매비와관리비")
                    if name == "당기순이익":
                        return _calc_major("영업이익") + _sum_for("영업외수익") - _sum_for("영업외비용") - _sum_for("법인세비용")
                    return _sum_for(name)
                return _sum_for(name)

            for major in major_list:
                mask = _mask_for_major(major)
                matched_idx.update(sub[mask].index.tolist())
                sum_val = _calc_major(major)
                major_sum[major] = sum_val
                summary_rows.append({"재무제표": _stmt_label(stmt), "주요계정": major, "금액": sum_val})
                with st.expander(f"{major} | {sum_val:,.0f}", expanded=False):
                    if stmt in computed and major in computed[stmt]:
                        parts = computed[stmt][major]
                        part_rows = [{"구성": p, "금액": major_sum.get(p, _sum_for(p))} for p in parts]
                        part_df = pd.DataFrame(part_rows)
                        render_table(part_df.rename(columns={"구성": "구성항목"}), number_cols=["금액"])
                    if mask.any():
                        view_cols = ["period", "account_name", "amount"]
                        if "corp_name" in sub.columns:
                            view_cols.insert(0, "corp_name")
                        view = sub.loc[mask, view_cols].copy()
                        view = view.rename(
                            columns={
                                "corp_name": "법인",
                                "period": "기간",
                                "account_name": "계정",
                                "amount": "금액",
                            }
                        )
                        render_table(view, number_cols=["금액"])
                    else:
                        st.info("해당 주요계정에 속한 데이터가 없습니다.")

            others = sub.drop(index=list(matched_idx)) if matched_idx else sub
            sum_val = float(pd.to_numeric(others["amount"], errors="coerce").fillna(0).sum())
            summary_rows.append({"재무제표": _stmt_label(stmt), "주요계정": "기타", "금액": sum_val})
            with st.expander(f"기타 | {sum_val:,.0f}", expanded=False):
                if not others.empty:
                    view_cols = ["period", "account_name", "amount"]
                    if "corp_name" in others.columns:
                        view_cols.insert(0, "corp_name")
                    view = others[view_cols].rename(
                        columns={
                            "corp_name": "법인",
                            "period": "기간",
                            "account_name": "계정",
                            "amount": "금액",
                        }
                    )
                    render_table(view, number_cols=["금액"])
                else:
                    st.info("기타로 분류된 데이터가 없습니다.")

    if summary_rows:
        st.subheader("주요계정 요약")
        summary_df = pd.DataFrame(summary_rows)
        render_table(summary_df, number_cols=["금액"])


def show_finance_page():
    st.header("재무제표 분석")

    try:
        query_df("SELECT 1")
    except Exception as e:
        st.error("DB 연결 실패로 재무제표 화면을 불러오지 못했습니다.")
        st.caption(str(e))
        return

    ensure_finance_tables()

    if not _table_exists("financial_statement"):
        st.error("재무제표 테이블이 없습니다. 업로드를 먼저 진행해주세요.")
        return

    corp_df = _load_corps()
    corp_names = corp_df["corp_name"].tolist() if not corp_df.empty else []

    with st.expander("법인 관리", expanded=False):
        c1, c2 = st.columns([3, 1])
        new_corp = c1.text_input("법인명 추가", value="")
        if c2.button("등록"):
            if not new_corp.strip():
                st.warning("법인명을 입력해주세요.")
            else:
                exec_sql(
                    "INSERT INTO financial_corp (corp_name) VALUES (%s) ON DUPLICATE KEY UPDATE corp_name=corp_name",
                    (new_corp.strip(),),
                )
                st.success("등록 완료")
                st.rerun()

        if not corp_df.empty:
            render_table(corp_df.rename(columns={"corp_name": "법인명"}), number_cols=["id"])

    with st.expander("지분율 관리", expanded=False):
        if corp_df.empty:
            st.info("법인을 먼저 등록해주세요.")
        else:
            parent = st.selectbox("모회사", corp_names, index=0)
            child_candidates = [c for c in corp_names if c != parent]
            child = st.selectbox("자회사", child_candidates, index=0)
            pct = st.number_input("지분율(%)", min_value=0.0, max_value=100.0, value=100.0, step=1.0)
            if st.button("지분율 저장"):
                parent_id = int(corp_df[corp_df["corp_name"] == parent].iloc[0]["id"])
                child_id = int(corp_df[corp_df["corp_name"] == child].iloc[0]["id"])
                exec_sql(
                    """
                    INSERT INTO financial_ownership (parent_corp_id, child_corp_id, ownership_pct)
                    VALUES (%s, %s, %s)
                    ON DUPLICATE KEY UPDATE ownership_pct=VALUES(ownership_pct)
                    """,
                    (parent_id, child_id, float(pct)),
                )
                st.success("저장 완료")
                st.rerun()

            own_df = _load_ownership()
            if not own_df.empty:
                render_table(
                    own_df.rename(columns={"parent": "모회사", "child": "자회사", "ownership_pct": "지분율"}),
                    number_cols=["지분율"],
                )

    with st.expander("계정 매핑", expanded=False):
        map_df = _load_account_map()
        stmt_options = ["IS", "BS", "CF"]
        majors = _major_accounts_map()
        if not corp_df.empty:
            corp_options = ["(공통)"] + corp_df["corp_name"].tolist()
        else:
            corp_options = ["(공통)"]

        c1, c2, c3 = st.columns(3)
        map_corp = c1.selectbox("법인", corp_options, index=0)
        map_stmt = c2.selectbox("재무제표", stmt_options, index=0, format_func=_stmt_label)
        map_major = c3.selectbox("주요계정", majors.get(map_stmt, []), index=0)

        acc_rows = query_df(
            """
            SELECT DISTINCT account_name
            FROM financial_statement
            WHERE statement_type = %s
            ORDER BY account_name
            """,
            (map_stmt,),
        )
        acc_list = acc_rows["account_name"].astype(str).tolist() if not acc_rows.empty else []
        mapped_rows = query_df(
            """
            SELECT detail_account
            FROM financial_account_map
            WHERE statement_type = %s
            """,
            (map_stmt,),
        )
        mapped_set = set(mapped_rows["detail_account"].astype(str).tolist()) if not mapped_rows.empty else set()
        show_mapped = st.checkbox("이미 매핑된 계정도 표시", value=False)
        if not show_mapped:
            acc_list = [a for a in acc_list if a not in mapped_set]
        acc_q = st.text_input("세부계정 검색", value="")
        if acc_q.strip():
            acc_list = [a for a in acc_list if acc_q.strip() in a]
        detail_sel = st.selectbox("세부계정 선택", acc_list if acc_list else ["(없음)"])

        if st.button("매핑 저장"):
            if detail_sel == "(없음)":
                st.warning("세부계정을 선택해주세요.")
            else:
                corp_id = None
                if map_corp != "(공통)":
                    corp_id = int(corp_df[corp_df["corp_name"] == map_corp].iloc[0]["id"])
                exec_sql(
                    """
                    INSERT INTO financial_account_map (corp_id, statement_type, major_account, detail_account)
                    VALUES (%s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE major_account=VALUES(major_account)
                    """,
                    (corp_id, map_stmt, map_major, detail_sel),
                )
                st.success("매핑 저장 완료")
                st.rerun()

        st.subheader("매핑 목록")
        if map_df is not None and not map_df.empty:
            show_map = map_df.rename(
                columns={
                    "corp_name": "법인",
                    "statement_type": "재무제표",
                    "major_account": "주요계정",
                    "detail_account": "세부계정",
                }
            )
            show_map["재무제표"] = show_map["재무제표"].apply(_stmt_label)
            render_table(show_map, number_cols=["id"])

            del_corp = st.selectbox("삭제 대상 법인", ["(공통)"] + corp_df["corp_name"].tolist(), index=0)
            del_stmt = st.selectbox("삭제 대상 재무제표", stmt_options, index=0, format_func=_stmt_label)
            del_detail = st.text_input("삭제 대상 세부계정", value="")
            if st.button("매핑 삭제"):
                if not del_detail.strip():
                    st.warning("삭제할 세부계정을 입력해주세요.")
                else:
                    if del_corp == "(공통)":
                        exec_sql(
                            """
                            DELETE FROM financial_account_map
                            WHERE corp_id IS NULL AND statement_type=%s AND detail_account=%s
                            """,
                            (del_stmt, del_detail.strip()),
                        )
                    else:
                        del_corp_id = int(corp_df[corp_df["corp_name"] == del_corp].iloc[0]["id"])
                        exec_sql(
                            """
                            DELETE FROM financial_account_map
                            WHERE corp_id=%s AND statement_type=%s AND detail_account=%s
                            """,
                            (del_corp_id, del_stmt, del_detail.strip()),
                        )
                    st.success("매핑 삭제 완료")
                    st.rerun()

    with st.expander("세부계정 관리", expanded=False):
        stmt_filter = st.selectbox("재무제표 필터", ["(전체)"] + stmt_options, index=0, format_func=_stmt_label)
        period_filter = st.text_input("기간 필터(부분일치)", value="")
        corp_filter = st.selectbox("법인 필터", ["(전체)"] + corp_names, index=0)
        account_filter = st.text_input("계정 필터(부분일치)", value="")

        where = ["1=1"]
        params = []
        if stmt_filter != "(전체)":
            where.append("s.statement_type = %s")
            params.append(stmt_filter)
        if period_filter.strip():
            where.append("s.period LIKE %s")
            params.append(f"%{period_filter.strip()}%")
        if corp_filter != "(전체)":
            where.append("c.corp_name = %s")
            params.append(corp_filter)
        if account_filter.strip():
            where.append("s.account_name LIKE %s")
            params.append(f"%{account_filter.strip()}%")

        stmt_rows = query_df(
            f"""
            SELECT s.id, c.corp_name, s.period, s.statement_type, s.account_name, s.amount
            FROM financial_statement s
            JOIN financial_corp c ON c.id = s.corp_id
            WHERE {' AND '.join(where)}
            ORDER BY s.period DESC, c.corp_name, s.account_name
            LIMIT 500
            """,
            tuple(params),
        )
        if stmt_rows.empty:
            st.info("조건에 맞는 세부계정이 없습니다.")
        else:
            view = stmt_rows.copy()
            view["statement_type"] = view["statement_type"].apply(_stmt_label)
            view = view.rename(
                columns={
                    "id": "ID",
                    "corp_name": "법인",
                    "period": "기간",
                    "statement_type": "재무제표",
                    "account_name": "계정",
                    "amount": "금액",
                }
            )
            render_table(view, number_cols=["ID", "금액"])

            edit_id = st.number_input("수정/삭제 ID", min_value=0, value=0, step=1)
            new_account = st.text_input("수정 계정명(선택)", value="")
            new_amount = st.text_input("수정 금액(선택)", value="")
            new_period = st.text_input("수정 기간(선택)", value="")
            c1, c2 = st.columns(2)
            if c1.button("세부계정 수정"):
                if edit_id <= 0:
                    st.warning("ID를 입력해주세요.")
                else:
                    updates = []
                    update_params = []
                if new_account.strip():
                    updates.append("account_name=%s")
                    update_params.append(new_account.strip())
                if new_amount.strip():
                    updates.append("amount=%s")
                    update_params.append(float(new_amount.replace(",", "")))
                if new_period.strip():
                    updates.append("period=%s")
                    update_params.append(new_period.strip())
                    if not updates:
                        st.warning("수정할 값을 입력해주세요.")
                    else:
                        update_params.append(int(edit_id))
                        exec_sql(
                            f"UPDATE financial_statement SET {', '.join(updates)} WHERE id=%s",
                            tuple(update_params),
                        )
                        st.success("수정 완료")
                        st.rerun()
            if c2.button("세부계정 삭제"):
                if edit_id <= 0:
                    st.warning("ID를 입력해주세요.")
                else:
                    exec_sql("DELETE FROM financial_statement WHERE id=%s", (int(edit_id),))
                    st.success("삭제 완료")
                    st.rerun()

    st.divider()
    st.subheader("재무제표 조회")

    periods = query_df(
        "SELECT DISTINCT period FROM financial_statement ORDER BY period DESC"
    )["period"].astype(str).tolist()
    if not periods:
        st.info("재무제표 데이터가 없습니다. 업로드 후 다시 확인해주세요.")
        return

    selected_periods = st.multiselect("기간 선택", periods, default=periods[:1])
    stmt_types = ["IS", "BS", "CF"]
    selected_types = st.multiselect("재무제표 구분", stmt_types, default=stmt_types)

    view_mode = st.radio("조회 모드", ["개별", "연결"], horizontal=True)

    if view_mode == "개별":
        selected_corps = st.multiselect("법인 선택", corp_names, default=corp_names[:1] if corp_names else [])
        if not selected_corps:
            st.info("법인을 선택해주세요.")
            return
        corp_ids = corp_df[corp_df["corp_name"].isin(selected_corps)]["id"].astype(int).tolist()
        rows = query_df(
            """
            SELECT c.corp_name, s.period, s.statement_type, s.account_name, s.amount
            FROM financial_statement s
            JOIN financial_corp c ON c.id = s.corp_id
            WHERE s.corp_id IN ({})
              AND s.period IN ({})
              AND s.statement_type IN ({})
            """.format(
                ",".join(["%s"] * len(corp_ids)),
                ",".join(["%s"] * len(selected_periods)),
                ",".join(["%s"] * len(selected_types)),
            ),
            tuple(corp_ids + selected_periods + selected_types),
        )
        if rows.empty:
            st.info("해당 조건의 데이터가 없습니다.")
            return
        detail_df = rows.copy()
        detail_df["statement_type"] = detail_df["statement_type"].apply(_stmt_label)
        view = detail_df.rename(
            columns={
                "corp_name": "법인",
                "period": "기간",
                "statement_type": "재무제표",
                "account_name": "계정",
                "amount": "금액",
            }
        )
        st.caption("재무제표 주요계정 기준으로 보기")
        _compute_kpis(rows, selected_periods)
        _render_major_accounts(rows)
        csv_bytes = view.to_csv(index=False).encode("utf-8-sig")
        st.download_button("엑셀 다운로드(CSV)", data=csv_bytes, file_name="financial.xlsx.csv", mime="text/csv")
        return

    parent = st.selectbox("모회사 선택", corp_names, index=0)
    if not parent:
        st.info("모회사를 선택해주세요.")
        return
    parent_id = int(corp_df[corp_df["corp_name"] == parent].iloc[0]["id"])
    own = query_df(
        """
        SELECT child_corp_id, ownership_pct
        FROM financial_ownership
        WHERE parent_corp_id = %s
        """,
        (parent_id,),
    )
    weights = {parent_id: 1.0}
    if not own.empty:
        for r in own.itertuples(index=False):
            weights[int(r.child_corp_id)] = float(r.ownership_pct) / 100.0

    corp_ids = list(weights.keys())
    rows = query_df(
        """
        SELECT s.corp_id, s.period, s.statement_type, s.account_name, s.amount
        FROM financial_statement s
        WHERE s.corp_id IN ({})
          AND s.period IN ({})
          AND s.statement_type IN ({})
        """.format(
            ",".join(["%s"] * len(corp_ids)),
            ",".join(["%s"] * len(selected_periods)),
            ",".join(["%s"] * len(selected_types)),
        ),
        tuple(corp_ids + selected_periods + selected_types),
    )
    if rows.empty:
        st.info("해당 조건의 데이터가 없습니다.")
        return

    rows["weight"] = rows["corp_id"].map(weights)
    rows["amount"] = pd.to_numeric(rows["amount"], errors="coerce").fillna(0) * rows["weight"]
    cons = (
        rows.groupby(["period", "statement_type", "account_name"], as_index=False)["amount"]
        .sum()
        .sort_values(["period", "statement_type", "amount"], ascending=[False, True, False])
    )
    detail_df = cons.copy()
    detail_df["statement_type"] = detail_df["statement_type"].apply(_stmt_label)
    view = detail_df.rename(
        columns={"period": "기간", "statement_type": "재무제표", "account_name": "계정", "amount": "금액"}
    )
    st.caption("재무제표 주요계정 기준으로 보기")
    _compute_kpis(cons, selected_periods)
    _render_major_accounts(cons)
    csv_bytes = view.to_csv(index=False).encode("utf-8-sig")
    st.download_button("엑셀 다운로드(CSV)", data=csv_bytes, file_name="financial_consolidated.csv", mime="text/csv")
