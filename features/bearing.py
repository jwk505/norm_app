# features/bearing.py
from __future__ import annotations

import re
from typing import Optional

import pandas as pd
import streamlit as st


def _norm_key(x: str) -> str:
    return re.sub(r"[\s_]+", "", str(x)).strip().lower()


def _detect_item_col(columns) -> Optional[str]:
    cols = list(columns)
    col_map = {_norm_key(c): c for c in cols}
    tokens = [
        "????",
        "???",
        "??",
        "??",
        "???",
        "item",
        "itemname",
        "item_name",
    ]
    token_keys = [_norm_key(t) for t in tokens]

    for tk in token_keys:
        if tk in col_map:
            return col_map[tk]
    for c in cols:
        ck = _norm_key(c)
        if any(tk and tk in ck for tk in token_keys):
            return c
    return None


def _detect_maker_col(columns) -> Optional[str]:
    cols = list(columns)
    col_map = {_norm_key(c): c for c in cols}
    tokens = [
        "maker",
        "brand",
        "manufacturer",
        "make",
        "???",
        "???",
        "???",
    ]
    token_keys = [_norm_key(t) for t in tokens]

    for tk in token_keys:
        if tk in col_map:
            return col_map[tk]
    for c in cols:
        ck = _norm_key(c)
        if any(tk and tk in ck for tk in token_keys):
            return c
    return None


def _normalize_maker(val: Optional[str]) -> str:
    if not val:
        return ""
    s = str(val).strip().upper()
    s = re.split(r"\s+", s)[0]
    aliases = {
        "SCHAEFFLER": "FAG",
        "INA": "INA",
        "FAG": "FAG",
        "SKF": "SKF",
        "NSK": "NSK",
        "NTN": "NTN",
        "KOYO": "KOYO",
        "JTEKT": "KOYO",
        "TIMKEN": "TIMKEN",
        "NACHI": "NACHI",
    }
    return aliases.get(s, s)


def _infer_bearing_type_from_number(num: str) -> str:
    if not num:
        return ""
    s = str(num)
    if re.match(r"^(302|303|320|322|323|329|330|331|332|333)", s):
        return "TaperedRoller"
    if s.startswith(("22", "23", "24")):
        return "SphericalRoller"
    if s.startswith(("70", "72", "73", "74", "7")):
        return "AngularContactBall"
    if s.startswith(("60", "62", "63", "64", "68", "69", "6")):
        return "DeepGrooveBall"
    if s.startswith(("5",)):
        return "ThrustBall"
    if s.startswith(("3",)):
        return "CylindricalRoller"
    return ""


def _parse_bearing_name(name: str, maker: Optional[str] = None) -> dict:
    s = str(name).upper().strip()
    s = s.replace("(미사용)", "")
    s = s.replace("(???)", "")
    s = s.strip()
    s_clean = re.sub(r"[\\/,_\-.]+", " ", s)
    tokens = [t for t in re.split(r"\s+", s_clean) if t]

    bearing_type = ""
    bearing_number = ""
    m = re.search(r"\b(\d{2,5}(?:/\d{2,3})?)\b", s)
    if m:
        bearing_number = m.group(1)
        prefix = s[: m.start()].strip()
        mtype = re.search(r"([A-Z]{1,4})$", prefix)
        if mtype:
            prefix_code = mtype.group(1)
            prefix_map = {
                "NU": "CylindricalRoller",
                "NJ": "CylindricalRoller",
                "N": "CylindricalRoller",
                "NA": "NeedleRoller",
                "NK": "NeedleRoller",
                "RNA": "NeedleRoller",
                "HK": "NeedleRoller",
                "KT": "NeedleRoller",
                "T": "TaperedRoller",
            }
            bearing_type = prefix_map.get(prefix_code, prefix_code)
        if not bearing_type:
            bearing_type = _infer_bearing_type_from_number(bearing_number)

    maker_norm = _normalize_maker(maker)

    seal_type = ""
    seal_map = {
        "2RS1": "2RS",
        "2RSH": "2RS",
        "2RS": "2RS",
        "RS": "RS",
        "2RZ": "2RZ",
        "RZ": "RZ",
        "ZZ": "ZZ",
        "2Z": "ZZ",
        "Z": "Z",
        "LLU": "2RS",
        "LLB": "2RS",
        "LLH": "2RS",
        "DDU": "2RS",
        "VV": "VV",
        "2RSR": "2RS",
        "2ZR": "ZZ",
        "OPEN": "OPEN",
    }
    seal_tokens = tokens
    if m:
        suffix = s[m.end():]
        suffix_tokens = [t for t in re.split(r"\s+", re.sub(r"[\\/,_\-.]+", " ", suffix)) if t]
        seal_tokens = suffix_tokens
    for t in seal_tokens + tokens:
        if t in seal_map:
            seal_type = seal_map[t]
            break

    taper_diameter = ""
    tm = re.search(r"\bK(\d{2,3})\b", s)
    if tm:
        taper_diameter = tm.group(1)
    else:
        tm2 = re.search(r"\b(\d{2,3})K\b", s)
        if tm2:
            taper_diameter = tm2.group(1)

    cage_material = ""
    for t in tokens:
        if t.startswith(("TN", "TV")):
            cage_material = t
            break
    if not cage_material:
        for t in tokens:
            if t in {"M", "J", "P", "F", "Y", "A", "E", "TVP", "TN9"}:
                cage_material = t
                break

    clearnace = ""
    cm = re.search(r"\b(C[2-5]|CN)\b", s)
    if cm:
        clearnace = cm.group(1)

    grease_type = ""
    gm = re.search(r"\b(LT|HT|G\d?|EP|MOLY|NLGI\d)\b", s)
    if gm:
        grease_type = gm.group(1)

    precision = ""
    pm = re.search(r"\bP[0-6]\b", s)
    if pm:
        precision = pm.group(0)
    else:
        am = re.search(r"\bABEC\s?-?\s?([1-9])\b", s)
        if am:
            precision = f"ABEC{am.group(1)}"

    return {
        "BearingType": bearing_type,
        "BearingNumber": bearing_number,
        "SealType": seal_type,
        "TaperDiameter": taper_diameter,
        "cageMaterial": cage_material,
        "Clearnace": clearnace,
        "greaseType": grease_type,
        "precision": precision,
        "Maker": maker_norm,
    }


def show_bearing_standard_page():
    st.header("Bearing Standard Classification")
    st.caption("Upload an Excel file with item names to classify standard bearing fields.")

    up = st.file_uploader("Bearing items Excel (.xlsx)", type=["xlsx"], key="bearing_upload")
    if up is None:
        st.info("Upload an Excel file to start classification.")
        return

    try:
        status = st.status("Loading Excel...", expanded=True)
        xls = pd.ExcelFile(up, engine="openpyxl")
        sheet_name = st.selectbox("Sheet", xls.sheet_names, index=0, key="bearing_sheet")
        df = pd.read_excel(up, sheet_name=sheet_name, engine="openpyxl")
        status.write("Excel loaded")
    except Exception as e:
        st.error(f"Failed to read Excel: {e}")
        return

    df.columns = [str(c).strip() if pd.notna(c) else "" for c in df.columns]
    auto_item = _detect_item_col(df.columns)
    item_idx = list(df.columns).index(auto_item) if auto_item in df.columns else 0
    item_col = st.selectbox("Item name column", list(df.columns), index=item_idx, key="bearing_item_col")
    if not item_col:
        st.error("Select the item name column.")
        return

    auto_maker = _detect_maker_col(df.columns)
    maker_options = ["(none)"] + list(df.columns)
    maker_idx = maker_options.index(auto_maker) if auto_maker in maker_options else 0
    maker_col = st.selectbox("Maker column (optional)", maker_options, index=maker_idx, key="bearing_maker_col")
    if maker_col == "(none)":
        maker_col = None

    base_cols = [item_col] + ([maker_col] if maker_col else [])
    base = df[base_cols].copy()
    base = base.rename(columns={item_col: "ItemName"})
    if maker_col:
        base = base.rename(columns={maker_col: "Maker"})

    base["ItemName"] = base["ItemName"].astype("string").str.strip()
    base["ItemName"] = base["ItemName"].str.replace(r"^[\\.-]+$", "", regex=True)
    base = base[base["ItemName"].notna() & (base["ItemName"] != "")]

    status.write("Classifying bearing items...")
    if maker_col:
        parsed = base.apply(lambda r: _parse_bearing_name(r["ItemName"], r["Maker"]), axis=1).apply(pd.Series)
    else:
        parsed = base["ItemName"].apply(_parse_bearing_name).apply(pd.Series)
    out = pd.concat([base, parsed], axis=1)
    status.update(label="Classification complete", state="complete", expanded=False)

    st.subheader("Preview")
    st.dataframe(out.head(200), use_container_width=True, hide_index=True)

    csv_bytes = out.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Download CSV",
        data=csv_bytes,
        file_name="bearing_standard_items.csv",
        mime="text/csv",
    )
