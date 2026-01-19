# features/bearing.py
from __future__ import annotations

import re
import json
from typing import Optional

import pandas as pd
import streamlit as st

from core.db import exec_many, exec_sql, query_df


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


def _detect_code_col(columns) -> Optional[str]:
    cols = list(columns)
    col_map = {_norm_key(c): c for c in cols}
    tokens = [
        "code",
        "itemcode",
        "item_code",
        "sku",
        "????",
        "??",
        "??",
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
    if s.startswith(("11", "12", "13")):
        return "SelfAligningBall"
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
    if "#" in s:
        s = s.split("#", 1)[0].strip()
    s = s.replace("(미사용)", "")
    s = s.replace("(???)", "")
    s = s.strip()
    s_clean = re.sub(r"[\\/,_\-.]+", " ", s)
    tokens = [t for t in re.split(r"\s+", s_clean) if t]

    bearing_type = ""
    bearing_number = ""
    bearing_prefix = ""
    m = re.search(r"\b(\d{2,5}(?:/\d{2,3})?)(?=[A-Z]|$)", s)
    if m:
        bearing_number = m.group(1)
        prefix = s[: m.start()].strip()
        mtype = re.search(r"([A-Z]{1,4})$", prefix)
        if mtype:
            prefix_code = mtype.group(1)
            bearing_prefix = prefix_code
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
    if not bearing_number:
        m_alt = re.search(r"(\d{2,5})([A-Z]{1,5})", s)
        if m_alt:
            bearing_number = m_alt.group(1)
            if not bearing_type:
                bearing_type = _infer_bearing_type_from_number(bearing_number)
    if not bearing_number:
        for t in tokens:
            if re.fullmatch(r"\d{2,5}(?:/\d{2,3})?", t):
                bearing_number = t
                if not bearing_type:
                    bearing_type = _infer_bearing_type_from_number(bearing_number)
                break

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
    suffix_tokens = []
    seal_tokens = tokens
    if m:
        suffix = s[m.end():]
        suffix_tokens = [t for t in re.split(r"\s+", re.sub(r"[\\/,_\-.]+", " ", suffix)) if t]
        seal_tokens = suffix_tokens
    if not seal_tokens:
        suffix_tokens = [t for t in re.split(r"\s+", s_clean) if t]
        seal_tokens = suffix_tokens
    # handle suffix appended right after bearing number (e.g., 6206ZZ)
    if not seal_tokens:
        m2 = re.search(r"\b(\d{2,5}(?:/\d{2,3})?)([A-Z0-9]{1,5})\b", s)
        if m2:
            seal_tokens = [m2.group(2)]
            suffix_tokens = [m2.group(2)]
    if not seal_tokens:
        for t in tokens:
            m3 = re.search(r"(\d{2,5}(?:/\d{2,3})?)([A-Z0-9]{1,5})", t)
            if m3:
                seal_tokens = [m3.group(2)]
                suffix_tokens = [m3.group(2)]
                break
    if not suffix_tokens and tokens:
        suffix_tokens = [t for t in tokens if t != bearing_number]
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
            if t.startswith(("TV", "TN")):
                cage_material = t
                break
    if not cage_material:
        m2 = re.search(r"\b(\d{2,5}(?:/\d{2,3})?)([A-Z]{1,6})\b", s)
        if m2:
            suf = m2.group(2)
            if suf.startswith(("TN", "TV")):
                cage_material = suf
    if not cage_material:
        for t in tokens:
            if t in {"M", "J", "P", "F", "Y", "A", "E", "TVP", "TN9"}:
                cage_material = t
                break
    if not cage_material:
        cage_material = "STEEL"

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

    # If number was detected but type not, infer from number only
    if bearing_number and not bearing_type:
        bearing_type = _infer_bearing_type_from_number(bearing_number)
    suffix_display_tokens = [t for t in suffix_tokens if t and t != bearing_number]
    if bearing_number:
        cleaned = []
        for t in suffix_display_tokens:
            if bearing_number in t and t != bearing_number:
                t = t.replace(bearing_number, "")
            if t:
                cleaned.append(t)
        suffix_display_tokens = cleaned
    suffix_raw = " ".join(suffix_display_tokens).strip()
    suffix_desc_map = {
        "ZZ": "Steel shields both sides",
        "2Z": "Steel shields both sides",
        "Z": "Steel shield one side",
        "2RS": "Rubber seals both sides",
        "RS": "Rubber seal one side",
        "2RZ": "Low friction seals both sides",
        "RZ": "Low friction seal one side",
        "VV": "Contact seals both sides",
        "OPEN": "Open bearing",
        "TV": "Polyamide cage",
        "TVH": "Polyamide cage",
        "TVP": "Polyamide cage",
        "TN": "Polyamide cage",
        "TN9": "Polyamide cage",
        "M": "Machined brass cage",
        "J": "Pressed steel cage",
        "C2": "Reduced clearance",
        "CN": "Normal clearance",
        "C3": "Increased clearance",
        "C4": "Greater clearance",
        "C5": "Very large clearance",
        "P0": "Precision class P0",
        "P5": "Precision class P5",
        "P6": "Precision class P6",
    }
    suffix_desc_parts = []
    for t in suffix_display_tokens:
        desc = suffix_desc_map.get(t)
        if desc:
            suffix_desc_parts.append(desc)
    suffix_desc = ", ".join(suffix_desc_parts)
    return {
        "BearingType": bearing_type,
        "BearingPrefix": bearing_prefix,
        "BearingNumber": bearing_number,
        "Suffix": suffix_raw,
        "SuffixDesc": suffix_desc,
        "SealType": seal_type,
        "TaperDiameter": taper_diameter,
        "cageMaterial": cage_material,
        "Clearnace": clearnace,
        "greaseType": grease_type,
        "precision": precision,
        "???": maker_norm,
    }


def ensure_catalog_tables() -> None:
    exec_sql(
        """
        CREATE TABLE IF NOT EXISTS catalog_source (
          id BIGINT AUTO_INCREMENT PRIMARY KEY,
          source_name VARCHAR(200) NOT NULL,
          source_type VARCHAR(50) NOT NULL,
          uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
          notes TEXT,
          UNIQUE KEY uq_catalog_source (source_name, source_type)
        )
        """
    )
    exec_sql(
        """
        CREATE TABLE IF NOT EXISTS catalog_item_raw (
          id BIGINT AUTO_INCREMENT PRIMARY KEY,
          source_id BIGINT NOT NULL,
          item_name_raw VARCHAR(500) NOT NULL,
          item_code_raw VARCHAR(200),
          maker_raw VARCHAR(200),
          extra_json JSON,
          created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
          UNIQUE KEY uq_source_item (source_id, item_name_raw(190), item_code_raw(64), maker_raw(64)),
          INDEX idx_raw_name (item_name_raw),
          INDEX idx_raw_source (source_id)
        )
        """
    )
    exec_sql(
        """
        CREATE TABLE IF NOT EXISTS catalog_item_norm (
          id BIGINT AUTO_INCREMENT PRIMARY KEY,
          item_name_std VARCHAR(300) NOT NULL,
          item_code_std VARCHAR(200),
          maker_std VARCHAR(200),
          category VARCHAR(100),
          status VARCHAR(30) NOT NULL DEFAULT 'active',
          created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
          updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
          UNIQUE KEY uq_norm (item_name_std(190), item_code_std(64), maker_std(64))
        )
        """
    )
    exec_sql(
        """
        CREATE TABLE IF NOT EXISTS catalog_item_map (
          id BIGINT AUTO_INCREMENT PRIMARY KEY,
          raw_item_id BIGINT NOT NULL,
          norm_item_id BIGINT NOT NULL,
          map_source VARCHAR(30) NOT NULL,
          confidence DECIMAL(5,2),
          mapped_by VARCHAR(100),
          mapped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
          UNIQUE KEY uq_raw_map (raw_item_id),
          INDEX idx_norm_map (norm_item_id)
        )
        """
    )
    exec_sql(
        """
        CREATE TABLE IF NOT EXISTS catalog_bearing_attr (
          norm_item_id BIGINT PRIMARY KEY,
          bearing_type VARCHAR(100),
          bearing_number VARCHAR(50),
          suffix VARCHAR(200),
          suffix_desc VARCHAR(300),
          seal_type VARCHAR(50),
          cage_material VARCHAR(50),
          clearance VARCHAR(20),
          precision_class VARCHAR(20),
          grease_type VARCHAR(50),
          taper_diameter VARCHAR(20)
        )
        """
    )
    exec_sql(
        """
        CREATE TABLE IF NOT EXISTS catalog_rule (
          id BIGINT AUTO_INCREMENT PRIMARY KEY,
          rule_name VARCHAR(200) NOT NULL,
          match_regex VARCHAR(300) NOT NULL,
          target_norm_item_id BIGINT,
          target_category VARCHAR(100),
          priority INT NOT NULL DEFAULT 100,
          enabled TINYINT NOT NULL DEFAULT 1,
          created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )


def _get_or_create_source(source_name: str, source_type: str, notes: str) -> Optional[int]:
    row = query_df(
        """
        SELECT id
        FROM catalog_source
        WHERE source_name=%s AND source_type=%s
        """,
        (source_name, source_type),
    )
    if not row.empty:
        return int(row.iloc[0]["id"])
    exec_sql(
        """
        INSERT INTO catalog_source (source_name, source_type, notes)
        VALUES (%s, %s, %s)
        """,
        (source_name, source_type, notes or None),
    )
    row = query_df(
        """
        SELECT id
        FROM catalog_source
        WHERE source_name=%s AND source_type=%s
        """,
        (source_name, source_type),
    )
    if row.empty:
        return None
    return int(row.iloc[0]["id"])


def _load_sources() -> pd.DataFrame:
    return query_df(
        """
        SELECT id, source_name, source_type, uploaded_at
        FROM catalog_source
        ORDER BY uploaded_at DESC, source_name
        """
    )


def _load_raw_items(source_id: int, only_unmapped: bool, search_text: str) -> pd.DataFrame:
    search_like = f"%{search_text.strip()}%" if search_text else None
    if only_unmapped:
        sql = """
            SELECT
              r.id,
              r.item_name_raw,
              r.item_code_raw,
              r.maker_raw,
              m.norm_item_id,
              n.item_name_std
            FROM catalog_item_raw r
            LEFT JOIN catalog_item_map m ON m.raw_item_id = r.id
            LEFT JOIN catalog_item_norm n ON n.id = m.norm_item_id
            WHERE r.source_id=%s AND m.raw_item_id IS NULL
        """
        params = [source_id]
    else:
        sql = """
            SELECT
              r.id,
              r.item_name_raw,
              r.item_code_raw,
              r.maker_raw,
              m.norm_item_id,
              n.item_name_std
            FROM catalog_item_raw r
            LEFT JOIN catalog_item_map m ON m.raw_item_id = r.id
            LEFT JOIN catalog_item_norm n ON n.id = m.norm_item_id
            WHERE r.source_id=%s
        """
        params = [source_id]
    if search_like:
        sql += " AND (r.item_name_raw LIKE %s OR r.item_code_raw LIKE %s OR r.maker_raw LIKE %s)"
        params += [search_like, search_like, search_like]
    sql += " ORDER BY r.item_name_raw LIMIT 500"
    return query_df(sql, tuple(params))


def _load_norm_items(search_text: str) -> pd.DataFrame:
    search_like = f"%{search_text.strip()}%" if search_text else None
    sql = """
        SELECT id, item_name_std, item_code_std, maker_std, category
        FROM catalog_item_norm
    """
    params: list = []
    if search_like:
        sql += " WHERE item_name_std LIKE %s OR item_code_std LIKE %s OR maker_std LIKE %s"
        params = [search_like, search_like, search_like]
    sql += " ORDER BY item_name_std LIMIT 500"
    return query_df(sql, tuple(params))


def _load_mapped_results(source_id: int) -> pd.DataFrame:
    return query_df(
        """
        SELECT
          r.id AS raw_id,
          r.item_name_raw,
          r.item_code_raw,
          r.maker_raw,
          m.norm_item_id,
          n.item_name_std,
          n.item_code_std,
          n.maker_std,
          n.category,
          a.bearing_type,
          a.bearing_number,
          a.suffix,
          a.suffix_desc,
          a.seal_type,
          a.cage_material,
          a.clearance,
          a.precision_class,
          a.grease_type,
          a.taper_diameter,
          m.map_source,
          m.confidence,
          m.mapped_at
        FROM catalog_item_raw r
        LEFT JOIN catalog_item_map m ON m.raw_item_id = r.id
        LEFT JOIN catalog_item_norm n ON n.id = m.norm_item_id
        LEFT JOIN catalog_bearing_attr a ON a.norm_item_id = n.id
        WHERE r.source_id=%s
        ORDER BY r.item_name_raw
        """,
        (source_id,),
    )


def _load_master_items() -> pd.DataFrame:
    return query_df(
        """
        SELECT
          n.id AS norm_item_id,
          n.item_name_std,
          n.item_code_std,
          n.maker_std,
          n.category,
          a.bearing_type,
          a.bearing_number,
          a.suffix,
          a.suffix_desc,
          a.seal_type,
          a.cage_material,
          a.clearance,
          a.precision_class,
          a.grease_type,
          a.taper_diameter,
          COUNT(m.raw_item_id) AS mapped_raw_count
        FROM catalog_item_norm n
        LEFT JOIN catalog_bearing_attr a ON a.norm_item_id = n.id
        LEFT JOIN catalog_item_map m ON m.norm_item_id = n.id
        GROUP BY
          n.id, n.item_name_std, n.item_code_std, n.maker_std, n.category,
          a.bearing_type, a.bearing_number, a.suffix, a.suffix_desc, a.seal_type,
          a.cage_material, a.clearance, a.precision_class, a.grease_type, a.taper_diameter
        ORDER BY n.item_name_std
        """
    )


def _load_fag_master_lookup() -> dict:
    master_df = query_df(
        """
        SELECT id, item_name_std
        FROM catalog_item_norm
        WHERE category='bearing'
        """
    )
    if master_df.empty:
        return {}
    master_df["item_name_std"] = master_df["item_name_std"].fillna("").astype(str).str.strip().str.upper()
    return {row["item_name_std"]: int(row["id"]) for _, row in master_df.iterrows() if row["item_name_std"]}


def _build_fag_std_name(row: pd.Series) -> str:
    prefix = str(row.get("BearingPrefix", "")).strip()
    number = str(row.get("BearingNumber", "")).strip()
    if prefix and number:
        base = f"{prefix}{number}"
    else:
        base = number or str(row.get("ItemName", "")).strip()
    suffix = str(row.get("Suffix", "")).strip()
    if suffix:
        tokens = [t for t in re.split(r"[\s,]+", suffix) if t]
        if tokens:
            return f"{base}-{'-'.join(tokens)}"
    return base


def _mapping_summary(source_id: int) -> tuple[int, int, int, float]:
    df = query_df(
        """
        SELECT
          COUNT(*) AS total_cnt,
          SUM(CASE WHEN m.raw_item_id IS NOT NULL THEN 1 ELSE 0 END) AS mapped_cnt
        FROM catalog_item_raw r
        LEFT JOIN catalog_item_map m ON m.raw_item_id = r.id
        WHERE r.source_id=%s
        """,
        (source_id,),
    )
    if df.empty:
        return 0, 0, 0, 0.0
    total_cnt = int(df.iloc[0]["total_cnt"] or 0)
    mapped_cnt = int(df.iloc[0]["mapped_cnt"] or 0)
    unmapped_cnt = max(0, total_cnt - mapped_cnt)
    pct = (mapped_cnt / total_cnt * 100) if total_cnt else 0.0
    return total_cnt, mapped_cnt, unmapped_cnt, pct


def _load_rules() -> pd.DataFrame:
    return query_df(
        """
        SELECT
          r.id,
          r.rule_name,
          r.match_regex,
          r.target_norm_item_id,
          n.item_name_std,
          r.priority,
          r.enabled
        FROM catalog_rule r
        LEFT JOIN catalog_item_norm n ON n.id = r.target_norm_item_id
        ORDER BY r.priority, r.rule_name
        """
    )


def _apply_rules(source_id: int) -> int:
    rules = query_df(
        """
        SELECT id, match_regex, target_norm_item_id
        FROM catalog_rule
        WHERE enabled=1 AND target_norm_item_id IS NOT NULL
        ORDER BY priority, id
        """
    )
    if rules.empty:
        return 0
    total = 0
    for _, rule in rules.iterrows():
        affected = exec_sql(
            """
            INSERT INTO catalog_item_map
              (raw_item_id, norm_item_id, map_source, confidence, mapped_by)
            SELECT
              r.id, %s, 'rule', 0.9, 'rule'
            FROM catalog_item_raw r
            LEFT JOIN catalog_item_map m ON m.raw_item_id = r.id
            WHERE r.source_id=%s
              AND m.raw_item_id IS NULL
              AND r.item_name_raw REGEXP %s
            """,
            (int(rule["target_norm_item_id"]), source_id, rule["match_regex"]),
        )
        total += max(0, affected)
    return total


def _reprocess_source(source_id: int, reset_maps: bool, status) -> None:
    if reset_maps:
        exec_sql(
            """
            DELETE m
            FROM catalog_item_map m
            JOIN catalog_item_raw r ON r.id = m.raw_item_id
            WHERE r.source_id=%s
            """,
            (source_id,),
        )
        status.write("기존 매핑 초기화 완료")

    raw_df = query_df(
        """
        SELECT
          r.id,
          r.item_name_raw,
          r.item_code_raw,
          r.maker_raw,
          m.raw_item_id AS mapped_id
        FROM catalog_item_raw r
        LEFT JOIN catalog_item_map m ON m.raw_item_id = r.id
        WHERE r.source_id=%s
        """,
        (source_id,),
    )
    if raw_df.empty:
        status.write("원본 데이터가 없습니다.")
        return

    if not reset_maps:
        raw_df = raw_df[raw_df["mapped_id"].isna()]
        if raw_df.empty:
            status.write("미매핑 데이터가 없습니다.")
            return

    raw_df = raw_df.fillna("")
    raw_df["ItemName"] = raw_df["item_name_raw"].astype(str)
    raw_df["ItemCode"] = raw_df["item_code_raw"].astype(str)
    raw_df["Maker"] = raw_df["maker_raw"].astype(str)

    parsed = raw_df.apply(lambda r: _parse_bearing_name(r["ItemName"], r["Maker"]), axis=1).apply(pd.Series)
    merged = pd.concat([raw_df, parsed], axis=1)

    master_lookup = _load_fag_master_lookup()
    if not master_lookup:
        status.write("FAG 마스터가 없습니다. 먼저 마스터를 등록하세요.")
        return

    merged["item_name_std"] = merged.apply(_build_fag_std_name, axis=1).astype(str).str.strip().str.upper()
    merged["master_id"] = merged["item_name_std"].map(master_lookup)

    map_rows = []
    attr_rows = []
    for _, row in merged.iterrows():
        raw_id = row.get("id")
        norm_id = row.get("master_id")
        if pd.isna(raw_id) or pd.isna(norm_id):
            continue
        map_rows.append(
            (
                int(raw_id),
                int(norm_id),
                "auto",
                0.8,
                "system",
            )
        )
        attr_rows.append(
            (
                int(norm_id),
                str(row.get("BearingType", "")).strip() or None,
                str(row.get("BearingNumber", "")).strip() or None,
                str(row.get("Suffix", "")).strip() or None,
                str(row.get("SuffixDesc", "")).strip() or None,
                str(row.get("SealType", "")).strip() or None,
                str(row.get("cageMaterial", "")).strip() or None,
                str(row.get("Clearnace", "")).strip() or None,
                str(row.get("precision", "")).strip() or None,
                str(row.get("greaseType", "")).strip() or None,
                str(row.get("TaperDiameter", "")).strip() or None,
            )
        )

    if map_rows:
        map_sql = """
            INSERT INTO catalog_item_map
              (raw_item_id, norm_item_id, map_source, confidence, mapped_by)
            VALUES (%s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
              norm_item_id=VALUES(norm_item_id),
              map_source=VALUES(map_source),
              confidence=VALUES(confidence),
              mapped_by=VALUES(mapped_by),
              mapped_at=CURRENT_TIMESTAMP
        """
        exec_many(map_sql, map_rows, batch_size=500)
    if attr_rows:
        attr_sql = """
            INSERT INTO catalog_bearing_attr
              (norm_item_id, bearing_type, bearing_number, suffix, suffix_desc,
               seal_type, cage_material, clearance, precision_class, grease_type, taper_diameter)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
              bearing_type=VALUES(bearing_type),
              bearing_number=VALUES(bearing_number),
              suffix=VALUES(suffix),
              suffix_desc=VALUES(suffix_desc),
              seal_type=VALUES(seal_type),
              cage_material=VALUES(cage_material),
              clearance=VALUES(clearance),
              precision_class=VALUES(precision_class),
              grease_type=VALUES(grease_type),
              taper_diameter=VALUES(taper_diameter)
        """
        exec_many(attr_sql, attr_rows, batch_size=500)

    status.write("재정규화 완료")

def _backfill_bearing_attrs(source_id: int, only_missing: bool = True, status=None) -> int:
    where_clause = ""
    if only_missing:
        where_clause = """
          AND (
            a.norm_item_id IS NULL
            OR (
              a.bearing_type IS NULL
              AND a.bearing_number IS NULL
              AND a.suffix IS NULL
            )
          )
        """
    raw_df = query_df(
        f"""
        SELECT
          r.item_name_raw,
          r.item_code_raw,
          r.maker_raw,
          m.norm_item_id
        FROM catalog_item_raw r
        JOIN catalog_item_map m ON m.raw_item_id = r.id
        LEFT JOIN catalog_bearing_attr a ON a.norm_item_id = m.norm_item_id
        WHERE r.source_id=%s
        {where_clause}
        """,
        (source_id,),
    )
    if raw_df.empty:
        if status:
            status.write("채울 속성이 없습니다.")
        return 0

    raw_df = raw_df.fillna("")
    parsed = raw_df.apply(
        lambda r: _parse_bearing_name(r["item_name_raw"], r["maker_raw"]),
        axis=1,
    ).apply(pd.Series)
    merged = pd.concat([raw_df, parsed], axis=1)

    attr_rows = []
    for _, row in merged.iterrows():
        norm_id = row.get("norm_item_id")
        if pd.isna(norm_id):
            continue
        attr_rows.append(
            (
                int(norm_id),
                str(row.get("BearingType", "")).strip() or None,
                str(row.get("BearingNumber", "")).strip() or None,
                str(row.get("Suffix", "")).strip() or None,
                str(row.get("SuffixDesc", "")).strip() or None,
                str(row.get("SealType", "")).strip() or None,
                str(row.get("cageMaterial", "")).strip() or None,
                str(row.get("Clearnace", "")).strip() or None,
                str(row.get("precision", "")).strip() or None,
                str(row.get("greaseType", "")).strip() or None,
                str(row.get("TaperDiameter", "")).strip() or None,
            )
        )

    if not attr_rows:
        if status:
            status.write("채울 속성이 없습니다.")
        return 0

    attr_sql = """
        INSERT INTO catalog_bearing_attr
          (norm_item_id, bearing_type, bearing_number, suffix, suffix_desc,
           seal_type, cage_material, clearance, precision_class, grease_type, taper_diameter)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
          bearing_type=VALUES(bearing_type),
          bearing_number=VALUES(bearing_number),
          suffix=VALUES(suffix),
          suffix_desc=VALUES(suffix_desc),
          seal_type=VALUES(seal_type),
          cage_material=VALUES(cage_material),
          clearance=VALUES(clearance),
          precision_class=VALUES(precision_class),
          grease_type=VALUES(grease_type),
          taper_diameter=VALUES(taper_diameter)
    """
    exec_many(attr_sql, attr_rows, batch_size=500)
    if status:
        status.write(f"속성 채움 완료: {len(attr_rows):,}건")
    return len(attr_rows)


def _save_catalog_upload(
    df: pd.DataFrame,
    source_name: str,
    source_type: str,
    notes: str,
    auto_map: bool,
    status,
) -> None:
    def _clean_str(val) -> str:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return ""
        return str(val).strip()

    def _safe_json(val) -> str:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return "{}"
        if isinstance(val, (dict, list)):
            return json.dumps(val, ensure_ascii=True)
        if isinstance(val, str):
            try:
                json.loads(val)
                return val
            except Exception:
                return "{}"
        return "{}"

    source_id = _get_or_create_source(source_name, source_type, notes)
    if not source_id:
        raise RuntimeError("Failed to create source.")

    if "Maker" not in df.columns:
        df["Maker"] = ""
    if "ItemCode" not in df.columns:
        df["ItemCode"] = ""

    raw_rows = []
    for _, row in df.iterrows():
        raw_rows.append(
            (
                source_id,
                _clean_str(row.get("ItemName", "")),
                _clean_str(row.get("ItemCode", "")) or None,
                _clean_str(row.get("Maker", "")) or None,
                _safe_json(row.get("extra_json", {})),
            )
        )
    insert_raw_sql = """
        INSERT IGNORE INTO catalog_item_raw
          (source_id, item_name_raw, item_code_raw, maker_raw, extra_json)
        VALUES (%s, %s, %s, %s, %s)
    """
    exec_many(insert_raw_sql, raw_rows, batch_size=500)

    raw_df = query_df(
        """
        SELECT id, item_name_raw, item_code_raw, maker_raw
        FROM catalog_item_raw
        WHERE source_id=%s
        """,
        (source_id,),
    )
    if raw_df.empty:
        return
    raw_df["item_name_raw"] = raw_df["item_name_raw"].fillna("")
    raw_df["item_code_raw"] = raw_df["item_code_raw"].fillna("")
    raw_df["maker_raw"] = raw_df["maker_raw"].fillna("")
    merge_df = df.copy()
    merge_df["item_name_raw"] = merge_df["ItemName"].fillna("").astype(str)
    merge_df["item_code_raw"] = merge_df["ItemCode"].fillna("").astype(str)
    merge_df["maker_raw"] = merge_df["Maker"].fillna("").astype(str)
    merged = merge_df.merge(
        raw_df,
        on=["item_name_raw", "item_code_raw", "maker_raw"],
        how="left",
        suffixes=("", "_raw"),
    )

    if not auto_map:
        status.write("DB ?? ?? (??? ???? ??)")
        return

    master_lookup = _load_fag_master_lookup()
    if not master_lookup:
        status.write("FAG ???? ????. ?? ???? ?????.")
        return

    merged["item_name_std"] = merged.apply(_build_fag_std_name, axis=1).astype(str).str.strip().str.upper()
    merged["master_id"] = merged["item_name_std"].map(master_lookup)

    map_rows = []
    attr_rows = []
    for _, row in merged.iterrows():
        raw_id = row.get("id")
        norm_id = row.get("master_id")
        if pd.isna(raw_id) or pd.isna(norm_id):
            continue
        map_rows.append(
            (
                int(raw_id),
                int(norm_id),
                "auto",
                0.8,
                "system",
            )
        )
        attr_rows.append(
            (
                int(norm_id),
                str(row.get("BearingType", "")).strip() or None,
                str(row.get("BearingNumber", "")).strip() or None,
                str(row.get("Suffix", "")).strip() or None,
                str(row.get("SuffixDesc", "")).strip() or None,
                str(row.get("SealType", "")).strip() or None,
                str(row.get("cageMaterial", "")).strip() or None,
                str(row.get("Clearnace", "")).strip() or None,
                str(row.get("precision", "")).strip() or None,
                str(row.get("greaseType", "")).strip() or None,
                str(row.get("TaperDiameter", "")).strip() or None,
            )
        )

    if map_rows:
        map_sql = """
            INSERT INTO catalog_item_map
              (raw_item_id, norm_item_id, map_source, confidence, mapped_by)
            VALUES (%s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
              norm_item_id=VALUES(norm_item_id),
              map_source=VALUES(map_source),
              confidence=VALUES(confidence),
              mapped_by=VALUES(mapped_by),
              mapped_at=CURRENT_TIMESTAMP
        """
        exec_many(map_sql, map_rows, batch_size=500)
    if attr_rows:
        attr_sql = """
            INSERT INTO catalog_bearing_attr
              (norm_item_id, bearing_type, bearing_number, suffix, suffix_desc,
               seal_type, cage_material, clearance, precision_class, grease_type, taper_diameter)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
              bearing_type=VALUES(bearing_type),
              bearing_number=VALUES(bearing_number),
              suffix=VALUES(suffix),
              suffix_desc=VALUES(suffix_desc),
              seal_type=VALUES(seal_type),
              cage_material=VALUES(cage_material),
              clearance=VALUES(clearance),
              precision_class=VALUES(precision_class),
              grease_type=VALUES(grease_type),
              taper_diameter=VALUES(taper_diameter)
        """
        exec_many(attr_sql, attr_rows, batch_size=500)
    status.write("DB ?? + ?? ?? ??")

def show_bearing_standard_page():
    st.header("베어링 표준품목")
    st.caption("엑셀 품목을 업로드해 표준 분류합니다.")

    db_ready = True
    try:
        ensure_catalog_tables()
    except Exception as e:
        db_ready = False
        st.warning(f"DB 연결 실패로 매핑 기능이 비활성화됩니다: {e}")

    source_id = None
    if db_ready:
        st.subheader("소스 선택")
        sources = _load_sources()
        if sources.empty:
            st.info("적재된 소스가 없습니다. 아래에서 업로드해 주세요.")
        else:
            master_df = _load_master_items()
            master_bytes = master_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "전체 마스터 다운로드",
                data=master_bytes,
                file_name="bearing_master_items.csv",
                mime="text/csv",
            )
            source_label = sources.apply(
                lambda r: f"{r['id']} | {r['source_name']} ({r['source_type']})", axis=1
            ).tolist()
            source_pick = st.selectbox("소스", source_label, index=0, key="bearing_source_pick")
            source_id = int(source_pick.split("|", 1)[0].strip())
            total_cnt, mapped_cnt, unmapped_cnt, mapped_pct = _mapping_summary(source_id)
            st.caption(
                f"원본 {total_cnt:,}건 | 매핑 {mapped_cnt:,}건 | 미매핑 {unmapped_cnt:,}건 | 매핑률 {mapped_pct:.1f}%"
            )
            only_unmapped = st.checkbox("미매핑만 보기", value=True)
            search_text = st.text_input("검색 (품목명/코드/메이커)")
            raw_items = _load_raw_items(source_id, only_unmapped, search_text)

            st.caption(f"표시 {len(raw_items):,}건 (최대 500건)")
            st.dataframe(raw_items, use_container_width=True, hide_index=True)

            mapped_df = _load_mapped_results(source_id)
            csv_bytes = mapped_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "매핑 결과 다운로드",
                data=csv_bytes,
                file_name="bearing_mapped_results.csv",
                mime="text/csv",
            )

            if raw_items.empty:
                st.info("표시할 원본이 없습니다.")
            else:
                raw_pick_options = raw_items.apply(
                    lambda r: f"{r['id']} | {r['item_name_raw']}", axis=1
                ).tolist()
                raw_pick = st.selectbox("원본 품목", raw_pick_options, index=0)
                raw_id = int(raw_pick.split("|", 1)[0].strip())
                raw_row = raw_items[raw_items["id"] == raw_id].iloc[0]
                st.write(
                    f"선택: {raw_row['item_name_raw']} / {raw_row.get('item_code_raw','')} / {raw_row.get('maker_raw','')}"
                )

                parsed = _parse_bearing_name(str(raw_row["item_name_raw"]), str(raw_row.get("maker_raw") or ""))
                default_std = _build_fag_std_name(
                    pd.Series(
                        {
                            "BearingPrefix": parsed.get("BearingPrefix", ""),
                            "BearingNumber": parsed.get("BearingNumber", ""),
                            "Suffix": parsed.get("Suffix", ""),
                            "ItemName": str(raw_row["item_name_raw"]),
                        }
                    )
                )

                st.subheader("기존 표준 품목에 매핑")
                norm_search = st.text_input("표준 품목 검색", key="bearing_norm_search")
                norm_items = _load_norm_items(norm_search)
                norm_pick = None
                norm_options = []
                if not norm_items.empty:
                    norm_options = norm_items.apply(
                        lambda r: f"{r['id']} | {r['item_name_std']} ({r.get('maker_std','')})", axis=1
                    ).tolist()
                    norm_pick = st.selectbox("표준 품목", norm_options, index=0)
                if st.button("선택한 표준 품목으로 매핑"):
                    if not norm_pick:
                        st.warning("표준 품목을 선택하세요.")
                    else:
                        norm_id = int(norm_pick.split("|", 1)[0].strip())
                        exec_sql(
                            """
                            INSERT INTO catalog_item_map
                              (raw_item_id, norm_item_id, map_source, confidence, mapped_by)
                            VALUES (%s, %s, 'manual', 1.0, 'user')
                            ON DUPLICATE KEY UPDATE
                              norm_item_id=VALUES(norm_item_id),
                              map_source='manual',
                              confidence=1.0,
                              mapped_by='user',
                              mapped_at=CURRENT_TIMESTAMP
                            """,
                            (raw_id, norm_id),
                        )
                        st.success("매핑 완료.")

                st.subheader("새 표준 품목 생성 후 매핑")
                new_name = st.text_input("표준 품목명", value=default_std, key="bearing_new_name")
                new_code = st.text_input("표준 품목 코드", value=str(raw_row.get("item_code_raw") or ""), key="bearing_new_code")
                new_maker = st.text_input("메이커", value=str(raw_row.get("maker_raw") or ""), key="bearing_new_maker")
                new_category = st.text_input("카테고리", value="bearing", key="bearing_new_cat")
                if st.button("표준 품목 생성 + 매핑"):
                    exec_sql(
                        """
                        INSERT IGNORE INTO catalog_item_norm
                          (item_name_std, item_code_std, maker_std, category)
                        VALUES (%s, %s, %s, %s)
                        """,
                        (new_name, new_code or None, new_maker or None, new_category or None),
                    )
                    norm_row = query_df(
                        """
                        SELECT id FROM catalog_item_norm
                        WHERE item_name_std=%s AND item_code_std=%s AND maker_std=%s
                        """,
                        (new_name, new_code or None, new_maker or None),
                    )
                    if norm_row.empty:
                        st.error("표준 품목 생성 실패")
                    else:
                        norm_id = int(norm_row.iloc[0]["id"])
                        exec_sql(
                            """
                            INSERT INTO catalog_item_map
                              (raw_item_id, norm_item_id, map_source, confidence, mapped_by)
                            VALUES (%s, %s, 'manual', 1.0, 'user')
                            ON DUPLICATE KEY UPDATE
                              norm_item_id=VALUES(norm_item_id),
                              map_source='manual',
                              confidence=1.0,
                              mapped_by='user',
                              mapped_at=CURRENT_TIMESTAMP
                            """,
                            (raw_id, norm_id),
                        )
                        exec_sql(
                            """
                            INSERT INTO catalog_bearing_attr
                              (norm_item_id, bearing_type, bearing_number, suffix, suffix_desc,
                               seal_type, cage_material, clearance, precision_class, grease_type, taper_diameter)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON DUPLICATE KEY UPDATE
                              bearing_type=VALUES(bearing_type),
                              bearing_number=VALUES(bearing_number),
                              suffix=VALUES(suffix),
                              suffix_desc=VALUES(suffix_desc),
                              seal_type=VALUES(seal_type),
                              cage_material=VALUES(cage_material),
                              clearance=VALUES(clearance),
                              precision_class=VALUES(precision_class),
                              grease_type=VALUES(grease_type),
                              taper_diameter=VALUES(taper_diameter)
                            """,
                            (
                                norm_id,
                                parsed.get("BearingType") or None,
                                parsed.get("BearingNumber") or None,
                                parsed.get("Suffix") or None,
                                parsed.get("SuffixDesc") or None,
                                parsed.get("SealType") or None,
                                parsed.get("cageMaterial") or None,
                                parsed.get("Clearnace") or None,
                                parsed.get("precision") or None,
                                parsed.get("greaseType") or None,
                                parsed.get("TaperDiameter") or None,
                            ),
                        )
                        st.success("표준 품목 생성 및 매핑 완료.")

                if st.button("선택 원본 매핑 해제"):
                    exec_sql("DELETE FROM catalog_item_map WHERE raw_item_id=%s", (raw_id,))
                    st.success("매핑 해제 완료.")

            st.divider()
            st.subheader("재정규화")
            reset_maps = st.checkbox("기존 매핑 초기화 후 재정규화", value=False)
            if st.button("재정규화 (미매핑만)", disabled=reset_maps or source_id is None):
                status = st.status("재정규화 처리 중...", expanded=True)
                _reprocess_source(source_id, reset_maps=False, status=status)
                status.update(label="재정규화 완료", state="complete", expanded=False)
            if st.button("재정규화 (초기화 후 전체)", disabled=(not reset_maps) or source_id is None):
                status = st.status("재정규화 처리 중...", expanded=True)
                _reprocess_source(source_id, reset_maps=True, status=status)
                status.update(label="재정규화 완료", state="complete", expanded=False)
            if st.button("매핑된 항목 속성 채우기", disabled=source_id is None):
                status = st.status("속성 채움 중...", expanded=True)
                _backfill_bearing_attrs(source_id, only_missing=False, status=status)
                status.update(label="속성 채움 완료", state="complete", expanded=False)

            st.divider()
            st.subheader("규칙 기반 자동 매핑")
            rule_df = _load_rules()
            if not rule_df.empty:
                st.dataframe(rule_df, use_container_width=True, hide_index=True)
            else:
                st.info("등록된 규칙이 없습니다.")

            st.markdown("규칙 추가")
            rule_name = st.text_input("규칙명", key="bearing_rule_name")
            rule_regex = st.text_input("정규식 (원본 품목명)", key="bearing_rule_regex")
            rule_priority = st.number_input("우선순위(낮을수록 우선)", min_value=1, value=100, step=1, key="bearing_rule_priority")
            rule_target_pick = None
            rule_norm_search = st.text_input("표준 품목 검색", key="bearing_rule_norm_search")
            rule_norm_items = _load_norm_items(rule_norm_search)
            if not rule_norm_items.empty:
                rule_norm_options = rule_norm_items.apply(
                    lambda r: f"{r['id']} | {r['item_name_std']} ({r.get('maker_std','')})", axis=1
                ).tolist()
                rule_target_pick = st.selectbox("대상 표준 품목", rule_norm_options, index=0, key="bearing_rule_target")
            if st.button("규칙 저장"):
                if not rule_name or not rule_regex or not rule_target_pick:
                    st.warning("규칙명, 정규식, 대상 품목을 입력하세요.")
                else:
                    target_id = int(rule_target_pick.split("|", 1)[0].strip())
                    exec_sql(
                        """
                        INSERT INTO catalog_rule
                          (rule_name, match_regex, target_norm_item_id, priority, enabled)
                        VALUES (%s, %s, %s, %s, 1)
                        """,
                        (rule_name, rule_regex, target_id, int(rule_priority)),
                    )
                    st.success("규칙 저장 완료.")

            if st.button("규칙 적용 (미매핑만)", disabled=source_id is None):
                applied = _apply_rules(source_id)
                st.success(f"규칙 적용 완료: {applied}건 매핑")
    else:
        st.info("DB 연결이 없어 매핑 UI를 사용할 수 없습니다.")

    st.divider()
    with st.expander("엑셀 업로드", expanded=False):
        up = st.file_uploader("베어링 품목 엑셀 (.xlsx)", type=["xlsx"], key="bearing_upload")
        if up is None:
            st.info("새 소스를 추가하려면 파일을 업로드하세요.")
            return

        try:
            status = st.status("엑셀 로딩 중...", expanded=True)
            xls = pd.ExcelFile(up, engine="openpyxl")
            sheet_name = st.selectbox("시트", xls.sheet_names, index=0, key="bearing_sheet")
            df = pd.read_excel(up, sheet_name=sheet_name, engine="openpyxl")
            status.write("엑셀 로딩 완료")
        except Exception as e:
            st.error(f"엑셀 읽기 실패: {e}")
            return

        df.columns = [str(c).strip() if pd.notna(c) else "" for c in df.columns]
        col_labels = [f"{i}:{c}" for i, c in enumerate(df.columns)]
        auto_item = _detect_item_col(df.columns)
        item_idx = list(df.columns).index(auto_item) if auto_item in df.columns else 0
        item_pick = st.selectbox("품목명 컬럼", col_labels, index=item_idx, key="bearing_item_col")
        if not item_pick:
            st.error("품목명 컬럼을 선택하세요.")
            return
        item_col_idx = int(item_pick.split(":", 1)[0])

        auto_maker = _detect_maker_col(df.columns)
        maker_options = ["(none)"] + col_labels
        maker_idx = maker_options.index(f"{list(df.columns).index(auto_maker)}:{auto_maker}") if auto_maker in df.columns else 0
        maker_pick = st.selectbox("메이커 컬럼(선택)", maker_options, index=maker_idx, key="bearing_maker_col")
        if maker_pick == "(none)":
            maker_col = None
        else:
            maker_col_idx = int(maker_pick.split(":", 1)[0])
            maker_col = maker_col_idx
        if maker_col == item_col_idx:
            maker_col = None

        auto_code = _detect_code_col(df.columns)
        code_options = ["(none)"] + col_labels
        code_idx = code_options.index(f"{list(df.columns).index(auto_code)}:{auto_code}") if auto_code in df.columns else 0
        code_pick = st.selectbox("품목코드 컬럼(선택)", code_options, index=code_idx, key="bearing_code_col")
        if code_pick == "(none)":
            code_col = None
        else:
            code_col_idx = int(code_pick.split(":", 1)[0])
            code_col = code_col_idx
        if code_col == item_col_idx:
            code_col = None

        base = pd.DataFrame()
        base["ItemName"] = df.iloc[:, item_col_idx]
        if maker_col is not None:
            base["Maker"] = df.iloc[:, maker_col]
        if code_col is not None:
            base["ItemCode"] = df.iloc[:, code_col]

        base["ItemName"] = base["ItemName"].astype("string").str.strip()
        base["ItemName"] = base["ItemName"].str.replace(r"^[\.-]+$", "", regex=True)
        base = base[base["ItemName"].notna() & (base["ItemName"] != "")]

        if db_ready:
            st.subheader("DB 적재 옵션")
            source_name = st.text_input("소스명", value=up.name)
            source_type = st.selectbox("소스 타입", ["customer", "vendor", "file"], index=2)
            source_notes = st.text_area("비고(선택)", height=60)
            save_to_db = st.checkbox("DB 적재", value=True)
            auto_map = st.checkbox("자동 매핑(베어링 파서)", value=True)
        else:
            source_name = ""
            source_type = "file"
            source_notes = ""
            save_to_db = False
            auto_map = False

        status.write("분류 중...")
        if maker_col is not None:
            parsed = base.apply(lambda r: _parse_bearing_name(r["ItemName"], r["Maker"]), axis=1).apply(pd.Series)
        else:
            parsed = base["ItemName"].apply(_parse_bearing_name).apply(pd.Series)
        out = pd.concat([base, parsed], axis=1)
        if out.columns.duplicated().any():
            cols = []
            seen = {}
            for c in out.columns:
                if c in seen:
                    seen[c] += 1
                    cols.append(f"{c}_{seen[c]}")
                else:
                    seen[c] = 1
                    cols.append(c)
            out.columns = cols
        status.update(label="분류 완료", state="complete", expanded=False)

        st.subheader("미리보기")
        total_cnt = len(out)
        bt_col = "BearingType"
        if bt_col not in out.columns:
            bt_col = next((c for c in out.columns if str(c).lower().startswith("bearingtype")), None)
        classified_cnt = int(out[bt_col].astype(str).str.strip().ne("").sum()) if bt_col else 0
        classified_pct = (classified_cnt / total_cnt * 100) if total_cnt else 0.0
        st.caption(
            f"전체 {total_cnt:,} | 분류 {classified_cnt:,} | {classified_pct:.1f}%"
        )
        unclassified = out[out[bt_col].astype(str).str.strip() == ""] if bt_col else out
        if not unclassified.empty:
            samples = unclassified["ItemName"].astype(str).head(20).tolist()
            st.info("미분류 샘플: " + ", ".join(samples))
        st.dataframe(out.head(200), use_container_width=True, hide_index=True)

        if db_ready and save_to_db:
            db_df = out.copy()
            if "ItemCode" not in db_df.columns:
                db_df["ItemCode"] = ""
            db_df["extra_json"] = {}
            _save_catalog_upload(db_df, source_name, source_type, source_notes, auto_map, status)

        csv_bytes = out.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "CSV 다운로드",
            data=csv_bytes,
            file_name="bearing_standard_items.csv",
            mime="text/csv",
        )

