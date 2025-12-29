import re
import pandas as pd

def normalize_customer_name(s: str) -> str:
    """
    거래처명 정규화 (자동 매핑용)
    """
    if not s or (isinstance(s, float) and pd.isna(s)):
        return ""
    s = str(s).upper()
    s = re.sub(r"\(.*?\)", "", s)
    s = s.replace("주식회사", "").replace("(주)", "")
    s = re.sub(r"\s+", "", s)
    return s

def safe_int(v, default=0):
    try:
        return int(v)
    except Exception:
        return default
# core/utils.py (추가)

# core/utils.py

import re
from difflib import SequenceMatcher

# ✅ 1) 우리 조직/업무에서 흔히 붙는 '내부 접두어' (지점/채널/조직 구분)
#    예: "SKB-종로에프에이" / "서원_종로에프에이" / "직납-OOO" ...
INTERNAL_PREFIXES = [
    "skb", "sk_b", "sk",          # 내부 법인/조직
    "서원", "seowon", "sw",        # 회사/조직
    "본사", "지점", "센터",
    "영업", "영업1", "영업2", "영업팀", "영업부",
    "내수", "수출", "시판", "직납", "온라인", "오프라인",
    "a팀", "b팀", "c팀",
]

# ✅ 2) 법인형태/자주 등장하는 불용어(업계 공통)
CORP_TOKENS = [
    "주식회사", "㈜", "(주)", "유한회사", "유한", "합자회사", "합명회사",
    "코리아", "한국", "대한",
    "trading", "trade", "co", "company", "corp", "corporation", "inc", "ltd",
    "상사", "공업", "산업", "기계", "유통", "물산", "상공", "무역",
    "테크", "테크놀로지", "테크놀러지",
    "베어링", "bearing", "자동차", "모터", "parts", "파트", "부품",
]

# ✅ 3) 지점/사업장 접미어(있으면 제거해도 무방한 경우가 많음)
#    (너무 공격적이면 주석처리해도 됨)
BRANCH_SUFFIXES = [
    "지점", "영업소", "물류센터", "센터", "공장", "창고", "사업부", "사업팀",
]


def similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _strip_internal_prefix(s: str) -> str:
    """
    내부 접두어 제거:
    - "SKB-xxx", "SKB xxx", "서원/xxx", "직납_xxx" 등
    """
    if not s:
        return s

    # (1) 앞쪽의 "접두어 + 구분자(-_/)" 제거 (반복 적용)
    prefix_alt = "|".join(sorted({re.escape(p) for p in INTERNAL_PREFIXES}, key=len, reverse=True))
    # 예: ^(skb|서원)\s*[-_/]\s*
    pat1 = re.compile(rf"^({prefix_alt})\s*[-_/]\s*", flags=re.IGNORECASE)
    while True:
        new = pat1.sub("", s).strip()
        if new == s:
            break
        s = new

    # (2) 앞쪽의 "접두어 + 공백" 제거 (반복 적용)
    pat2 = re.compile(rf"^({prefix_alt})\s+", flags=re.IGNORECASE)
    while True:
        new = pat2.sub("", s).strip()
        if new == s:
            break
        s = new

    return s


def normalize_customer_name_strict(name: str) -> str:
    """
    ✅ 거래처명 비교용 강한 표준화 (업무/도메인 맞춤)
    - 내부 접두어(SKB-, 서원-, 직납- 등) 제거
    - 괄호/특수문자 정리
    - 불용어(CORP_TOKENS) 제거
    - 지점/사업장 접미어(BRANCH_SUFFIXES) 제거(선택적)
    - 최종 key: 공백 제거한 문자열
    """
    if name is None:
        return ""

    s = str(name).strip().lower()

    # 0) 앞의 내부 접두어 제거 (가장 중요!)
    s = _strip_internal_prefix(s)

    # 1) 괄호내용 제거: "OOO(본사)" → "OOO"
    s = re.sub(r"\([^)]*\)", " ", s)

    # 2) 구분자 통일: 하이픈/슬래시/언더바는 공백으로
    s = re.sub(r"[-_/]+", " ", s)

    # 3) 특수문자 제거(한글/영문/숫자만)
    s = re.sub(r"[^0-9a-z가-힣\s]+", " ", s)

    # 4) 토큰화 후 불용어 제거
    tokens = [t for t in s.split() if t]

    cleaned = []
    for t in tokens:
        # 법인/업계 불용어 제거
        if t in CORP_TOKENS:
            continue

        # 지점/사업장 접미어 제거 (예: "종로지점" → "종로")
        for suf in BRANCH_SUFFIXES:
            if t.endswith(suf) and len(t) > len(suf):
                t = t[: -len(suf)]
                break

        cleaned.append(t)

    # 5) 붙여서 비교키 생성
    key = "".join(cleaned).strip()
    return key

