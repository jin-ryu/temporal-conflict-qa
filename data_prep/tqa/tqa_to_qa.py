"""
TQA(Temporal Wiki, Özer 2025) 엔티티 JSON → 평가용 QA 레코드(기존 eval 스키마).

TQA는 엔티티마다 연도별 incident을 주고, 각 incident에 그 시점의 실제 위키 revision
문단(body_par)·연도별 질문·결정론적 답(Wikidata)이 들어있다. 여기서 한 엔티티의
'옛 연도'와 '새 연도'를 골라 충돌 컨텍스트(두 시점 문단 공존)를 만든다.
LLM 호출이 없는 순수 변환(비용 0).

출력 레코드(= 01_sample_eval_set.py가 소비하는 스키마):
  {id, source_idx, mode:"outdated_0"|"current", new_question, target_answer,
   evidence_chunk_id, chunks:[{chunk_id,label,text,last_modified_time}]}

엔티티당 2개 레코드 생성:
  - outdated_0 (과거 지향, 주력): 질문=옛 연도, 정답=옛 답, 근거=옛 문단
  - current   (현재 지향, 대조군): 질문=새 연도, 정답=새 답, 근거=새 문단
두 레코드 모두 같은 충돌 청크쌍[옛=outdated_0, 새=current]을 공유한다.

usage:
  python3 data_prep/tqa/tqa_to_qa.py                 # 전체(data/tqa/source → data/tqa/qa_tqa.jsonl)
  python3 data_prep/tqa/tqa_to_qa.py --max 100        # 100 엔티티만
"""
import argparse
import glob
import json
import os
import re
from collections import Counter
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]      # data_prep/tqa/ → repo 루트
DEFAULT_IN = _REPO / "data" / "tqa" / "source"   # TQA 엔티티 JSON 878개
DEFAULT_OUT = _REPO / "data" / "tqa" / "qa_tqa.jsonl"

_CSS_RE = re.compile(r"\.mw-parser-output[^{}]*\{[^{}]*\}")
_WS_RE = re.compile(r"\s+")


def clean_body(text: str, cap: int = 2000) -> str:
    """위키 revision 텍스트에서 인라인 CSS(.mw-parser-output{...})·잡공백 제거 후 길이 제한."""
    if not text:
        return ""
    t = _CSS_RE.sub(" ", text)
    t = t.replace("​", "").replace("\xa0", " ")
    t = _WS_RE.sub(" ", t).strip()
    return t[:cap]


def ans_name(inc: dict) -> str:
    a = inc.get("answer") or []
    if isinstance(a, list) and a and isinstance(a[0], dict):
        return (a[0].get("name") or "").strip()
    return ""


def q_text(inc: dict) -> str:
    q = (inc.get("question") or "").strip()
    return re.sub(r"^Question:\s*", "", q).strip()


def body_of(inc: dict) -> str:
    return clean_body((inc.get("dump") or {}).get("body_par", ""))


# 역할·조직 표기에 흔한 generic 단어 — 변별력이 없어 근거 매칭에서 제외
_GENERIC = {
    "national", "international", "football", "soccer", "team", "association", "club",
    "league", "city", "united", "states", "kingdom", "european", "commissioner",
    "secretary", "minister", "ministry", "member", "party", "university", "college",
    "school", "district", "county", "council", "federal", "republic", "democratic",
    "general", "office", "department", "committee", "company", "corporation", "group",
    "under", "women", "academy", "institute", "society", "service", "board",
}


def answer_in_evidence(ans: str, body: str) -> bool:
    """근거 문단이 답을 뒷받침하는지. 정식명↔약칭 차이를 흡수하기 위해
    generic 단어를 뺀 '변별적 토큰'이 하나라도 근거에 있으면 통과."""
    a, b = ans.lower(), body.lower()
    if not a:
        return False
    if a in b:                       # 정식명이 그대로 있으면 통과
        return True
    spec = [w for w in re.findall(r"[a-z0-9]+", a)
            if len(w) > 3 and w not in _GENERIC]
    if not spec:                     # 답이 전부 generic어 → 정식명 일치만 인정
        return a in b
    return any(w in b for w in spec)  # 변별 토큰 ≥1 (약칭 Milan 등 흡수)


# "클럽 vs 국가대표/유스팀" 모호성 — "which team" 질문이 두 범주를 섞으면 단일정답 깨짐
_NAT_TEAM = re.compile(r"\bnational\b.{0,30}\bteam\b", re.I)
_YOUTH_TEAM = re.compile(r"\bunder[-\s]?\d+\b|\bu[-\s]?\d{2}\b", re.I)


def is_ambiguous_team(*answers: str) -> bool:
    return any(_NAT_TEAM.search(a or "") or _YOUTH_TEAM.search(a or "") for a in answers)


def pick_pair(incidents: dict):
    """유효한 옛/새 연도 쌍을 고른다. 새=가장 최근, 옛=답이 다른 가장 이른 연도."""
    valid = {y: inc for y, inc in incidents.items()
             if y.isdigit() and ans_name(inc) and body_of(inc)}
    years = sorted(valid, key=int)
    if len(years) < 2:
        return None
    new_y = years[-1]
    new_ans = ans_name(valid[new_y])
    for old_y in years[:-1]:
        if ans_name(valid[old_y]).lower() != new_ans.lower():
            return old_y, valid[old_y], new_y, valid[new_y]
    return None  # 모든 연도가 같은 답 → 실제 변화 없음(충돌 아님)


def convert_entity(path: str, cap: int):
    """엔티티 파일 → (outdated 레코드, current 레코드) 또는 (None, 사유)."""
    try:
        data = json.load(open(path, encoding="utf-8"))
    except Exception:
        return None, "parse_error"
    incidents = data.get("incidents") or {}
    pair = pick_pair(incidents)
    if not pair:
        return None, "no_drift_pair"
    old_y, old_inc, new_y, new_inc = pair
    old_body, new_body = clean_body(body_of(old_inc), cap), clean_body(body_of(new_inc), cap)
    old_ans, new_ans = ans_name(old_inc), ans_name(new_inc)
    # 스포츠 클럽↔국가대표/유스 혼합 제외("which team" 단일정답 모호)
    if is_ambiguous_team(old_ans, new_ans):
        return None, "team_ambiguous"
    # 품질 게이트: 각 근거 문단이 자기 시점 답을 뒷받침해야 함(시점-유효 근거 보장)
    if not (answer_in_evidence(old_ans, old_body) and answer_in_evidence(new_ans, new_body)):
        return None, "evidence_mismatch"

    sid = str(data.get("event_id") or Path(path).stem)
    chunks = [
        {"chunk_id": 0, "label": "outdated_0", "text": old_body, "last_modified_time": old_y},
        {"chunk_id": 1, "label": "current",    "text": new_body, "last_modified_time": new_y},
    ]
    rec_outdated = {
        "id": f"tqa_{sid}_outdated_0", "source_idx": sid, "mode": "outdated_0",
        "new_question": q_text(old_inc), "target_answer": old_ans,
        "evidence_chunk_id": 0, "chunks": chunks,
    }
    rec_current = {
        "id": f"tqa_{sid}_current", "source_idx": sid, "mode": "current",
        "new_question": q_text(new_inc), "target_answer": new_ans,
        "evidence_chunk_id": 1, "chunks": chunks,
    }
    return (rec_outdated, rec_current), ""


def main():
    ap = argparse.ArgumentParser(description="TQA 엔티티 → 충돌 평가셋 QA 변환")
    ap.add_argument("--input", default=str(DEFAULT_IN), help="TQA 엔티티 JSON 디렉토리")
    ap.add_argument("--output", default=str(DEFAULT_OUT), help="출력 qa jsonl")
    ap.add_argument("--max", type=int, default=None, help="엔티티 수 제한(테스트용)")
    ap.add_argument("--cap", type=int, default=2000, help="청크 텍스트 최대 길이")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.input, "*.json")))
    if args.max:
        files = files[:args.max]

    out_recs, reasons = [], Counter()
    for f in files:
        res, reason = convert_entity(f, args.cap)
        if res is None:
            reasons[reason] += 1
            continue
        out_recs.extend(res)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fout:
        for r in out_recs:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_ent = len(out_recs) // 2
    print(f"엔티티 {len(files)}개 중 {n_ent}개 변환 → {args.output}")
    print(f"  레코드 {len(out_recs)}개 (outdated {n_ent} + current {n_ent})")
    print(f"  제외 사유: {dict(reasons)}")


if __name__ == "__main__":
    main()
