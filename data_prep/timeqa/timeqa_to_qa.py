"""
TimeQA(Chen et al., NeurIPS 2021 Datasets&Benchmarks) annotated → 평가용 QA 레코드.

TimeQA annotated 데이터는 엔티티마다 위키 페이지 문단(paras)과, 시점 구간별로
{정답 + gold 근거 문단 인덱스(para) + 글자 span}을 **사람이 라벨**해 제공한다.
여기서 답이 다른 옛/새 두 시점을 골라 충돌 컨텍스트(두 시점의 gold 문단 공존)를 만든다.
근거 문단이 라벨돼 있어 TQA와 달리 추론 게이트가 거의 불필요. LLM 호출 없음(무료).

입력(annotated_*.json) 레코드: {index, type(Wikidata Pxx), link, paras:[str],
  questions:[ [[time_start,time_end], [{para,from,end,answer}, ...]] , ... ]}
관계→질문 템플릿: relations.json ({Pxx: {meaning, template:[...]}}, $1=주체, $2/$4=시점)

출력(= 01_sample_eval_set.py 소비 스키마):
  {id, source_idx, mode:"outdated_0"|"current", new_question, target_answer,
   evidence_chunk_id, chunks:[{chunk_id,label,text,last_modified_time}]}

usage:
  python3 data_prep/timeqa/timeqa_to_qa.py                    # 전체(annotated_*.json)
  python3 data_prep/timeqa/timeqa_to_qa.py --split dev        # dev만
  python3 data_prep/timeqa/timeqa_to_qa.py --max 200
"""
import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]          # data_prep/timeqa/ → repo 루트
DEFAULT_IN = _REPO / "data" / "timeqa" / "source"
DEFAULT_OUT = _REPO / "data" / "timeqa" / "qa_timeqa.jsonl"

_YEAR = re.compile(r"\d{4}")
_WS = re.compile(r"\s+")
# 스포츠 클럽↔국가대표/유스 모호성 (TQA와 동일 기준)
_NAT_TEAM = re.compile(r"\bnational\b.{0,30}\bteam\b", re.I)
_YOUTH_TEAM = re.compile(r"\bunder[-\s]?\d+\b|\bu[-\s]?\d{2}\b", re.I)
# generic 역할/조직 단어 — 근거 매칭 변별력 없음
_GENERIC = {
    "national", "international", "football", "soccer", "team", "association", "club",
    "league", "city", "united", "states", "kingdom", "european", "commissioner",
    "secretary", "minister", "ministry", "member", "party", "university", "college",
    "school", "district", "county", "council", "federal", "republic", "democratic",
    "general", "office", "department", "committee", "company", "corporation", "group",
    "under", "women", "academy", "institute", "society", "service", "board",
}


def entity_name(link: str) -> str:
    n = link.split("/")[-1].replace("_", " ")
    return re.sub(r"\s*\([^)]*\)", "", n).strip()   # "(politician)" 등 제거


def year_of(t: str):
    m = _YEAR.search(t or "")
    return int(m.group()) if m else None


def clean(text: str, cap: int = 2000) -> str:
    return _WS.sub(" ", text or "").strip()[:cap]


def is_ambiguous_team(*answers: str) -> bool:
    return any(_NAT_TEAM.search(a or "") or _YOUTH_TEAM.search(a or "") for a in answers)


def answer_in_evidence(ans: str, body: str) -> bool:
    """안전망: gold 라벨이라 대개 통과하나, 주석 노이즈 방어용 약한 검사."""
    a, b = ans.lower(), body.lower()
    if not a:
        return False
    if a in b:
        return True
    spec = [w for w in re.findall(r"[a-z0-9]+", a) if len(w) > 3 and w not in _GENERIC]
    return any(w in b for w in spec) if spec else a in b


def valid_periods(ex: dict):
    """[(year, answer, para)] — 단일 비공백 답 + 시작연도를 가진 시점만."""
    out = []
    for q in ex.get("questions", []):
        if not (isinstance(q, list) and len(q) == 2):
            continue
        time_range, evs = q
        ans = [e for e in (evs or []) if (e.get("answer") or "").strip()]
        if not ans:
            continue
        if len({a["answer"].strip() for a in ans}) != 1:   # 다중 답 시점 제외(모호)
            continue
        y = year_of(time_range[0] if time_range else "")
        if y is None:
            continue
        out.append((y, ans[0]["answer"].strip(), ans[0].get("para", 0)))
    return out


def pick_pair(periods):
    """새=가장 늦은 시점, 옛=답이 다른 가장 이른 시점."""
    periods = sorted(periods)
    if len(periods) < 2:
        return None
    new = periods[-1]
    for old in periods[:-1]:
        if old[1].lower() != new[1].lower():
            return old, new
    return None


def make_question(template: str, name: str, year: int) -> str:
    t, time = template, f"in {year}"
    for ph in ("$4", "$2", "$3", "$5"):     # 관계마다 시점 placeholder가 다름
        if ph in t:
            t = t.replace(ph, time)
            break
    return _WS.sub(" ", t.replace("$1", name)).strip()


def convert_entity(ex: dict, relations: dict, cap: int):
    templates = (relations.get(ex.get("type")) or {}).get("template") or []
    if not templates:
        return None, "no_template"
    pair = pick_pair(valid_periods(ex))
    if not pair:
        return None, "no_drift_pair"
    (oy, oa, op), (ny, na, npa) = pair
    if is_ambiguous_team(oa, na):
        return None, "team_ambiguous"
    paras = ex.get("paras") or []
    if not (0 <= op < len(paras) and 0 <= npa < len(paras)):
        return None, "bad_para"
    old_body, new_body = clean(paras[op], cap), clean(paras[npa], cap)
    if not (old_body and new_body):
        return None, "empty_evidence"
    if not (answer_in_evidence(oa, old_body) and answer_in_evidence(na, new_body)):
        return None, "evidence_mismatch"

    name = entity_name(ex.get("link", ""))
    sid = re.sub(r"[^A-Za-z0-9]+", "_", ex.get("index") or ex.get("link", "")).strip("_")
    chunks = [
        {"chunk_id": 0, "label": "outdated_0", "text": old_body, "last_modified_time": str(oy)},
        {"chunk_id": 1, "label": "current",    "text": new_body, "last_modified_time": str(ny)},
    ]
    tmpl = templates[0]
    rec_outdated = {
        "id": f"timeqa_{sid}_outdated_0", "source_idx": sid, "mode": "outdated_0",
        "new_question": make_question(tmpl, name, oy), "target_answer": oa,
        "evidence_chunk_id": 0, "chunks": chunks,
    }
    rec_current = {
        "id": f"timeqa_{sid}_current", "source_idx": sid, "mode": "current",
        "new_question": make_question(tmpl, name, ny), "target_answer": na,
        "evidence_chunk_id": 1, "chunks": chunks,
    }
    return (rec_outdated, rec_current), ""


def main():
    ap = argparse.ArgumentParser(description="TimeQA annotated → 충돌 평가셋 QA 변환")
    ap.add_argument("--input", default=str(DEFAULT_IN), help="annotated_*.json + relations.json 디렉토리")
    ap.add_argument("--output", default=str(DEFAULT_OUT))
    ap.add_argument("--split", default="all", choices=["all", "train", "dev", "test"])
    ap.add_argument("--max", type=int, default=None)
    ap.add_argument("--cap", type=int, default=2000, help="청크 텍스트 최대 길이")
    args = ap.parse_args()

    relations = json.load(open(os.path.join(args.input, "relations.json"), encoding="utf-8"))
    splits = ["train", "dev", "test"] if args.split == "all" else [args.split]
    entities = []
    for sp in splits:
        path = os.path.join(args.input, f"annotated_{sp}.json")
        if os.path.exists(path):
            entities.extend(json.load(open(path, encoding="utf-8")))
    if args.max:
        entities = entities[:args.max]

    out_recs, reasons = [], Counter()
    seen = set()
    for ex in entities:
        res, reason = convert_entity(ex, relations, args.cap)
        if res is None:
            reasons[reason] += 1
            continue
        if res[0]["source_idx"] in seen:    # 분할 간 중복 엔티티 제거
            reasons["dup_entity"] += 1
            continue
        seen.add(res[0]["source_idx"])
        out_recs.extend(res)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fout:
        for r in out_recs:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(out_recs) // 2
    print(f"엔티티 {len(entities)}개 중 {n}개 변환 → {args.output}")
    print(f"  레코드 {len(out_recs)}개 (outdated {n} + current {n})")
    print(f"  제외 사유: {dict(reasons)}")


if __name__ == "__main__":
    main()
