"""
HoH(chunks) → 충돌 평가셋 QA (결정론적·LLM 없음).

HoH는 위키 편집의 *바뀐 사실 부분*만 옛/새 청크로 뽑아 → **충돌이 날카롭다**
(current 청크엔 새 답만, outdated 청크엔 옛 답만 — 서로 답이 안 샌다). + distractor
청크까지 있어 현실적. 단점이던 "질문 생성"은 여기서 **LLM 대신 날짜 prepend 템플릿**으로
결정론적으로 만든다(예전 LLM 생성의 메타-anchor 문제 회피).

입력(`data/hoh/chunks/chunks_0_600.jsonl`, hoh_to_chunks.py 산출):
  {id, hoh_source_idx, question, answers:[{label:current|outdated, answer, last_modified_time,
   outdated_index?}], chunks:[{chunk_id, label:current|outdated|distractor, last_modified_time, text}]}

출력(`data/hoh/qa_hoh.jsonl`, 01_sample 소비 스키마): 엔티티당 outdated_0 + current 레코드.

사실(fact) 정제 — 청크 모양은 *안 건드리고* 어떤 문항을 남길지만 거른다:
  · Fix1 단일전환만: outdated 답이 정확히 1개인 것만(다중=편집전쟁/churn 드롭). 남은 답은
    outdated_index로 *정확한 청크에 매칭*(예전 '마지막답↔첫청크' 미스매치 버그 해소).
  · Fix2 엄격 근거게이트: 답의 변별토큰이 *자기 청크*에 과반 이상 실재해야 통과(동명이인·엉뚱근거 컷).
  · Fix3-자동: clean_answer로 숫자·측정값 답(체중 440→350류 오타정정) 배제.
⚠️ 자동이 못 거르는 *애매한 엔티티 정정*은 **수동 검수(validation_sheet)**가 최종 안전망.
   오래된/잘못된 버전은 버리지 않고 *distractor 청크*로 남아 현실적 노이즈 역할.

usage:
  python3 data_prep/hoh/hoh_to_qa.py
  python3 data_prep/hoh/hoh_to_qa.py --max 200
"""
import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
DEFAULT_IN = _REPO / "data" / "hoh" / "chunks" / "chunks_0_600.jsonl"
DEFAULT_OUT = _REPO / "data" / "hoh" / "qa_hoh.jsonl"
_MONTHS = ["", "January", "February", "March", "April", "May", "June",
           "July", "August", "September", "October", "November", "December"]
_GENERIC = {"the", "of", "and", "a", "an", "in", "at", "for", "over", "under", "about",
            "former", "formerly", "current", "currently", "with", "from", "than", "more"}
_UNITS = {"lbs", "lb", "kg", "kgs", "km", "mi", "miles", "mile", "ft", "feet", "cm", "mm",
          "mph", "kmh", "percent", "million", "billion", "thousand", "kilometres", "kilometers"}


def month_year(ts):
    m = re.match(r"(\d{4})-(\d{2})", str(ts or ""))
    return f"{_MONTHS[int(m.group(2))]} {m.group(1)}" if m else None


def as_of_q(question, ts):
    """원 질문 앞에 '시점' 명시(결정론적). 예: 'As of June 2024, what is ...'"""
    my = month_year(ts)
    q = (question or "").strip()
    if not q:
        return q
    return f"As of {my}, {q[0].lower() + q[1:]}" if my else q


def answer_in_strict(ans, text):
    """Fix2: 정식명 전체 포함 OR 변별토큰 *과반 이상* 실재 → 약한 우연매칭 컷."""
    a, b = ans.lower(), text.lower()
    if a in b:
        return True
    spec = [w for w in re.findall(r"[a-z0-9]+", a) if len(w) > 3 and w not in _GENERIC]
    if not spec:
        return a in b
    hit = sum(1 for w in spec if w in b)
    return hit * 2 >= len(spec)


def clean_answer(a):
    """깨끗한 엔티티 답만 — 사상자수·체중·법령번호 등 *수치/측정값 정정류* 배제(Fix3-자동)."""
    if not a or len(a.split()) > 5:
        return False
    toks = re.findall(r"[a-z0-9]+", a.lower())
    if any(t in _UNITS for t in toks):                       # 측정단위 → 정정/수치
        return False
    if any(t.isdigit() and len(t) >= 2 for t in toks):       # 2자리+ 순수 숫자 토큰(체중·연도·카운트)
        return False
    if sum(c.isdigit() for c in a) / max(1, len(a)) > 0.15:
        return False
    return "," not in a and " and " not in a.lower()


def is_near_dup(a, b):
    al, bl = a.lower(), b.lower()
    if al in bl or bl in al:
        return True
    t1, t2 = set(re.findall(r"[a-z0-9]+", al)), set(re.findall(r"[a-z0-9]+", bl))
    return bool(t1 and t2) and len(t1 & t2) / len(t1 | t2) >= 0.6


def convert(rec):
    chunks = []
    for c in rec.get("chunks", []):
        chunks.append({"chunk_id": int(c["chunk_id"]), "label": c["label"], "text": c["text"],
                       "last_modified_time": c.get("last_modified_time"),
                       "outdated_index": c.get("outdated_index")})
    cur_chunks = [c for c in chunks if c["label"] == "current"]
    out_chunks = [c for c in chunks if c["label"] == "outdated"]
    if not cur_chunks or not out_chunks:
        return None, "no_conflict_chunk"

    # Fix1: 단일전환만 — outdated 답이 정확히 1개(다중=편집전쟁 churn 드롭)
    cur_as = [a for a in rec.get("answers", []) if a.get("label") == "current"]
    out_as = [a for a in rec.get("answers", []) if a.get("label") == "outdated"]
    if not cur_as or len(out_as) != 1:
        return None, "multi_or_no_transition"
    cur_a, old_a = cur_as[0], out_as[0]

    # Fix1: outdated_index로 *정확한* 청크 매칭
    oi = old_a.get("outdated_index", 0)
    old_cand = [c for c in out_chunks if c["outdated_index"] == oi]
    if not old_cand:
        return None, "no_matching_outdated_chunk"
    old_c = old_cand[0]
    # current 청크: 답을 담는 것 우선(여러 current 청크 가능)
    cur_c = next((c for c in cur_chunks if answer_in_strict(cur_a["answer"], c["text"])), cur_chunks[0])

    if cur_c["text"] == old_c["text"]:                 # 옛/새 청크 동일 → 충돌 아님
        return None, "identical_evidence"
    # Fix3-자동: 깨끗한 엔티티 답만(숫자·측정값 정정류 배제)
    if not (clean_answer(cur_a["answer"]) and clean_answer(old_a["answer"])):
        return None, "messy_answer"
    # Fix2: 엄격 근거게이트 — 각 답이 *자기* 청크에 과반 실재
    if not (answer_in_strict(cur_a["answer"], cur_c["text"]) and answer_in_strict(old_a["answer"], old_c["text"])):
        return None, "evidence_mismatch"
    # 옛/새 답이 실질적으로 같으면(정식명↔약칭 포함) 충돌 아님
    if is_near_dup(cur_a["answer"], old_a["answer"]):
        return None, "near_dup"

    sid = str(rec.get("hoh_source_idx", rec.get("id")))
    q = rec.get("question", "")
    rec_o = {"id": f"hoh_{sid}_outdated_0", "source_idx": sid, "mode": "outdated_0",
             "new_question": as_of_q(q, old_a["last_modified_time"]),
             "target_answer": old_a["answer"], "evidence_chunk_id": old_c["chunk_id"], "chunks": chunks}
    rec_c = {"id": f"hoh_{sid}_current", "source_idx": sid, "mode": "current",
             "new_question": as_of_q(q, cur_a["last_modified_time"]),
             "target_answer": cur_a["answer"], "evidence_chunk_id": cur_c["chunk_id"], "chunks": chunks}
    return (rec_o, rec_c), ""


def main():
    ap = argparse.ArgumentParser(description="HoH chunks → 충돌 평가셋 (LLM 없음)")
    ap.add_argument("--input", default=str(DEFAULT_IN))
    ap.add_argument("--output", default=str(DEFAULT_OUT))
    ap.add_argument("--max", type=int, default=None)
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(args.input, encoding="utf-8") if l.strip()]
    if args.max:
        recs = recs[:args.max]
    out, reasons = [], Counter()
    seen = set()
    for r in recs:
        res, why = convert(r)
        if res is None:
            reasons[why] += 1
            continue
        if res[0]["source_idx"] in seen:
            reasons["dup"] += 1
            continue
        seen.add(res[0]["source_idx"])
        out.extend(res)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fo:
        for r in out:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"레코드 {len(recs)} → {len(out)//2} 변환 ({len(out)} 레코드) → {args.output}")
    print(f"  제외 사유: {dict(reasons)}")


if __name__ == "__main__":
    main()
