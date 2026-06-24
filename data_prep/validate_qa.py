"""
변환 산출 QA 데이터(qa_*.jsonl)의 무결성 검증. tqa·timeqa 공용.

실험03 평가셋이 되기 전에 데이터가 깨지지 않았는지 자동 점검한다:
  - 스키마: 필수 필드 + chunks 형식
  - 충돌 공존: current 청크 ≥1 + outdated 청크 ≥1
  - evidence_chunk_id 유효 + mode와 일치(과거→outdated, 현재→current)
  - 옛/새 청크 비동일(내용 충돌 존재)
  - 시점-유효 근거가 답을 뒷받침(변별 토큰 매칭)
  - 엔티티쌍: 옛/새 답이 다르고 near-dup 아님

usage:
  python3 data_prep/validate_qa.py data/tqa/qa_tqa.jsonl
  python3 data_prep/validate_qa.py data/timeqa/qa_timeqa.jsonl
"""
import json
import re
import sys
from collections import Counter, defaultdict

_REQ = ("id", "source_idx", "mode", "new_question", "target_answer",
        "evidence_chunk_id", "chunks")
_GENERIC = {"national", "international", "football", "team", "association", "club",
            "league", "city", "united", "states", "european", "commissioner",
            "secretary", "minister", "member", "party", "university", "college",
            "school", "district", "county", "council", "company", "group"}


def spec_tokens(s):
    return [w for w in re.findall(r"[a-z0-9]+", s.lower()) if len(w) > 3 and w not in _GENERIC]


def answer_supported(ans, text):
    a, b = ans.lower(), text.lower()
    if a in b:
        return True
    spec = spec_tokens(ans)
    return any(w in b for w in spec) if spec else a in b


def is_near_dup(a, b):
    al, bl = a.lower(), b.lower()
    if al in bl or bl in al:
        return True
    t1, t2 = set(re.findall(r"[a-z0-9]+", al)), set(re.findall(r"[a-z0-9]+", bl))
    return bool(t1 and t2) and len(t1 & t2) / len(t1 | t2) >= 0.6


def check_record(r):
    """레코드 1건 → 실패한 검사 이름 리스트."""
    fails = []
    for k in _REQ:
        if k not in r:
            return [f"missing:{k}"]
    chunks = r["chunks"]
    if not isinstance(chunks, list) or not chunks:
        return ["chunks_empty"]
    for c in chunks:
        if not all(k in c for k in ("chunk_id", "label", "text", "last_modified_time")):
            return ["chunk_schema"]
    labels = [c["label"] for c in chunks]
    if not (any(l == "current" for l in labels) and any(l.startswith("outdated") for l in labels)):
        fails.append("no_conflict")
    by_id = {c["chunk_id"]: c for c in chunks}
    ev = by_id.get(r["evidence_chunk_id"])
    if ev is None:
        fails.append("bad_evidence_id")
    else:
        side = "outdated" if r["mode"].startswith("outdated") else "current"
        if not ev["label"].startswith(side):
            fails.append("evidence_side_mismatch")
        if not answer_supported(r["target_answer"], ev["text"]):
            fails.append("answer_not_in_evidence")
    if not str(r["new_question"]).strip():
        fails.append("empty_question")
    if not str(r["target_answer"]).strip():
        fails.append("empty_answer")
    texts = {c["text"] for c in chunks}
    if len(texts) < len(chunks):
        fails.append("identical_chunks")
    return fails


def main():
    if len(sys.argv) < 2:
        print("usage: python3 data_prep/validate_qa.py <qa_*.jsonl>")
        sys.exit(2)
    path = sys.argv[1]
    recs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]

    fail_counts = Counter()
    bad = []
    pairs = defaultdict(dict)
    for r in recs:
        f = check_record(r)
        if f:
            fail_counts.update(f)
            bad.append((r.get("id", "?"), f))
        pairs[r.get("source_idx")][r.get("mode", "")] = r.get("target_answer", "")

    # 엔티티쌍 검사: 옛/새 답 다름 + near-dup 아님
    pair_fail = 0
    for sid, d in pairs.items():
        oa = next((v for k, v in d.items() if k.startswith("outdated")), None)
        na = d.get("current")
        if oa and na and (oa.lower() == na.lower() or is_near_dup(oa, na)):
            pair_fail += 1

    print(f"[validate] {path}")
    print(f"  레코드 {len(recs)} | 엔티티 {len(pairs)}")
    print(f"  레코드 검사 통과: {len(recs) - len(bad)}/{len(recs)}")
    if fail_counts:
        print(f"  실패 항목: {dict(fail_counts)}")
        for rid, f in bad[:8]:
            print(f"    ✗ {rid}: {f}")
    print(f"  엔티티쌍 약한충돌(옛≈새): {pair_fail}/{len(pairs)}")
    ok = not bad and pair_fail == 0
    print("  ==> ✅ PASS" if ok else "  ==> ❌ FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
