"""
HoH 정제본(qa_hoh.jsonl) → 사람검수 CSV.

목적: 152개에서 *진짜 시간변화(genuine)* vs *정적사실 정정(correction)* 을 사람이 30분에 가린다.
각 행은 자기완결 — 질문·옛/새 답·시점 + *정답 주변 근거 스니펫*까지 담아 파일 안 열고 판단.

검수법: `verdict` 칸에
   g = genuine 시간변화(유지)   /   c = 정적사실 정정(버림)   /   ? = 애매
   → 판단 기준 상세: construct_filter_guideline.md (g/c 기준표·예시·LLM rubric)
저장 후 apply_review.py 로 g만 추려 eval 후보 확정.

usage:
  python3 data_prep/hoh/make_review_sheet.py
출력: data/hoh/review_sheet_hoh.csv
"""
import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
DEFAULT_IN = _REPO / "data" / "hoh" / "qa_hoh.jsonl"
DEFAULT_OUT = _REPO / "data" / "hoh" / "review_sheet_hoh.csv"
_NUMWORDS = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
             "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
             "seventeen", "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty",
             "hundred", "thousand", "million", "first", "second", "third", "fourth", "fifth"}


def base_q(q):
    """앞의 'As of <시점>, ' 제거 → 원 질문."""
    return re.sub(r"^As of [^,]+,\s*", "", q or "").strip()


def snippet(text, answer, width=200):
    """정답(또는 변별토큰) 주변 ~width자 창 — 정적/변화 판단용 맥락."""
    t = " ".join((text or "").split())
    low, a = t.lower(), (answer or "").lower()
    i = low.find(a)
    if i == -1:
        for w in sorted(re.findall(r"[a-z0-9]+", a), key=len, reverse=True):
            if len(w) > 3:
                i = low.find(w)
                if i != -1:
                    break
    if i == -1:
        i = 0
    start = max(0, i - width // 3)
    end = start + width
    return ("…" if start > 0 else "") + t[start:end] + ("…" if end < len(t) else "")


def flag(old_a, new_a):
    """자동 힌트(판정 아님): 카운트/숫자류면 정정 의심."""
    toks = re.findall(r"[a-z0-9]+", (old_a + " " + new_a).lower())
    if any(t in _NUMWORDS for t in toks) or any(t.isdigit() for t in toks):
        return "수치?"
    return ""


def main():
    ap = argparse.ArgumentParser(description="HoH 정제본 → 사람검수 CSV")
    ap.add_argument("--input", default=str(DEFAULT_IN))
    ap.add_argument("--output", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(args.input, encoding="utf-8") if l.strip()]
    by = defaultdict(dict)
    for r in recs:
        by[r["source_idx"]][r["mode"]] = r

    rows = []
    for sid, d in by.items():
        o, c = d.get("outdated_0"), d.get("current")
        if not (o and c):
            continue
        ev = {x["chunk_id"]: x for x in o["chunks"]}
        old_ev = ev.get(o["evidence_chunk_id"], {})
        new_ev = ev.get(c["evidence_chunk_id"], {})
        rows.append({
            "source_idx": sid,
            "question": base_q(o["new_question"]),
            "old_answer": o["target_answer"],
            "new_answer": c["target_answer"],
            "old_time": old_ev.get("last_modified_time", ""),
            "new_time": new_ev.get("last_modified_time", ""),
            "old_evidence": snippet(old_ev.get("text", ""), o["target_answer"]),
            "new_evidence": snippet(new_ev.get("text", ""), c["target_answer"]),
            "flag": flag(o["target_answer"], c["target_answer"]),
            "verdict": "",   # ← 여기에 g / c / ?
            "note": "",
        })

    cols = ["source_idx", "question", "old_answer", "new_answer", "old_time", "new_time",
            "old_evidence", "new_evidence", "flag", "verdict", "note"]
    with open(args.output, "w", newline="", encoding="utf-8-sig") as f:  # utf-8-sig: Excel 한글 OK
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for i, r in enumerate(rows, 1):
            w.writerow({"source_idx": r["source_idx"], **{k: r[k] for k in cols if k != "source_idx"}})

    nflag = sum(1 for r in rows if r["flag"])
    print(f"검수시트 생성: {len(rows)}행 → {args.output}")
    print(f"  자동 '수치?' 플래그: {nflag}행 (정정 의심 — 먼저 보세요)")
    print(f"  검수: verdict 칸에 g(유지)/c(버림)/?(애매) 입력 후 저장 → apply_review.py")


if __name__ == "__main__":
    main()
