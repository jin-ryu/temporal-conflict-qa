"""
검수 끝난 시트(review_sheet_hoh.csv) → genuine만 추린 평가셋.

verdict 칸 기준으로 qa_hoh.jsonl을 거른다:
   g / genuine / keep  → 유지(genuine 시간변화)
   c / correction / drop → 버림(정적사실 정정)
   ? / 빈칸             → 보류(미검수로 집계, 기본 제외)

usage:
  python3 data_prep/hoh/apply_review.py
출력: data/hoh/qa_hoh_genuine.jsonl  (01_sample 소비)
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
DEFAULT_SHEET = _REPO / "data" / "hoh" / "review_sheet_hoh.csv"
DEFAULT_QA = _REPO / "data" / "hoh" / "qa_hoh.jsonl"
DEFAULT_OUT = _REPO / "data" / "hoh" / "qa_hoh_genuine.jsonl"
_KEEP = {"g", "genuine", "keep", "유지", "o"}
_DROP = {"c", "correction", "drop", "버림", "x"}


def verdict_of(raw):
    v = (raw or "").strip().lower()
    if v in _KEEP:
        return "keep"
    if v in _DROP:
        return "drop"
    return "pending"  # ? 또는 빈칸


def main():
    ap = argparse.ArgumentParser(description="검수 시트 적용 → genuine 평가셋")
    ap.add_argument("--sheet", default=str(DEFAULT_SHEET))
    ap.add_argument("--qa", default=str(DEFAULT_QA))
    ap.add_argument("--output", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    verdicts = {}
    for r in csv.DictReader(open(args.sheet, encoding="utf-8-sig")):
        verdicts[str(r["source_idx"])] = verdict_of(r.get("verdict"))
    tally = Counter(verdicts.values())

    keep_ids = {sid for sid, v in verdicts.items() if v == "keep"}
    recs = [json.loads(l) for l in open(args.qa, encoding="utf-8") if l.strip()]
    kept = [r for r in recs if str(r["source_idx"]) in keep_ids]

    with open(args.output, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"검수 결과: keep {tally['keep']} / drop {tally['drop']} / 미검수(?·빈칸) {tally['pending']}")
    print(f"  → genuine {len(kept)//2} 엔티티 ({len(kept)} 레코드) → {args.output}")
    if tally["pending"]:
        print(f"  ⚠️ 미검수 {tally['pending']}건은 제외됨. 다 보려면 verdict 채우고 재실행.")


if __name__ == "__main__":
    main()
