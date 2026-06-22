"""
01 — 평가셋 샘플링 (§3.4).

입력: 기존 가공 QA (data/qa/*.jsonl). 데이터 생성 코드는 변경 불필요 — 이미
충돌 컨텍스트(current+outdated 공존) + 청크 label + evidence_chunk_id 포함.

mode별 개수를 --quota로 지정(일반화):
  - 키는 정확 일치 또는 접두 일치 — "outdated"는 outdated_0/1/2.. 전부, "outdated_1"은 그것만.
  - current_raw(시간 신호 없음→정답 모호)는 항상 제외.

mode 역할(설계 §3.4):
  - outdated_i (과거 지향)        : 주력(기여). 다단계(outdated_1+) 우선 포함.
  - current   (현재 지향)        : recency-bias 대조군.
  - current_raw                  : 제외.

출력: data/eval_set.jsonl + data/validation_sheet.csv (사람 검수용)

usage:
  python 01_sample_eval_set.py --input ../../../data/qa/qa_gemini_work.jsonl
  python 01_sample_eval_set.py --input ... --quota "outdated=40,current=10"
  python 01_sample_eval_set.py --input ... --quota "outdated_0=20,outdated_1=15,current=10"
"""
import argparse, csv, os, random
from collections import Counter
import pilot_common as pc

KEEP = ("id", "hoh_source_idx", "mode", "new_question", "target_answer",
        "evidence_chunk_id", "chunks")


def has_conflict(rec) -> bool:
    labels = [c["label"] for c in rec["chunks"]]
    return any(l == "current" for l in labels) and any(l.startswith("outdated") for l in labels)


def target_side(mode: str) -> str:
    return "outdated" if mode.startswith("outdated") else "current"


def matches(mode: str, key: str) -> bool:
    """정확 일치 또는 'key_' 접두 일치 (current_raw는 호출 전에 이미 제외)."""
    return mode == key or mode.startswith(key + "_")


def parse_quota(s: str):
    quota = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        k, _, v = part.partition("=")
        quota.append((k.strip(), int(v)))
    return quota


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out_dir", default=os.path.join(os.path.dirname(__file__), "..", "data"))
    ap.add_argument("--quota", default="outdated=40,current=10",
                    help="mode별 개수. 예: 'outdated=40,current=10' / 'outdated_0=20,outdated_1=15,current=10'")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    quota = parse_quota(args.quota)

    # 후보: current_raw 제외 + 충돌 컨텍스트 보유
    pool = [r for r in pc.read_jsonl(args.input)
            if r["mode"] != "current_raw" and has_conflict(r)]

    used, selected = set(), []
    for key, count in quota:
        cands = [r for r in pool if matches(r["mode"], key)]
        rng.shuffle(cands)
        # outdated는 다단계(outdated_1+)를 먼저 (stable sort로 셔플 순서 유지)
        cands.sort(key=lambda r: 0 if (r["mode"].startswith("outdated") and r["mode"] != "outdated_0") else 1)
        picked = 0
        for r in cands:
            if picked >= count:
                break
            if r["hoh_source_idx"] in used:   # leakage 방지: 소스 중복 금지
                continue
            used.add(r["hoh_source_idx"])
            row = {k: r[k] for k in KEEP}
            row["target_side"] = target_side(r["mode"])
            selected.append(row)
            picked += 1
        if picked < count:
            print(f"  ⚠️ '{key}': {picked}/{count}개만 확보 (후보 부족)")
    rng.shuffle(selected)

    out_dir = os.path.abspath(args.out_dir)
    pc.write_jsonl(os.path.join(out_dir, "eval_set.jsonl"), selected)

    # 수동 검증 시트
    with open(os.path.join(out_dir, "validation_sheet.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "mode", "new_question", "target_answer",
                    "gold_evidence_text", "OK_시점결정성", "OK_자연스러움", "비고"])
        for r in selected:
            gold = next((c["text"] for c in r["chunks"]
                         if c["chunk_id"] == r["evidence_chunk_id"]), "")
            w.writerow([r["id"], r["mode"], r["new_question"], r["target_answer"], gold, "", "", ""])

    dist = Counter(r["mode"] for r in selected)
    print(f"총 {len(selected)}문항 → {out_dir}/eval_set.jsonl")
    print(f"  mode 분포: {dict(dist)}")
    print(f"  target_side: {dict(Counter(r['target_side'] for r in selected))}")
    print("→ validation_sheet.csv 사람 검수 후, 통과 항목만 남겨 본실험에 사용 (§3.3 Step 3)")


if __name__ == "__main__":
    main()
