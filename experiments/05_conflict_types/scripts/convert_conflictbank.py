"""
ConflictBank (NeurIPS 2024) → 우리 평가셋 (충돌 *유형* 일반화 보조 실험).

목적: "틀린답에 좋은인용(mis-attribution)" 이 *시간 충돌 외*(오정보·의미)에서도 생기나 확인.
ConflictBank 한 항목 = 정답 evidence + 3유형 충돌 evidence(misinformation/temporal/semantic).
각 유형마다 별도 문항 생성:
  · 청크 = [정답 evidence] + [해당 유형 충돌 evidence] + [다른 항목의 evidence 1~2개=distractor]
  · 질문 = question, 정답 = correct_option 텍스트
  · evidence_chunk_id = 정답 evidence 청크
→ 모델이 충돌 evidence를 인용/사용해 틀리면 mis-attribution ★.

⚠️ ConflictBank는 *합성*(LLM 생성 evidence, 미래 날짜 등) → 통제·일반화용 보조. 현실성은 exp03/04.

usage: python convert_conflictbank.py --per_type 40
출력: ../data/eval_{type}.jsonl (유형별)
"""
import json, os, argparse, random

SRC = "/tmp/cb_qa.json"
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
TYPES = {
    "misinformation": ("misinformation_conflict_claim", "misinformation_conflict_evidence_evidence"),
    "temporal": ("temporal_conflict_claim", "temporal_conflict_evidence"),
    "semantic": ("semantic_conflict_claim", "semantic_conflict_evidence"),
}


def opt_text(r, letter):
    """correct_option 'D' → options 텍스트."""
    idx = "ABCD".find(letter)
    opts = r.get("options", [])
    return opts[idx] if 0 <= idx < len(opts) else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_type", type=int, default=40)
    ap.add_argument("--max_scan", type=int, default=4000)
    args = ap.parse_args()
    rng = random.Random(0)

    # 일부만 스캔 (2.9GB 전체 안 읽음)
    pool = []
    with open(SRC) as f:
        for i, line in enumerate(f):
            if i >= args.max_scan:
                break
            try:
                pool.append(json.loads(line))
            except Exception:
                continue
    print(f"스캔 {len(pool)}건")

    # distractor용 evidence 모음 (다른 항목의 default_evidence)
    distractors = [r["default_evidence"] for r in pool if r.get("default_evidence")]

    os.makedirs(os.path.abspath(OUT_DIR), exist_ok=True)
    for tkey, (claim_f, ev_f) in TYPES.items():
        out = []
        cand = [r for r in pool if r.get(ev_f) and r.get("default_evidence") and r.get("correct_option")]
        rng.shuffle(cand)
        for r in cand[:args.per_type]:
            gold = opt_text(r, r["correct_option"])
            if not gold:
                continue
            conflict_ev = r[ev_f]
            if isinstance(conflict_ev, list):
                conflict_ev = " ".join(str(x) for x in conflict_ev)
            chunks = [
                {"chunk_id": 0, "label": "current", "last_modified_time": "default",
                 "text": r["default_evidence"]},                      # 정답
                {"chunk_id": 1, "label": "outdated", "last_modified_time": tkey,
                 "text": conflict_ev},                                # 충돌(오답 유발)
            ]
            # distractor 2개 (다른 항목)
            for j, d in enumerate(rng.sample(distractors, min(2, len(distractors))), start=2):
                if d != r["default_evidence"]:
                    chunks.append({"chunk_id": j, "label": "distractor",
                                   "last_modified_time": "other", "text": d})
            rng.shuffle(chunks)
            for k, c in enumerate(chunks):
                c["chunk_id"] = k
            ev_id = next(c["chunk_id"] for c in chunks if c["label"] == "current")
            out.append({
                "id": f"cb_{tkey}_{len(out)}", "source_idx": f"cb_{tkey}_{len(out)}",
                "mode": "current", "target_side": "current", "conflict_type": tkey,
                "new_question": r["question"], "target_answer": gold,
                "evidence_chunk_id": ev_id, "chunks": chunks})
        path = os.path.join(os.path.abspath(OUT_DIR), f"eval_{tkey}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for r in out:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  {tkey}: {len(out)}건 → {path}")


if __name__ == "__main__":
    main()
