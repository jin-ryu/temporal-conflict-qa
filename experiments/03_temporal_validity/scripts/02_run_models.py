"""
02 — 모델 실행 (§5, §6).

각 문항 × {conflict, current_only, outdated_only} × 모델 → 답·인용 수집.
인용은 inline self-citation([k])으로 추출(§5.1). 반사실 3조건은 §5.2.

출력: results/raw_<model>.jsonl

usage:
  python 02_run_models.py --model gpt
  python 02_run_models.py --model claude
  python 02_run_models.py --model llama70b      # vLLM 서빙 필요(VLLM_BASE_URL)
"""
import argparse, os, random
import pilot_common as pc

CONDITIONS = ["conflict", "current_only", "outdated_only"]


def run_one(model, rec, rng):
    out = {"id": rec["id"], "model": model, "mode": rec["mode"],
           "target_side": rec["target_side"], "evidence_chunk_id": rec["evidence_chunk_id"]}
    for cond in CONDITIONS:
        chunks = pc.filter_chunks(rec["chunks"], cond)
        user, idx2chunk = pc.build_user_message(rec["new_question"], chunks, rng)
        try:
            raw = pc.call_model(model, pc.SYSTEM_PROMPT, user)
            parsed = pc.parse_output(raw)
            cite_chunk_ids = [idx2chunk[i] for i in parsed["cite_indices"] if i in idx2chunk]
            out[cond] = {"answer": parsed["answer"], "reasoning": parsed["reasoning"],
                         "cite_chunk_ids": cite_chunk_ids, "raw": parsed["raw"]}
        except Exception as e:  # noqa
            out[cond] = {"error": str(e), "answer": "", "cite_chunk_ids": []}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(pc.MODELS))
    ap.add_argument("--eval", default=os.path.join(os.path.dirname(__file__), "..", "data", "eval_set.jsonl"))
    ap.add_argument("--out_dir", default=os.path.join(os.path.dirname(__file__), "..", "results"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    recs = list(pc.read_jsonl(os.path.abspath(args.eval)))
    rows = []
    for i, rec in enumerate(recs, 1):
        rows.append(run_one(args.model, rec, rng))
        if i % 10 == 0:
            print(f"  {i}/{len(recs)}")
    out = os.path.join(os.path.abspath(args.out_dir), f"raw_{args.model}.jsonl")
    pc.write_jsonl(out, rows)
    print(f"{len(rows)}문항 × 3조건 완료 → {out}")
    print("\n[이번 실행 사용량/비용]")
    print(pc.cost_report())
    pc.append_ledger("02_run_models")
    print("\n[전체 누적 — repo 루트 usage/usage_summary.txt]")
    print(pc.ledger_total())


if __name__ == "__main__":
    main()
