"""
02 — 모델 실행 (§5, §6).

각 문항 × {conflict, current_only, outdated_only} × 모델 → 답·인용 수집.
인용은 inline self-citation([k])으로 추출(§5.1). 반사실 3조건은 §5.2.

출력: results/raw_<model>.jsonl (문항마다 즉시 append — 중간에 끊겨도 보존)

재개(resume): 같은 명령 재실행 시 기존 파일의 *에러 없는* 문항은 건너뛰고 나머지만 실행.
  → 네트워크 끊김·키 오류로 중단돼도, 키 고치고 다시 돌리면 못 한 것만 이어서 함.
진행상황: 실행 중 `wc -l results/raw_<model>.jsonl` 로 실시간 확인.

usage:
  python 02_run_models.py --model gpt
  python 02_run_models.py --model claude
  python 02_run_models.py --model llama70b      # vLLM 서빙 필요(VLLM_BASE_URL)
"""
import argparse, json, os, random
import pilot_common as pc

CONDITIONS = ["conflict", "current_only", "outdated_only"]


def is_done(row):
    """모든 조건이 에러 없이 끝났나 (재개 시 재실행 불필요 판정)."""
    return "id" in row and all(not row.get(c, {}).get("error") for c in CONDITIONS)


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
    out = os.path.join(os.path.abspath(args.out_dir), f"raw_{args.model}.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    # 재개: 기존 파일에서 '에러 없는' 행만 보존, 나머지(에러·미실행)는 다시 실행
    done = {}
    if os.path.exists(out):
        for l in open(out, encoding="utf-8"):
            l = l.strip()
            if not l:
                continue
            try:
                r = json.loads(l)
            except Exception:
                continue
            if is_done(r):
                done[r["id"]] = r
        with open(out, "w", encoding="utf-8") as f:   # 좋은 행만 남겨 재작성(중복·에러행 제거)
            for r in done.values():
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        if done:
            print(f"  재개: 기존 {len(done)}문항 보존, 나머지만 실행")

    # 문항마다 즉시 append + flush → 중간에 끊겨도 보존, 진행상황 실시간(wc -l) 확인 가능
    todo = [rec for rec in recs if rec["id"] not in done]
    with open(out, "a", encoding="utf-8") as f:
        for i, rec in enumerate(todo, 1):
            f.write(json.dumps(run_one(args.model, rec, rng), ensure_ascii=False) + "\n")
            f.flush()
            print(f"  {i}/{len(todo)}  {rec['id']}", flush=True)
    print(f"\n완료 → {out} (총 {len(done) + len(todo)}문항 × 3조건)")
    print("\n[이번 실행 사용량/비용]")
    print(pc.cost_report())
    pc.append_ledger("02_run_models")
    print("\n[전체 누적 — repo 루트 usage/usage_summary.txt]")
    print(pc.ledger_total())


if __name__ == "__main__":
    main()
