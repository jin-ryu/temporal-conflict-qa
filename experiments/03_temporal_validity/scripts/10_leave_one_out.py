"""
10 — 진짜 leave-one-out counterfactual (faithfulness 엄밀 검증).

동기: 기존 faithfulness 분류(eval_unified)는 3조건(conflict/current_only/outdated_only)
      답 비교에 기반한 *근사*다 (컨텍스트 구성이 달라 완전한 leave-one-out 아님).
      여기서는 Wallat 2025식 진짜 leave-one-out: 틀린 답을 지지한 문서 *하나만* 빼고
      *같은 conflict 컨텍스트*에서 재실행 → 답이 바뀌면 그 문서에 진짜 의존(faithful-wrong),
      안 바뀌면 사후인용(post-rationalization).

절차 (각 문항):
  1) 기존 raw의 conflict 답을 읽어 EM 채점. 틀린 답(EM=0)만 대상.
  2) 뺄 문서 결정: 모델이 *인용한 문서 중 그 틀린 답을 지지하는 것*
     (인용 없으면: conflict 청크 중 답 지지하는 것 → 그래도 없으면 skip).
  3) 그 문서만 제거한 conflict 컨텍스트로 재실행 → 새 답.
  4) 판정: 새 답이 원래 틀린 답과 *다름* → faithful_wrong(그 문서 의존)
           같음 → post_rat(문서 없어도 같은 답 = parametric/사후인용).

주의: 오픈모델(vLLM) 전용 권장(무료). 프론티어는 비용 발생 → 별도 허락 필요.
출력: results/loo_<model>.jsonl  (문항별 원답·제거문서·새답·판정)

usage: python 10_leave_one_out.py --model qwen3_8b [--side outdated]
"""
import argparse, json, os, random, re, sys
import pilot_common as pc

_G = {"the", "of", "and", "a", "an", "in", "at", "for", "is", "was", "to", "on", "as"}


def S(x):
    return " ".join(str(i) for i in x) if isinstance(x, list) else str(x or "")


def supports(passage, answer):
    a, b = S(answer).lower().strip(), S(passage).lower()
    if not a:
        return False
    if a in b:
        return True
    s = [w for w in re.findall(r"[a-z0-9]+", a) if len(w) > 3 and w not in _G]
    return bool(s) and sum(1 for w in s if w in b) * 2 >= len(s)


def pick_remove(rec, conf, ans):
    """뺄 문서 chunk_id 결정: 인용문서 중 답 지지 > conflict 청크 중 답 지지."""
    ct = {c["chunk_id"]: c["text"] for c in rec["chunks"]}
    cites = conf.get("cite_chunk_ids", [])
    # 1순위: 인용했고 그 답을 지지하는 문서
    for cid in cites:
        if cid in ct and supports(ct[cid], ans):
            return cid
    # 2순위: 인용은 했으나(지지판정 실패해도) 첫 인용 문서
    if cites and cites[0] in ct:
        return cites[0]
    # 3순위: 무인용 → conflict 청크 중 답 지지 문서
    for c in rec["chunks"]:
        if supports(c["text"], ans):
            return c["chunk_id"]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(pc.MODELS))
    ap.add_argument("--side", default="outdated", help="target_side 필터 (기본 as-of-past)")
    ap.add_argument("--eval", default=os.path.join(os.path.dirname(__file__), "..", "data", "eval_set.jsonl"))
    ap.add_argument("--raw_dir", default=os.path.join(os.path.dirname(__file__), "..", "results"))
    args = ap.parse_args()

    if pc.MODELS[args.model][0] != "vllm":
        print(f"⚠️  {args.model}은 vLLM(오픈) 아님 → 유료 호출 발생. 중단.")
        print("    오픈모델(qwen3_8b/qwen3_32b/mistral_small)만 허용.")
        return

    ev = {r["id"]: r for r in pc.read_jsonl(os.path.abspath(args.eval))}
    raw_path = os.path.join(os.path.abspath(args.raw_dir), f"raw_{args.model}.jsonl")
    orig = [json.loads(l) for l in open(raw_path) if l.strip()]
    out_path = os.path.join(os.path.abspath(args.raw_dir), f"loo_{args.model}.jsonl")

    # resume
    done = set()
    if os.path.exists(out_path):
        for l in open(out_path):
            try:
                done.add(json.loads(l)["id"])
            except Exception:
                pass
        if done:
            print(f"  재개: {len(done)}건 완료")

    rng = random.Random(0)
    todo = []
    for o in orig:
        rec = ev.get(o["id"])
        if not rec or rec.get("target_side") != args.side or o["id"] in done:
            continue
        conf = o["conflict"]
        ans = S(conf.get("answer", ""))
        if not ans.strip() or pc.ans_equiv(ans, S(rec["target_answer"])):
            continue        # EM=1 또는 빈답 제외 → 틀린 답만
        todo.append((rec, conf, ans))

    print(f"  대상(틀린 답): {len(todo)}건")
    n_fw = n_pr = n_skip = 0
    with open(out_path, "a", encoding="utf-8") as f:
        for i, (rec, conf, ans) in enumerate(todo, 1):
            rm = pick_remove(rec, conf, ans)
            if rm is None:
                n_skip += 1
                continue
            kept = [c for c in rec["chunks"] if c["chunk_id"] != rm]
            user, idx2 = pc.build_user_message(rec["new_question"], kept, rng)
            try:
                raw = pc.call_model(args.model, pc.SYSTEM_PROMPT, user)
                new_ans = pc.parse_output(raw)["answer"]
            except Exception as e:  # noqa
                json.dump({"id": rec["id"], "error": str(e)}, f); f.write("\n"); f.flush()
                continue
            changed = not pc.ans_equiv(new_ans, ans)
            verdict = "faithful_wrong" if changed else "post_rat"
            n_fw += changed; n_pr += (not changed)
            rec_out = {"id": rec["id"], "orig_wrong_answer": ans, "removed_chunk": rm,
                       "new_answer": new_ans, "answer_changed": changed, "verdict": verdict}
            f.write(json.dumps(rec_out, ensure_ascii=False) + "\n"); f.flush()
            print(f"  {i}/{len(todo)} {rec['id']}: {ans[:20]!r} → {new_ans[:20]!r} [{verdict}]", flush=True)

    tot = n_fw + n_pr
    print(f"\n완료 → {out_path}")
    print(f"  faithful_wrong(답 바뀜, 진짜 의존): {n_fw}/{tot} ({n_fw/tot:.0%})" if tot else "  대상 0")
    print(f"  post_rat(답 그대로, 사후인용):      {n_pr}/{tot} ({n_pr/tot:.0%})" if tot else "")
    print(f"  skip(뺄 문서 못 정함): {n_skip}")
    print(pc.cost_report())


if __name__ == "__main__":
    main()
