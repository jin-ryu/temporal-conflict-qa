"""
04 — 위치 ablation (리랭커 vs 셀렉터 결정용).

질문: *충돌 컨텍스트에서 청크 *순서*가 모델의 시점 선택을 바꾸나?*
같은 as-of-past 항목을, 시점-유효(evidence) 청크를 **맨 앞 / 맨 뒤**에 강제 배치해 비교.
  · promoted(맨앞) ≫ demoted(맨뒤)  → 위치가 핵심 = *리랭커가 통함*.
  · 차이 없음                         → 모델이 순서 무시 = 리랭커 말고 *셀렉터(틀린시점 제거)* 필요.
random 기준선은 기존 raw_<model>.jsonl(conflict, 셔플)에서 가져옴.

지표: TV_cite(시점-유효 인용율), Acc(시점-유효 답 정확도, ans_equiv).

usage:
  python 04_position_ablation.py --model qwen3_32b     # (open이면 VLLM_BASE_URL 환경변수)
출력: results/ablation_<model>.json
"""
import argparse, json, os, random
import pilot_common as pc


def run_order(model, recs, order, rng):
    """order: 'first'|'last' — evidence 청크를 그 위치에 두고 conflict 조건 실행."""
    tv = acc = n = 0
    for rec in recs:
        chunks = pc.filter_chunks(rec["chunks"], "conflict")
        ev_id = rec["evidence_chunk_id"]
        ev = [c for c in chunks if c["chunk_id"] == ev_id]
        rest = [c for c in chunks if c["chunk_id"] != ev_id]
        rng.shuffle(rest)                       # 나머지는 섞되 evidence 위치만 고정
        ordered = ev + rest if order == "first" else rest + ev
        user, idx2chunk = pc.build_user_message(rec["new_question"], ordered, rng, shuffle=False)
        try:
            parsed = pc.parse_output(pc.call_model(model, pc.SYSTEM_PROMPT, user))
            cites = [idx2chunk[i] for i in parsed["cite_indices"] if i in idx2chunk]
            tv += int(ev_id in cites)
            acc += int(pc.ans_equiv(parsed["answer"], rec["target_answer"]))
        except Exception as e:
            print(f"  err {rec['id']}: {str(e)[:60]}")
        n += 1
    return {"order": order, "n": n, "TV_cite": round(tv / (n or 1), 4), "Acc": round(acc / (n or 1), 4)}


def baseline_random(model, recs):
    """기존 conflict(셔플) 결과에서 random 기준선 계산."""
    raw = {r["id"]: r for r in pc.read_jsonl(
        os.path.join(os.path.dirname(__file__), "..", "results", f"raw_{model}.jsonl"))}
    tv = acc = n = 0
    for rec in recs:
        o = raw.get(rec["id"])
        if not o:
            continue
        conf = o["conflict"]
        tv += int(rec["evidence_chunk_id"] in conf.get("cite_chunk_ids", []))
        acc += int(pc.ans_equiv(conf.get("answer", ""), rec["target_answer"]))
        n += 1
    return {"order": "random(기존)", "n": n, "TV_cite": round(tv / (n or 1), 4), "Acc": round(acc / (n or 1), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(pc.MODELS))
    ap.add_argument("--eval", default=os.path.join(os.path.dirname(__file__), "..", "data", "eval_set.jsonl"))
    args = ap.parse_args()
    rng = random.Random(0)
    recs = [r for r in pc.read_jsonl(os.path.abspath(args.eval)) if r["target_side"] == "outdated"]
    print(f"as-of-past {len(recs)}문항 × 위치 2조건\n")

    rows = [baseline_random(args.model, recs),
            run_order(args.model, recs, "first", rng),
            run_order(args.model, recs, "last", rng)]
    print(f"\n{'배치':14}{'n':>4}{'TV_cite':>10}{'Acc':>9}")
    print("-" * 38)
    for r in rows:
        print(f"{r['order']:14}{r['n']:>4}{r['TV_cite']:>10.3f}{r['Acc']:>9.3f}")
    first = next(r for r in rows if r["order"] == "first")
    last = next(r for r in rows if r["order"] == "last")
    print(f"\nΔ(first−last) TV_cite = {first['TV_cite']-last['TV_cite']:+.3f}, Acc = {first['Acc']-last['Acc']:+.3f}")
    print("→ 크면 위치 영향=리랭커 통함 / ~0이면 순서 무시=셀렉터 필요")
    out = os.path.join(os.path.abspath(os.path.dirname(args.eval)), "..", "results", f"ablation_{args.model}.json")
    json.dump(rows, open(os.path.abspath(out), "w"), ensure_ascii=False, indent=2)
    print(f"\n→ {os.path.abspath(out)}")
    print(pc.cost_report())


if __name__ == "__main__":
    main()
