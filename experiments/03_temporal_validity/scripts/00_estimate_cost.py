"""
00 — 예상 비용 (dry-run, 실제 API 호출 없음).

eval_set의 *실제 프롬프트*로 입력 토큰을 정확히 세고(출력은 가정치), 모델별 예상 비용을 출력.
입력 토큰은 정확, 출력 토큰만 추정이므로 비용의 변동은 주로 출력 길이에서 발생.

가격은 환경변수로 주입(USD per 1M tokens):
  GPT_IN_PRICE/GPT_OUT_PRICE, CLAUDE_IN_PRICE/CLAUDE_OUT_PRICE   ← 최신 가격으로 설정
가격 미설정 시 $는 0으로 나오니, 토큰 수를 보고 직접 계산해도 됨.

usage:
  python 00_estimate_cost.py --models gpt,claude --out_tokens 150 --avg_cited 1.5 --judge gpt
"""
import argparse, os, random
import pilot_common as pc

JUDGE_INSTRUCTION_TOKENS = 40  # judge 프롬프트 고정부 근사
JUDGE_OUT_TOKENS = 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default=os.path.join(os.path.dirname(__file__), "..", "data", "eval_set.jsonl"))
    ap.add_argument("--models", default="gpt,claude", help="쉼표구분 (pilot_common.MODELS 키)")
    ap.add_argument("--out_tokens", type=int, default=150, help="메인콜 출력 토큰 가정치")
    ap.add_argument("--avg_cited", type=float, default=1.5, help="문항당 CitePrec judge 호출 수 가정")
    ap.add_argument("--judge", default="gpt")
    ap.add_argument("--max_tokens", type=int, default=512, help="worst-case 출력 상한(call_model 기본)")
    args = ap.parse_args()

    recs = list(pc.read_jsonl(os.path.abspath(args.eval)))
    rng = random.Random(0)
    sys_tok = pc.count_tokens(pc.SYSTEM_PROMPT)

    # 메인콜: 문항 × 3조건 입력 토큰(정확)
    main_in = 0
    n_main = 0
    for r in recs:
        for cond in ("conflict", "current_only", "outdated_only"):
            chunks = pc.filter_chunks(r["chunks"], cond)
            user, _ = pc.build_user_message(r["new_question"], chunks, rng)
            main_in += sys_tok + pc.count_tokens(user)
            n_main += 1

    # judge(CitePrec): conflict 청크 중 인용분 — 평균 청크 토큰으로 추정
    all_chunk_tok = [pc.count_tokens(c["text"]) for r in recs for c in r["chunks"]]
    avg_chunk_tok = sum(all_chunk_tok) / max(1, len(all_chunk_tok))
    n_judge = int(round(len(recs) * args.avg_cited))
    judge_in = int(n_judge * (avg_chunk_tok + JUDGE_INSTRUCTION_TOKENS))

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    print(f"[예상 비용 — dry run]  문항 {len(recs)} | 메인콜 {n_main} | judge ~{n_judge} "
          f"(avg_cited={args.avg_cited}, avg_chunk≈{avg_chunk_tok:.0f}tok)\n")
    print(f"{'model':<10}{'in_tok':>10}{'out~tok':>10}{'$typical':>11}{'$worst':>10}")
    grand = 0.0
    for m in models:
        out_typ = n_main * args.out_tokens
        out_wst = n_main * args.max_tokens
        c_typ = pc.cost_usd(m, main_in, out_typ)
        c_wst = pc.cost_usd(m, main_in, out_wst)
        grand += c_typ
        warn = "  ⚠가격미설정" if pc.PRICING.get(m, (0, 0)) == (0, 0) and not m.startswith("llama") else ""
        print(f"{m:<10}{main_in:>10}{out_typ:>10}{('$%.3f'%c_typ):>11}{('$%.3f'%c_wst):>10}{warn}")

    # judge 비용(별도)
    jc = pc.cost_usd(args.judge, judge_in, n_judge * JUDGE_OUT_TOKENS)
    grand += jc
    print(f"{'judge('+args.judge+')':<10}{judge_in:>10}{n_judge*JUDGE_OUT_TOKENS:>10}{('$%.3f'%jc):>11}")
    print(f"\n합계(typical, 위 model 들 + judge) ≈ ${grand:.3f}")
    print("※ 입력 토큰은 정확, 출력은 가정치. 가격은 환경변수(GPT_IN_PRICE 등)로 최신값 주입.")


if __name__ == "__main__":
    main()
