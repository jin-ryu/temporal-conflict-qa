"""
03 — 채점 및 집계 (§7, §8, §9).

지표:
  EM        = 1[conflict 답 == target_answer]
  TV_cite   = 1[evidence_chunk_id ∈ conflict 인용]                 (신규, 인용 기준)
  TV_behav  = 1[behav 가 목표 시점 쪽]                              (신규, 반사실 기준)
              behav: conflict 답이 outdated_only와 같고 current_only와 다르면 outdated, 그 반대면 current
  CitePrec  = 1[인용 청크 중 하나라도 conflict 답을 함의] (ALCE식, 기존 표준 foil)

산출:
  results/metrics_<model>.json   : 전체 + (current / as-of-past)별 율, 맹점비율
  results/contingency_<model>.csv: TV_cite × CitePrec 2×2 (§9 핵심 그림)

usage:
  python 03_evaluate.py --model gpt --judge gpt
"""
import argparse, csv, json, os, re
import pilot_common as pc

_G = {"the", "of", "and", "a", "an", "in", "at", "for", "is", "was", "to"}


def support_proxy(passage, answer):
    """무료 결정론적 CitePrec (ALCE의 NLI 판정 근사 — 답이 짧은 엔티티라 유효).
    답 정식명이 통째로 포함되거나, 변별토큰(>3자)의 과반이 청크에 있으면 '뒷받침'."""
    a, b = (answer or "").lower().strip(), (passage or "").lower()
    if not a:
        return 0
    if a in b:
        return 1
    spec = [w for w in re.findall(r"[a-z0-9]+", a) if len(w) > 3 and w not in _G]
    if not spec:
        return int(a in b)
    return int(sum(1 for w in spec if w in b) * 2 >= len(spec))


def behav_side(conflict_ans, cur_ans, old_ans):
    """conflict 답이 어느 단일문서 답을 따라갔나 (① 의미매칭 ans_equiv 사용)."""
    eq_old = pc.ans_equiv(conflict_ans, old_ans)
    eq_cur = pc.ans_equiv(conflict_ans, cur_ans)
    if eq_old and not eq_cur:
        return "outdated"
    if eq_cur and not eq_old:
        return "current"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--judge", default="proxy",
                    help="CitePrec 판정: 'proxy'(무료 결정론적·기본) 또는 모델키(gpt/claude…, 유료 LLM)")
    ap.add_argument("--eval", default=os.path.join(os.path.dirname(__file__), "..", "data", "eval_set.jsonl"))
    ap.add_argument("--raw_dir", default=os.path.join(os.path.dirname(__file__), "..", "results"))
    args = ap.parse_args()

    eval_by_id = {r["id"]: r for r in pc.read_jsonl(os.path.abspath(args.eval))}
    raw_path = os.path.join(os.path.abspath(args.raw_dir), f"raw_{args.model}.jsonl")

    per_item = []
    for o in pc.read_jsonl(raw_path):
        rec = eval_by_id[o["id"]]
        chunk_text = {c["chunk_id"]: c["text"] for c in rec["chunks"]}
        conf = o["conflict"]
        ans = conf.get("answer", "")
        cites = conf.get("cite_chunk_ids", [])

        em = int(pc.normalize(ans) == pc.normalize(rec["target_answer"]))
        tv_cite = int(rec["evidence_chunk_id"] in cites)
        # CitePrec: 인용 청크 중 하나라도 답을 함의 (proxy=무료 결정론, 그 외=유료 LLM judge)
        use_proxy = args.judge in ("proxy", "none")
        cite_prec = 0
        for cid in cites:
            if cid not in chunk_text:
                continue
            ok = support_proxy(chunk_text[cid], ans) if use_proxy else pc.judge_support(args.judge, chunk_text[cid], ans)
            if ok:
                cite_prec = 1
                break
        # 반사실 (행동 기준)
        cur_ans = o["current_only"].get("answer", "")
        old_ans = o["outdated_only"].get("answer", "")
        side = behav_side(ans, cur_ans, old_ans)
        # ② 유효성: 단일문서 답이 서로 갈려야(=문서가 답을 가른다) TV_behav가 의미있음.
        #    옛=새 단일문서 답이 동등하면(두 문서가 같은 답 지지/모델이 불변) 반사실 판정 불가 → 제외.
        behav_valid = bool(ans.strip()) and not pc.ans_equiv(cur_ans, old_ans)

        per_item.append({"id": o["id"], "target_side": o["target_side"], "EM": em,
                         "TV_cite": tv_cite, "CitePrec": cite_prec, "behav": side,
                         "behav_valid": behav_valid, "cites": cites, "answer": ans})

    def agg(items):
        n = len(items) or 1
        m = lambda k: round(sum(it[k] for it in items) / n, 4)
        tv0 = [it for it in items if it["TV_cite"] == 0]
        blind = round(sum(it["CitePrec"] for it in tv0) / (len(tv0) or 1), 4)
        # TV_behav: 반사실이 유효한 문항(단일문서 답이 갈림)에서만 집계
        bv = [it for it in items if it["behav_valid"]]
        tv_behav = round(sum(1 for it in bv if it["behav"] == it["target_side"]) / (len(bv) or 1), 4)
        return {"n": len(items), "EM": m("EM"), "TV_cite": m("TV_cite"),
                "TV_behav": tv_behav, "TV_behav_n_valid": len(bv),
                "CitePrec": m("CitePrec"),
                "wrong_time_cite_rate": round(1 - m("TV_cite"), 4),
                "blind_spot_rate_P(CitePrec=1|TV_cite=0)": blind}

    metrics = {"model": args.model, "overall": agg(per_item),
               "as_of_past": agg([it for it in per_item if it["target_side"] == "outdated"]),
               "current": agg([it for it in per_item if it["target_side"] == "current"])}

    # 2×2 분할표 (TV_cite × CitePrec)
    cells = {(tv, cp): 0 for tv in (1, 0) for cp in (1, 0)}
    for it in per_item:
        cells[(it["TV_cite"], it["CitePrec"])] += 1

    out_dir = os.path.abspath(args.raw_dir)
    with open(os.path.join(out_dir, f"metrics_{args.model}.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, f"contingency_{args.model}.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["", "CitePrec=PASS", "CitePrec=FAIL"])
        w.writerow(["TV=PASS(시점-유효 인용)", cells[(1, 1)], cells[(1, 0)]])
        w.writerow(["TV=FAIL(틀린 시점 인용)", cells[(0, 1)], cells[(0, 0)]])

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"\n★ 맹점 셀 (TV=FAIL & CitePrec=PASS) = {cells[(0,1)]}건  ← C2·C3 핵심")
    print(f"산출 → metrics_{args.model}.json, contingency_{args.model}.csv")
    if args.judge in ("proxy", "none"):
        print("\n[judge=proxy] 무료 결정론적 채점 — API 비용 0")
    else:
        print("\n[judge 사용량/비용]")
        print(pc.cost_report())
        pc.append_ledger("03_evaluate")
        print("\n[전체 누적 — repo 루트 usage/usage_summary.txt]")
        print(pc.ledger_total())


if __name__ == "__main__":
    main()
