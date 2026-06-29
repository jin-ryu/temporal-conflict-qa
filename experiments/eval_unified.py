"""
3실험 통일 채점 — mis-attribution + faithfulness (exp03/04/05 공용).

지표 (셋 다 동일):
  EM        = 1[모델 답 ≈ 정답]                            (의미매칭 ans_equiv)
  CitePrec  = 1[인용 청크 중 하나라도 모델 답을 담음]       (표준, 시점무관, 내용기준 sup)
  ★/mis-attr= P(CitePrec=1 | EM=0)  = 틀린 답인데 인용평가 통과 (헤드라인)
  faithfulness 분류 (틀린 답·틀린시점 인용 중, 문서제거 counterfactual):
    faithful_wrong = conflict답이 outdated_only와 같고 current_only와 다름 (옛 문서 진짜 사용)
    post_rat       = 어느 단일조건과도 불일치 (사후 인용)
    invariant      = 두 단일조건 답 동일 (반사실 불가)

usage:
  python eval_unified.py --eval <eval_set.jsonl> --raw <raw_model.jsonl> [--ap_only]
  (ap_only: as-of-past 항목만 — exp03 주력. exp04/05는 생략)
"""
import json, os, re, argparse, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "03_temporal_validity", "scripts"))
import pilot_common as pc
_G = {"the", "of", "and", "a", "an", "in", "at", "for", "is", "was", "to", "on", "as"}


def S(x):
    return " ".join(str(i) for i in x) if isinstance(x, list) else str(x or "")


def sup(passage, answer):
    """인용 청크가 답을 담았나 — 정식 포함 OR 변별토큰(4자+) 과반."""
    a, b = S(answer).lower().strip(), S(passage).lower()
    if not a:
        return 0
    if a in b:
        return 1
    s = [w for w in re.findall(r"[a-z0-9]+", a) if len(w) > 3 and w not in _G]
    return int(bool(s) and sum(1 for w in s if w in b) * 2 >= len(s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--ap_only", action="store_true", help="as-of-past(target_side=outdated)만")
    args = ap.parse_args()

    ev = {r["id"]: r for r in pc.read_jsonl(args.eval)}
    rows = [json.loads(l) for l in open(args.raw)]
    n = em_cnt = cp_cnt = 0
    cells = {(e, c): 0 for e in (1, 0) for c in (1, 0)}     # EM × CitePrec
    fbuckets = {}
    for o in rows:
        rec = ev.get(o["id"])
        if not rec:
            continue
        if args.ap_only and rec.get("target_side") != "outdated":
            continue
        conf = o["conflict"]
        cites = conf.get("cite_chunk_ids", [])
        ans = S(conf.get("answer", ""))
        if not ans.strip():
            continue
        n += 1
        ct = {c["chunk_id"]: c["text"] for c in rec["chunks"]}
        em = int(pc.ans_equiv(ans, S(rec["target_answer"])))
        cp = int(any(sup(ct.get(cid, ""), ans) for cid in cites))
        em_cnt += em
        cp_cnt += cp
        cells[(em, cp)] += 1
        # faithfulness: 틀린 답(EM=0)에 대해 문서제거 counterfactual.
        # 방향 통일: 틀린 답이 *어느 한쪽 단일조건*과 일치하면 그 문서에 *행동 의존* = faithful(진짜 사용).
        # (exp03=과거질문→틀린답은 최신측, exp04/05=현재질문→틀린답은 옛측. 어느 쪽이든 "단일조건 의존"이면 faithful.)
        if em == 0:
            cur = S(o.get("current_only", {}).get("answer", ""))
            old = S(o.get("outdated_only", {}).get("answer", ""))
            eq_cur, eq_old = pc.ans_equiv(ans, cur), pc.ans_equiv(ans, old)
            if (eq_cur and not eq_old) or (eq_old and not eq_cur):
                b = "faithful_wrong"     # 한쪽 단일문서에 행동 의존 = 진짜 그 문서 보고 틀림
            elif eq_cur and eq_old:
                b = "invariant"          # 두 단일조건 답 동일 → 반사실 불가
            else:
                b = "post_rat"           # 어느 단일조건과도 불일치 = 사후인용/parametric
            fbuckets[b] = fbuckets.get(b, 0) + 1

    em0 = cells[(0, 1)] + cells[(0, 0)]
    misattr = cells[(0, 1)] / em0 if em0 else 0
    res = {
        "n": n, "EM": round(em_cnt / n, 3) if n else 0, "CitePrec": round(cp_cnt / n, 3) if n else 0,
        "star_cell_EMxCP": cells[(0, 1)],
        "mis_attribution_P(CitePrec=1|EM=0)": round(misattr, 3),
        "2x2_EMxCitePrec": {"EM1_CP1": cells[(1, 1)], "EM1_CP0": cells[(1, 0)],
                            "EM0_CP1": cells[(0, 1)], "EM0_CP0": cells[(0, 0)]},
        "faithfulness_of_wrong": fbuckets,
    }
    fw = fbuckets.get("faithful_wrong", 0)
    fwtot = sum(fbuckets.values()) or 1
    res["faithful_wrong_rate"] = round(fw / fwtot, 3)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return res


if __name__ == "__main__":
    main()
