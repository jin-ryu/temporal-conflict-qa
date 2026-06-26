"""
05 — 틀린시점 인용 실패의 faithfulness 분류 (Wallat 2025 대비 차별 입증).

대상: as-of-past 중 conflict에서 *틀린시점* 인용한 항목 (TV_cite=0).
질문: 모델이 인용한 (틀린시점=current측) 문서를 *실제로 사용*했나, 아니면 사후 갖다붙였나?
방법(black-box counterfactual, 추가 호출 0 — 기존 3조건 raw 재사용):
  · conflict 답 = current_only 답 == outdated_only 답?  (단일조건 답 = 문서 제거 효과)
  - conflict 답이 current_only(새문서만)와 같고 outdated_only(옛문서만)와 다르면
      → 새 문서에 *행동 의존* = **faithful-but-wrong-time** (우리 신규 범주: 진짜 썼는데 시점 틀림)
  - conflict 답이 두 단일조건과 *무관*(특히 옛문서만 줘도 같은 답)
      → 새 문서 없이도 그 답 = **post-rationalization 의심** (Wallat 범주: parametric/사후인용)
  - 그 외(parametric 등) → other

→ "우리 실패의 다수가 faithful-but-wrong-time" 이면 Wallat의 post-rat과 *구별되는* 범주임을 입증.

usage: python 05_faithfulness_split.py            (전체 모델)
출력: results/faithfulness_split.json + 콘솔 표
"""
import json, os, glob
import pilot_common as pc

HERE = os.path.dirname(__file__)
EVAL = os.path.join(HERE, "..", "data", "eval_set.jsonl")
RES = os.path.join(HERE, "..", "results")


def classify(conf_ans, cur_ans, old_ans):
    """문서 제거 counterfactual로 실패 유형 분류 (ans_equiv 의미매칭)."""
    eq_cur = pc.ans_equiv(conf_ans, cur_ans)   # 새문서만 답과 같나
    eq_old = pc.ans_equiv(conf_ans, old_ans)   # 옛문서만 답과 같나
    if eq_cur and not eq_old:
        return "faithful_wrong_time"   # 새 문서에 행동 의존 → 진짜 썼는데 시점 틀림 (우리 신규)
    if eq_old and not eq_cur:
        return "faithful_correct_time"  # 옛(유효) 문서 의존 (TV_cite=0이지만 답은 과거 따라감 = 인용/행동 불일치)
    if eq_cur and eq_old:
        return "invariant"              # 두 단일조건 답 동일 → 문서 안 가림(반사실 불가)
    return "other_parametric"           # 어느 단일조건과도 불일치 → 새 답이 문서서 안 나옴 = post-rat/parametric 의심


def analyze(model):
    eval_by = {r["id"]: r for r in pc.read_jsonl(EVAL)}
    rows = [json.loads(l) for l in open(os.path.join(RES, f"raw_{model}.jsonl"))]
    out = {"total_ap": 0, "wrong_time": 0, "buckets": {}}
    for o in rows:
        rec = eval_by.get(o["id"])
        if not rec or rec["target_side"] != "outdated":
            continue
        out["total_ap"] += 1
        conf = o["conflict"]
        cites = conf.get("cite_chunk_ids", [])
        if rec["evidence_chunk_id"] in cites:   # 맞는시점 인용 → 대상 아님
            continue
        out["wrong_time"] += 1
        b = classify(conf.get("answer", ""),
                     o["current_only"].get("answer", ""),
                     o["outdated_only"].get("answer", ""))
        out["buckets"][b] = out["buckets"].get(b, 0) + 1
    return out


def main():
    models = [os.path.basename(p)[4:-6] for p in sorted(glob.glob(os.path.join(RES, "raw_*.jsonl")))
              if "_tqa" not in p]
    allres = {}
    order = ["faithful_wrong_time", "other_parametric", "invariant", "faithful_correct_time"]
    label = {"faithful_wrong_time": "faithful-wrong-time(우리)", "other_parametric": "post-rat의심(Wallat)",
             "invariant": "불변(반사실불가)", "faithful_correct_time": "옛문서의존"}
    print(f"\n{'model':16}{'wrong_time':>11}", end="")
    for k in order:
        print(f"{label[k]:>22}", end="")
    print()
    print("-" * 95)
    for m in models:
        r = analyze(m)
        allres[m] = r
        wt = r["wrong_time"] or 1
        print(f"{m:16}{r['wrong_time']:>11}", end="")
        for k in order:
            c = r["buckets"].get(k, 0)
            print(f"{c:>8} ({c/wt:>4.0%})      ", end="")
        print()
    json.dump(allres, open(os.path.join(RES, "faithfulness_split.json"), "w"), ensure_ascii=False, indent=2)
    print(f"\n→ {os.path.join(RES, 'faithfulness_split.json')}")
    print("\n해석: faithful-wrong-time 비율이 높으면 = Wallat의 post-rationalization과 *구별되는* 실패범주.")


if __name__ == "__main__":
    main()
