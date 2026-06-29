"""
exp04 faithfulness 분류 (exp03의 05번을 현실 웹 충돌에 적용).

대상: 틀린시점 인용(TV_cite=0 = 인용 청크가 정답[최신값]을 안 담음) 항목.
질문: 모델이 *outdated 문서*를 인용해 틀릴 때 — 그 옛 문서를 *진짜 사용*했나(faithful-wrong-time),
      아니면 사후 갖다붙였나(post-rationalization)?
방법(black-box counterfactual, 추가 호출 0 — 기존 3조건 raw 재사용):
  conflict 답 vs current_only(최신문서만) 답 vs outdated_only(옛문서만 — distractor 제거된 셋) 답.
  · conflict 답이 outdated_only와 같고 current_only와 다름 → 옛 문서에 *행동 의존* = faithful_wrong_time
  · conflict 답이 current_only와 같고 outdated_only와 다름 → 최신문서 의존(맞는 방향이나 TV_cite=0이면 인용↔행동 불일치)
  · 두 단일조건 답 동일 → invariant(반사실 불가)
  · 어느 것과도 불일치 → other/parametric

usage: python faithfulness_split_rc.py            (전체 모델)
"""
import json, os, glob, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "03_temporal_validity", "scripts"))
import pilot_common as pc
import re

HERE = os.path.dirname(__file__)
EVAL = os.path.join(HERE, "..", "data", "eval_set.jsonl")
RES = os.path.join(HERE, "..", "results")
_G = {"the", "of", "and", "a", "an", "in", "at", "for", "is", "was", "to", "on", "as"}


def S(x):
    return " ".join(str(i) for i in x) if isinstance(x, list) else str(x or "")


def contains(passage, answer):
    a, b = S(answer).lower().strip(), S(passage).lower()
    if not a:
        return False
    if a in b:
        return True
    s = [w for w in re.findall(r"[a-z0-9]+", a) if len(w) > 3 and w not in _G]
    return bool(s) and sum(1 for w in s if w in b) * 2 >= len(s)


def classify(conf_ans, cur_ans, old_ans):
    eq_cur = pc.ans_equiv(conf_ans, cur_ans)
    eq_old = pc.ans_equiv(conf_ans, old_ans)
    if eq_old and not eq_cur:
        return "faithful_wrong_time"     # 옛(outdated) 문서에 행동 의존 = 진짜 사용했는데 시점 틀림
    if eq_cur and not eq_old:
        return "faithful_correct_time"   # 최신 의존(인용은 틀시점인데 답은 최신 따라감 = 불일치)
    if eq_cur and eq_old:
        return "invariant"               # 두 단일조건 동일 → 반사실 불가
    return "other_parametric"            # 어느 것도 아님 = post-rat/parametric 의심


def analyze(model):
    eval_by = {r["id"]: r for r in pc.read_jsonl(EVAL)}
    rows = [json.loads(l) for l in open(os.path.join(RES, f"raw_{model}.jsonl"))]
    out = {"wrong_time": 0, "buckets": {}}
    for o in rows:
        rec = eval_by.get(o["id"])
        if not rec:
            continue
        conf = o["conflict"]
        cites = conf.get("cite_chunk_ids", [])
        ans = S(conf.get("answer", ""))
        if not ans.strip():
            continue
        gold = S(rec["target_answer"])
        ctext = {c["chunk_id"]: c["text"] for c in rec["chunks"]}
        # TV_cite=0 = 인용 청크가 정답(최신값) 안 담음 (= 틀린시점 인용)
        tv_cite = int(any(contains(ctext.get(cid, ""), gold) for cid in cites))
        if tv_cite == 1:
            continue
        out["wrong_time"] += 1
        b = classify(ans, S(o["current_only"].get("answer", "")), S(o["outdated_only"].get("answer", "")))
        out["buckets"][b] = out["buckets"].get(b, 0) + 1
    return out


def main():
    models = [os.path.basename(p)[4:-6] for p in sorted(glob.glob(os.path.join(RES, "raw_*.jsonl")))]
    order = ["faithful_wrong_time", "other_parametric", "invariant", "faithful_correct_time"]
    label = {"faithful_wrong_time": "faithful-wrong-time", "other_parametric": "post-rat의심",
             "invariant": "불변", "faithful_correct_time": "최신의존(불일치)"}
    allres = {}
    print(f"\n{'model':16}{'wrong_t':>8}", end="")
    for k in order:
        print(f"{label[k]:>22}", end="")
    print()
    print("-" * 96)
    for m in models:
        r = analyze(m)
        allres[m] = r
        wt = r["wrong_time"] or 1
        print(f"{m:16}{r['wrong_time']:>8}", end="")
        for k in order:
            c = r["buckets"].get(k, 0)
            print(f"{c:>8} ({c/wt:>4.0%})      ", end="")
        print()
    json.dump(allres, open(os.path.join(RES, "faithfulness_split.json"), "w"), ensure_ascii=False, indent=2)
    print(f"\n→ {os.path.join(RES, 'faithfulness_split.json')}")
    print("해석: faithful-wrong-time 높으면 = 현실서도 모델이 옛 문서를 *진짜 사용*해 틀림 (Wallat post-rat과 구별).")


if __name__ == "__main__":
    main()
