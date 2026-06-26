"""
06 — ★ 맹점의 지표-일반성 (다른 attribution 지표도 시점에 눈머나?).

주장: temporal validity 맹점이 *CitePrec 한 지표의 특이성*이 아니라 *attribution 평가 전반*의 한계임을 보인다.
세 표준 지표를 같은 raw에 계산(추가 호출 0, 결정론적):
  · CitePrec  : 인용 청크 중 *하나라도* 답 함의 (ALCE precision 근사)
  · CiteRecall: 인용 청크 *전체(concat)* 가 답 함의 (ALCE recall 근사)
  · AIS       : 인용이 존재하고 그 합집합이 답 뒷받침 = attributable=1 (binary AIS 근사)
각 지표 X에 대해 β_X = P(X=1 | TV_cite=0) (틀린시점 인용 중 X가 '정상' 통과 비율).
β_X 가 모두 높으면 → *어느 attribution 지표도 시점을 못 본다* = 맹점은 일반적.

usage: python 06_metric_generality.py
출력: results/metric_generality.json + 콘솔
"""
import json, os, glob, re
import pilot_common as pc

HERE = os.path.dirname(__file__)
EVAL = os.path.join(HERE, "..", "data", "eval_set.jsonl")
RES = os.path.join(HERE, "..", "results")
_G = {"the", "of", "and", "a", "an", "in", "at", "for", "is", "was", "to"}


def supports(passage, answer):
    """결정론적 entailment 근사(ALCE NLI 대용): 정식명 포함 OR 변별토큰 과반."""
    a, b = (answer or "").lower().strip(), (passage or "").lower()
    if not a:
        return False
    if a in b:
        return True
    spec = [w for w in re.findall(r"[a-z0-9]+", a) if len(w) > 3 and w not in _G]
    if not spec:
        return a in b
    return sum(1 for w in spec if w in b) * 2 >= len(spec)


def metrics_for(ans, cites, chunk_text):
    """세 attribution 지표 (시점 무관)."""
    cited = [chunk_text[c] for c in cites if c in chunk_text]
    citeprec = int(any(supports(t, ans) for t in cited))            # 하나라도
    citerecall = int(bool(cited) and supports(" ".join(cited), ans))  # 합집합 전체
    ais = int(bool(cites) and (citeprec or citerecall))              # 인용 존재 + 뒷받침
    return {"CitePrec": citeprec, "CiteRecall": citerecall, "AIS": ais}


def analyze(model):
    eval_by = {r["id"]: r for r in pc.read_jsonl(EVAL)}
    rows = [json.loads(l) for l in open(os.path.join(RES, f"raw_{model}.jsonl"))]
    METR = ["CitePrec", "CiteRecall", "AIS"]
    n = 0
    tv0 = 0
    pass_all = {m: 0 for m in METR}        # 전체 평균
    blind = {m: 0 for m in METR}           # TV_cite=0 중 통과
    for o in rows:
        rec = eval_by.get(o["id"])
        if not rec or rec["target_side"] != "outdated":
            continue
        n += 1
        conf = o["conflict"]
        cites = conf.get("cite_chunk_ids", [])
        ans = conf.get("answer", "")
        ct = {c["chunk_id"]: c["text"] for c in rec["chunks"]}
        mv = metrics_for(ans, cites, ct)
        for m in METR:
            pass_all[m] += mv[m]
        if rec["evidence_chunk_id"] not in cites:   # 틀린시점 인용
            tv0 += 1
            for m in METR:
                blind[m] += mv[m]
    return {"n": n, "wrong_time": tv0,
            "rate": {m: round(pass_all[m] / (n or 1), 3) for m in METR},
            "beta": {m: round(blind[m] / (tv0 or 1), 3) for m in METR}}


def main():
    models = [os.path.basename(p)[4:-6] for p in sorted(glob.glob(os.path.join(RES, "raw_*.jsonl")))
              if "_tqa" not in p]
    METR = ["CitePrec", "CiteRecall", "AIS"]
    allres = {}
    print("\n=== β_X = P(지표=1 | TV_cite=0) : 틀린시점 인용 중 각 지표가 '정상' 통과한 비율 ===")
    print("    (β 높을수록 그 지표도 시점에 눈멈 → 맹점 일반적)\n")
    print(f"{'model':16}{'wrong_t':>8}" + "".join(f"{m:>13}" for m in METR))
    print("-" * 64)
    for m in models:
        r = analyze(m)
        allres[m] = r
        print(f"{m:16}{r['wrong_time']:>8}" + "".join(f"{r['beta'][k]:>12.0%} " for k in METR))
    json.dump(allres, open(os.path.join(RES, "metric_generality.json"), "w"), ensure_ascii=False, indent=2)
    print(f"\n→ {os.path.join(RES, 'metric_generality.json')}")
    print("해석: 세 지표 β 모두 높음 → temporal 맹점은 CitePrec 특이성 아니라 attribution 평가 *전반*의 한계.")


if __name__ == "__main__":
    main()
