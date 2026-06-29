"""
exp05 채점 — 유형별 mis-attribution (EM × CitePrec).

mis-attribution = P(CitePrec=1 | EM=0) = 답 틀렸는데 표준 인용평가는 통과시킨 비율.
"시간 외 충돌(오정보·의미)에서도 mis-attribution 생기나" 를 유형별로 확인.

usage: python eval_misattr.py --model qwen3_8b
"""
import json, os, re, argparse, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "03_temporal_validity", "scripts"))
import pilot_common as pc
_G = {"the", "of", "and", "a", "an", "in", "at", "for", "is", "was", "to", "on", "as"}


def S(x):
    return " ".join(str(i) for i in x) if isinstance(x, list) else str(x or "")


def sup(p, a):
    a, b = S(a).lower().strip(), S(p).lower()
    if not a:
        return 0
    if a in b:
        return 1
    s = [w for w in re.findall(r"[a-z0-9]+", a) if len(w) > 3 and w not in _G]
    return int(bool(s) and sum(1 for w in s if w in b) * 2 >= len(s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    a = ap.parse_args()
    here = os.path.dirname(__file__)
    ev = {r["id"]: r for r in pc.read_jsonl(os.path.join(here, "..", "data", "eval_set.jsonl"))}
    rows = [json.loads(l) for l in open(os.path.join(here, "..", "results", f"raw_{a.model}.jsonl"))]

    by_type = {}
    for o in rows:
        rec = ev[o["id"]]
        t = rec.get("conflict_type", "?")
        conf = o["conflict"]
        cites = conf.get("cite_chunk_ids", [])
        ans = S(conf.get("answer", ""))
        if not ans.strip():
            continue
        ct = {c["chunk_id"]: c["text"] for c in rec["chunks"]}
        em = int(pc.ans_equiv(ans, S(rec["target_answer"])))
        cp = int(any(sup(ct.get(cid, ""), ans) for cid in cites))
        d = by_type.setdefault(t, {(e, c): 0 for e in (1, 0) for c in (1, 0)})
        d[(em, cp)] += 1

    print(f"\n=== exp05 mis-attribution (EM×CitePrec) — {a.model} ===")
    print(f"{'유형':16}{'n':>4}{'EM정답률':>9}{'CitePrec':>9}{'★(EM0CP1)':>11}{'mis-attr':>10}")
    print("-" * 60)
    allres = {}
    for t, c in by_type.items():
        n = sum(c.values())
        em = (c[(1, 1)] + c[(1, 0)]) / n
        cp = (c[(1, 1)] + c[(0, 1)]) / n
        em0 = c[(0, 1)] + c[(0, 0)]
        ma = c[(0, 1)] / em0 if em0 else 0
        allres[t] = {"n": n, "EM": round(em, 3), "CitePrec": round(cp, 3),
                     "star": c[(0, 1)], "mis_attr": round(ma, 3)}
        print(f"{t:16}{n:>4}{em:>8.0%}{cp:>9.0%}{c[(0,1)]:>9}{ma:>10.0%}")
    json.dump(allres, open(os.path.join(here, "..", "results", f"misattr_{a.model}.json"), "w"),
              ensure_ascii=False, indent=2)
    print("\nmis-attr 높으면 = 그 충돌유형에서도 '틀린답에 좋은인용' 발생 (시간 외 일반화).")


if __name__ == "__main__":
    main()
