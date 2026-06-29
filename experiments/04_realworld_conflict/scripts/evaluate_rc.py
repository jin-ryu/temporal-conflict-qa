"""
04 채점 — "인용한 문서가 정답(최신값)을 담았나" 기준 (exp03와 개념 동일, 단일인용에 맞춤).

exp04는 정답(최신값)이 여러 문서에 담길 수 있고 모델이 인용을 1개만 뽑으므로,
evidence_chunk_id(단일 지정) 대신 *내용 기준*으로 시점유효성을 판정:
  · TV_cite = 1[인용한 청크가 정답(target_answer)을 담음]   ← 맞는 시점(최신) 문서 인용
  · wrong_time = 인용했는데 정답 안 담음(= 옛/다른 문서)
  · CitePrec = 1[인용 청크가 모델 *답*을 뒷받침]            ← 표준(시점무관)
  · ★ = TV_cite=0 & CitePrec=1  (틀린시점 인용인데 표준은 통과)
  · β = P(CitePrec=1 | TV_cite=0)

usage: python evaluate_rc.py --model mistral_small
"""
import json, os, re, argparse, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "03_temporal_validity", "scripts"))
import pilot_common as pc
_G = {"the","of","and","a","an","in","at","for","is","was","to","on","as"}


def S(x):
    return " ".join(str(i) for i in x) if isinstance(x, list) else str(x or "")


def contains(passage, answer):
    """문서가 답을 담았나 (정식 포함 OR 변별토큰 과반)."""
    a, b = S(answer).lower().strip(), S(passage).lower()
    if not a:
        return False
    if a in b:
        return True
    s = [w for w in re.findall(r"[a-z0-9]+", a) if len(w) > 3 and w not in _G]
    return bool(s) and sum(1 for w in s if w in b) * 2 >= len(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    a = ap.parse_args()
    here = os.path.dirname(__file__)
    ev = {r["id"]: r for r in pc.read_jsonl(os.path.join(here, "..", "data", "eval_set.jsonl"))}
    rows = [json.loads(l) for l in open(os.path.join(here, "..", "results", f"raw_{a.model}.jsonl"))]

    n = tv = cp = star = em = no_cite = 0
    cells = {(t, c): 0 for t in (1, 0) for c in (1, 0)}
    for o in rows:
        rec = ev[o["id"]]
        conf = o["conflict"]
        cites = conf.get("cite_chunk_ids", [])
        ans = S(conf.get("answer", ""))
        if not ans.strip():
            continue
        n += 1
        gold = S(rec["target_answer"])           # 정답 = 최신값
        ctext = {c["chunk_id"]: c["text"] for c in rec["chunks"]}
        em += int(pc.ans_equiv(ans, gold))
        if not cites:
            no_cite += 1
            continue
        # TV_cite: 인용한 청크 *중 하나라도* 정답(최신값)을 담았나 (맞는 시점 문서)
        tv_cite = int(any(contains(ctext.get(cid, ""), gold) for cid in cites))
        # CitePrec(표준): 인용 청크가 모델 답을 뒷받침 (시점무관)
        cite_prec = int(any(contains(ctext.get(cid, ""), ans) for cid in cites))
        tv += tv_cite
        cp += cite_prec
        cells[(tv_cite, cite_prec)] += 1
        if tv_cite == 0 and cite_prec == 1:
            star += 1

    tv0 = cells[(0, 1)] + cells[(0, 0)]
    beta = round(cells[(0, 1)] / tv0, 3) if tv0 else 0.0
    res = {"model": a.model, "n": n, "no_cite": no_cite,
           "EM": round(em / n, 3), "TV_cite": round(tv / n, 3),
           "wrong_time_cite_rate": round(1 - tv / n, 3),
           "CitePrec": round(cp / n, 3),
           "star_cell": cells[(0, 1)],
           "blind_spot_beta_P(CitePrec=1|TV_cite=0)": beta}
    json.dump(res, open(os.path.join(here, "..", "results", f"metrics_{a.model}.json"), "w"),
              ensure_ascii=False, indent=2)
    # 2x2
    with open(os.path.join(here, "..", "results", f"contingency_{a.model}.csv"), "w") as f:
        f.write(",CitePrec=PASS,CitePrec=FAIL\n")
        f.write(f"TV=PASS,{cells[(1,1)]},{cells[(1,0)]}\n")
        f.write(f"TV=FAIL,{cells[(0,1)]},{cells[(0,0)]}\n")
    print(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"\n2x2: TV=PASS [{cells[(1,1)]} | {cells[(1,0)]}]  TV=FAIL [{cells[(0,1)]}★ | {cells[(0,0)]}]")


if __name__ == "__main__":
    main()
