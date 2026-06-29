"""04 — 현실 웹 충돌 ★ 측정 (rag_conflicts, 현재지향). usage: python eval_star.py --model qwen3_8b"""
import json, os, re, argparse, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "03_temporal_validity", "scripts"))
import pilot_common as pc
_G = {"the","of","and","a","an","in","at","for","is","was","to","on","as"}
def S(x): return " ".join(str(i) for i in x) if isinstance(x, list) else str(x or "")
def sup(p, a):
    a, b = S(a).lower().strip(), S(p).lower()
    if not a: return 0
    if a in b: return 1
    s = [w for w in re.findall(r"[a-z0-9]+", a) if len(w) > 3 and w not in _G]
    return int(bool(s) and sum(1 for w in s if w in b) * 2 >= len(s))
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); a = ap.parse_args()
    here = os.path.dirname(__file__)
    ev = {r["id"]: r for r in pc.read_jsonl(os.path.join(here, "..", "data", "eval_set.jsonl"))}
    rows = [json.loads(l) for l in open(os.path.join(here, "..", "results", f"raw_{a.model}.jsonl"))]
    n=tv=cp=star=em=0
    for o in rows:
        rec=ev[o["id"]]; conf=o["conflict"]; cites=conf.get("cite_chunk_ids",[]); ans=S(conf.get("answer",""))
        if not ans.strip(): continue
        n+=1; ct={c["chunk_id"]:c["text"] for c in rec["chunks"]}
        t=int(rec["evidence_chunk_id"] in cites); tv+=t
        c=int(any(sup(ct.get(cid,""),ans) for cid in cites)); cp+=c
        em+=int(pc.ans_equiv(ans,S(rec["target_answer"])))
        if t==0 and c==1: star+=1
    res={"model":a.model,"n":n,"EM":round(em/n,3),"TV_cite":round(tv/n,3),
         "wrong_time":round((n-tv)/n,3),"CitePrec":round(cp/n,3),"star":star,"star_rate":round(star/n,3)}
    json.dump(res, open(os.path.join(here,"..","results",f"star_{a.model}.json"),"w"), indent=2)
    print(json.dumps(res, indent=2))
if __name__=="__main__": main()
