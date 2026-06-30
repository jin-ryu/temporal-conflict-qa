"""
틀린 답의 *오류 모드* 통합 분석 — 2x2(EM×CitePrec) + 인용 대상 분류를 함께.

질문: 모델이 틀릴 때 (a)무엇을 인용하고 (b)그게 ★(mis-attr)로 가나 정직오류로 가나?

인용 대상 4분류:
  · conflict_doc : 충돌 청크(엉뚱한 시점/주) 인용 → 보통 틀린답 받침 → ★
  · distractor   : 무관 청크 인용 → 안 받침 → 정직오류
  · evidence     : 정답 청크 인용했는데도 답 틀림 → 해석실패 → 정직오류
  · none         : 무인용 → 정직오류

오류모드 2분류 (틀린답 중):
  · star (mis-attr) : CitePrec=1 (인용이 틀린답 받침) — 평가가 못 잡는 위험한 오류
  · honest          : CitePrec=0 (인용 빗나감) — 평가가 잡는 정직한 오류

usage:
  python analyze_error_modes.py --eval <eval> --raw <raw> [--binary] [--side outdated|current]
    --binary : Yes/No 축(exp05). 기본은 열린답(exp03/04).
    --side   : exp03 target_side 필터(as-of-past=outdated, current). 생략시 전체.
"""
import json, re, argparse, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "03_temporal_validity", "scripts"))
import pilot_common as pc

_G = {"the", "of", "and", "a", "an", "in", "at", "for", "is", "was", "to", "on", "as"}
_YN = re.compile(r"\b(yes|no)\b", re.I)


def S(x):
    return " ".join(str(i) for i in x) if isinstance(x, list) else str(x or "")


def yn(s):
    m = _YN.search(S(s))
    return m.group(1).lower() if m else ""


def sup(passage, answer):
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
    ap.add_argument("--binary", action="store_true")
    ap.add_argument("--side", choices=["outdated", "current"])
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    ev = {r["id"]: r for r in pc.read_jsonl(args.eval)}
    rows = [json.loads(l) for l in open(args.raw) if l.strip()]

    n = wrong = 0
    cite_mode = {"conflict_doc": 0, "distractor": 0, "evidence": 0, "none": 0}
    # 각 인용대상이 star/honest로 어떻게 갈리는지
    cross = {k: {"star": 0, "honest": 0} for k in cite_mode}
    star_total = honest_total = 0

    for o in rows:
        rec = ev.get(o["id"])
        if not rec:
            continue
        if args.side and rec.get("target_side") != args.side:
            continue
        conf = o["conflict"]
        ans = S(conf.get("answer", ""))
        if args.binary:
            a, gold = yn(ans), yn(rec["target_answer"])
            if not a:
                continue
            em = int(a == gold)
        else:
            if not ans.strip():
                continue
            em = int(pc.ans_equiv(ans, S(rec["target_answer"])))
        n += 1
        if em:
            continue
        wrong += 1

        cites = conf.get("cite_chunk_ids", [])
        # CitePrec: 인용청크가 모델의 (틀린)답을 받치나
        if args.binary:
            stance = {c["chunk_id"]: yn(c.get("chunk_answer", ""))
                      for c in rec["chunks"] if c["label"] in ("current", "outdated")}
            cp = int(any(stance.get(cid) == a for cid in cites))
        else:
            ct = {c["chunk_id"]: c["text"] for c in rec["chunks"]}
            cp = int(any(sup(ct.get(cid, ""), ans) for cid in cites))
        mode = "star" if cp else "honest"
        if cp:
            star_total += 1
        else:
            honest_total += 1

        # 인용 대상 분류 (우선순위: 정답청크 > 충돌(틀린시점/주) > distractor > none)
        # 주의: HoH는 current 라벨 청크가 여러개 → '정답청크'는 evidence_chunk_id 정확히 일치만.
        #       그 외 current(정답아닌 최신)·outdated 는 '충돌 문서'(엉뚱한 시점) 로 묶음.
        lab = {c["chunk_id"]: c["label"] for c in rec["chunks"]}
        ev_id = rec.get("evidence_chunk_id")
        cl = [lab.get(c, "?") for c in cites]
        if not cites:
            tgt = "none"
        elif ev_id in cites:
            tgt = "evidence"                                  # 정답 청크를 인용
        elif any(l in ("outdated", "current") for l in cl):
            tgt = "conflict_doc"                              # 정답아닌 시점-문서(틀린시점) 인용
        elif any(l == "distractor" for l in cl):
            tgt = "distractor"
        else:
            tgt = "none"
        cite_mode[tgt] += 1
        cross[tgt][mode] += 1

    lbl = args.label or os.path.basename(args.raw)
    print(f"[{lbl}]  n={n}  틀린답={wrong}")
    if wrong:
        print(f"  오류모드:  ★mis-attr {star_total} ({star_total/wrong:.0%})  |  정직오류 {honest_total} ({honest_total/wrong:.0%})")
        print(f"  인용대상별 (→ ★ / 정직):")
        for k in ["conflict_doc", "evidence", "distractor", "none"]:
            v = cite_mode[k]
            if v:
                print(f"     {k:13} {v:3} ({v/wrong:3.0%})  → ★{cross[k]['star']:3} / 정직{cross[k]['honest']:3}")
    return {"n": n, "wrong": wrong, "star": star_total, "honest": honest_total,
            "cite_mode": cite_mode, "cross": cross}


if __name__ == "__main__":
    main()
