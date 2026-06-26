"""
07 — Timestamp-셀렉터 baseline (DSG의 ②만 timestamp로, 추가 호출 0).

질문 as-of 시점 t* 가 주어졌을 때, 각 청크의 last_modified_time τ 중
*t* 를 덮는(τ ≤ t*에 가장 가까운, 즉 t* 시점에 유효한)* 청크만 남기는 결정론적 selector.
→ 그 selector가 고른 청크집합이 시점-유효(evidence) 청크를 포함하나(TV_cite_oracle)?
→ "우리 셋에서 단순 timestamp 매칭이 오라클 셀렉터에 얼마나 근접하나" 측정.

평가 (생성 없이, selector 단독):
  · sel_hit  : selector가 고른 청크에 evidence_chunk_id 포함? (= 셀렉터가 맞는 문서 골랐나)
  · sel_only : selector가 *evidence만* 골랐나 (틀린시점 문서 다 걸렀나)
  · 비교: baseline(전체 청크 주고 모델이 인용) TV_cite vs selector sel_hit

usage: python 07_timestamp_selector.py
출력: results/timestamp_selector.json + 콘솔
"""
import json, os, re

HERE = os.path.dirname(__file__)
EVAL = os.path.join(HERE, "..", "data", "eval_set.jsonl")
RES = os.path.join(HERE, "..", "results")
_MONTHS = {m: i for i, m in enumerate(
    ["", "January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"])}


def as_of_ym(question):
    """질문의 'As of <Month> <Year>' → (year, month) 정수쌍."""
    m = re.search(r"As of (\w+) (\d{4})", question or "")
    if not m or m.group(1) not in _MONTHS:
        return None
    return int(m.group(2)), _MONTHS[m.group(1)]


def chunk_ym(ts):
    """'2024-10-01 ...' → (year, month)."""
    m = re.match(r"(\d{4})-(\d{2})", str(ts or ""))
    return (int(m.group(1)), int(m.group(2))) if m else None


def select(chunks, t):
    """t*(질문시점)에 *유효한* 청크: τ ≤ t* 중 t*에 가장 가까운 timestamp의 청크들.
    (그 timestamp가 '그 시점에 살아있던 마지막 버전'.) τ가 다 t*보다 크면 가장 이른 것."""
    cand = []
    for c in chunks:
        ym = chunk_ym(c.get("last_modified_time"))
        if ym:
            cand.append((ym, c))
    if not cand:
        return []
    le = [(ym, c) for ym, c in cand if ym <= t]
    if le:
        best = max(ym for ym, _ in le)               # t* 이하 중 가장 최근
        return [c for ym, c in le if ym == best]
    best = min(ym for ym, _ in cand)                 # 다 미래면 가장 이른
    return [c for ym, c in cand if ym == best]


def main():
    recs = [json.loads(l) for l in open(EVAL)]
    ap = [r for r in recs if r["target_side"] == "outdated"]
    n = len(ap)
    sel_hit = sel_only = no_asof = 0
    misses = []
    for r in ap:
        t = as_of_ym(r["new_question"])
        if not t:
            no_asof += 1
            continue
        sel = select(r["chunks"], t)
        sel_ids = {c["chunk_id"] for c in sel}
        ev = r["evidence_chunk_id"]
        if ev in sel_ids:
            sel_hit += 1
            if sel_ids == {ev} or all(c["chunk_id"] == ev or c["label"] == "outdated" for c in sel):
                sel_only += 1
        else:
            misses.append({"id": r["id"], "as_of": t, "ev": ev,
                           "selected": [(c["chunk_id"], c.get("last_modified_time")) for c in sel]})
    res = {"n": n, "no_asof": no_asof,
           "sel_hit": sel_hit, "sel_hit_rate": round(sel_hit / (n or 1), 3),
           "sel_only": sel_only, "sel_only_rate": round(sel_only / (n or 1), 3),
           "n_miss": len(misses)}
    print("\n=== Timestamp-셀렉터 baseline (as-of-past n=%d) ===\n" % n)
    print(f"  as-of 추출 실패: {no_asof}")
    print(f"  sel_hit  (evidence 청크를 선택에 포함): {sel_hit}/{n} = {res['sel_hit_rate']:.1%}")
    print(f"  sel_only (틀린시점 청크 다 거름)       : {sel_only}/{n} = {res['sel_only_rate']:.1%}")
    print(f"  miss (evidence 못 고름): {len(misses)}")
    for m in misses[:8]:
        print(f"    {m['id']}: as_of={m['as_of']} ev={m['ev']} 골라진것={m['selected']}")
    json.dump({**res, "misses": misses}, open(os.path.join(RES, "timestamp_selector.json"), "w"),
              ensure_ascii=False, indent=2)
    print(f"\n→ {os.path.join(RES, 'timestamp_selector.json')}")
    print("\n해석: sel_hit가 ~오라클(높음)이면 = 우리 셋은 timestamp매칭으로 거의 풀림(셀렉터 자명).")
    print("      → DSG 신규성은 'timestamp≠유효구간' 어려운 케이스에서 content-추론으로 차별화해야.")


if __name__ == "__main__":
    main()
