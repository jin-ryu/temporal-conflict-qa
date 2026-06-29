"""
rag_conflicts (Google DRAGged, 2025) → 우리 평가셋 (현실 웹 충돌, 변형 없음).

대상: conflict_type == "Conflict due to outdated information" (62개).
설계 (as-of-past로 *변형하지 않음* — 원래 현재지향 질문 그대로):
  · 질문   = 원래 question (현재지향, 자연스러운 일상 질문)
  · 청크   = search_results (short_text), date를 last_modified_time으로
  · 정답   = correct_answer (최신/정답)
  · evidence_chunk_id = 정답을 담은 *가장 최신* 문서 (= 시점-유효).
              모델이 *옛(outdated) 문서*를 인용/사용해 틀리면 → 틀린시점 인용(★ 후보).
  · mode='current' (현재지향), target_side='current'.

→ "자연스러운 현재 질문 + 진짜 웹 충돌에서, 모델이 outdated 문서에 속아 틀리고
   표준 인용평가가 못 잡나(★)" 를 *현실적 난이도*로 측정.

필터: 정답을 담은 문서가 있어야(evidence 매핑 가능) + 충돌(2개+ 날짜) 있어야 채택.

usage: python convert_rag_conflicts.py
출력: ../data/eval_set.jsonl
"""
import json, os, re

SRC = "/tmp/rag_conflicts/conflicts.jsonl"
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "eval_set.jsonl")
_G = {"the", "of", "and", "a", "an", "in", "at", "for", "is", "was", "to", "on", "as"}


def contains_ans(text, ans):
    """정답이 문서에 담겨있나 (정식 포함 OR 변별토큰 과반)."""
    t, a = (text or "").lower(), (ans or "").lower().strip()
    if not a:
        return False
    if a in t:
        return True
    spec = [w for w in re.findall(r"[a-z0-9]+", a) if len(w) > 3 and w not in _G]
    if not spec:
        return a in t
    return sum(1 for w in spec if w in t) * 2 >= len(spec)


def ymd(date_str):
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", str(date_str or ""))
    return m.group(0) if m else (str(date_str)[:10] if date_str else None)


def main():
    rows = [json.loads(l) for l in open(SRC)]
    out_items = rows  # 전체에서 outdated만
    outdated = [r for r in rows if r.get("conflict_type") == "Conflict due to outdated information"]
    out = []
    reasons = {"no_ans_doc": 0, "no_conflict_dates": 0, "too_few_docs": 0, "no_outdated_doc": 0, "ok": 0}
    for i, r in enumerate(outdated):
        srs = r.get("search_results", [])
        if len(srs) < 2:
            reasons["too_few_docs"] += 1
            continue
        ans = r.get("correct_answer", "")
        # 청크 구성: short_text 우선, 날짜 파싱
        chunks = []
        ans_docs = []   # (chunk_id, date) 정답 담은 문서
        for j, s in enumerate(srs):
            txt = (s.get("short_text") or s.get("snippet") or "").strip()
            if not txt:
                continue
            d = ymd(s.get("date"))
            cid = len(chunks)
            chunks.append({"chunk_id": cid, "label": "doc", "last_modified_time": d,
                           "text": txt, "title": s.get("title", ""), "url": s.get("url", "")})
            if contains_ans(txt, ans):
                ans_docs.append((cid, d))
        if len(chunks) < 2:
            reasons["too_few_docs"] += 1
            continue
        # 충돌: 날짜가 2개 이상 갈려야
        dates = {c["last_modified_time"] for c in chunks if c["last_modified_time"]}
        if len(dates) < 2:
            reasons["no_conflict_dates"] += 1
            continue
        if not ans_docs:
            reasons["no_ans_doc"] += 1
            continue
        # outdated 문서(정답 안 담은)가 *하나라도* 있어야 충돌 — 없으면 측정 불가
        n_outdated = len(chunks) - len(ans_docs)
        if n_outdated < 1:
            reasons["no_outdated_doc"] += 1
            continue
        # evidence = 정답 담은 문서 중 *가장 최신* (= 시점-유효, 정답 출처)
        ans_docs.sort(key=lambda x: x[1] or "", reverse=True)
        ev_id = ans_docs[0][0]
        # label 보정: evidence=current, 정답없는 문서=outdated, 정답담은 나머지=current_dup
        for c in chunks:
            if c["chunk_id"] == ev_id:
                c["label"] = "current"
            else:
                c["label"] = "current_dup" if contains_ans(c["text"], ans) else "outdated"
        sid = f"rc_{i}"
        out.append({"id": f"{sid}_current", "source_idx": sid, "mode": "current",
                    "target_side": "current", "domain": "realworld",
                    "new_question": r["question"], "target_answer": ans,
                    "evidence_chunk_id": ev_id, "chunks": chunks})
        reasons["ok"] += 1

    os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
    with open(os.path.abspath(OUT), "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"outdated 충돌 {len(outdated)}개 → {len(out)} 변환")
    print(f"  사유: {reasons}")
    # 통계
    import statistics
    nc = [len(r["chunks"]) for r in out]
    print(f"  청크수: 평균 {statistics.mean(nc):.1f}, 범위 {min(nc)}~{max(nc)}")
    print(f"→ {os.path.abspath(OUT)}")


if __name__ == "__main__":
    main()
