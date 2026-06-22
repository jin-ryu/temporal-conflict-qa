"""
전체 토큰 사용량·비용 모니터 (읽기 전용, 프로젝트 전역).

생성(chunks_to_qa) · 실험 테스트(02) · judge(03) 등 모든 스크립트가 같은 원장
repo 루트 usage/usage_ledger.jsonl 에 누적 append 한다. 이 스크립트는 그 합계를 보여준다.
(pilot_common 등 어떤 실험 코드에도 의존하지 않는다 — 어디서든 실행 가능)

usage:
  python3 scripts/monitor_cost.py             # 모델별 + 총계
  python3 scripts/monitor_cost.py --by-script # 스크립트별 분해도 함께
"""
import argparse
import json
import os
from pathlib import Path

# repo 루트 usage/ 에 고정 (실행 위치 무관)
LEDGER = Path(__file__).resolve().parents[1] / "usage" / "usage_ledger.jsonl"


def _aggregate(path, key):
    """원장을 key('model' 또는 'script') 기준으로 합산. (agg, total) 반환."""
    agg, tot = {}, {"calls": 0, "input": 0, "output": 0, "cost": 0.0}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            a = agg.setdefault(r.get(key, "?"), {"calls": 0, "input": 0, "output": 0, "cost": 0.0})
            for k in ("calls", "input", "output", "cost"):
                a[k] += r.get(k, 0)
                tot[k] += r.get(k, 0)
    return agg, tot


def _table(agg, tot, label):
    rows = [f"{label:<26}{'calls':>7}{'in_tok':>12}{'out_tok':>12}{'$cost':>10}"]
    for name, a in sorted(agg.items()):
        rows.append(f"{name:<26}{a['calls']:>7}{a['input']:>12}{a['output']:>12}{('$%.4f' % a['cost']):>10}")
    rows.append("-" * 67)
    rows.append(f"{'TOTAL':<26}{tot['calls']:>7}{tot['input']:>12}{tot['output']:>12}{('$%.4f' % tot['cost']):>10}")
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser(description="전체 토큰·비용 합계 보기")
    ap.add_argument("--by-script", action="store_true", help="스크립트별 분해도 함께 출력")
    args = ap.parse_args()

    if not LEDGER.exists():
        print(f"(원장 없음 — 아직 실행 기록 없음: {LEDGER})")
        return

    print(f"[전체 토큰·비용 — {LEDGER}]\n")
    agg_m, tot = _aggregate(LEDGER, "model")
    print(_table(agg_m, tot, "model"))

    if args.by_script:
        print("\n[스크립트별]")
        agg_s, tot_s = _aggregate(LEDGER, "script")
        print(_table(agg_s, tot_s, "script"))


if __name__ == "__main__":
    main()
