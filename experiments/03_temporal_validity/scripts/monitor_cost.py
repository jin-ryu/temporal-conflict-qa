"""
전체 토큰 사용량·비용 모니터 (읽기 전용).

모든 스크립트(chunks_to_qa 생성 · 02 테스트 · 03 judge)가 같은 원장
repo 루트 usage/usage_ledger.jsonl 에 누적 append 한다. 이 스크립트는 그 합계를 보여준다.

usage:
  python monitor_cost.py            # 모델별 + 총계
  python monitor_cost.py --by-script  # 스크립트별 분해도 함께
"""
import argparse, json, os
import pilot_common as pc


def by_script(path):
    if not os.path.exists(path):
        return "(원장 없음)"
    agg = {}
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        a = agg.setdefault(r["script"], {"calls": 0, "input": 0, "output": 0, "cost": 0.0})
        for k in ("calls", "input", "output", "cost"):
            a[k] += r.get(k, 0)
    out = [f"{'script':<22}{'calls':>7}{'in_tok':>12}{'out_tok':>12}{'$cost':>10}"]
    for s, a in sorted(agg.items()):
        out.append(f"{s:<22}{a['calls']:>7}{a['input']:>12}{a['output']:>12}{('$%.4f'%a['cost']):>10}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--by-script", action="store_true")
    args = ap.parse_args()
    print(f"[전체 토큰·비용 — {os.path.relpath(pc.LEDGER)}]\n")
    print(pc.ledger_total())
    if args.by_script:
        print("\n[스크립트별]")
        print(by_script(pc.LEDGER))


if __name__ == "__main__":
    main()
