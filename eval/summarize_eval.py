"""
평가 결과 집계 스크립트.

data/eval/ 의 eval_*.jsonl 파일을 읽어 조건별 지표를 집계하고
data/eval_summary/ 에 저장한다.

사용법:
  python eval/summarize_eval.py
  python eval/summarize_eval.py --input data/eval/eval_gpt41_conflict_hoh_qa_gpt_0_50.jsonl
"""

import argparse
import json
import re
import string
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DIR_EVAL, DIR_EVAL_SUMMARY, setup_logging

logger = setup_logging("summarize_eval")


def _normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    return " ".join(s.split())


def compute_summary(records: list[dict]) -> dict:
    """레코드 목록에서 평균 지표를 계산한다."""
    keys = ["answer_em", "answer_f1", "evidence_accuracy", "combined_score"]
    scores = {k: [r[k] for r in records if k in r and not r.get("parse_error")] for k in keys}
    n = len([r for r in records if not r.get("parse_error")])
    n_errors = len([r for r in records if r.get("parse_error")])
    summary = {"n": n, "n_parse_error": n_errors}
    for k, vals in scores.items():
        summary[k] = round(sum(vals) / len(vals), 4) if vals else 0.0
    return summary


def summarize(input_paths: list[Path]) -> None:
    DIR_EVAL_SUMMARY.mkdir(parents=True, exist_ok=True)

    for path in input_paths:
        records = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        if not records:
            logger.warning("%s: 레코드 없음, skip", path.name)
            continue

        # 파일명에서 메타 정보 추출 — eval_{model}_{condition}_{qa_stem}.jsonl
        meta = {"source_file": path.name}
        m = re.match(r"eval_(.+?)_(no_conflict|conflict|ambiguous)_(.+)\.jsonl", path.name)
        if m:
            meta["model"] = records[0].get("model", m.group(1))
            meta["condition"] = m.group(2)
            meta["qa_stem"] = m.group(3)
            meta["provider"] = records[0].get("provider", "")

        summary = {**meta, **compute_summary(records)}

        logger.info("%s: n=%d  EM=%.4f  F1=%.4f  Evidence=%.4f  Combined=%.4f",
                    path.name, summary["n"],
                    summary["answer_em"], summary["answer_f1"],
                    summary["evidence_accuracy"], summary["combined_score"])

        stem = path.stem[len("eval_"):] if path.stem.startswith("eval_") else path.stem
        out_path = DIR_EVAL_SUMMARY / f"summary_{stem}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        logger.info("Summary → %s", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="eval 결과 집계")
    parser.add_argument(
        "--input", type=str, default=None, nargs="+",
        help="집계할 jsonl 파일 경로. 생략 시 data/eval/ 전체"
    )
    args = parser.parse_args()

    if args.input:
        paths = [Path(p) for p in args.input]
    else:
        paths = sorted(p for p in DIR_EVAL.glob("eval_*.jsonl")
                       if "_errors" not in p.name)

    if not paths:
        logger.error("data/eval/ 에 eval_*.jsonl 파일이 없습니다.")
        exit(1)

    summarize(paths)
