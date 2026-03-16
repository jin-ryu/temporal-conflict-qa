"""
평가 결과 집계 스크립트.

data/eval/ 의 eval_*.jsonl 파일을 읽어 조건별 지표를 집계하고
data/eval_summary/ 에 저장한다.

사용법:
  python eval/summarize_eval.py
  python eval/summarize_eval.py --input data/eval/eval_gpt41_conflict_hoh_qa_gpt_0_50.jsonl
  python eval/summarize_eval.py --input file1.jsonl file2.jsonl --subset 100
"""

import argparse
import json
import re
import string
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DIR_EVAL, DIR_EVAL_SUMMARY, DIR_QA, setup_logging

logger = setup_logging("summarize_eval")


def _mode_group(mode: str) -> str:
    """outdated_0, outdated_1 등을 'outdated'로 합산."""
    if mode.startswith("outdated"):
        return "outdated"
    return mode


def _load_expected_ids(qa_stem: str, condition: str) -> dict[str, set[str]]:
    """원본 QA 파일에서 condition에 해당하는 id를 mode_group별로 반환한다."""
    qa_path = DIR_QA / f"{qa_stem}.jsonl"
    if not qa_path.exists():
        return {}

    # condition별 대상 mode 필터
    if condition == "conflict":
        target_groups = {"current", "outdated"}
    elif condition == "no_conflict":
        target_groups = {"current"}
    elif condition == "ambiguous":
        target_groups = {"current_raw"}
    else:
        return {}

    ids_by_group: dict[str, set[str]] = {}
    with qa_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            group = _mode_group(r.get("mode", "unknown"))
            if group in target_groups:
                ids_by_group.setdefault(group, set()).add(r["id"])

    return ids_by_group


def _normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    return " ".join(s.split())


def _avg_scores(records: list[dict]) -> dict:
    """레코드 목록에서 평균 지표를 계산한다."""
    keys = ["answer_em", "answer_f1", "evidence_accuracy", "combined_score"]
    valid = [r for r in records if not r.get("parse_error")]
    scores = {k: [r[k] for r in valid if k in r] for k in keys}
    result = {"n": len(valid), "n_parse_error": len(records) - len(valid)}
    for k, vals in scores.items():
        result[k] = round(sum(vals) / len(vals), 4) if vals else 0.0
    return result


def compute_summary(
    records: list[dict],
    expected_ids: dict[str, set[str]] | None = None,
) -> dict:
    """전체 + mode_group별 지표를 계산한다."""
    n_expected_total = sum(len(ids) for ids in expected_ids.values()) if expected_ids else None
    summary = _avg_scores(records)
    if n_expected_total is not None:
        summary["n_expected"] = n_expected_total

    # mode_group별 breakdown
    groups: dict[str, list[dict]] = {}
    for r in records:
        group = _mode_group(r.get("mode", "unknown"))
        groups.setdefault(group, []).append(r)

    if len(groups) > 1 or expected_ids:
        by_mode = {}
        all_group_names = set(groups.keys())
        if expected_ids:
            all_group_names |= set(expected_ids.keys())
        for group in sorted(all_group_names):
            recs = groups.get(group, [])
            entry = _avg_scores(recs)
            if expected_ids and group in expected_ids:
                entry["n_expected"] = len(expected_ids[group])
            # 레코드도 없고 expected도 없는 그룹은 제외
            if entry["n"] == 0 and "n_expected" not in entry:
                continue
            by_mode[group] = entry
        summary["by_mode"] = by_mode

    return summary


def _load_records(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _fmt_n(s: dict) -> str:
    """n 또는 n/n_expected 형태의 문자열을 반환한다."""
    if "n_expected" in s:
        return f"{s['n']}/{s['n_expected']}"
    return str(s["n"])


def _log_summary(label: str, summary: dict) -> None:
    logger.info("%s: n=%s  EM=%.4f  F1=%.4f  Evidence=%.4f  Combined=%.4f",
                label, _fmt_n(summary),
                summary["answer_em"], summary["answer_f1"],
                summary["evidence_accuracy"], summary["combined_score"])

    if "by_mode" in summary:
        for mode, ms in summary["by_mode"].items():
            logger.info("  ├─ %s: n=%s  EM=%.4f  F1=%.4f  Evidence=%.4f  Combined=%.4f",
                        mode, _fmt_n(ms).ljust(8), ms["answer_em"], ms["answer_f1"],
                        ms["evidence_accuracy"], ms["combined_score"])


def summarize(input_paths: list[Path], subset: int | None = None) -> None:
    DIR_EVAL_SUMMARY.mkdir(parents=True, exist_ok=True)

    # --subset: 여러 파일의 공통 성공 id 교집합으로 필터링
    if subset is not None:
        all_file_records: dict[Path, list[dict]] = {}
        all_success_ids: list[set[str]] = []

        for path in input_paths:
            records = _load_records(path)
            all_file_records[path] = records
            success_ids = {r["id"] for r in records if not r.get("parse_error")}
            all_success_ids.append(success_ids)

        if not all_success_ids:
            logger.error("입력 파일이 없습니다.")
            return

        common_ids = set.intersection(*all_success_ids)
        logger.info("공통 성공 id: %d건 (파일 %d개)", len(common_ids), len(input_paths))

        if len(common_ids) < subset:
            logger.warning("공통 id %d건이 subset %d보다 적음, 전체 사용", len(common_ids), subset)
            subset_ids = common_ids
        else:
            subset_ids = set(sorted(common_ids)[:subset])

        logger.info("subset=%d건으로 집계", len(subset_ids))

        for path in input_paths:
            records = [r for r in all_file_records[path] if r["id"] in subset_ids]

            if not records:
                logger.warning("%s: subset 필터 후 레코드 없음, skip", path.name)
                continue

            meta = {"source_file": path.name, "subset": len(subset_ids)}
            expected_ids: dict[str, set[str]] | None = None
            m = re.match(r"eval_(.+?)_(no_conflict|conflict|ambiguous)_(.+)\.jsonl", path.name)
            if m:
                meta["model"] = records[0].get("model", m.group(1))
                meta["condition"] = m.group(2)
                meta["qa_stem"] = m.group(3)
                meta["provider"] = records[0].get("provider", "")
                expected_ids = _load_expected_ids(m.group(3), m.group(2))

            summary = {**meta, **compute_summary(records, expected_ids)}
            _log_summary(path.name, summary)

            stem = path.stem[len("eval_"):] if path.stem.startswith("eval_") else path.stem
            out_path = DIR_EVAL_SUMMARY / f"summary_{stem}_subset{len(subset_ids)}.json"
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)

            logger.info("Summary → %s", out_path)
        return

    # 기본 모드: 파일별 독립 집계
    for path in input_paths:
        records = _load_records(path)

        if not records:
            logger.warning("%s: 레코드 없음, skip", path.name)
            continue

        meta = {"source_file": path.name}
        expected_ids: dict[str, set[str]] | None = None
        m = re.match(r"eval_(.+?)_(no_conflict|conflict|ambiguous)_(.+)\.jsonl", path.name)
        if m:
            meta["model"] = records[0].get("model", m.group(1))
            meta["condition"] = m.group(2)
            meta["qa_stem"] = m.group(3)
            meta["provider"] = records[0].get("provider", "")
            expected_ids = _load_expected_ids(m.group(3), m.group(2))

        summary = {**meta, **compute_summary(records, expected_ids)}
        _log_summary(path.name, summary)

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
    parser.add_argument(
        "--subset", type=int, default=None,
        help="모델 간 비교 시 공통 성공 id 교집합에서 N건만 사용 (파일명에 _subsetN 추가)"
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

    summarize(paths, subset=args.subset)
