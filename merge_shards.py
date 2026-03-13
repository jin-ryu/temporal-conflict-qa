"""
범위별로 나뉜 JSONL shard 파일을 하나로 병합한다.

사용법:
  python merge_shards.py --step 1   # hoh_chunks_*.jsonl → hoh_chunks.jsonl
  python merge_shards.py --step 2   # hoh_qa_*.jsonl     → hoh_qa.jsonl
  python merge_shards.py --step 3   # hoh_sft_*.jsonl    → hoh_sft.jsonl

  # 패턴과 출력 파일을 직접 지정
  python merge_shards.py --pattern "hoh_chunks_*.jsonl" --output hoh_chunks.jsonl
"""

import argparse
import json
from pathlib import Path

from config import DIR_CHUNKS, DIR_QA, DIR_QA_REASONING, setup_logging

logger = setup_logging("merge_shards")

STEP_CONFIG = {
    1: {
        "dir":     DIR_CHUNKS,
        "pattern": "hoh_chunks_[0-9]*.jsonl",
        "output":  DIR_CHUNKS / "hoh_chunks.jsonl",
    },
    2: {
        "dir":     DIR_QA,
        "pattern": "hoh_qa_*_[0-9]*.jsonl",
        "output":  DIR_QA / "hoh_qa.jsonl",
    },
    3: {
        "dir":     DIR_QA_REASONING,
        "pattern": "hoh_sft_*_[0-9]*.jsonl",
        "output":  DIR_QA_REASONING / "hoh_sft.jsonl",
    },
}


def merge(pattern: str, output_path: Path, search_dir: Path = Path(".")) -> None:
    shard_files = sorted(search_dir.glob(pattern))

    if not shard_files:
        logger.warning("파일 없음: %s/%s", search_dir, pattern)
        return

    logger.info("%d개 shard 파일:", len(shard_files))
    for f in shard_files:
        logger.info("  %s", f)

    seen_ids: set[str] = set()
    total_written = 0
    total_dup = 0

    with output_path.open("w", encoding="utf-8") as fout:
        for shard_file in shard_files:
            file_count = 0
            with shard_file.open(encoding="utf-8") as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    record_id = record.get("id", "")

                    if record_id in seen_ids:
                        total_dup += 1
                        continue

                    seen_ids.add(record_id)
                    fout.write(line + "\n")
                    file_count += 1
                    total_written += 1

            logger.info("  %s: %d건", shard_file.name, file_count)

    logger.info("완료 → %s", output_path)
    logger.info("  총 %d건 (중복 제거: %d건)", total_written, total_dup)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="JSONL shard 파일 병합")
    parser.add_argument("--step", type=int, choices=[1, 2, 3], help="Step 번호")
    parser.add_argument("--pattern", type=str, help="glob 패턴 (직접 지정)")
    parser.add_argument("--output", type=str, help="출력 파일 경로 (직접 지정)")
    args = parser.parse_args()

    if args.step:
        cfg = STEP_CONFIG[args.step]
        merge(cfg["pattern"], cfg["output"], search_dir=cfg["dir"])
    elif args.pattern and args.output:
        merge(args.pattern, Path(args.output))
    else:
        parser.print_help()
