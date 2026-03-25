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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DIR_CHUNKS, DIR_QA, DIR_QA_REASONING, setup_logging

logger = setup_logging("merge_shards")

STEP_CONFIG = {
    1: {
        "dir":     DIR_CHUNKS,
        "pattern": "chunks_[0-9]*.jsonl",
        "output":  DIR_CHUNKS / "chunks.jsonl",
    },
    2: {
        "dir":     DIR_QA,
        "pattern": "qa_*_[0-9]*.jsonl",
        "output":  DIR_QA / "qa.jsonl",
    },
    3: {
        "dir":     DIR_QA_REASONING,
        "pattern": "qa_*_reasoning_*.jsonl",
        "output":  DIR_QA_REASONING / "qa_reasoning.jsonl",
    },
}


def _build_range_name(shard_files: list[Path]) -> str | None:
    """shard 파일명에서 공통 prefix + 전체 숫자 범위를 추출.

    예: [hoh_qa_gpt_0_25.jsonl, hoh_qa_gpt_25_60.jsonl]
      → 'hoh_qa_gpt_0_60'
    """
    import re
    # 공통 prefix 추출: 첫 번째 숫자 직전까지
    stems = [f.stem for f in shard_files]
    prefix_match = re.match(r"^(.*?_)\d", stems[0])
    if not prefix_match:
        return None
    prefix = prefix_match.group(1)  # e.g. "hoh_qa_gpt_"

    # 모든 shard에서 숫자 수집
    nums: list[int] = []
    for stem in stems:
        found = re.findall(r"(\d+)", stem)
        nums.extend(int(n) for n in found)
    if not nums:
        return None

    return f"{prefix}{min(nums)}_{max(nums)}"


def merge(pattern: str, output_path: Path, search_dir: Path = Path("."),
          auto_range: bool = False) -> None:
    shard_files = sorted(search_dir.glob(pattern))

    if not shard_files:
        logger.warning("파일 없음: %s/%s", search_dir, pattern)
        return

    if auto_range:
        range_name = _build_range_name(shard_files)
        if range_name:
            output_path = output_path.with_name(f"{range_name}{output_path.suffix}")

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
    parser.add_argument("--auto-range", action="store_true",
                        help="출력 파일명에 shard 범위를 자동 추가 (예: hoh_qa_0_60.jsonl)")
    args = parser.parse_args()

    if args.step:
        cfg = STEP_CONFIG[args.step]
        merge(cfg["pattern"], cfg["output"], search_dir=cfg["dir"],
              auto_range=args.auto_range)
    elif args.pattern and args.output:
        merge(args.pattern, Path(args.output), auto_range=args.auto_range)
    else:
        parser.print_help()
