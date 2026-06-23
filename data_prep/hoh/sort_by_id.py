"""
JSONL 파일을 id 필드 기준으로 정렬한다.

사용법:
  python scripts/sort_by_id.py data/qa/hoh_qa_vllm_0_500.jsonl
  python scripts/sort_by_id.py data/qa/hoh_qa_vllm_0_500.jsonl -o data/qa/sorted.jsonl
"""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="JSONL 파일을 id 기준으로 정렬")
    parser.add_argument("input", type=Path, help="입력 JSONL 파일")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="출력 파일 (미지정 시 입력 파일을 덮어씀)")
    args = parser.parse_args()

    output_path = args.output or args.input

    records = []
    with args.input.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    records.sort(key=lambda r: r["id"])

    with output_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Sorted {len(records)} records → {output_path}")


if __name__ == "__main__":
    main()
