"""
데이터셋 생성 파이프라인 실행기.

hoh_to_chunks.py → chunks_to_qa.py 를 순차 실행한다.

사용법:
  python run_pipeline.py --start 0 --end 25 --provider gpt
  python run_pipeline.py --start 0 --end 100 --provider gemini

generate_reasoning.py 는 별도 실행:
  python generate_reasoning.py --provider gpt --input data/qa/hoh_qa_gpt_0_25.jsonl --ratio 0.2
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DIR_CHUNKS, DIR_QA, get_model_alias, setup_logging

_HERE = Path(__file__).resolve().parent

logger = setup_logging("run_pipeline")


def run(cmd: list[str]) -> None:
    logger.info("=" * 60)
    logger.info("  %s", " ".join(cmd))
    logger.info("=" * 60)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        logger.error("실패: %s", " ".join(cmd))
        sys.exit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="데이터셋 생성 파이프라인 (hoh_to_chunks → chunks_to_qa)")
    parser.add_argument("--start", type=int, default=0, help="시작 인덱스 (inclusive)")
    parser.add_argument("--end",   type=int, required=True, help="종료 인덱스 (exclusive)")
    parser.add_argument(
        "--provider", type=str, default="gemini", choices=["gemini", "gpt", "vllm"],
        help="LLM provider (기본값: gemini)"
    )
    parser.add_argument("--gpt-model",    type=str, default=None, help="GPT 모델명 오버라이드")
    parser.add_argument("--gemini-model", type=str, default=None, help="Gemini 모델명 오버라이드")
    parser.add_argument("--vllm-model",   type=str, default=None, help="vLLM 모델명 (예: Qwen/Qwen3-32B-AWQ)")
    args = parser.parse_args()

    if args.vllm_model and args.provider != "vllm":
        args.provider = "vllm"

    if args.provider == "vllm" and not args.vllm_model:
        parser.error("--vllm-model is required when --provider is vllm")

    chunks_path = DIR_CHUNKS / f"chunks_{args.start}_{args.end}.jsonl"

    # ── hoh_to_chunks ─────────────────────────────────────────────────
    run([
        sys.executable, str(_HERE / "hoh_to_chunks.py"),
        "--start", str(args.start),
        "--end",   str(args.end),
    ])

    if not chunks_path.exists():
        logger.error("청크 파일 없음: %s", chunks_path)
        sys.exit(1)

    # ── chunks_to_qa ──────────────────────────────────────────────────
    cmd2 = [
        sys.executable, str(_HERE / "chunks_to_qa.py"),
        "--provider", args.provider,
        "--input", str(chunks_path),
    ]
    if args.gpt_model:
        cmd2 += ["--gpt-model", args.gpt_model]
    if args.gemini_model:
        cmd2 += ["--gemini-model", args.gemini_model]
    if args.vllm_model:
        cmd2 += ["--vllm-model", args.vllm_model]
    run(cmd2)

    # ── 완료 ──────────────────────────────────────────────────────────
    suffix  = f"_{args.start}_{args.end}"
    if args.provider == "vllm" and args.vllm_model:
        model_tag = get_model_alias(args.vllm_model)
    else:
        model_tag = get_model_alias(args.provider)
    qa_path = DIR_QA / f"qa_{model_tag}{suffix}.jsonl"

    logger.info("=" * 60)
    logger.info("  파이프라인 완료!")
    logger.info("  청크: %s", chunks_path)
    logger.info("  QA:   %s", qa_path)
    logger.info("")
    logger.info("  reasoning 생성이 필요하면:")
    logger.info("  python generate_reasoning.py --provider gpt --input %s --ratio 0.2", qa_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
