"""
LLM 평가 스크립트 — temporal conflict QA 실험.

조건 A/B/C에 따라 context 청크를 필터링하고,
LLM에 <thought>/<relevance>/<answer> 형식의 응답을 요청하여
Answer EM, Token F1, Evidence Accuracy, Combined Score를 산출한다.

사용법:
  python evaluate_llm.py --input data/qa/hoh_qa_gpt_0_35.jsonl --condition A
  python evaluate_llm.py --input data/qa/hoh_qa_gpt_0_35.jsonl --condition B --provider gemini
  python evaluate_llm.py --input data/qa/hoh_qa_gpt_0_35.jsonl --condition C
"""

import argparse
import hashlib
import json
import random
import re
import string
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

from config import (
    DIR_EVAL, DIR_QA,
    EVAL_GPT_MODEL, EVAL_GEMINI_MODEL,
    VLLM_CONCURRENCY,
    MAX_API_RETRIES, MAX_PARTIAL_RETRIES,
    get_model_alias, setup_logging,
)
from llm_client import (
    make_client, set_rpm,
    rate_limit_wait, handle_api_error,
)

load_dotenv()
logger = setup_logging("evaluate_llm")


# ---------------------------------------------------------------------------
# API calls
# ---------------------------------------------------------------------------

def call_gpt(client, system: str, prompt: str, model: str) -> str | None:
    for attempt in range(MAX_API_RETRIES):
        try:
            rate_limit_wait()
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.0,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if not handle_api_error(e, attempt):
                return None
    logger.error("API max retries exhausted")
    return None


def call_gemini(client, system: str, prompt: str, model: str) -> str | None:
    from google.genai import types
    for attempt in range(MAX_API_RETRIES):
        try:
            rate_limit_wait()
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=0.0,
                ),
            )
            return response.text.strip()
        except Exception as e:
            if not handle_api_error(e, attempt):
                return None
    logger.error("API max retries exhausted")
    return None


def call_vllm(client, system: str, prompt: str, model: str) -> str | None:
    for attempt in range(MAX_API_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.0,
                max_tokens=1024,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if not handle_api_error(e, attempt):
                return None
    logger.error("API max retries exhausted")
    return None


def call_llm(client, system: str, prompt: str, provider: str, model: str) -> str | None:
    if provider == "gpt":
        return call_gpt(client, system, prompt, model)
    elif provider == "gemini":
        return call_gemini(client, system, prompt, model)
    elif provider == "vllm":
        return call_vllm(client, system, prompt, model)
    else:
        raise ValueError(f"Unknown provider: {provider}")




# ---------------------------------------------------------------------------
# Chunk sampling (k 고정)
# ---------------------------------------------------------------------------

def _build_distractor_pool(records: list[dict]) -> dict[int, list[dict]]:
    """hoh_source_idx별 distractor 청크를 수집해 풀을 만든다."""
    pool: dict[int, list[dict]] = {}
    for r in records:
        src = r["hoh_source_idx"]
        if src not in pool:
            pool[src] = []
        for ch in r["chunks"]:
            if ch["label"] == "distractor":
                if not any(c["chunk_id"] == ch["chunk_id"] for c in pool[src]):
                    pool[src].append(ch)
    return pool


def _record_rng(record_id: str) -> random.Random:
    """record id 기반 deterministic RNG — resume 시에도 동일한 결과 보장."""
    seed = int(hashlib.md5(record_id.encode()).hexdigest(), 16) % (2 ** 32)
    return random.Random(seed)


def _get_chunks_for_condition(
    record: dict,
    condition: str,
    distractor_pool: dict[int, list[dict]],
    num_chunks: int = 10,
) -> list[dict]:
    """조건별 청크를 num_chunks개로 구성한다.

    필수 청크(current, 조건 B/C는 outdated 포함)를 유지한 뒤
    나머지 슬롯을 distractor로 채운다.
    record id 기반 deterministic RNG를 사용해 resume 시에도 동일한 구성을 보장한다.
    """
    rng = _record_rng(record["id"])
    chunks = record["chunks"]
    src = record["hoh_source_idx"]

    current_time = next(
        (ch["last_modified_time"] for ch in chunks if ch["label"] == "current"),
        "N/A",
    )

    # 필수 청크 결정
    mandatory = [ch for ch in chunks if ch["label"] == "current"]
    if condition != "no_conflict":
        outdated = [ch for ch in chunks if ch["label"] == "outdated"]
        # mandatory가 num_chunks를 초과하면 outdated를 샘플링
        max_outdated = num_chunks - len(mandatory)
        if len(outdated) > max_outdated > 0:
            outdated = rng.sample(outdated, max_outdated)
        mandatory += outdated

    n_distractor = max(num_chunks - len(mandatory), 0)

    # distractor 후보: 레코드 내 distractor 우선, 부족하면 다른 source에서 보충
    used_ids = {ch["chunk_id"] for ch in mandatory}
    own_distractors = [ch for ch in chunks if ch["label"] == "distractor"]
    candidates = [ch for ch in own_distractors if ch["chunk_id"] not in used_ids]

    if n_distractor > 0 and len(candidates) < n_distractor:
        for other_src, pool_chunks in distractor_pool.items():
            if other_src == src:
                continue
            for ch in pool_chunks:
                if ch["chunk_id"] not in used_ids:
                    candidates.append(ch)

    sampled = rng.sample(candidates, min(n_distractor, len(candidates)))

    # 외부 distractor는 timestamp를 current와 맞춤 (shortcut 방지)
    own_ids = {ch["chunk_id"] for ch in own_distractors}
    padded = [
        ch if ch["chunk_id"] in own_ids
        else dict(ch, label="distractor", last_modified_time=current_time)
        for ch in sampled
    ]

    result = mandatory + padded
    rng.shuffle(result)
    return result


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

EVAL_SYSTEM_PROMPT = (
    "You are an assistant that answers questions using the provided documents. "
    "Each document has a modification timestamp. "
    "Use the timestamps to identify the most temporally relevant document for the question.\n\n"
    "You MUST respond using ONLY the following XML tags, in this exact order:\n\n"
    "<thought>\n"
    "Step-by-step reasoning about which document is most relevant based on the timestamps.\n"
    "</thought>\n"
    "<relevance>\n"
    "The number of the most relevant document (e.g. 3)\n"
    "</relevance>\n"
    "<answer>\n"
    "ONLY the exact short answer (a name, number, date, or short phrase). "
    "Do NOT include explanations, full sentences, or extra context.\n"
    "</answer>\n\n"
    "Do not include any text outside these tags.\n\n"
    "Example:\n"
    "<thought>\nThe question asks about the name before the mid-2024 update. "
    "Document 3 (modified: 2024-07-01) states 'Maudiozyma bulderi' and Document 5 (modified: 2024-06-01) states 'Saccharomyces bulderi'. "
    "Since the question asks about the name before the update, Document 5 predates the change and is more relevant.\n</thought>\n"
    "<relevance>\n5\n</relevance>\n"
    "<answer>\nSaccharomyces bulderi\n</answer>"
)


def build_eval_prompt(record: dict, condition: str, chunks: list[dict]) -> str:
    """조건별 프롬프트 구성. chunks는 _get_chunks_for_condition 결과."""
    if condition == "ambiguous":
        question = record["original_question"]
    else:
        question = record["new_question"]

    lines: list[str] = []
    lines.append(f"[Query] {question}")
    lines.append("")

    for i, ch in enumerate(chunks, 1):
        lmt = ch.get("last_modified_time") or "N/A"
        lines.append(f"[Document {i}] [modified: {lmt}]")
        lines.append(ch["text"])
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

def parse_response(text: str) -> dict | None:
    """<thought>/<relevance>/<answer> 파싱. 태그가 하나라도 빠지면 None 반환."""
    if not text:
        return None

    thought_m   = re.search(r"<thought>(.*?)</thought>", text, re.DOTALL)
    relevance_m = re.search(r"<relevance>\s*\[?\s*(\d+)\s*\]?\s*</relevance>", text, re.DOTALL)
    answer_m    = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)

    if not answer_m or not relevance_m:
        return None

    return {
        "thought": thought_m.group(1).strip() if thought_m else "",
        "relevance_doc_num": int(relevance_m.group(1)),
        "answer": answer_m.group(1).strip(),
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _normalize(s: str) -> str:
    """소문자화, 관사 제거, 구두점 제거, 공백 정규화."""
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    return " ".join(s.split())


def token_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = _normalize(prediction).split()
    gt_tokens = _normalize(ground_truth).split()
    if not pred_tokens or not gt_tokens:
        return float(pred_tokens == gt_tokens)
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall = num_common / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def score_record(parsed: dict, record: dict, chunks: list[dict]) -> dict:
    """EM, F1, evidence, combined 계산. chunks는 실제 프롬프트에 사용된 청크 목록."""
    target = record["target_answer"]
    pred_answer = parsed["answer"]

    em = float(_normalize(pred_answer) == _normalize(target))
    f1 = token_f1(pred_answer, target)

    evidence_correct = 0.0
    if parsed["relevance_doc_num"] is not None:
        doc_idx = parsed["relevance_doc_num"] - 1
        if 0 <= doc_idx < len(chunks):
            if chunks[doc_idx]["chunk_id"] == record["evidence_chunk_id"]:
                evidence_correct = 1.0

    combined = f1 * evidence_correct

    return {
        "answer_em": em,
        "answer_f1": f1,
        "evidence_accuracy": evidence_correct,
        "combined_score": combined,
    }


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------

def load_done_ids(output_path: Path) -> set[str]:
    done: set[str] = set()
    if not output_path.exists():
        return done
    with output_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    done.add(json.loads(line)["id"])
                except Exception:
                    pass
    return done


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def filter_records_by_condition(records: list[dict], condition: str) -> list[dict]:
    """조건에 맞는 mode의 레코드만 필터링.

    no_conflict: current 모드만 (outdated 청크 제거 후 현재 시점 질문만 평가)
    conflict:    current + outdated 모드 (충돌 청크 포함, 양쪽 시점 질문 모두)
    ambiguous:   current_raw 모드만 (시간 힌트 없는 원본 질문)
    """
    if condition == "no_conflict":
        return [r for r in records if r.get("mode") == "current"]
    elif condition == "conflict":
        return [r for r in records if r.get("mode") not in ("current_raw",)]
    else:  # ambiguous
        return [r for r in records if r.get("mode") == "current_raw"]


def evaluate(
    input_path: Path,
    condition: str,
    provider: str,
    model: str,
    num_chunks: int = 10,
) -> None:
    set_rpm(provider)

    DIR_EVAL.mkdir(parents=True, exist_ok=True)

    model_short = get_model_alias(model)
    qa_stem     = input_path.stem
    output_path = DIR_EVAL / f"eval_{model_short}_{condition}_{qa_stem}.jsonl"

    # 전체 레코드 로드 (distractor pool 구성용)
    all_records = []
    with input_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_records.append(json.loads(line))

    records = filter_records_by_condition(all_records, condition)
    if not records:
        logger.error("조건 %s에 해당하는 레코드가 없습니다.", condition)
        return

    distractor_pool = _build_distractor_pool(all_records)

    client = make_client(provider)
    done = load_done_ids(output_path)
    logger.info("=== evaluate_llm (condition=%s, provider=%s, model=%s, num_chunks=%d) input=%s: %d done, %d target ===",
                condition, provider, model, num_chunks, input_path.name, len(done), len(records))


    # ── 대기 태스크 수집 ─────────────────────────────────────────────
    pending = [(r, _get_chunks_for_condition(r, condition, distractor_pool, num_chunks))
               for r in records if r["id"] not in done]

    logger.info("pending: %d eval tasks", len(pending))

    def _process_one(record, chunks):
        record_id = record["id"]
        prompt = build_eval_prompt(record, condition, chunks)

        parsed = None
        response_text = None
        current_prompt = prompt
        for attempt in range(MAX_PARTIAL_RETRIES):
            response_text = call_llm(client, EVAL_SYSTEM_PROMPT, current_prompt, provider, model)
            if response_text is None:
                logger.warning("[%s] API 호출 실패, skipping", record_id)
                break
            parsed = parse_response(response_text)
            if parsed is not None:
                break
            logger.warning("[%s] 포맷 오류, 재시도 %d/%d: %s",
                           record_id, attempt + 1, MAX_PARTIAL_RETRIES, repr(response_text[-80:]))
            current_prompt = (
                prompt
                + f"\n\n[Previous response was invalid — missing required tags.]\n"
                + f"Your previous response:\n{response_text}\n\n"
                + "You MUST wrap your answer in <thought>, <relevance>, and <answer> tags. "
                + "Do not include any text outside these tags."
            )

        if parsed is None:
            return None

        sc = score_record(parsed, record, chunks)
        return {
            "id": record_id,
            "condition": condition,
            "provider": provider,
            "model": model,
            "mode": record.get("mode"),
            "question": record.get("new_question") if condition != "ambiguous" else record.get("original_question"),
            "target_answer": record["target_answer"],
            "predicted_answer": parsed["answer"],
            "evidence_chunk_id": record["evidence_chunk_id"],
            "predicted_relevance": parsed["relevance_doc_num"],
            "thought": parsed["thought"],
            "raw_response": response_text,
            **sc,
        }

    # ── 병렬 (vLLM) 또는 순차 (gemini/gpt) ────────────────────────
    write_lock = threading.Lock()
    use_parallel = provider == "vllm" and VLLM_CONCURRENCY > 1
    max_workers = VLLM_CONCURRENCY if use_parallel else 1

    completed = 0
    with output_path.open("a", encoding="utf-8") as fout:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_process_one, record, chunks): record
                for record, chunks in pending
            }
            for future in as_completed(futures):
                record = futures[future]
                record_id = record["id"]
                try:
                    result = future.result()
                except Exception:
                    logger.exception("[%s] unexpected error", record_id)
                    result = None

                if result is not None:
                    with write_lock:
                        fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                        fout.flush()
                        done.add(record_id)

                completed += 1
                if completed % 10 == 0:
                    if result is not None:
                        logger.info("[%d/%d] %s — EM=%.2f F1=%.2f Ev=%.2f",
                                    completed, len(pending), record_id,
                                    result["answer_em"], result["answer_f1"], result["evidence_accuracy"])
                    else:
                        logger.info("[%d/%d] %s — failed", completed, len(pending), record_id)

    logger.info("Done → %s", output_path)
    logger.info("summary 생성: python eval/summarize_eval.py")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM 평가: temporal conflict QA 실험")
    parser.add_argument(
        "--input", type=str, default=None,
        help="입력 파일 경로. 생략 시 data/qa/ 전체"
    )
    parser.add_argument(
        "--condition", type=str, required=True, choices=["no_conflict", "conflict", "ambiguous"],
        help="실험 조건: no_conflict (outdated 제거), conflict (전체 청크), ambiguous (전체 청크 + 원본 질문)"
    )
    parser.add_argument(
        "--provider", type=str, default="gpt", choices=["gpt", "gemini", "vllm"],
        help="LLM provider (기본값: gpt)"
    )
    parser.add_argument(
        "--gpt-model", type=str, default=None,
        help=f"GPT 모델명 (기본값: {EVAL_GPT_MODEL})"
    )
    parser.add_argument(
        "--gemini-model", type=str, default=None,
        help=f"Gemini 모델명 (기본값: {EVAL_GEMINI_MODEL})"
    )
    parser.add_argument(
        "--vllm-model", type=str, default=None,
        help="vLLM 모델명 (--provider vllm 시 필수)"
    )
    parser.add_argument(
        "--num-chunks", type=int, default=10,
        help="프롬프트에 넣을 청크 수 (기본값: 5)"
    )
    args = parser.parse_args()

    if args.vllm_model and args.provider != "vllm":
        args.provider = "vllm"

    if args.provider == "gpt":
        model = args.gpt_model or EVAL_GPT_MODEL
    elif args.provider == "gemini":
        model = args.gemini_model or EVAL_GEMINI_MODEL
    elif args.provider == "vllm":
        if not args.vllm_model:
            parser.error("--vllm-model is required when --provider is vllm")
        model = args.vllm_model

    if args.input:
        input_paths = [Path(args.input)]
    else:
        input_paths = sorted(DIR_QA.glob("hoh_qa_*.jsonl"))
        if not input_paths:
            logger.error("%s/ 에 hoh_qa_*.jsonl 파일이 없습니다.", DIR_QA)
            exit(1)

    for input_path in input_paths:
        evaluate(input_path=input_path, condition=args.condition, provider=args.provider, model=model, num_chunks=args.num_chunks)
