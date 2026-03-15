"""
hoh_chunks.jsonl → data/qa/ (GRPO 학습용 QA 샘플)

hoh_to_chunks.py 결과물을 입력으로 받아 LLM을 호출하고,
시간적 충돌 QA 샘플을 생성하여 data/qa/ 에 저장한다.

생성되는 mode
-------------
- current     : current 시점 기준, LLM이 narrative 시간 힌트 포함 질문 생성
- current_raw : HOH 원본 질문 그대로 사용 (LLM 미사용, 시간 힌트 없음)
- outdated_i  : i번째 outdated 시점 기준, LLM이 narrative 시간 힌트 포함 질문 생성

질문 생성 규칙
--------------
- 질문에 연도/월 숫자(e.g. 2024, July 2024) 포함 금지
- 'when the policy changed', 'at the time of the announcement' 등 서술적 시간 힌트만 허용
- 숫자 날짜 포함 시 validate_pair()에서 자동 거부

target_answer는 hoh_to_chunks.py 결과에서 고정 사용 (LLM 자유 생성 없음).

hoh_source_idx 필드: HoH 원본 데이터셋 인덱스 (정수). 학습/평가 split 시 데이터 오염 방지용.

Provider
--------
--provider gemini  : Gemini API (기본값, GEMINI_API_KEY)
--provider gpt     : OpenAI API (OPENAI_API_KEY)
"""

import argparse
import json
import os
import random
import re
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

from config import (
    DIR_CHUNKS, DIR_QA, CHUNKS_PATH,
    GEMINI_MODEL, GPT_MODEL, GEMINI_RPM, GPT_RPM,
    MAX_API_RETRIES, MAX_PARTIAL_RETRIES,
    setup_logging,
)

load_dotenv()
logger = setup_logging("chunks_to_qa")


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _answer_for_mode(answers: list[dict], mode: str) -> str:
    if mode == "current":
        for a in answers:
            if a["label"] == "current":
                return a["answer"]
    else:
        idx = int(mode.split("_")[-1])
        for a in answers:
            if a["label"] == "outdated" and a.get("outdated_index") == idx:
                return a["answer"]
    return ""


def expected_modes(record: dict) -> list[str]:
    modes = []
    for ans in record["answers"]:
        if ans["label"] == "current":
            modes.append("current")
        elif ans["label"] == "outdated":
            modes.append(f"outdated_{ans.get('outdated_index', 0)}")
    return modes


def build_prompt(record: dict, mode: str) -> str:
    target = _answer_for_mode(record["answers"], mode)
    lines: list[str] = []

    lines.append("[ORIGINAL QUESTION]")
    lines.append(record["question"])
    lines.append("")

    lines.append("[ANSWER CANDIDATES]")
    for ans in record["answers"]:
        tag = "current" if ans["label"] == "current" else f"outdated (index {ans.get('outdated_index', 0)})"
        lines.append(
            f"  - label={tag} | "
            f"last_modified_time={ans['last_modified_time']} | "
            f"answer={ans['answer']}"
        )
    lines.append("")

    lines.append("[CHUNKS]")
    for ch in record["chunks"]:
        label_tag = (
            f"outdated (index {ch.get('outdated_index', '?')})"
            if ch["label"] == "outdated" else ch["label"]
        )
        lines.append(
            f"chunk_id={ch['chunk_id']} | "
            f"label={label_tag} | "
            f"last_modified_time={ch['last_modified_time'] or 'N/A'}"
        )
        lines.append(ch["text"])
        lines.append("")

    lines.append("[TARGET MODE]")
    lines.append(f"mode: \"{mode}\"")
    lines.append(f"target_answer (fixed, use exactly as-is): \"{target}\"")
    lines.append("")

    lines.append("[INSTRUCTIONS]")
    lines.append(
        "Generate a single QA pair for the TARGET MODE above.\n"
        "1. new_question: Write a new question inspired by the original question that asks about "
        "information specific to that time period. "
        "Use a NARRATIVE temporal hint — a descriptive phrase such as 'when the policy was first announced', "
        "'at the time of the merger', 'shortly after the election', 'before the regulation took effect', etc.\n"
        "   *** CRITICAL: The new_question must NOT contain ANY numbers that look like years (e.g. 2024, 1999), "
        "month names with years (e.g. July 2024), quarter references (e.g. Q3 2023), "
        "or date formats (e.g. 07/2024). Use only descriptive, narrative phrases for temporal hints. "
        "If you include any year or date number, the output will be rejected. ***\n"
        "2. target_answer: Use the exact value specified in TARGET MODE above. Do not modify it.\n"
        "3. evidence_chunk_id: The chunk_id (integer) of the chunk that supports the target_answer. "
        "Must be a chunk_id that exists in the CHUNKS list above.\n"
        "4. reasoning: Explain why the evidence chunk supports the answer for this mode, "
        "and why chunks from other time periods would give incorrect answers — "
        "using date-based temporal reasoning."
    )
    lines.append("")
    lines.append("Respond strictly in the following JSON schema:")
    lines.append(
        '{"mode": str, "new_question": str, '
        '"target_answer": str, "evidence_chunk_id": int, "reasoning": str}'
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

_last_call_time: float = 0.0
_requests_per_minute: int = GEMINI_RPM


def _rate_limit_wait() -> None:
    global _last_call_time
    min_interval = 60.0 / _requests_per_minute
    elapsed = time.time() - _last_call_time
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)
    _last_call_time = time.time()


# ---------------------------------------------------------------------------
# API calls
# ---------------------------------------------------------------------------

def call_gemini(client, prompt: str) -> dict | None:
    from google.genai import types
    for attempt in range(MAX_API_RETRIES):
        try:
            _rate_limit_wait()
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.7,
                ),
            )
            return json.loads(response.text.strip())
        except Exception as e:
            if not _handle_api_error(e, attempt):
                return None
    logger.error("API max retries exhausted")
    return None


def call_gpt(client, prompt: str) -> dict | None:
    for attempt in range(MAX_API_RETRIES):
        try:
            _rate_limit_wait()
            response = client.chat.completions.create(
                model=GPT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.7,
            )
            return json.loads(response.choices[0].message.content.strip())
        except Exception as e:
            if not _handle_api_error(e, attempt):
                return None
    logger.error("API max retries exhausted")
    return None


def call_llm(client, prompt: str, provider: str) -> dict | None:
    if provider == "gemini":
        return call_gemini(client, prompt)
    elif provider == "gpt":
        return call_gpt(client, prompt)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def make_client(provider: str):
    if provider == "gemini":
        from google import genai
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY가 설정되지 않았습니다.")
        return genai.Client(api_key=api_key)
    elif provider == "gpt":
        import openai
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY가 설정되지 않았습니다.")
        return openai.OpenAI(api_key=api_key)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def _handle_api_error(e: Exception, attempt: int) -> bool:
    err_str  = str(e)
    err_type = type(e).__name__

    is_rate_limit  = any(x in err_str for x in ("429", "ResourceExhausted", "RESOURCE_EXHAUSTED", "RateLimitError"))
    is_server_err  = "500" in err_str or "503" in err_str
    is_network_err = any(t in err_type for t in (
        "ConnectError", "TimeoutException", "ReadTimeout",
        "ConnectTimeout", "RemoteProtocolError", "NetworkError",
    )) or "httpcore" in err_str or "httpx" in err_str

    if is_rate_limit:
        wait = min(30.0 * (2 ** attempt) + random.uniform(0, 2), 300.0)
        logger.warning("rate limit, retrying in %.1fs (attempt %d/%d)", wait, attempt + 1, MAX_API_RETRIES)
        time.sleep(wait)
        return True
    elif is_server_err or is_network_err:
        wait = min(5.0 * (2 ** attempt) + random.uniform(0, 2), 120.0)
        label = "network" if is_network_err else "server"
        logger.warning("%s error (%s), retrying in %.1fs (attempt %d/%d)", label, err_type, wait, attempt + 1, MAX_API_RETRIES)
        time.sleep(wait)
        return True
    else:
        logger.error("API fatal error (%s): %s", err_type, err_str[:200])
        return False


# ---------------------------------------------------------------------------
# current_raw: LLM 없이 원본 질문 그대로
# ---------------------------------------------------------------------------

def make_current_raw_pair(record: dict) -> dict | None:
    current_answer = _answer_for_mode(record["answers"], "current")
    if not current_answer:
        return None
    current_chunk = next(
        (ch for ch in record["chunks"] if ch["label"] == "current"), None
    )
    if current_chunk is None:
        return None
    return {
        "mode": "current_raw",
        "new_question": record["question"],
        "target_answer": current_answer,
        "evidence_chunk_id": current_chunk["chunk_id"],
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

_DATE_PATTERN = re.compile(
    r"\b(19|20)\d{2}\b"
    r"|\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*\d{4}\b"
    r"|\b(q[1-4]|quarter\s*[1-4])\s*(of\s*)?(19|20)\d{2}\b"
    r"|\b\d{1,2}/\d{4}\b",
    re.IGNORECASE,
)


def validate_pair(
    raw: dict,
    mode: str,
    valid_chunk_ids: set[int],
    record_answers: list[dict],
) -> tuple[dict | None, str]:
    """검증 통과 시 (pair, ""), 실패 시 (None, rejection_reason)."""
    if not isinstance(raw, dict):
        return None, "invalid response format"

    chunk_id = raw.get("evidence_chunk_id")
    if not isinstance(chunk_id, int) or chunk_id not in valid_chunk_ids:
        reason = f"invalid evidence_chunk_id={chunk_id}"
        logger.warning("mode=%s: %s", mode, reason)
        return None, reason

    new_q     = raw.get("new_question", "").strip()
    reasoning = raw.get("reasoning", "").strip()
    if not new_q or not reasoning:
        reason = "empty new_question or reasoning"
        logger.warning("mode=%s: %s", mode, reason)
        return None, reason

    date_match = _DATE_PATTERN.search(new_q)
    if date_match:
        reason = f"new_question contains explicit date/year '{date_match.group()}' — use narrative hints only"
        logger.warning("mode=%s: %s", mode, reason)
        return None, reason

    return {
        "mode": mode,
        "new_question": new_q,
        "target_answer": _answer_for_mode(record_answers, mode),
        "evidence_chunk_id": chunk_id,
        "reasoning": reasoning,
    }, ""


# ---------------------------------------------------------------------------
# Resume: 출력 파일에서 기처리 id 읽기
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
# Main pipeline
# ---------------------------------------------------------------------------

def chunks_to_qa(
    input_path: Path,
    output_path: Path,
    provider: str,
) -> None:
    global _requests_per_minute
    _requests_per_minute = GEMINI_RPM if provider == "gemini" else GPT_RPM

    output_path.parent.mkdir(parents=True, exist_ok=True)

    client = make_client(provider)
    done   = load_done_ids(output_path)
    logger.info("=== chunks_to_qa (%s) input=%s: %d already done ===", provider, input_path.name, len(done))

    with input_path.open(encoding="utf-8") as fin, \
         output_path.open("a", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue

            record    = json.loads(line)
            record_id = record["id"]

            if record_id in done:
                continue

            modes = expected_modes(record)
            if not modes:
                logger.warning("[%s] no valid modes, skipping", record_id)
                done.add(record_id)
                continue

            valid_chunk_ids = {ch["chunk_id"] for ch in record["chunks"]}

            # ── LLM 호출: current / outdated_* ──────────────────────────
            for mode in modes:
                pair_id = f"{record_id}_{mode}"
                if pair_id in done:
                    continue

                pair = None
                rejection = ""
                for attempt in range(MAX_PARTIAL_RETRIES):
                    prompt = build_prompt(record, mode)
                    if rejection:
                        prompt += (
                            f"\n\n[PREVIOUS ATTEMPT REJECTED]\n"
                            f"Reason: {rejection}\n"
                            f"Fix this issue and regenerate."
                        )
                    raw = call_llm(client, prompt, provider)
                    if raw is None:
                        break
                    pair, rejection = validate_pair(raw, mode, valid_chunk_ids, record["answers"])
                    if pair is not None:
                        break
                    logger.info("[%s] validation failed (%s), retry %d/%d", pair_id, rejection, attempt + 1, MAX_PARTIAL_RETRIES)

                if pair is None:
                    logger.warning("[%s] failed mode=%s, skipping", pair_id, mode)
                    done.add(pair_id)
                    continue

                fout.write(json.dumps({
                    "id": pair_id,
                    "hoh_source_idx": int(record_id.split("_")[1]),
                    "provider": provider,
                    "mode": pair["mode"],
                    "original_question": record["question"],
                    "new_question": pair["new_question"],
                    "target_answer": pair["target_answer"],
                    "evidence_chunk_id": pair["evidence_chunk_id"],
                    "chunks": record["chunks"],
                }, ensure_ascii=False) + "\n")
                fout.flush()
                done.add(pair_id)

            # ── LLM 없이: current_raw ────────────────────────────────────
            raw_pair_id = f"{record_id}_current_raw"
            if raw_pair_id not in done:
                raw_pair = make_current_raw_pair(record)
                if raw_pair is None:
                    logger.warning("[%s] current 청크 없음, skipping", raw_pair_id)
                else:
                    fout.write(json.dumps({
                        "id": raw_pair_id,
                        "hoh_source_idx": int(record_id.split("_")[1]),
                        "provider": "none",
                        "mode": "current_raw",
                        "original_question": record["question"],
                        "new_question": raw_pair["new_question"],
                        "target_answer": raw_pair["target_answer"],
                        "evidence_chunk_id": raw_pair["evidence_chunk_id"],
                        "chunks": record["chunks"],
                    }, ensure_ascii=False) + "\n")
                    fout.flush()
                done.add(raw_pair_id)

            if len(done) % 50 == 0:
                logger.info("%d pairs done", len(done))

    logger.info("Done → %s", output_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="hoh_chunks.jsonl → data/qa/ QA pairs")
    parser.add_argument(
        "--provider", type=str, default="gemini", choices=["gemini", "gpt"],
        help="LLM provider (기본값: gemini)"
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="입력 파일 경로. 생략 시 data/chunks/hoh_chunks.jsonl"
    )
    parser.add_argument(
        "--gpt-model", type=str, default=None,
        help=f"GPT 모델명 (기본값: {GPT_MODEL})"
    )
    parser.add_argument(
        "--gemini-model", type=str, default=None,
        help=f"Gemini 모델명 (기본값: {GEMINI_MODEL})"
    )
    args = parser.parse_args()

    if args.gpt_model:
        GPT_MODEL = args.gpt_model
    if args.gemini_model:
        GEMINI_MODEL = args.gemini_model

    input_path = Path(args.input) if args.input else CHUNKS_PATH
    if not input_path.exists():
        logger.error("입력 파일 없음: %s — 먼저 hoh_to_chunks.py 를 실행하세요.", input_path)
        exit(1)

    stem   = input_path.stem
    suffix = stem.replace("hoh_chunks", "")
    p      = args.provider

    chunks_to_qa(
        input_path=input_path,
        output_path=DIR_QA / f"hoh_qa_{p}{suffix}.jsonl",
        provider=p,
    )
