"""
data/qa/ → data/qa-reasoning/ (SFT cold-start용 reasoning 생성)

chunks_to_qa.py 결과물에서 LLM으로 reasoning을 생성하여
data/qa-reasoning/ 에 저장한다.

SFT 학습 시 <thought> 블록의 cold-start로 사용되며,
GRPO 단계에서 품질이 보정되므로 teacher reasoning이 완벽할 필요 없다.

reasoning 내용
--------------
  - 질문의 시간적 맥락과 각 chunk의 timestamp 비교
  - evidence chunk가 시간적으로 적합한 이유
  - 다른 chunk가 부적합한 이유 (시간적 충돌)

current_raw 모드는 narrative 시간 힌트가 없어 reasoning 학습에 부적합하므로 제외.

Provider
--------
--provider gpt    : GPT (기본값)
--provider gemini : Gemini
"""

import argparse
import json
import os
import random
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

from config import (
    DIR_QA, DIR_QA_REASONING,
    GPT_MODEL, GEMINI_MODEL, GPT_RPM, GEMINI_RPM,
    MAX_API_RETRIES, MAX_PARTIAL_RETRIES,
    setup_logging,
)

load_dotenv()
logger = setup_logging("generate_reasoning")


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_reasoning_prompt(record: dict) -> str:
    lines: list[str] = []

    lines.append("[QUESTION]")
    lines.append(record["new_question"])
    lines.append("")

    lines.append("[CHUNKS]")
    for ch in record["chunks"]:
        lmt = ch.get("last_modified_time") or "N/A"
        lines.append(
            f"chunk_id={ch['chunk_id']} | "
            f"label={ch['label']} | "
            f"last_modified_time={lmt}"
        )
        lines.append(ch["text"])
        lines.append("")

    lines.append("[CORRECT ANSWER]")
    lines.append(f"answer: \"{record['target_answer']}\"")
    lines.append(f"evidence_chunk_id: {record['evidence_chunk_id']}")
    lines.append("")

    lines.append("[INSTRUCTIONS]")
    lines.append(
        "You are a teacher model helping train a student to resolve temporal conflicts in RAG.\n"
        "Given the question, the chunks (each with a last_modified_time timestamp), "
        "and the correct answer with its evidence chunk, generate a step-by-step reasoning "
        "that a student model should produce.\n\n"
        "The reasoning must:\n"
        "1. Identify the temporal context implied by the question's narrative hint\n"
        "2. Compare each chunk's last_modified_time against that context\n"
        "3. Explain why the evidence chunk is temporally consistent with the question\n"
        "4. Explain why other chunks are temporally inconsistent (temporal conflict)\n"
        "5. Conclude with the correct answer selection\n\n"
        "Write the reasoning as a coherent inner monologue (2–5 sentences). "
        "Do NOT include explicit year-month numbers that are not already present in the chunks. "
        "Output only the reasoning string, no JSON wrapper."
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

_last_call_time: float = 0.0
_requests_per_minute: int = GPT_RPM


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

def call_gpt(client, prompt: str) -> str | None:
    for attempt in range(MAX_API_RETRIES):
        try:
            _rate_limit_wait()
            response = client.chat.completions.create(
                model=GPT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if not _handle_api_error(e, attempt):
                return None
    logger.error("API max retries exhausted")
    return None


def call_gemini(client, prompt: str) -> str | None:
    from google.genai import types
    for attempt in range(MAX_API_RETRIES):
        try:
            _rate_limit_wait()
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.7),
            )
            return response.text.strip()
        except Exception as e:
            if not _handle_api_error(e, attempt):
                return None
    logger.error("API max retries exhausted")
    return None


def call_llm(client, prompt: str, provider: str) -> str | None:
    if provider == "gpt":
        return call_gpt(client, prompt)
    elif provider == "gemini":
        return call_gemini(client, prompt)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def make_client(provider: str):
    if provider == "gpt":
        import openai
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY가 설정되지 않았습니다.")
        return openai.OpenAI(api_key=api_key)
    elif provider == "gemini":
        from google import genai
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY가 설정되지 않았습니다.")
        return genai.Client(api_key=api_key)
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

def generate_reasoning(
    input_path: Path,
    output_path: Path,
    provider: str,
    sample_ratio: float,
    sample_seed: int,
) -> None:
    global _requests_per_minute
    _requests_per_minute = GPT_RPM if provider == "gpt" else GEMINI_RPM

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 전체 레코드 로드 (current_raw 제외)
    records = []
    with input_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                if rec.get("mode") != "current_raw":
                    records.append(rec)

    total_before = len(records)

    # 샘플링
    if sample_ratio < 1.0:
        rng = random.Random(sample_seed)
        k = max(1, int(len(records) * sample_ratio))
        records = rng.sample(records, k)
        logger.info("샘플링: %d/%d records (ratio=%.2f, seed=%d)", k, total_before, sample_ratio, sample_seed)

    client = make_client(provider)
    done   = load_done_ids(output_path)
    logger.info("=== generate_reasoning (%s) input=%s: %d already done, %d target ===",
                provider, input_path.name, len(done), len(records))

    with output_path.open("a", encoding="utf-8") as fout:
        for i, record in enumerate(records, 1):
            record_id = record["id"]
            if record_id in done:
                continue

            reasoning = None
            for attempt in range(MAX_PARTIAL_RETRIES):
                result = call_llm(client, build_reasoning_prompt(record), provider)
                if result is None:
                    break
                if len(result) < 20:
                    logger.info("[%s] reasoning too short, retry %d/%d", record_id, attempt + 1, MAX_PARTIAL_RETRIES)
                    continue
                reasoning = result
                break

            if reasoning is None:
                logger.warning("[%s] reasoning 생성 실패, skipping", record_id)
                done.add(record_id)
                continue

            fout.write(json.dumps({**record, "reasoning": reasoning}, ensure_ascii=False) + "\n")
            fout.flush()
            done.add(record_id)

            if i % 10 == 0:
                logger.info("[%d/%d] %s done", i, len(records), record_id)

    logger.info("Done → %s", output_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="data/qa/ → data/qa-reasoning/ (SFT reasoning 생성)")
    parser.add_argument(
        "--provider", type=str, default="gpt", choices=["gpt", "gemini"],
        help="LLM provider (기본값: gpt)"
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="입력 파일 경로. 생략 시 data/qa/ 전체"
    )
    parser.add_argument(
        "--ratio", type=float, default=1.0,
        help="샘플링 비율 0.0~1.0 (기본값: 1.0). SFT는 10~30%% 권장"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="샘플링 seed (기본값: 42)"
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

    p = args.provider

    if args.input:
        input_paths = [Path(args.input)]
    else:
        input_paths = sorted(DIR_QA.glob("hoh_qa_*.jsonl"))
        if not input_paths:
            logger.error("%s/ 에 hoh_qa_*.jsonl 파일이 없습니다. chunks_to_qa.py 를 먼저 실행하세요.", DIR_QA)
            exit(1)

    for input_path in input_paths:
        # hoh_qa_gpt_0_25.jsonl → hoh_qa_gpt_0_25_reasoning_gpt.jsonl
        generate_reasoning(
            input_path=input_path,
            output_path=DIR_QA_REASONING / f"{input_path.stem}_reasoning_{p}.jsonl",
            provider=p,
            sample_ratio=args.ratio,
            sample_seed=args.seed,
        )
