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
--provider vllm    : vLLM 로컬 서버 (OPENAI_BASE_URL, --vllm-model 필수)
"""

import argparse
import json
import os
import random
import re
import threading
import time
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

from config import (
    DIR_CHUNKS, DIR_QA, CHUNKS_PATH,
    GEMINI_MODEL, GPT_MODEL, CLAUDE_MODEL, VLLM_CONCURRENCY,
    MAX_API_RETRIES, MAX_PARTIAL_RETRIES,
    get_model_alias, setup_logging,
)
from llm_client import (
    make_client, set_rpm,
    rate_limit_wait, handle_api_error,
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


def _evidence_chunk_id_for_mode(chunks: list[dict], mode: str) -> int | None:
    """mode에 해당하는 label의 청크 id를 자동 결정한다."""
    if mode == "current":
        for ch in chunks:
            if ch["label"] == "current":
                return ch["chunk_id"]
    else:
        idx = int(mode.split("_")[-1])
        for ch in chunks:
            if ch["label"] == "outdated" and ch.get("outdated_index") == idx:
                return ch["chunk_id"]
    return None


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
        "1. new_question: Rewrite a natural question (inspired by the original) whose answer is exactly the "
        "TARGET ANSWER, asking about the value at the TARGET MODE's time period.\n"
        "   - Anchor the time with ONE concrete, identifiable event or milestone from that period "
        "(e.g. 'before the team was relegated', 'when the company first turned a profit'). "
        "Do NOT use vague anchors like 'shortly after the summer' or 'around that time'.\n"
        "   - The question MUST be fully self-contained: do NOT refer to 'the following/named/listed players' "
        "or any list, table, or entity that is not written out inside the question itself.\n"
        "   - Do NOT put a temporal frame on attributes that never change (e.g. someone's nationality).\n"
        "   - Write the way a real person would naturally ask. One sentence.\n"
        "   *** CRITICAL: The new_question must NOT contain ANY year or explicit date (e.g. 2024, 1999, "
        "July 2024, Q3 2023, 07/2024) — use narrative words only. "
        "If you include any year or date number, the output will be rejected. ***\n"
        "2. target_answer: Use the exact value specified in TARGET MODE above. Do not modify it.\n"
        "3. reasoning: Explain why the evidence chunk supports the answer for this mode, "
        "and why chunks from other time periods would give incorrect answers."
    )
    lines.append("")
    lines.append("Respond strictly in the following JSON schema:")
    lines.append(
        '{"mode": str, "new_question": str, '
        '"target_answer": str, "reasoning": str}'
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# API calls
# ---------------------------------------------------------------------------

# 생성 토큰 누적 → 실험 03 중앙 원장에 기록 (전체 비용 모니터링용)
_GEN = {"input": 0, "output": 0, "calls": 0}


def _acc(in_tok, out_tok) -> None:
    _GEN["input"] += int(in_tok or 0)
    _GEN["output"] += int(out_tok or 0)
    _GEN["calls"] += 1


def _append_gen_ledger(provider: str, model_id: str) -> None:
    """생성 사용량을 repo 루트 usage/usage_ledger.jsonl 에 append (best-effort)."""
    if _GEN["calls"] == 0:
        return
    try:
        from datetime import datetime
        price = {"gpt": ("GPT_IN_PRICE", "GPT_OUT_PRICE"),
                 "gemini": ("GEMINI_IN_PRICE", "GEMINI_OUT_PRICE"),
                 "anthropic": ("CLAUDE_IN_PRICE", "CLAUDE_OUT_PRICE")}.get(provider)
        pin = float(os.environ.get(price[0], "0")) if price else 0.0
        pout = float(os.environ.get(price[1], "0")) if price else 0.0
        cost = _GEN["input"] / 1e6 * pin + _GEN["output"] / 1e6 * pout
        ledger = Path(__file__).resolve().parents[1] / "usage" / "usage_ledger.jsonl"
        ledger.parent.mkdir(parents=True, exist_ok=True)
        with ledger.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": datetime.now().isoformat(timespec="seconds"),
                                "script": "chunks_to_qa(gen)", "model": f"{provider}:{model_id}",
                                "calls": _GEN["calls"], "input": _GEN["input"], "output": _GEN["output"],
                                "cost": round(cost, 4)}, ensure_ascii=False) + "\n")
        logger.info("usage ledger: %d calls in=%d out=%d $%.4f", _GEN["calls"], _GEN["input"], _GEN["output"], cost)
        # 사람이 읽는 합계 파일(usage_summary.txt)도 갱신
        agg, tot = {}, [0, 0, 0, 0.0]
        for ln in ledger.open(encoding="utf-8"):
            if not ln.strip():
                continue
            r = json.loads(ln)
            a = agg.setdefault(r["model"], [0, 0, 0, 0.0])
            for i, k in enumerate(("calls", "input", "output", "cost")):
                a[i] += r.get(k, 0); tot[i] += r.get(k, 0)
        rows = [f"# 전체 토큰·비용 누적 (자동 갱신: {datetime.now().isoformat(timespec='seconds')})", "",
                f"{'model':<26}{'calls':>7}{'in_tok':>12}{'out_tok':>12}{'$cost':>10}"]
        for m, a in sorted(agg.items()):
            rows.append(f"{m:<26}{a[0]:>7}{a[1]:>12}{a[2]:>12}{('$%.4f' % a[3]):>10}")
        rows.append(f"{'TOTAL':<26}{tot[0]:>7}{tot[1]:>12}{tot[2]:>12}{('$%.4f' % tot[3]):>10}")
        (ledger.parent / "usage_summary.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
    except Exception:
        logger.exception("usage ledger 기록 실패(무시)")


def _loads_json_lenient(text: str) -> dict | None:
    """모델이 코드펜스(```json)로 감싸거나 앞뒤 잡텍스트를 붙여도 JSON 객체를 파싱한다.
    실패 시 원본 앞부분을 로그로 남기고 None을 반환(→ 상위에서 재시도/스킵)."""
    if not text:
        logger.warning("[gemini] 빈 응답")
        return None
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s).strip()

    def _coerce(obj):
        # 배열로 감싸 오면 첫 dict 원소를 꺼낸다
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, list):
            return next((el for el in obj if isinstance(el, dict)), None)
        return None

    try:
        got = _coerce(json.loads(s))
        if got is not None:
            return got
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", s, re.S)  # 가장 바깥 {...} 추출 후 재시도
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    logger.warning("[gemini] JSON 파싱 실패. raw head ↓\n%s", s[:800])
    return None


def call_gemini(client, prompt: str) -> dict | None:
    from google.genai import types
    # 단일 JSON 객체 형식을 강제(배열·코드펜스·형식 흔들림 방지)
    schema = types.Schema(
        type=types.Type.OBJECT,
        properties={
            "mode":          types.Schema(type=types.Type.STRING),
            "new_question":  types.Schema(type=types.Type.STRING),
            "target_answer": types.Schema(type=types.Type.STRING),
            "reasoning":     types.Schema(type=types.Type.STRING),
        },
        required=["new_question", "reasoning"],
    )
    for attempt in range(MAX_API_RETRIES):
        try:
            rate_limit_wait()
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema,
                    temperature=0.7,
                ),
            )
            try:
                um = response.usage_metadata
                _acc(getattr(um, "prompt_token_count", 0), getattr(um, "candidates_token_count", 0))
            except Exception:
                pass
            return _loads_json_lenient(response.text)
        except Exception as e:
            if not handle_api_error(e, attempt):
                return None
    logger.error("API max retries exhausted")
    return None


def call_openai_compat(client, prompt: str, model: str) -> dict | None:
    """OpenAI 호환 API 호출 (GPT, vLLM 공용)."""
    for attempt in range(MAX_API_RETRIES):
        try:
            rate_limit_wait()
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.7,
                max_tokens=1024,
            )
            try:
                u = response.usage
                _acc(getattr(u, "prompt_tokens", 0), getattr(u, "completion_tokens", 0))
            except Exception:
                pass
            return json.loads(response.choices[0].message.content.strip())
        except Exception as e:
            if not handle_api_error(e, attempt):
                return None
    logger.error("API max retries exhausted")
    return None


def call_anthropic(client, prompt: str) -> dict | None:
    """Anthropic(Claude) 호출. JSON mode가 없어 텍스트에서 JSON을 추출한다."""
    for attempt in range(MAX_API_RETRIES):
        try:
            rate_limit_wait()
            response = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=1024,
                temperature=0.7,
                messages=[{"role": "user", "content": prompt + "\n\nReturn ONLY the JSON object, nothing else."}],
            )
            try:
                u = response.usage
                _acc(getattr(u, "input_tokens", 0), getattr(u, "output_tokens", 0))
            except Exception:
                pass
            text = "".join(b.text for b in response.content if getattr(b, "type", "") == "text")
            m = re.search(r"\{.*\}", text, re.S)
            return json.loads(m.group()) if m else None
        except Exception as e:
            if not handle_api_error(e, attempt):
                return None
    logger.error("API max retries exhausted")
    return None


# 현재 사용 중인 모델명 (entry point에서 설정)
VLLM_MODEL: str = ""


def call_llm(client, prompt: str, provider: str) -> dict | None:
    if provider == "gemini":
        return call_gemini(client, prompt)
    elif provider == "gpt":
        return call_openai_compat(client, prompt, GPT_MODEL)
    elif provider == "anthropic":
        return call_anthropic(client, prompt)
    elif provider == "vllm":
        return call_openai_compat(client, prompt, VLLM_MODEL)
    else:
        raise ValueError(f"Unknown provider: {provider}")


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
    chunks: list[dict],
    record_answers: list[dict],
) -> tuple[dict | None, str]:
    """검증 통과 시 (pair, ""), 실패 시 (None, rejection_reason)."""
    if not isinstance(raw, dict):
        return None, "invalid response format"

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

    evidence_chunk_id = _evidence_chunk_id_for_mode(chunks, mode)
    if evidence_chunk_id is None:
        reason = f"no evidence chunk found for mode={mode}"
        logger.warning("mode=%s: %s", mode, reason)
        return None, reason

    return {
        "mode": mode,
        "new_question": new_q,
        "target_answer": _answer_for_mode(record_answers, mode),
        "evidence_chunk_id": evidence_chunk_id,
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

def _process_one_task(
    client,
    provider: str,
    record: dict,
    mode: str,
) -> dict | None:
    """단일 (record, mode) 태스크를 처리하고 결과 dict 또는 None을 반환한다."""
    record_id = record["id"]
    pair_id = f"{record_id}_{mode}"
    logger.debug("[%s] 호출 중...", pair_id)

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
        pair, rejection = validate_pair(raw, mode, record["chunks"], record["answers"])
        if pair is not None:
            break
        logger.info("[%s] validation failed (%s), retry %d/%d", pair_id, rejection, attempt + 1, MAX_PARTIAL_RETRIES)

    if pair is None:
        logger.warning("[%s] failed mode=%s, skipping", pair_id, mode)
        return None

    return {
        "id": pair_id,
        "hoh_source_idx": int(record_id.split("_")[1]),
        "provider": provider,
        "model": VLLM_MODEL if provider == "vllm" else (GPT_MODEL if provider == "gpt" else GEMINI_MODEL),
        "mode": pair["mode"],
        "original_question": record["question"],
        "new_question": pair["new_question"],
        "target_answer": pair["target_answer"],
        "evidence_chunk_id": pair["evidence_chunk_id"],
        "reasoning_qa": pair["reasoning"],
        "chunks": record["chunks"],
    }


def chunks_to_qa(
    input_path: Path,
    output_path: Path,
    provider: str,
) -> None:
    set_rpm(provider)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 로그용 모델명
    if provider == "vllm":
        model_label = f"vllm/{VLLM_MODEL}"
    elif provider == "gpt":
        model_label = f"gpt/{GPT_MODEL}"
    else:
        model_label = f"gemini/{GEMINI_MODEL}"

    client = make_client(provider)
    done   = load_done_ids(output_path)
    logger.info("=== chunks_to_qa (%s) input=%s: %d already done ===", model_label, input_path.name, len(done))

    # ── 전체 태스크 수집 ──────────────────────────────────────────────
    tasks = []       # (record, mode) — LLM 호출 필요
    raw_tasks = []   # record         — current_raw (LLM 불필요)

    with input_path.open(encoding="utf-8") as fin:
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

            for mode in modes:
                pair_id = f"{record_id}_{mode}"
                if pair_id not in done:
                    tasks.append((record, mode))

            raw_pair_id = f"{record_id}_current_raw"
            if raw_pair_id not in done:
                raw_tasks.append(record)

    logger.info("pending: %d LLM tasks, %d current_raw tasks", len(tasks), len(raw_tasks))

    # ── 결과 기록 (thread-safe) ───────────────────────────────────────
    write_lock = threading.Lock()

    def _write_result(fout, result: dict) -> None:
        with write_lock:
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            fout.flush()
            done.add(result["id"])

    with output_path.open("a", encoding="utf-8") as fout:
        # ── current_raw (LLM 불필요, 즉시 처리) ──────────────────────
        for record in raw_tasks:
            raw_pair = make_current_raw_pair(record)
            raw_pair_id = f"{record['id']}_current_raw"
            if raw_pair is None:
                logger.warning("[%s] current 청크 없음, skipping", raw_pair_id)
            else:
                _write_result(fout, {
                    "id": raw_pair_id,
                    "hoh_source_idx": int(record["id"].split("_")[1]),
                    "provider": "none",
                    "mode": "current_raw",
                    "original_question": record["question"],
                    "new_question": raw_pair["new_question"],
                    "target_answer": raw_pair["target_answer"],
                    "evidence_chunk_id": raw_pair["evidence_chunk_id"],
                    "chunks": record["chunks"],
                })
            done.add(raw_pair_id)

        # ── LLM 호출: 병렬 (vLLM) 또는 순차 (gemini/gpt) ────────────
        use_parallel = provider == "vllm" and VLLM_CONCURRENCY > 1
        max_workers = VLLM_CONCURRENCY if use_parallel else 1

        completed = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_process_one_task, client, provider, record, mode): (record, mode)
                for record, mode in tasks
            }
            for future in as_completed(futures):
                record, mode = futures[future]
                pair_id = f"{record['id']}_{mode}"
                try:
                    result = future.result()
                except Exception:
                    logger.exception("[%s] unexpected error", pair_id)
                    result = None

                if result is not None:
                    _write_result(fout, result)
                else:
                    with write_lock:
                        done.add(pair_id)

                completed += 1
                status = "ok" if result is not None else "skip"
                logger.info("[%d/%d] %s (%s)", completed, len(tasks), pair_id, status)

    logger.info("Done → %s", output_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="hoh_chunks.jsonl → data/qa/ QA pairs")
    parser.add_argument(
        "--provider", type=str, default="gemini", choices=["gemini", "gpt", "anthropic", "vllm"],
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
    parser.add_argument(
        "--claude-model", type=str, default=None,
        help=f"Claude 모델명 (--provider anthropic 시, 기본값: {CLAUDE_MODEL})"
    )
    parser.add_argument(
        "--vllm-model", type=str, default=None,
        help="vLLM 모델명 (--provider vllm 시 필수, 예: Qwen/Qwen3-32B)"
    )
    args = parser.parse_args()

    if args.vllm_model and args.provider != "vllm":
        args.provider = "vllm"

    if args.gpt_model:
        GPT_MODEL = args.gpt_model
    if args.gemini_model:
        GEMINI_MODEL = args.gemini_model
    if args.claude_model:
        CLAUDE_MODEL = args.claude_model

    if args.provider == "vllm":
        if not args.vllm_model:
            parser.error("--provider vllm 사용 시 --vllm-model 이 필수입니다.")
        VLLM_MODEL = args.vllm_model

    input_path = Path(args.input) if args.input else CHUNKS_PATH
    if not input_path.exists():
        logger.error("입력 파일 없음: %s — 먼저 hoh_to_chunks.py 를 실행하세요.", input_path)
        exit(1)

    stem   = input_path.stem
    suffix = stem.replace("chunks", "")

    if args.provider == "vllm":
        model_tag = get_model_alias(args.vllm_model)
    else:
        model_tag = get_model_alias(args.provider)

    # 생성 사용량은 중단(Ctrl-C)·예외에도 항상 기록되도록 finally로 보장
    _model_id = {"gpt": GPT_MODEL, "gemini": GEMINI_MODEL,
                 "anthropic": CLAUDE_MODEL, "vllm": VLLM_MODEL}.get(args.provider, args.provider)
    try:
        chunks_to_qa(
            input_path=input_path,
            output_path=DIR_QA / f"qa_{model_tag}{suffix}.jsonl",
            provider=args.provider,
        )
    finally:
        _append_gen_ledger(args.provider, _model_id)
