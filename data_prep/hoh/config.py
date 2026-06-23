"""
공통 설정.

디렉토리 경로, LLM 모델명, API rate limit, 재시도 횟수 등
파이프라인 전체에서 공유하는 상수를 정의한다.
"""

import logging
import os
from datetime import datetime
from pathlib import Path

# repo 루트 = data_prep/hoh/config.py 에서 두 단계 위
_REPO = Path(__file__).resolve().parents[2]

# .env 로드 → 아래 os.getenv 들이 .env 값을 반영(생성 파이프라인도 .env의 모델/RPM 사용)
try:
    from dotenv import load_dotenv
    load_dotenv(_REPO / ".env")
except Exception:
    pass

# ---------------------------------------------------------------------------
# 디렉토리 (HoH 가공 입출력은 data/hoh/ 아래로 고정 — 실행 위치 무관)
# ---------------------------------------------------------------------------

_HOH             = _REPO / "data" / "hoh"
DIR_CHUNKS       = _HOH / "chunks"
DIR_QA           = _HOH / "qa"
DIR_QA_REASONING = _HOH / "qa-reasoning"
DIR_EVAL         = _REPO / "data" / "eval"          # 옛 실험(01) 평가 산출
DIR_EVAL_SUMMARY = _REPO / "data" / "eval_summary"
DIR_LOGS         = Path(__file__).resolve().parent / "logs"   # data_prep/hoh/logs (생성 코드 옆)

CHUNKS_PATH = DIR_CHUNKS / "chunks.jsonl"

# ---------------------------------------------------------------------------
# 모델 alias (파일명용)
# ---------------------------------------------------------------------------

MODEL_ALIAS = {
    "meta-llama-3.1-70b-instruct-awq-int4": "llama3_1-70b-awq",
    "meta-llama-31-70b-instruct-awq-int4": "llama3_1-70b-awq",
    "gpt-4.1": "gpt4_1",
    "gpt-41": "gpt4_1",
    "gpt-4.1-mini": "gpt4_1-mini",
    "anthropic": "claude",
}


def get_model_alias(model_name: str) -> str:
    """모델명을 파일명용 짧은 alias로 변환한다. 사전에 없으면 점(.)을 언더바(_)로 바꾼다."""
    tag = model_name.split("/")[-1].lower()
    if tag in MODEL_ALIAS:
        return MODEL_ALIAS[tag]
    return tag.replace(".", "_")

# ---------------------------------------------------------------------------
# 청킹
# ---------------------------------------------------------------------------

CHUNK_SIZE = 5
STRIDE     = 3
MAX_CHUNKS = 10   # 초과 시 distractor 우선 제거

# ---------------------------------------------------------------------------
# Wikipedia
# ---------------------------------------------------------------------------

WIKI_API        = "https://en.wikipedia.org/w/api.php"
WIKI_SLEEP      = 0.5   # seconds between API calls
WIKI_USER_AGENT = (
    "temporal-conflict-qa/1.0 "
    "(https://github.com/your-repo; your@email.com) python-requests"
)

# ---------------------------------------------------------------------------
# LLM 모델
# ---------------------------------------------------------------------------

# 데이터셋 생성용 (chunks_to_qa, generate_reasoning) — .env로 override 가능
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview")
GPT_MODEL    = os.getenv("GPT_MODEL", "gpt-5.5")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-4-8")

# 평가용 (evaluate_llm)
EVAL_GEMINI_MODEL = "gemini-2.5-flash"
EVAL_GPT_MODEL    = "gpt-4.1"

# ---------------------------------------------------------------------------
# Rate limit (requests per minute)
# ---------------------------------------------------------------------------

# 블랙박스 유료 기준 기본값. 실제 account tier에 맞게 .env로 override 가능.
GEMINI_RPM       = int(os.getenv("GEMINI_RPM", "60"))
GPT_RPM          = int(os.getenv("GPT_RPM", "100"))
ANTHROPIC_RPM    = int(os.getenv("ANTHROPIC_RPM", "50"))
VLLM_CONCURRENCY = int(os.getenv("VLLM_CONCURRENCY", "8"))  # vLLM 병렬 요청 수

# ---------------------------------------------------------------------------
# 재시도 / 체크포인트
# ---------------------------------------------------------------------------

MAX_API_RETRIES     = 6
MAX_PARTIAL_RETRIES = 3


# ---------------------------------------------------------------------------
# 로깅
# ---------------------------------------------------------------------------

def setup_logging(name: str) -> logging.Logger:
    """
    콘솔(INFO) + 파일(DEBUG) 핸들러를 가진 logger를 반환한다.
    로그 파일: logs/{name}/{YYYYMMDD_HHMMSS}.log
    """
    log_dir = DIR_LOGS / name
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file  = log_dir / f"{name}_{timestamp}.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger

    # 콘솔: INFO, 간결한 포맷
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console)

    # 파일: DEBUG, 타임스탬프 포함
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(fh)

    return logger
