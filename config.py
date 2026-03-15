"""
공통 설정.

디렉토리 경로, LLM 모델명, API rate limit, 재시도 횟수 등
파이프라인 전체에서 공유하는 상수를 정의한다.
"""

import logging
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# 디렉토리
# ---------------------------------------------------------------------------

DIR_CHUNKS       = Path("data/chunks")
DIR_QA           = Path("data/qa")
DIR_QA_REASONING = Path("data/qa-reasoning")
DIR_EVAL         = Path("data/eval")
DIR_EVAL_SUMMARY = Path("data/eval_summary")
DIR_LOGS         = Path("logs")

CHUNKS_PATH = DIR_CHUNKS / "hoh_chunks.jsonl"

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

# 데이터셋 생성용 (chunks_to_qa, generate_reasoning)
GEMINI_MODEL = "gemini-2.5-flash"
GPT_MODEL    = "gpt-4.1-mini"

# 평가용 (evaluate_llm)
EVAL_GEMINI_MODEL = "gemini-2.5-flash"
EVAL_GPT_MODEL    = "gpt-4.1"

# ---------------------------------------------------------------------------
# Rate limit (requests per minute)
# ---------------------------------------------------------------------------

GEMINI_RPM = 10   # free tier 기준
GPT_RPM    = 60

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
