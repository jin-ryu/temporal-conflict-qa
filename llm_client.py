"""
LLM API 공통 모듈.

make_client, rate limiter, error handling 등
scripts/chunks_to_qa.py와 eval/evaluate_llm.py에서 공유한다.
"""

import logging
import os
import random
import threading
import time

from config import GEMINI_RPM, GPT_RPM, MAX_API_RETRIES, VLLM_CONCURRENCY

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

_last_call_time: float = 0.0
_rate_lock = threading.Lock()
_requests_per_minute: int = GPT_RPM


def set_rpm(provider: str) -> None:
    """provider에 따라 RPM을 설정한다. vLLM은 로컬 서버이므로 제한 없음."""
    global _requests_per_minute
    if provider == "vllm":
        _requests_per_minute = 999_999
    elif provider == "gemini":
        _requests_per_minute = GEMINI_RPM
    else:
        _requests_per_minute = GPT_RPM


def rate_limit_wait() -> None:
    global _last_call_time
    with _rate_lock:
        min_interval = 60.0 / _requests_per_minute
        elapsed = time.time() - _last_call_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        _last_call_time = time.time()


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def handle_api_error(e: Exception, attempt: int) -> bool:
    """API 에러를 처리하고 재시도 여부를 반환한다."""
    err_str = str(e)
    err_type = type(e).__name__

    is_rate_limit = any(x in err_str for x in ("429", "ResourceExhausted", "RESOURCE_EXHAUSTED", "RateLimitError"))
    is_server_err = "500" in err_str or "503" in err_str
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
# Client factory
# ---------------------------------------------------------------------------

def make_client(provider: str):
    """provider에 따라 LLM 클라이언트를 생성한다."""
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
    elif provider == "vllm":
        import openai
        base_url = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
        return openai.OpenAI(api_key="dummy", base_url=base_url)
    else:
        raise ValueError(f"Unknown provider: {provider}")
