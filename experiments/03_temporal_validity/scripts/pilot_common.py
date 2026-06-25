"""
Temporal Validity 파일럿 — 공통 유틸 (프롬프트·파싱·정규화·모델 호출·judge).

데이터 레코드 스키마 (data/qa/*.jsonl):
  {id, source_idx, mode, new_question, target_answer, evidence_chunk_id,
   chunks:[{chunk_id, label:"current|outdated|outdated_i|distractor", text, last_modified_time}]}

설계 근거: 파일럿_실험계획_TemporalValidity.md §5~§7, §12
"""
from __future__ import annotations
import os, re, json, random, string
from typing import Any


def _load_dotenv():
    """의존성 없이 상위 경로의 .env를 로드(이미 설정된 환경변수는 보존 — shell export 우선)."""
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(6):
        p = os.path.join(d, ".env")
        if os.path.isfile(p):
            for line in open(p, encoding="utf-8"):
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
            return p
        d = os.path.dirname(d)
    return None


_DOTENV = _load_dotenv()  # 임포트 시 .env 자동 로드(있으면)

# ---------------------------------------------------------------- 정규화/파싱
_ARTICLES = {"a", "an", "the"}
_PUNCT = str.maketrans("", "", string.punctuation)


def normalize(s: str | None) -> str:
    """EM 정규화: 소문자 → 구두점·관사 제거 → 공백 정규화."""
    if not s:
        return ""
    s = s.lower().translate(_PUNCT)
    toks = [t for t in s.split() if t not in _ARTICLES]
    return " ".join(toks)


def ans_equiv(a: str | None, b: str | None) -> bool:
    """답 동등성(무료 의미매칭 근사): 정규화 동일 | 부분문자열 | 토큰 자카드≥0.6.
    정식명↔약칭("Tottenham Hotspur"↔"...F.C."), 뒷붙음("Jordi Pujol"↔"...i Soley") 흡수."""
    na, nb = normalize(a), normalize(b)
    if not na or not nb:
        return False
    if na == nb or na in nb or nb in na:
        return True
    ta, tb = set(na.split()), set(nb.split())
    return len(ta & tb) / len(ta | tb) >= 0.6


def f1(pred: str, gold: str) -> float:
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return float(p == g)
    common = {}
    for t in p:
        common[t] = min(p.count(t), g.count(t))
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    prec, rec = overlap / len(p), overlap / len(g)
    return 2 * prec * rec / (prec + rec)


_TAG = lambda name, txt: re.search(rf"<{name}>(.*?)</{name}>", txt, re.S | re.I)


def parse_output(text: str) -> dict:
    """<reasoning>, <answer> 추출 + answer 내 [k] 인용 번호 파싱."""
    reasoning = _TAG("reasoning", text)
    answer_blk = _TAG("answer", text)
    raw_ans = answer_blk.group(1).strip() if answer_blk else text.strip()
    cites = [int(x) for x in re.findall(r"\[(\d+)\]", raw_ans)]
    answer = re.sub(r"\[\d+\]", "", raw_ans).strip()
    return {
        "reasoning": reasoning.group(1).strip() if reasoning else "",
        "answer": answer,
        "cite_indices": cites,  # 1-based document display index
        "raw": text,
    }


# ---------------------------------------------------------------- 컨텍스트/조건
def _is_current(label: str) -> bool:
    return label == "current"


def _is_outdated(label: str) -> bool:
    return label.startswith("outdated")


def filter_chunks(chunks: list[dict], condition: str) -> list[dict]:
    """condition ∈ {conflict, current_only, outdated_only} (§5)."""
    if condition == "conflict":
        return list(chunks)
    if condition == "current_only":   # outdated 제거
        return [c for c in chunks if not _is_outdated(c["label"])]
    if condition == "outdated_only":  # current 제거
        return [c for c in chunks if not _is_current(c["label"])]
    raise ValueError(condition)


SYSTEM_PROMPT = (
    "You are given a set of documents, each with an index and a modification timestamp.\n"
    "Use the timestamps to identify the document whose information is valid for the time\n"
    "frame the question refers to, and base your answer on that document.\n"
    "Cite the supporting document inline using its index in square brackets.\n"
    "Output EXACTLY these two tags and nothing else:\n"
    "<reasoning> brief reasoning about which document is time-appropriate </reasoning>\n"
    "<answer> short answer [index] </answer>\n"
    "Example: <answer> Saccharomyces bulderi [2] </answer>"
)


def build_user_message(question: str, chunks: list[dict], rng: random.Random):
    """청크를 무작위 순서로 렌더링 + (1-based display index -> chunk_id) 매핑 반환."""
    shuffled = list(chunks)
    rng.shuffle(shuffled)
    idx2chunk = {}
    lines = [f"[Query] {question}", ""]
    for k, c in enumerate(shuffled, start=1):
        idx2chunk[k] = c["chunk_id"]
        ts = c.get("last_modified_time") or "N/A"
        lines.append(f"[Document {k}] [modified: {ts}]")
        lines.append(c["text"])
        lines.append("")
    return "\n".join(lines), idx2chunk


# ---------------------------------------------------------------- 모델 호출
# 친숙한 이름 -> (backend, model_id). 실제 ID는 환경에 맞게 조정.
MODELS: dict[str, tuple[str, str]] = {
    # 블랙박스 (최신 프론티어, 2026-06)
    "gpt":     ("openai",    os.getenv("GPT_MODEL", "gpt-5.5")),
    "claude":  ("anthropic", os.getenv("CLAUDE_MODEL", "claude-opus-4-8")),
    "gemini":  ("gemini",    os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview")),
    # 오픈 (단일 H100, vLLM) — eval 생성자가 Llama-3.1-70B → Llama 계열은 자가생성 편향으로 제외
    "qwen3_32b":     ("vllm", os.getenv("QWEN3_32B_MODEL", "Qwen/Qwen3-32B")),
    "qwen3_8b":      ("vllm", os.getenv("QWEN3_8B_MODEL", "Qwen/Qwen3-8B")),
    "mistral_small": ("vllm", os.getenv("MISTRAL_SMALL_MODEL", "mistralai/Mistral-Small-3.2-24B-Instruct-2506")),
}


# ---------------------------------------------------------------- 비용/토큰
# USD per 1M tokens (input, output). ★ 반드시 각 프로바이더 최신 가격으로 갱신할 것
# (환경변수로 주입 가능). 로컬 vLLM 오픈모델은 0.
# 모델 id·단가 모두 .env에서 관리(단일 소스). 코드엔 하드코딩 없음.
PRICING: dict[str, tuple[float, float]] = {
    "gpt":     (float(os.getenv("GPT_IN_PRICE", "0")),    float(os.getenv("GPT_OUT_PRICE", "0"))),
    "claude":  (float(os.getenv("CLAUDE_IN_PRICE", "0")), float(os.getenv("CLAUDE_OUT_PRICE", "0"))),
    "gemini":  (float(os.getenv("GEMINI_IN_PRICE", "0")), float(os.getenv("GEMINI_OUT_PRICE", "0"))),
    "qwen3_32b": (0.0, 0.0),
    "qwen3_8b": (0.0, 0.0),
    "mistral_small": (0.0, 0.0),
}

USAGE: dict[str, dict[str, int]] = {}  # model -> {input, output, calls}

try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text: str) -> int:
        return len(_ENC.encode(text or ""))
except Exception:  # tiktoken 미설치 시 근사(≈4 chars/token)
    def count_tokens(text: str) -> int:
        return max(1, len(text or "") // 4)


def record_usage(model: str, in_tok: int, out_tok: int):
    u = USAGE.setdefault(model, {"input": 0, "output": 0, "calls": 0})
    u["input"] += in_tok or 0
    u["output"] += out_tok or 0
    u["calls"] += 1


def cost_usd(model: str, in_tok: int, out_tok: int) -> float:
    pin, pout = PRICING.get(model, (0.0, 0.0))
    return in_tok / 1e6 * pin + out_tok / 1e6 * pout


def cost_report() -> str:
    lines = [f"{'model':<12}{'calls':>6}{'in_tok':>10}{'out_tok':>10}{'$est':>10}"]
    total = 0.0
    for m, u in USAGE.items():
        c = cost_usd(m, u["input"], u["output"]); total += c
        lines.append(f"{m:<12}{u['calls']:>6}{u['input']:>10}{u['output']:>10}{('$%.4f'%c):>10}")
    lines.append(f"{'TOTAL':<12}{'':>6}{'':>10}{'':>10}{('$%.4f'%total):>10}")
    return "\n".join(lines)


# ---------------------------------------------------------------- 중앙 원장(repo 루트 usage/, 실험 무관 전체 누적)
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
LEDGER = os.path.join(_REPO_ROOT, "usage", "usage_ledger.jsonl")


def append_ledger(script: str, path: str = LEDGER) -> None:
    """이번 실행의 USAGE를 중앙 원장에 append (모델별 1줄). 모든 스크립트가 같은 파일에 누적."""
    from datetime import datetime
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ts = datetime.now().isoformat(timespec="seconds")
    with open(path, "a", encoding="utf-8") as f:
        for m, u in USAGE.items():
            f.write(json.dumps({"ts": ts, "script": script, "model": m, "calls": u["calls"],
                                "input": u["input"], "output": u["output"],
                                "cost": round(cost_usd(m, u["input"], u["output"]), 4)},
                               ensure_ascii=False) + "\n")
    write_summary(path)  # 사람이 읽는 합계 파일도 자동 갱신


def ledger_total(path: str = LEDGER) -> str:
    """원장 전체 합계(모델별 + 총계)."""
    if not os.path.exists(path):
        return "(원장 없음 — 아직 실행 기록 없음)"
    agg, tot = {}, {"calls": 0, "input": 0, "output": 0, "cost": 0.0}
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        a = agg.setdefault(r["model"], {"calls": 0, "input": 0, "output": 0, "cost": 0.0})
        for k in ("calls", "input", "output", "cost"):
            a[k] += r.get(k, 0); tot[k] += r.get(k, 0)
    out = [f"{'model':<22}{'calls':>7}{'in_tok':>12}{'out_tok':>12}{'$cost':>10}"]
    for m, a in sorted(agg.items()):
        out.append(f"{m:<22}{a['calls']:>7}{a['input']:>12}{a['output']:>12}{('$%.4f'%a['cost']):>10}")
    out.append("-" * 63)
    out.append(f"{'TOTAL':<22}{tot['calls']:>7}{tot['input']:>12}{tot['output']:>12}{('$%.4f'%tot['cost']):>10}")
    return "\n".join(out)


# 사람이 읽는 합계 — 매 실행 끝에 자동 갱신(이 파일만 열면 됨). repo 루트 usage/
SUMMARY = os.path.join(_REPO_ROOT, "usage", "usage_summary.txt")


def write_summary(ledger_path: str = LEDGER, out_path: str = SUMMARY) -> None:
    from datetime import datetime
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# 전체 토큰·비용 누적 (자동 갱신: {datetime.now().isoformat(timespec='seconds')})\n\n")
        f.write(ledger_total(ledger_path) + "\n")


def _retry(fn, n=4):
    """일시적 Connection error/rate-limit/5xx 재시도 (백오프 1.5·3·4.5초).
    인증·권한·잘못된요청·모델없음(400/401/403/404)은 재시도 무의미 → 즉시 실패."""
    import time
    for i in range(n):
        try:
            return fn()
        except Exception as e:
            if getattr(e, "status_code", None) in (400, 401, 403, 404):
                raise   # 키 오류 등 — 재시도해도 동일, 27분 낭비 방지
            if i == n - 1:
                raise
            time.sleep(1.5 * (i + 1))


def call_model(name: str, system: str, user: str, temperature: float = 0.0,
               max_tokens: int = 4096) -> str:   # thinking 모델(Gemini 3.1 Pro 등): 추론+답+인용 여유 (2048→4096)
    """모델 호출 + usage(토큰) 자동 누적(USAGE). 비용은 cost_report()로 확인."""
    backend, model_id = MODELS[name]
    if backend in ("openai", "vllm", "gemini"):  # 모두 OpenAI 호환 client 재사용
        from openai import OpenAI
        if backend == "openai":
            client = OpenAI()  # OPENAI_API_KEY
            # GPT-5.x 등 추론모델: max_tokens→max_completion_tokens, temperature 커스텀 미지원(생략)
            extra = {"max_completion_tokens": max_tokens}
        elif backend == "gemini":  # Gemini의 OpenAI 호환 엔드포인트
            client = OpenAI(api_key=os.getenv("GEMINI_API_KEY"),
                            base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
            extra = {"max_tokens": max_tokens, "temperature": temperature}
        else:  # vllm (로컬 또는 ngrok 터널)
            client = OpenAI(base_url=os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
                            api_key=os.getenv("VLLM_API_KEY", "EMPTY"),
                            default_headers={"ngrok-skip-browser-warning": "1"})  # ngrok 무료 경고 우회
            extra = {"max_tokens": max_tokens, "temperature": temperature}
            if "qwen3" in model_id.lower():  # Qwen3 thinking 끄기 → <think> 없이 인용 파싱 깔끔
                extra["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        r = _retry(lambda: client.chat.completions.create(
            model=model_id,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}], **extra))
        u = getattr(r, "usage", None)
        record_usage(name, getattr(u, "prompt_tokens", 0), getattr(u, "completion_tokens", 0))
        return r.choices[0].message.content or ""
    if backend == "anthropic":
        import anthropic
        client = anthropic.Anthropic()  # ANTHROPIC_API_KEY
        r = _retry(lambda: client.messages.create(   # Opus 4.8은 temperature deprecated → 미전달
            model=model_id, max_tokens=max_tokens,
            system=system, messages=[{"role": "user", "content": user}]))
        u = getattr(r, "usage", None)
        record_usage(name, getattr(u, "input_tokens", 0), getattr(u, "output_tokens", 0))
        return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
    raise ValueError(backend)


def judge_support(judge_name: str, passage: str, answer: str) -> int:
    """Citation Precision foil (ALCE식): 인용 청크가 답을 함의(support)하는가? (시점 무관)"""
    prompt = (f'Passage: "{passage}"\n'
              f'Answer: "{answer}"\n'
              'Does the passage state or directly support the Answer? Reply only "yes" or "no".')
    out = call_model(judge_name, "You are a strict factual-support judge.", prompt,
                     temperature=0.0, max_tokens=4).strip().lower()
    return 1 if out.startswith("y") else 0


# ---------------------------------------------------------------- IO
def read_jsonl(path: str):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
