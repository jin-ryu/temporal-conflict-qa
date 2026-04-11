# Temporal Conflict QA Dataset

RAG 시스템에서 시간적 충돌(temporal conflict)을 다루는 능력을 평가/학습하기 위한 데이터셋 생성 파이프라인 및 실험 프레임워크.

[russwest404/HoH-QAs](https://huggingface.co/datasets/russwest404/HoH-QAs)를 기반으로, Wikipedia 히스토리컬 revision을 가져와 청크를 만들고, LLM(GPT/Gemini/vLLM)을 통해 시간적 추론이 필요한 새로운 QA pair를 생성한다.

---

## 파이프라인 개요

```
[HoH-QAs 데이터셋]
        │
        ▼
  hoh_to_chunks.py  --start N --end M
  Wikipedia historical revision fetch
  → 문장 분리 → 슬라이딩 윈도우 청크
  → current / outdated / distractor 라벨링
        │
        ▼  data/chunks/chunks_N_M.jsonl
        │
        ▼
  chunks_to_qa.py  --input data/chunks/chunks_N_M.jsonl --provider gpt|gemini|vllm
  LLM 호출 → 시간 힌트 포함 새 질문 생성 + evidence_chunk_id 할당
        │
        ▼  data/qa/qa_{model_alias}_N_M.jsonl
        │
        ▼
  generate_reasoning.py  --provider gpt|gemini [--ratio 0.3]
  Teacher LLM → SFT cold-start용 reasoning 생성
        │
        ▼  data/qa-reasoning/qa_{model_alias}_N_M_reasoning_{provider}.jsonl
        │
        ▼
  merge_shards.py  --step 1|2|3
  범위별 shard 파일을 하나로 병합
```

> **모델 alias**: 파일명에 전체 모델명 대신 짧은 alias를 사용한다. `config.py`의 `MODEL_ALIAS` 참조.
> 예: `meta-llama-3.1-70b-instruct-awq-int4` → `llama70b`, `gpt-4.1` → `gpt41`

---

## 디렉토리 구조

```
temporal-conflict-qa/
├── config.py                  # 공통 설정 (디렉토리, 모델, RPM, 청킹 등)
├── llm_client.py              # LLM API 공통 모듈 (rate limiter, 에러 핸들링)
├── scripts/                   # 데이터 생성 파이프라인 스크립트
│   ├── hoh_to_chunks.py       # Wikipedia 청크 생성
│   ├── chunks_to_qa.py        # LLM QA pair 생성 (GPT/Gemini/vLLM)
│   ├── generate_reasoning.py  # SFT reasoning 생성 (ablation용)
│   ├── merge_shards.py        # 범위별 결과 병합
│   ├── sort_by_id.py          # JSONL 파일 id 기준 정렬
│   └── run_pipeline.py        # 파이프라인 자동 실행 (chunks → qa)
├── experiments/               # 실험 및 평가 프레임워크 (독립적 관리)
│   ├── 01_pilot_initial_eval/ # [실험 1] 대규모 자동 평가 히스토리
│   │   ├── scripts/           # evaluate_llm.py, summarize_eval.py
│   │   ├── results/           # 개별 실험 결과 (jsonl)
│   │   └── metrics/           # 집계된 리포트 (json)
│   └── 02_pilot_closed_source/# [실험 2] Gemini 3 Flash 파일럿 (수동/정밀)
│       ├── scripts/           # 샘플링, 템플릿 생성, 평가 스크립트
│       ├── data/              # 샘플링된 100개 데이터
│       ├── results/           # 실제 기록될 결과 (exp_a.json, exp_b.json)
│       └── metrics/           # 분석 리포트 (summary_exp_a.json, summary_exp_b.json)
├── data/                      # 생성된 원본 데이터셋
├── docs/                      # 실험 계획 및 아키텍처 문서
└── logs/                      # 실행 로그
```

> `data/` 및 `logs/` 디렉토리는 스크립트 실행 시 자동 생성된다.

---

## 데이터 구조

### hoh_to_chunks.py 출력: `data/chunks/chunks_N_M.jsonl`

```json
{
  "id": "hoh_000000",
  "hoh_source_idx": 0,
  "question": "What is the name of the yeast that can ferment gluconolactone?",
  "answers": [
    {"label": "current",  "answer": "Maudiozyma bulderi",    "last_modified_time": "2024-07-01 00:00:00"},
    {"label": "outdated", "answer": "Saccharomyces bulderi", "last_modified_time": "2024-06-01", "outdated_index": 0}
  ],
  "chunks": [
    {"chunk_id": 0,  "label": "distractor", "text": "...",                        "last_modified_time": "2024-07-01 00:00:00"},
    {"chunk_id": 12, "label": "current",    "text": "...Maudiozyma bulderi...",    "last_modified_time": "2024-07-01 00:00:00"},
    {"chunk_id": 31, "label": "outdated",   "text": "...Saccharomyces bulderi...", "last_modified_time": "2024-06-01", "outdated_index": 0}
  ]
}
```

### chunks_to_qa.py 출력: `data/qa/qa_{model_alias}_N_M.jsonl`

```json
{
  "id": "hoh_000000_current",
  "hoh_source_idx": 0,
  "mode": "current",
  "original_question": "What is the name of the yeast...",
  "new_question": "Which yeast is currently recognized as capable of fermenting gluconolactone?",
  "target_answer": "Maudiozyma bulderi",
  "evidence_chunk_id": 12,
  "chunks": [...]
}
```

- record 1개당 `1 + n`개의 pair 생성 (mode: `current`, `current_raw`, `outdated_0`, ...)
- `target_answer`는 HoH answers에서 고정 (LLM 자유 생성 없음)
- `new_question`에 연월 숫자 금지 — 서술형 시간 힌트만 허용
- 검증 실패 시 rejection reason과 함께 재시도 (최대 3회)

### generate_reasoning.py 출력: `data/qa-reasoning/qa_{model_alias}_N_M_reasoning_{provider}.jsonl`

```json
{
  "id": "hoh_000000_current",
  "hoh_source_idx": 0,
  "mode": "current",
  "new_question": "Which yeast is currently recognized as capable of fermenting gluconolactone?",
  "target_answer": "Maudiozyma bulderi",
  "evidence_chunk_id": 12,
  "reasoning": "Document 12 (modified: 2024-07-01) and Document 31 (modified: 2024-06-01) both describe the same fermentation fact with different species names. Document 12 is more recent and states 'Maudiozyma bulderi'. The query asks for the 'currently' recognized name, which aligns with the latest modification.",
  "chunks": [...]
}
```

- `current_raw` 모드 제외 (narrative 시간 힌트 없어 reasoning 학습에 부적합)
- `--ratio`로 샘플링 비율 조절 가능 (SFT는 10~30% 권장)

---

## 설치 및 환경 설정

```bash
git clone <repo>
cd temporal-conflict-qa

# 가상환경 생성 + 패키지 설치 + .env 파일 생성
./setup.sh

# 가상환경 활성화
source .venv/bin/activate

# .env 파일에 API 키 입력
vi .env
# GEMINI_API_KEY=your-key-here
# OPENAI_API_KEY=your-key-here
# HF_TOKEN=your-token-here  (optional, HoH-QAs 접근용)
# VLLM_BASE_URL=http://localhost:8000/v1  (vLLM 사용 시)
```

---

## 실행 방법

### 자동 파이프라인 (hoh_to_chunks → chunks_to_qa)

```bash
python scripts/run_pipeline.py --start 0 --end 25 --provider gpt
python scripts/run_pipeline.py --start 0 --end 25 --vllm-model hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4
```

> `--vllm-model`을 지정하면 `--provider`가 자동으로 `vllm`으로 설정된다.

### 수동 실행

#### 1. Wikipedia 청크 생성

```bash
python scripts/hoh_to_chunks.py --start 0   --end 600
python scripts/hoh_to_chunks.py --start 600 --end 1200
```

중단 후 동일 명령어로 재실행하면 출력 파일에서 완료된 id를 읽어 자동으로 이어서 처리한다.

#### 2. LLM QA pair 생성

```bash
python scripts/chunks_to_qa.py --input data/chunks/chunks_0_600.jsonl --provider gpt
python scripts/chunks_to_qa.py --input data/chunks/chunks_0_600.jsonl --provider gemini
python scripts/chunks_to_qa.py --input data/chunks/chunks_0_600.jsonl --vllm-model hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4
```

#### 3. SFT Reasoning 생성

```bash
python scripts/generate_reasoning.py --provider gpt
python scripts/generate_reasoning.py --provider gpt --ratio 0.3 --seed 42
python scripts/generate_reasoning.py --provider gemini --input data/qa/qa_gpt41_0_600.jsonl
```

#### 4. Shard 병합

```bash
python scripts/merge_shards.py --step 1   # chunks → data/chunks/chunks.jsonl
python scripts/merge_shards.py --step 2   # qa     → data/qa/qa.jsonl
python scripts/merge_shards.py --step 3   # qa-reasoning → data/qa-reasoning/qa_reasoning.jsonl

python scripts/merge_shards.py --step 2 --auto-range   # → data/qa/qa_llama70b_0_600.jsonl
```

중복 id는 자동으로 제거된다.

#### 5. JSONL 정렬

병렬 생성 시 순서가 섞일 수 있으므로 id 기준으로 정렬한다.

```bash
python scripts/sort_by_id.py data/qa/qa_llama70b_0_600.jsonl                # 원본 덮어쓰기
python scripts/sort_by_id.py data/qa/qa_llama70b_0_600.jsonl -o sorted.jsonl # 별도 파일
```

---

## 실험 및 평가

### [실험 1] 대규모 자동 평가 (Initial Eval)
다양한 모델의 시간적 충돌 해결 능력을 자동으로 측정합니다.

```bash
# 평가 실행 (루트 디렉토리에서 실행)
python3 experiments/01_pilot_initial_eval/scripts/evaluate_llm.py --input data/qa/qa.jsonl --condition conflict

# 결과 집계
python3 experiments/01_pilot_initial_eval/scripts/summarize_eval.py
```
- 결과 데이터는 `results/`에, 최종 집계 리포트는 `metrics/`에 저장됩니다.

### [실험 2] Gemini 3 Flash 파일럿 (Closed-source Pilot)
닫힌 RAG 환경에서의 성능 급락을 실증하기 위한 수동/정밀 실험입니다.

1.  **실험 A (Open Web)**: `results/results_exp_a.json`의 질문을 Gemini 웹에 입력하고 결과를 기록합니다.
2.  **실험 B (Closed RAG)**: `results/results_exp_b.json`의 `no_conflict` 및 `conflict` 프롬프트를 AI Studio에 입력하고 결과를 기록합니다.
3.  **지표 계산**:
    ```bash
    python3 experiments/02_pilot_closed_source/scripts/evaluate_pilot.py
    ```
    - `metrics/summary_exp_a.json` 및 `summary_exp_b.json`으로 결과가 분리되어 저장됩니다.
    - `response` 필드만 입력해도 정규화된 EM(Exact Match) 로직으로 **자동 채점**이 지원됩니다.

---

## Wikipedia 청크 생성 전략

- MediaWiki API (`rvstart` + `rvdir=older`)로 `last_modified_time` 기준 직전 revision fetch.
- 슬라이딩 윈도우: `chunk_size=5`, `stride=3`.
- **current snapshot**: evidence chunk → `current`, 나머지 → `distractor`.
- **outdated snapshot**: evidence chunk 1개만 → `outdated`.
- 총 청크 수가 `MAX_CHUNKS=10` 초과 시 distractor를 균등 샘플링으로 축소.

## LLM 호출 전략

- record 1개당 mode별 1번 호출 (`current`, `current_raw`, `outdated_0`, ...).
- 생성된 질문에 연월 숫자 포함 시 rejection reason과 함께 재시도 (최대 3회).
- 429 / 5xx / 네트워크 에러: 지수 백오프 재시도 (최대 6회).
