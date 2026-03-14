# Temporal Conflict QA Dataset

RAG 시스템에서 시간적 충돌(temporal conflict)을 다루는 능력을 평가/학습하기 위한 데이터셋 생성 파이프라인.

[russwest404/HoH-QAs](https://huggingface.co/datasets/russwest404/HoH-QAs)를 기반으로, Wikipedia 히스토리컬 revision을 가져와 청크를 만들고, LLM(GPT/Gemini)을 통해 시간적 추론이 필요한 새로운 QA pair를 생성한다.

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
        ▼  data/chunks/hoh_chunks_N_M.jsonl
        │
        ▼
  chunks_to_qa.py  --input data/chunks/hoh_chunks_N_M.jsonl --provider gpt|gemini
  LLM 호출 → 시간 힌트 포함 새 질문 생성 + evidence_chunk_id 할당
        │
        ▼  data/qa/hoh_qa_{provider}_N_M.jsonl
        │
        ▼
  generate_reasoning.py  --provider gpt|gemini [--ratio 0.3]
  Teacher LLM → SFT cold-start용 reasoning 생성
        │
        ▼  data/qa-reasoning/hoh_qa_{provider}_N_M_reasoning_{provider}.jsonl
        │
        ▼
  merge_shards.py  --step 1|2|3
  범위별 shard 파일을 하나로 병합
```

---

## 디렉토리 구조

```
temporal-conflict-qa/
├── config.py                  # 공통 설정 (디렉토리, 모델, RPM, 청킹 등)
├── hoh_to_chunks.py           # Wikipedia 청크 생성
├── chunks_to_qa.py            # LLM QA pair 생성 (GPT/Gemini)
├── generate_reasoning.py      # SFT cold-start reasoning 생성
├── merge_shards.py            # 범위별 결과 병합
├── run_pipeline.py            # 파이프라인 자동 실행 (chunks → qa)
├── setup.sh                   # 가상환경 설치 스크립트
├── requirements.txt
├── .env                       # API 키 (git 제외)
├── .env.example               # 키 템플릿
├── data/
│   ├── chunks/                # hoh_to_chunks.py 결과
│   │   ├── hoh_chunks_0_500.jsonl
│   │   └── hoh_chunks.jsonl           ← merge_shards.py --step 1
│   ├── qa/                    # chunks_to_qa.py 결과
│   │   ├── hoh_qa_gpt_0_500.jsonl
│   │   └── hoh_qa.jsonl               ← merge_shards.py --step 2
│   └── qa-reasoning/          # generate_reasoning.py 결과
│       ├── hoh_qa_gpt_0_500_reasoning_gpt.jsonl
│       └── hoh_qa_reasoning.jsonl     ← merge_shards.py --step 3
└── logs/                      # 실행 로그 (스크립트별 폴더, 타임스탬프별 파일)
    ├── hoh_to_chunks/
    │   └── hoh_to_chunks_20260314_123456.log
    ├── chunks_to_qa/
    └── generate_reasoning/
```

> `data/` 및 `logs/` 디렉토리는 스크립트 실행 시 자동 생성된다.

---

## 데이터 구조

### hoh_to_chunks.py 출력: `data/chunks/hoh_chunks_N_M.jsonl`

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
    {"chunk_id": 0,  "label": "distractor", "text": "...", "sentence_indices": [0,1,2,3,4],     "last_modified_time": "2024-07-01 00:00:00"},
    {"chunk_id": 12, "label": "current",    "text": "...Maudiozyma bulderi...",    "sentence_indices": [38,39,40,41,42], "last_modified_time": "2024-07-01 00:00:00"},
    {"chunk_id": 31, "label": "outdated",   "text": "...Saccharomyces bulderi...", "sentence_indices": [38,39,40],       "last_modified_time": "2024-06-01", "outdated_index": 0}
  ]
}
```

- `hoh_source_idx`: HoH 원본 데이터셋 인덱스 (정수)
- `answers`: current 1개 + outdated n개 (n ≥ 1)
- `chunks`: current snapshot 전체 (current 1개 + distractor 다수) + outdated snapshot에서 evidence chunk 1개씩
- distractor 청크에도 `last_modified_time`이 부여됨 (current_time과 동일, timestamp 유무 shortcut 방지)

### chunks_to_qa.py 출력: `data/qa/hoh_qa_{provider}_N_M.jsonl`

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

### generate_reasoning.py 출력: `data/qa-reasoning/hoh_sft_{provider}_{suffix}.jsonl`

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
```

---

## 실행 방법

### 자동 파이프라인 (hoh_to_chunks → chunks_to_qa)

```bash
python run_pipeline.py --start 0 --end 25 --provider gpt
```

### 수동 실행

#### 1. Wikipedia 청크 생성

```bash
# 전체 처리
python hoh_to_chunks.py

# 범위 지정
python hoh_to_chunks.py --start 0   --end 500
python hoh_to_chunks.py --start 500 --end 1000
```

중단 후 동일 명령어로 재실행하면 출력 파일에서 완료된 id를 읽어 자동으로 이어서 처리한다.

#### 2. LLM QA pair 생성

```bash
# GPT (기본값)
python chunks_to_qa.py --input data/chunks/hoh_chunks_0_500.jsonl --provider gpt

# Gemini
python chunks_to_qa.py --input data/chunks/hoh_chunks_0_500.jsonl --provider gemini
```

#### 3. SFT Reasoning 생성

```bash
# 전체 (data/qa/ 내 모든 파일)
python generate_reasoning.py --provider gpt

# 샘플링 (30%)
python generate_reasoning.py --provider gpt --ratio 0.3 --seed 42

# 특정 파일 지정
python generate_reasoning.py --provider gemini --input data/qa/hoh_qa_gpt_0_500.jsonl
```

#### 4. Shard 병합

```bash
python merge_shards.py --step 1   # chunks → data/chunks/hoh_chunks.jsonl
python merge_shards.py --step 2   # qa     → data/qa/hoh_qa.jsonl
python merge_shards.py --step 3   # qa-reasoning → data/qa-reasoning/hoh_qa_reasoning.jsonl

# --auto-range: shard 파일명의 공통 prefix + 범위로 출력 파일명 자동 생성
python merge_shards.py --step 2 --auto-range   # → data/qa/hoh_qa_gpt_0_60.jsonl
```

중복 id는 자동으로 제거된다.

---

## Wikipedia 청크 생성 전략

- MediaWiki API (`rvstart` + `rvdir=older`)로 `last_modified_time` 기준 직전 revision fetch
- `mwparserfromhell`로 wikitext → 평문 변환
- 슬라이딩 윈도우: `chunk_size=5`, `stride=3`
- **current snapshot**: evidence chunk → `current`, 나머지 → `distractor` (모든 청크에 `last_modified_time` 부여)
- **outdated snapshot**: evidence chunk 1개만 → `outdated` (distractor 없음, 중복 방지)
- 총 청크 수가 `MAX_CHUNKS=10` 초과 시 distractor를 균등 샘플링으로 축소

## LLM 호출 전략

- GPT 모델: `gpt-4.1-mini` / Gemini 모델: `gemini-2.5-flash`
- record 1개당 mode별 1번 호출 (`current`, `current_raw`, `outdated_0`, ...)
- 생성된 질문에 연월 숫자 포함 시 rejection reason과 함께 재시도 (최대 3회)
- 429 / 5xx / 네트워크 에러: 지수 백오프 재시도 (최대 6회)

---

## 주요 의존성

| 패키지 | 용도 |
|--------|------|
| `datasets` | HoH-QAs 로드 |
| `requests` | Wikipedia MediaWiki API 호출 |
| `mwparserfromhell` | wikitext 파싱 |
| `openai` | GPT API |
| `google-genai` | Gemini API |
| `python-dotenv` | `.env` 파일 로드 |
| `huggingface-hub` | HF 인증 (optional) |
