# 실험 03 — Temporal Validity (시점 유효성)

## 한 줄 요약
RAG에서 **같은 사실의 옛/새 버전이 같이 검색될 때, LLM이 "맞는 시점의 문서"를 근거로 답하는지** 측정한다.
특히 **과거 시점이 정답인 경우(as-of-past)** 에 집중.

> 배경·기여: `연구_포지셔닝_TemporalValidity.md` · 설계·지표수식: `파일럿_실험계획_TemporalValidity.md` · 지표 쉬운설명: `지표_설명.md`
> 교수님 보고용 1장 요약: `../../docs/방향성_보고_TemporalValidity.md`

## 보이려는 것 (3가지) + 핵심 그림
- **C1** 최신 LLM도 *틀린 시점*의 문서를 인용해 답한다.
- **C2** 표준 지표(정답률·CitePrec)는 이 오류를 **못 잡는다**.
- **C3** 신규 지표 **TV(Temporal Validity)** 로는 잡힌다.

| | CitePrec=통과 | CitePrec=실패 |
|---|---|---|
| **TV=통과**(맞는 시점 인용) | 이상적 | 드묾 |
| **TV=실패**(틀린 시점 인용) | **★ 맹점** | 양쪽 다 잡음 |

→ **★ 칸**(틀린 시점 인용인데 표준지표는 "정상")이 비어있지 않음을 보이는 게 결과물.

---

## 전체 흐름 — 데이터부터 점수까지

```
[데이터 만들기 · data_prep/hoh/]              [실험 · experiments/03/scripts/]
 HoH원본                                       01_sample_eval_set.py
   │ hoh_to_chunks.py  (현실적 RAG 청크)          │  qa_hoh_genuine → eval_set.jsonl(58)
   ▼                                              ▼
 chunks_0_600.jsonl                            02_run_models.py
   │ hoh_to_qa.py  (무결성 정제, 무료)             │  모델이 3조건으로 답+인용
   ▼                                              ▼  → results/raw_<model>.jsonl  ← "모델 원응답"
 qa_hoh.jsonl(152)                             03_evaluate.py
   │ make_review_sheet.py → 사람/LLM 검수          │  raw 를 채점 (TV·CitePrec·★)
   │ apply_review.py  (genuine만 추림)             ▼  → results/metrics_<model>.json   ← "점수"
   ▼                                                 results/contingency_<model>.csv ← "2×2 표"
 qa_hoh_genuine.jsonl(58)  ───────────────────▶ (01의 입력)
```

### ⭐ raw vs metrics — 헷갈리기 쉬운 부분
- **`results/raw_<model>.jsonl`** = **02가 만든 "모델이 *실제로 답한* 내용"** (답 텍스트 + 인용한 청크번호 + reasoning). *점수 아님, 원자료.*
- **`results/metrics_<model>.json`** = **03이 raw를 *채점한* 점수** (TV_cite·CitePrec·★ 비율 등).
- **`results/contingency_<model>.csv`** = 03이 만든 **2×2 표**(★ 칸 포함).

즉 **raw(원응답) → `03_evaluate.py` → metrics(점수) + contingency(표)**.

### 데이터 소스
- **주력 = HoH** (Wikipedia 편집 이력 → 옛/새 버전이 한 컨텍스트에 공존). 3단계 정제 거쳐 **genuine 시간변화 58개**.
  - ① 무결성(`hoh_to_qa.py`): 단일전환·근거매칭·수치답 제거 → 152
  - ② 구성(`make_review_sheet`+`apply_review`): "진짜 시간변화 vs 정적사실 정정" 사람/LLM 검수 → 58
- **TQA**(`data_prep/tqa/`) = soft충돌 *대조용* 백업 (`results/*_tqa.*`).

### 역할 분리
| 역할 | 주체 |
|---|---|
| 질문·데이터 | **HoH**(위키 편집 기반, 모델 생성 아님) |
| 답변(테스트) | **GPT-5.5 · Gemini 3.1 Pro** + (예정) open: Qwen3-32B·Mistral-Small-3·Gemma-3-27B |
| 채점(CitePrec) | **proxy(무료 결정론, 기본)** 또는 Claude/오픈NLI |

> Llama 계열은 **데이터 생성자**라 자가편향 회피 위해 테스트 제외.

---

## 파일 지도 (어떤 걸 봐야 하나)

```
data_prep/hoh/
  hoh_to_chunks.py       원본 → 현실적 RAG 청크
  hoh_to_qa.py           청크 → 충돌 평가셋 (무결성 정제·무료)
  make_review_sheet.py   → data/hoh/review_sheet_hoh.csv (검수 시트)
  apply_review.py        검수결과 → data/hoh/qa_hoh_genuine.jsonl (genuine만)
data/hoh/
  qa_hoh.jsonl           무결성 통과 152 (중간 산출)
  review_sheet_hoh.csv   genuine/정정 검수 기록 (g/c)
  qa_hoh_genuine.jsonl   ★최종 genuine 58 = 평가셋 원천
experiments/03_temporal_validity/
  scripts/
    pilot_common.py      공통(프롬프트·파싱·지표·모델호출·비용)
    00_estimate_cost.py  실행 전 비용 추정
    01_sample_eval_set.py  genuine → eval_set.jsonl 샘플
    02_run_models.py     모델 답변 수집 → raw
    03_evaluate.py       채점 → metrics·contingency
  data/
    eval_set.jsonl       ★평가 문항 (58 = as-of-past 48 + current 10)
    validation_sheet.csv 문항 검수용
  results/
    raw_gemini.jsonl         ← 모델 원응답 (게이트)
    metrics_gemini.json      ← 점수 (TV·CitePrec·★)
    contingency_gemini.csv   ← 2×2 표
    *_tqa.*                  ← TQA soft 대조 백업
```

**빠르게 결과만 보려면**: `results/metrics_gemini.json`의 `as_of_past` 블록 + `contingency_gemini.csv`.

---

## 실행 방법

### A. 데이터 (루트에서, 무료·LLM 없음)
```bash
python3 data_prep/hoh/hoh_to_qa.py                 # 청크 → qa_hoh.jsonl(152)
python3 data_prep/hoh/make_review_sheet.py         # → review_sheet_hoh.csv
#   시트의 verdict 칸에 g(시간변화 유지)/c(정정 버림) 기입 후:
python3 data_prep/hoh/apply_review.py              # → qa_hoh_genuine.jsonl(58)
```

### B. 실험 (scripts/ 에서)
```bash
python 01_sample_eval_set.py --input ../../../data/hoh/qa_hoh_genuine.jsonl --quota "outdated=48,current=10"
python 02_run_models.py --model gemini             # ⚠️유료 — raw_gemini.jsonl
python 03_evaluate.py   --model gemini --judge proxy   # 무료 채점 → metrics·contingency
```
> 💡 `--judge proxy`(기본) = **무료 결정론적 CitePrec**(ALCE의 NLI 근사, 답이 짧은 엔티티라 유효). 유료 LLM judge 원하면 `--judge claude`.
> ⚠️ **02(모델 답변)는 유료 API.** 실행 전 `00_estimate_cost.py`로 비용 확인하고 진행.

---

## 결과 보는 법 + 현재 게이트 결과

`metrics_<model>.json`의 **`as_of_past` 블록**이 핵심:

| 지표 | 뜻 | Gemini 게이트(n=48) |
|---|---|---|
| EM | 답이 맞았나 | 0.583 |
| CitePrec (표준) | 인용문서가 답 뒷받침하나(*시점 무관*) | 0.938 |
| **TV_cite** (신규) | 인용문서가 *맞는 시점*인가 | 0.646 |
| **wrong_time_cite_rate** | 틀린 시점 인용율 (=1−TV_cite) | **0.354** |
| TV_behav | 답이 실제로 어느 시점 문서에 좌우되나 | 0.738 |
| **blind_spot_rate** | 틀린시점 인용 중 CitePrec 통과 비율 (★) | **0.824** |

**현재 게이트(깨끗한 HoH 58, Gemini)**: ★ 맹점 **14/48**, 틀린시점의 82%가 표준지표에 비가시. 대조군 current는 wrong_time 0.10(잘함) → 실패가 *과거질문에 특정* = 진짜 시간 grounding 실패. **게이트 통과.**

---

## 비용
모든 유료 실행의 토큰·비용은 루트 `usage/usage_summary.txt`에 자동 누적. `--judge proxy`·데이터 가공은 **무료**.
> ⚠️ 유료 API(02 모델답변, `--judge` 유료옵션) 실행은 **사전 비용추정 + 허락** 후 진행.

---

## 다음 단계
- [ ] **Open LLM 측정** (vLLM 서빙 → `02_run_models.py --model <open>`) → 보고서 part3 완성
- [ ] 데이터 수백 규모 확장 (HoH 샤드 추가 / Wikidata)
- [ ] 오픈 NLI(TRUE)로 CitePrec 교차검증
- [ ] **해결법**(시점-유효구간 grounding) 프로토타입 = 해결논문 본체
