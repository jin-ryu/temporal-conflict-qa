# 실험 03 — Temporal Validity (근거 시점 유효성) 파일럿

## 한 줄 요약

RAG에서 **같은 사실의 옛/새 버전이 같이 검색될 때, 최신 LLM이 "맞는 시점의 문서"를 근거로 답하는지** 측정한다.
특히 **과거 시점이 정답인 경우(as-of-past)** — 선행연구(GaRAGe, ACL 2025 Findings)가 안 다룬 칸 — 에 집중한다.

> 배경·기여·경쟁연구: `연구_포지셔닝_TemporalValidity.md`
> 설계 상세(지표 수식·프롬프트): `파일럿_실험계획_TemporalValidity.md`

## 무엇을 보이려는가 (3가지)

- **C1** 최신 LLM도 *틀린 시점*의 문서를 인용해 답한다.
- **C2** 기존 표준 지표(정답률·citation precision)는 이 오류를 **못 잡는다**.
- **C3** 새 지표 **Temporal Validity(TV)** 를 쓰면 잡힌다.

**핵심 결과물** = 아래 2×2 표에서 **★ 칸**(틀린 시점 인용인데 표준 지표는 "정상"이라 판정)이 비어있지 않음을 보이는 것.

| | citation precision = 통과 | = 실패 |
|---|---|---|
| **TV = 통과** (맞는 시점 인용) | 이상적 | 드묾 |
| **TV = 실패** (틀린 시점 인용) | **★ 맹점** | 양쪽 다 잡음 |

---

## 전체 흐름

```
데이터 변환             평가셋 만들기        모델에게 풀게 하기        채점·집계
tqa_to_qa.py     →    01_sample      →    02_run_models      →    03_evaluate
 (TQA→충돌, 무료)      (50문항 추출)        (GPT·Gemini 답변)        (TV·맹점 표)
                        ↓ 사람 검수
```

**베이스 데이터 = TQA / Temporal Wiki (Özer 2025, MIT)** — 엔티티마다 연도별
**실제 위키 revision 스냅샷** + Wikidata 답을 제공 → 옛/새 두 revision을 합쳐
**충돌 컨텍스트 조립**. 질문은 데이터 원본(Wikidata 템플릿)이라 **생성 편향·비용 없음**.
*(temporal 데이터셋 24종 전수조사 결과 "옛/새 실제 버전 + 깨끗한 답 + 시점라벨"이 한 문맥에 공존하는 건 TQA뿐 → 유일 적합. preprint 한계는 향후 Wikidata+위키 revision 직접빌드로 보강. TimeQA는 단일페이지·합성 타임스탬프라 제외.)*

**모델 역할 분리** (Claude를 테스트에서 빼 분석 편향 방지):

| 역할 | 주체 |
|---|---|
| 질문(데이터 원천) | **TQA**(위키 revision 기반, 모델 아님) |
| 채점(judge) | **Claude Opus 4.8** |
| 답변(테스트 대상) | **GPT-5.5 + Gemini 3.1 Pro** |

데이터 규모: **50문항** = 과거지향 40 + 현재지향(대조군) 10.
예상 비용: 테스트 ~$4.5 (**데이터 변환은 무료 — LLM 호출 없음**).

---

## 파일 구조 (프로젝트 루트 기준)

```
temporal-conflict-qa/
├── .env                     # API 키·모델 id·단가·RPM (자동 로드)
│
├── data_prep/               # ── 데이터셋 가공 (실험과 별개, 데이터셋별 분리) ──
│   ├── tqa/                #   ★주력 — TQA/Temporal Wiki 파이프라인
│   │   ├── tqa_to_qa.py    #     TQA 엔티티 → 충돌 QA 변환 (LLM 없음, 무료)
│   │   └── README.md       #     입력·출력 구조·스텝·필터 상세
│   ├── timeqa/            #   (조사 후 제외 — 단일페이지·합성타임스탬프)
│   │   └── timeqa_to_qa.py
│   ├── validate_qa.py     #   산출 데이터 무결성 검증 (tqa·timeqa 공용)
│   └── hoh/               #   legacy — HoH 파이프라인 (자체 config 포함)
│       ├── config.py · llm_client.py
│       └── hoh_to_chunks.py · chunks_to_qa.py · ...
│
├── tools/
│   └── monitor_cost.py      #   전체 비용 보기 (어디서든 실행)
│
├── data/
│   ├── tqa/                #   ★주력
│   │   ├── source/          #     TQA 엔티티 JSON 878개 (gitignored, MIT)
│   │   └── qa_tqa.jsonl     #     변환 산출 = 평가셋 원천 (722 레코드)
│   ├── timeqa/            #   (제외)
│   └── hoh/               #   legacy (chunks/ · qa/ · qa-reasoning/ · original/)
│
├── usage/                   # ── 전체 비용 (gitignored) ──
│   ├── usage_summary.txt    #   사람이 읽는 합계 (자동 갱신) ← 이거만 열면 됨
│   └── usage_ledger.jsonl   #   원시 로그
│
└── experiments/03_temporal_validity/        # ── 이 실험 ──
    ├── README.md
    ├── 연구_포지셔닝_TemporalValidity.md       #   동기·지형도·기여
    ├── 파일럿_실험계획_TemporalValidity.md     #   설계·지표·프롬프트 명세
    ├── scripts/
    │   ├── pilot_common.py       #   공통(프롬프트·파싱·지표·모델호출·비용)
    │   ├── 00_estimate_cost.py   #   실행 전 예상 비용 (API 호출 X)
    │   ├── 01_sample_eval_set.py #   평가셋 50문항 추출
    │   ├── 02_run_models.py      #   테스트 모델에게 풀게 함
    │   └── 03_evaluate.py        #   채점·2×2 집계
    ├── data/
    │   ├── eval_set.jsonl        #   (01 산출) 평가 문항
    │   └── validation_sheet.csv  #   (01 산출) 사람 검수용
    └── results/                  # ── 실험 결과만 (비용은 루트 usage/에) ──
        ├── raw_<model>.jsonl     #   (02) 모델 원응답·인용
        ├── metrics_<model>.json  #   (03) 정답률·TV·맹점비율
        └── contingency_<model>.csv  # (03) 2×2 표
```

---

## 실행 방법

`.env`에 키·모델·단가가 들어 있어 자동 로드된다. 0단계는 루트에서, 1~4단계는
`experiments/03_temporal_validity/scripts`에서 실행.

### 0단계 데이터 변환 (TQA → 충돌 QA, LLM 없음·무료)
TQA 엔티티에서 답이 다른 옛/새 두 연도의 **실제 위키 revision 본문**을 옛/새 청크로
조립한다. 상세(입력·출력 구조·스텝·필터)는 `data_prep/tqa/README.md`.
```bash
# (루트에서) data/tqa/source/*.json → data/tqa/qa_tqa.jsonl
python3 data_prep/tqa/tqa_to_qa.py
python3 data_prep/validate_qa.py data/tqa/qa_tqa.jsonl   # (선택) 데이터 무결성 검증
```
> TQA 데이터가 없으면: `git clone https://github.com/atahanoezer/TQA` 후
> `full_data_filtered/`를 `data/tqa/source/`로 복사.

### 1단계 예상 비용 확인
```bash
python 00_estimate_cost.py --models gpt,gemini --judge claude
```

### 2단계 평가셋 만들기 → **사람 검수**
```bash
python 01_sample_eval_set.py --input ../../../data/tqa/qa_tqa.jsonl
#   mode별 개수 지정: --quota "outdated=40,current=10" (기본). current=recency-bias 대조군.
```
→ 생성된 `data/validation_sheet.csv`를 열어, 각 문항이 **"과거 시점이 모호 없이 정해지고 + 질문이 자연스러운가"** 를 사람이 확인. **통과한 것만 남긴다.** (이 실험의 신뢰성은 여기서 갈림.)

### 3단계 테스트 모델 실행 (답변)
```bash
python 02_run_models.py --model gpt       # GPT-5.5
python 02_run_models.py --model gemini    # Gemini 3.1 Pro
```

### 4단계 채점 (judge = Claude, 답변 모델과 다름)
```bash
python 03_evaluate.py --model gpt    --judge claude
python 03_evaluate.py --model gemini --judge claude
```
→ `results/metrics_<model>.json`, `results/contingency_<model>.csv` 생성.

> 💡 처음엔 `eval_set.jsonl`을 몇 줄로 줄여 3~4단계를 돌려 형식·비용만 확인 후 전체 실행 권장.

---

## 결과 보는 법

> 📖 **각 지표가 뭔지 모르겠으면 `지표_설명.md`** — 모든 지표를 실제 수치 예시로 풀어 설명.

`results/metrics_<model>.json`에서 — 특히 **`as_of_past` 블록**:
- `wrong_time_cite_rate` : 틀린 시점 문서를 인용한 비율 (높을수록 C1 강함)
- `blind_spot_rate` : 그중 표준 지표가 "정상"으로 통과시킨 비율 (C2)

지표 의미 (쉽게):

| 지표 | 뜻 |
|---|---|
| **EM** | 답이 맞았나 |
| **CitePrec** (기존 표준) | 인용 문서가 답을 뒷받침하나 (*시점은 안 봄*) |
| **TV_cite** (신규) | 인용 문서가 *맞는 시점 버전*인가 |
| **TV_behav** | (문서를 빼고 넣어보며) 답이 실제로 어느 시점 문서에 좌우되나 |

`contingency_<model>.csv` = 위 2×2 표. **★ 칸(TV 실패 & CitePrec 통과)이 핵심.**

---

## 비용 보는 법

생성·테스트·채점 **모든 실행의 토큰·비용이 repo 루트 `usage/`에 자동 누적**된다. (실험 폴더 밖 — 생성은 실험과 별개라서.)

- **`usage/usage_summary.txt`** 를 열면 전체 합계가 보인다 (매 실행 자동 갱신, 스크립트 실행 불필요).
- 스크립트별 분해가 필요하면(루트 어디서든): `python3 tools/monitor_cost.py --by-script`

---

## 결과 판정과 다음 단계

`metrics`의 as_of_past + 2×2 ★칸을 보고:

| 결과 | 판정 | 다음 |
|---|---|---|
| **틀린 시점 인용 > 25%** & ★칸 유의 | C1·C2·C3 성립 | **진행** |
| **TV > 85%** & ★칸 거의 빔 | 프론티어가 이미 잘함 | **중단(step-out)** — 영역 포화, 주제 재검토 |
| 그 사이 | Findings급은 가능 | 규모 확대 + 방법론 결합 검토 |

**[진행] 시 할 일**
- [ ] 규모 확대(100+), 모델 추가
- [ ] **GaRAGe 대비 차별점 3개 못박기**: ① 과거지향(as-of-past) ② 통제된 충돌 ③ "표준 지표가 틀린 시점 인용을 통과시킴"의 명시적 입증(★칸)
- [ ] `TV_behav`(행동)와 `TV_cite`(선언) 일치율 → 인용 신뢰성 근거
- [ ] 현실성 보강(WikiContradict 등) 검토
- [ ] (선택) 개선 방법론 — 단 기존 연구(Time-Travel 등) 대비 차별화 필요

**[중단] 시 할 일**
- [ ] 지도교수와 지형도(`연구_포지셔닝` §5.3) 공유 후 영역 전환 결정
- [ ] RAG 평가 역량은 유지, 덜 붐비는 문제로 재설정

> ⚠️ 어느 경로든 **2단계 사람 검수**가 선행돼야 결과가 유효하다. 모호한 문항이 섞이면 비율이 왜곡된다.
