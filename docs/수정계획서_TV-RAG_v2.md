# 수정 연구계획서 v2 — 시간 충돌 RAG: 오답 인용을 막는 올바른 근거 선택 및 기권 학습 (TV-RAG)

**문서 개요**
*   **시스템명 (가칭):** TV-RAG (Temporal Verification-RAG) — 확정 전, 대안 후보 GRACE
*   **사용 데이터셋:** TeCoQA (Temporal Conflict QA) 외 (구 명칭 "TC-RAG"에서 개명)
*   **목표 학회:** ACL / EMNLP Main Track (또는 TACL)
*   **핵심 요지:** 시간 충돌 RAG 환경에서 발생하는 치명적 오류(틀린 답과 그럴듯한 근거의 결합)를 기존 평가 지표가 포착하지 못함을 규명함(진단). 이를 해결하기 위해, 능동적 기권(Abstain) 메커니즘과 정답-앵커(Gold-anchor) 집합 기반의 비대칭 보상 함수를 적용한 GRPO(Group Relative Policy Optimization) 학습 방법론을 제안함(처방). 진단과 처방이 "앵커 교체(답-앵커 → 정답-앵커)"라는 하나의 원리로 관통됨.
*   **대상 독자:** 레포지토리 협업자. 본 문서를 이후 모든 실험·집필의 기준(single source of truth)으로 삼음.

**주요 설계 결정 (확정)**
1.  **보상 함수:** 비대칭 페널티를 제안 보상으로 채택하고, 단순 곱셈은 ablation 비교군으로 둠 → "비대칭이 왜 필요한가"를 실험으로 증명.
2.  **능동적 기권(Abstain):** 핵심 기여로 포함하되, 별도 모듈이 아닌 `<conflict>` 프로토콜의 특수사례로 편입하여 스코프를 관리함.
3.  **데이터 믹스:** 시간 충돌 80% + 일반 QA 20%(망각 방지)의 v2 구조를 채택함.

---

## 1. 연구 배경 및 문제 정의

### 1.1. 시점-조건부 QA의 한계 및 검색기 단독 접근의 문제점
본 연구는 문제를 최신성(Freshness) 확보가 아니라 **시점-조건부 QA(Temporal-conditional QA)** — 임의의 시점 t를 기준으로 유효한 답을 요구하는 과제 — 로 정의함. Freshness("최신 찾기")는 t=now인 퇴화 특수사례에 불과하며, 과거 시점 질의(As-of-past)를 처리하지 못함. 선행 연구(HoH; arXiv:2503.04800)는 시간 인식 검색(Time-aware Retrieval)이 오래된 문서를 필터링할 때 정답 문서 회수율(Recall)마저 붕괴하는 상충 관계(Trade-off)를 겪음을 보였고(MRAG는 relevant hit@5가 0.73→0.32로 하락), 나아가 현재 날짜·타임스탬프·"최신 우선" 지시를 모두 포함한 time-aware 프롬프트 기본 설정에서도 오래된 문서 1개로 생성 점수가 20%p 이상 하락함을 보임. 따라서 문제의 핵심은 검색이 아니라, 검색된 문맥 내에서 **시간적 유효성을 직접 추론하는 생성기(Generator)**에 있음.

### 1.2. 기존 평가 지표(CitePrec)의 맹점: Mis-attribution 현상
기존의 생성 기반 인용 평가 지표 CitePrec(답-앵커)는 모델이 생성한 '답변'을 기준으로 인용 적절성을 평가함. 이는 충돌 코퍼스에서 모델이 '틀린 시점의 문서'를 인용해 '틀린 답'을 생성했을 때, 그 문서가 자기 답을 지지하므로 이를 올바른 인용으로 간주하는 치명적 맹점(순환성)을 지님. 본 연구는 CitePrec과, 시점-유효 문서 집합에 기반한 정답-앵커 지표 **GoldCite** 간의 괴리를 실증함.

*   진단 실험(exp03/04)에서 mis-attribution rate는 5모델·2데이터셋에서 **81~100%**로 나타남.
*   CitePrec은 97~100%(만점처럼 보임)이나 GoldCite는 44~82%로 급락, **괴리 18~54%p**(약한 모델일수록 큼).
*   **헤드라인은 "괴리·랭킹 역전"으로 배치함**: "mis-attr가 높다"는 충돌 세팅상 상당 부분 필연이므로(리뷰어보다 먼저 인정), CitePrec 기준 모델 랭킹과 GoldCite 기준 랭킹이 역전됨(=지표 선택이 실제 결론을 바꿈)을 헤드라인으로 삼음.

### 1.3. 왜 하필 "시간 충돌"인가 — 정답이 존재하는 충돌
동시대 벤치마크 CONFRAG(ACL 2026)는 실제 웹의 **논쟁형 충돌**("감기 때 유제품을 피해야 하나?" 같은 다관점 대립)을 다루며, 정답이 하나로 정해지지 않아 태스크를 "관점 클러스터링·설명"으로 정의함(그들 스스로 "gold 답 하나가 아니라 어느 문서가 같은 관점인지 식별"이라 명시). 즉 **논쟁형 충돌에서는 '틀린 답'이 정의되지 않아 mis-attribution·인용 정확성을 측정할 수 없음.** 이와 대비하여 **시간 충돌은 질문 시점 t가 유효 답을 하나로 결정**하므로, 답 정확성과 인용 정확성(GoldCite)이 비로소 측정 가능해지는 유일하게 깨끗한 무대임. 이 대비가 "왜 시간 충돌인가"에 대한 원리적 정당화를 제공함.

---

## 2. 핵심 지표·보상 정의 (코드 반영 필수)

### 2.1. 시점-유효 evidence 집합 E\* (점→집합 전환, GRPO 전 필수 수정)
```
E* = { c : contains_answer(c, a*, alias_table) ∧ in_valid_window(c.timestamp, t_q) }
```
기존 `evidence_chunk_id`(유일 gold) 정의를 폐기함. 이유는 세 층위임.
*   **지표 층위 (false negative):** 유효 문서가 복수일 때(실험 B 51건 중 46건=90%, TeCoQA도 슬라이딩 윈도우[5문장/3겹침]로 evidence 문장이 인접 청크 2~3개에 중복. 실측상 정답 문서 복수 비율 A 63%·B 90%) 모델이 다른 유효 문서를 인용해도 0점 처리됨.
*   **보상 층위 (오염, 더 심각):** GRPO 그룹 비교에서 "옳은 추론(다른 유효 문서 인용)"이 "틀린 추론"과 같은 0점을 받아, 모델이 시점 판단이 아니라 주석 아티팩트(주석자가 고른 청크 패턴)를 학습하게 됨. 곱셈 구조가 이를 증폭함.
*   **개념 층위:** 실제 구인(construct)은 "유일 정답 문서"가 아니라 "시점-유효 문서의 집합"임.

**구축:** 정답 문자열+별칭 매칭 ∧ 시점 창 매칭으로 자동 라벨 전파 + 경량 NLI 재검("이 청크가 'Q의 답은 X'를 함의하는가") + 샘플 200건 사람 검증(오염률 보고). **strict E\***(타임스탬프 기준) / **lenient E\***(내용 기준) 두 버전으로 전 지표 민감도 분석하여 야생 데이터 타임스탬프 노이즈에 대응함. 코드 수정은 `cited == gold_id` → `cited in gold_set`. **본 수정을 GRPO 학습보다 반드시 먼저 완료함** (유일-gold로 학습한 결과는 논문에 사용 불가).

### 2.2. 평가 지표

| 지표 | 정의 | 비고 |
|---|---|---|
| EM / Token-F1 | 정답 문자열 대비 (별칭 정규화) | 답 정확성 |
| **GoldCite-R** | 1[S ∩ E\* ≠ ∅], S=모델 인용 집합 | 헤드라인 evidence 지표 |
| **GoldCite-P** | \|S ∩ E\*\| / \|S\| | 과잉 인용 페널티 |
| Combined | Token-F1 × GoldCite-R | KILT score의 생성측·시점조건 확장으로 명시 |
| **mis-attribution rate** | P(CitePrec=1 \| EM=0) | 논문 시그니처. baseline 대비 감소가 핵심 |
| 2×2 (EM × CitePrec) | Grounded / Under-cited / ★Mis-attr / Transparent-fail | 진단 섹션 |
| Conflict-report Acc | `<conflict>` 블록의 충돌 고지·기권 정확도 | 신규 |

### 2.3. 비대칭 보상 함수 (Asymmetric Reward Function) — 제안 보상
단순 곱셈($R_{answer} \times R_{evidence}$)이 유발하는 부분 점수 소실을 넘어, **진단에서 규명한 "가장 위험한 오류(mis-attribution)"를 보상에 직접 인코딩**하기 위해 오답 유형별 비대칭 페널티를 적용함. 이로써 진단(무엇이 위험한가)과 처방(무엇을 벌하는가)이 봉합됨.

```python
# 보상 산출 구조
R_total = R_format + R_conflict + R_main

# R_main 설계 (제안 보상)
if R_evidence == 1 and R_answer == 1:
    R_main = +1.0  # Grounded: 유효 문서 선택 + 정답
elif R_evidence == 1 and R_answer == 0:
    R_main = +0.2  # 탐색 장려: 유효 문서는 골랐으나 생성 실패
elif R_evidence == 0 and R_answer == 1:
    R_main = -1.0  # ★Mis-attribution: 틀린 문서 기반의 우연한 정답 → 최고 페널티
else:  # R_evidence == 0 and R_answer == 0
    R_main = -0.5  # Transparent-fail: 둘 다 실패 (정직한 오류)
```
*   **핵심 논리:** 위험한 오답(mis-attribution, −1.0)을 정직한 오답(transparent-fail, −0.5)보다 *더 강하게* 벌함. 단순 곱셈은 두 경우를 모두 0으로 처리하여 이 구분을 못 함.
*   **ablation:** 곱셈($R_{answer} \times R_{evidence}$)을 비교군으로 두어 "비대칭 페널티가 mis-attribution 감소에 기여함"을 직접 증명함(§5 시그니처 ablation과 연동).
*   **보상 해킹 감시:** R_answer(스팬 복사), R_evidence(형식 요행), 다중 인용 스팸(→GoldCite-P로 견제). 시드 3개·KL 규제·체크포인트별 reasoning 샘플 수동 감사.

---

## 3. 제안 방법론 (TV-RAG 아키텍처)

검색기를 동결(Frozen)하되, 생성기에 능동적 검증 프로토콜과 비대칭 보상을 결합함. 유지 요소: frozen bi-encoder retriever(k=10~20) → 생성기(SFT cold-start[형식만] → GRPO), narrative 시간 힌트 데이터.

### 3.1. Input → Output 및 능동적 기권(Abstain) 프로토콜
생성기는 단일 패스(Single-pass)로 추론하며, 검색 문서 집합의 유효성에 따라 두 행동을 학습함.

**Input** (검색기가 가져온 문서, 순서 무작위):
```
[Query] As of 2024-09, LA 시의회 의장은?
[Doc 1] [modified: 2024-10] "...의장 Blumenfield..."    ← 최신(틀린시점)
[Doc 2] [modified: 2024-09] "...의장 Harris-Dawson..."  ← 9월(유효)
[Doc 3] [modified: 2024-09] "...Harris-Dawson 의장..."  ← 9월(유효, 또 있음)
...
```
**Output (정상 추론 — 유효 문서 존재):**
```
<conflict> 9월엔 Harris-Dawson, 10월엔 Blumenfield로 값이 바뀜 </conflict>
<evidence> [2] </evidence>              ← 시점-유효 문서 선택(집합 E* 중 하나면 정답)
<answer>   Harris-Dawson </answer>       ← 질문 시점의 답 하나
```
**Output (능동적 기권 — 유효 문서 부재):** 검색 결과 내에 질문 시점에 유효한 문서가 없을 경우, 파라메트릭 지식에 의존한 환각을 생성하는 대신 기권을 선언함.
```
<conflict> 유효한 시점의 문서가 없습니다. 재검색이 필요합니다. </conflict>
```
*   **설계 원칙:** 질문이 시점을 지정(As-of {t})하므로 답은 하나로 고름(다관점 나열은 질문에 답하지 않은 것). 단 `<conflict>`로 충돌을 숨기지 않고 드러냄. 기권은 이 프로토콜의 특수사례(유효 문서 = ∅)로 자연 편입되어 별도 모듈이 필요 없음.
*   **기여 관점:** ① 충돌 고지는 CONFRAG의 기대행동과 정합하여 "단일 답 강제가 잘못된 목표를 측정한다"는 공격을 차단함. ② 기권은 CONFRAG(정답 클러스터 수 k를 강제로 제공, 기권 없음)와 차별화되는 **신뢰성(reliability) 축**을 추가함. ③ R_conflict를 별도 보상 항으로 분리하여 진단의 "인지 실패 vs 선택 실패"(detection ≠ resolution, monitoring–control gap)를 학습 신호에도 반영함.

### 3.2. 아키텍처 정당화 스코프 (서론·실험 필수 반영)
"순수 최신성 문제라면 결정론 규칙이 더 낫다"는 반박에 대응하여, RL 생성기가 정당화되는 잔여 난제 4종 — 규칙(결정론적 recency·메타데이터 필터)이 **원리적으로 실패**하는 영역 — 으로 scope를 명시함.
1.  과거-앵커 질문(양방향) 2. 발행일 ≠ 서술 대상 시점(focus time) 3. narrative 시간 힌트(명시 날짜 없음) 4. 타임스탬프 노이즈(내용 기반 시점 추론 필요)

"규칙이 이기는 곳(t=now, 깨끗한 타임스탬프)에선 비슷하고, 규칙이 죽는 곳(outdated mode)에서 우리만 산다"는 그림을 **Phase 2 파일럿에서 선(先)확인**함.

### 3.3. 배포 이중 모드
*   **Standalone:** 단독 QA 시스템.
*   **Plug-in 가드(Guard):** TV-RAG가 선별한 시점-유효 문서만 임의의 대형 모델(72B, 상용 API) 앞단에 전달하는 listwise 필터 — "중간 아키텍처"가 등장할 유일하게 올바른 자리(방법이 아니라 배포 형태). 호스트 모델의 시간 환각 방어를 실증함.

> **프레이밍 금지사항.** (a) "모델이 시간 충돌에 약하다"를 첫 펀치로 쓰지 않음(known). (b) "중간 모듈(리랭커/충돌감지) 추가"라 쓰지 않음(서론의 중간개입 비판과 자기모순·ConflictRAG와 정면 경쟁 유발). (c) "emergence"는 reasoning 블록에만 한정하고, evidence selection은 감독됨을 정직하게 기술. (d) Combined/GoldCite를 발명품으로 쓰지 않고 KILT 계보를 선언한 뒤 "충돌 조건에서의 필요성 증명"에 기여를 배치함.

---

## 4. 데이터셋 구축 계획

모델이 텍스트 유사도·타임스탬프 숫자 비교(Shortcut)에 과적합되는 것을 방지하고 범용성을 확보하기 위해 복합 데이터셋을 구성함.

### 4.1. 학습 데이터 구성

1.  **시간 충돌 특화 데이터 (80%)**
    *   **TeCoQA:** HoH(96K) 파생, 위키 기반 고유사도(~96%) 충돌 데이터 (미세한 사실 변경 추론 학습). Train/Test 8:2, `hoh_source_idx` 기준 분할.
    *   **StreamingQA 충돌 합성:** 뉴스(WMT07–20) 기반 저유사도 충돌 데이터. 같은 질문의 시점별 evidence 문서를 짝지어(질문 시점 이전 답 vs 이후 답, 미래 문서 함정 포함) 수만 건 규모로 합성.
    *   **CLARK-News (ERASE, Li et al. 2024):** 과거 사실을 명시적으로 부정하는 형태("replacing ...")의 실제 뉴스. cross-style 검증용(+학습 믹스 후보).
    *   **Hard Negative Set:** 질문 시점의 유효 문서 없이 틀린 시점 문서만 제공하여 능동적 기권(Abstain) 행동을 유도하는 특수 목적 데이터.
2.  **일반화 검증 데이터 (20%)**
    *   **General QA (NQ, HotpotQA):** 시간 충돌이 없는 일반 RAG 환경에서의 성능 하락(Catastrophic Forgetting)을 방지하기 위한 통제군.

**학습 믹스 원칙:** 고유사(TeCoQA) : 저유사(StreamingQA) 균형으로 "diff 탐지만 학습했다"는 공격을 차단하고, 한 유형 학습 → 다른 유형 평가의 교차 전이 표를 작성함.

### 4.2. 평가 6단 일반화 사다리
in-domain(TeCoQA test) → near(HoH 원본) → cross-style(CLARK-News) → cross-domain(후보) → wild(rag_conflicts 51) → OOD 형식(SituatedQA temp ~520, narrative→명시 연도 전이).

### 4.3. 철회·주의
*   **evolveQA 사용 철회:** RAG 벤치마크가 아니라 파라메트릭 지식 프로빙(저자 명시)이며 데이터 미공개("coming soon"). superseding-knowledge 개념 인용만 유지함.
*   실험 B 타임스탬프 노이즈(예: 2016년 문서가 2025년 사실 서술) → strict/lenient E\*로 방어 + 한계 명시. 해당 예시는 논문 본문에 그대로 싣지 않음.
*   TeCoQA 생성 질문 200건 사람 검수: 힌트 품질·정답 누수율 보고.
*   **(신규 기회) CONFRAG 구축 파이프라인 개조:** CONFRAG의 실제 웹 데이터 구축 레시피(SerpAPI 검색 → Jina reader 본문추출 → 답/근거 추출 → 클러스터링)를 "논쟁 질문 필터 → 시간에 따라 답이 바뀌는 사실 필터", "클러스터링 → 시점-유효 답 추출"로 개조하면, 실제 웹 기반 시간 충돌 데이터를 대규모로 구축하는 검증된 템플릿으로 활용 가능함(rag_conflicts 51건 규모 한계 극복 경로).

---

## 5. 실험 설계 및 평가 계획

### 5.1. Exp 1 — 진단 (논문 §3, 기존 exp03/04 재배치 + 신규)
*   기존: 2×2/mis-attr(5모델×A·B), **CitePrec vs GoldCite 괴리·랭킹 역전 표(헤드라인)**, recency-pull 대칭 그림 1장, 지표 일반성(CiteRecall/AIS)은 1문단+부록.
*   신규(P0): **closed-book 기저선**(문서 없이 질문만, 전 모델×전 데이터 — recency-pull vs memory-pull 분리, 파라메트릭 오염 방어).
*   진단 섹션은 학습과 **독립 완결**로 작성함(학습 부진 시 벤치마크+진단 논문으로 전환 가능한 보험).

### 5.2. Exp 2 — 메인: TV-RAG vs Baselines (논문 §5)

| # | Baseline | 목적 |
|---|---|---|
| 1 | Standard RAG (Qwen2.5-7B) | 하한 |
| 2 | Time-aware Reranking (recency decay) | outdated mode에서 원리적 실패 시연 |
| 3 | Time-aware CoT Prompting (지시문 2~3 변형 민감도 포함) | "프롬프트로 되지 않냐" 차단 |
| 4 | Llama-3.1-8B (동규모) | 아키텍처 효과 분리 |
| 5 | DeepSeek-R1-Distill-7B | 범용 RL reasoning 불충분 |
| 6 | **TRACE-7B** | 핵심: 동일 RL-RAG 구조서 temporal 데이터+gold-앵커 보상 기여 분리 |
| 7 | Llama-70B | 규모로 안 됨 |
| 8 | 재현 가능한 시간인식 RAG 1종 (결정론적 시점필터 or ConflictRAG류) | "쉬운 상대만 골랐다" 차단 |
| 9 | **무학습 에이전트** (Time-aware ReAct, LangChain 기반; 날짜 파싱 도구+시점 필터 반복) | "학습 없이 에이전트로 되지 않냐" 차단 + 지연시간/비용(경제성) 대조 |

*   **보고:** 4지표(F1 / GoldCite-R / Combined / **mis-attr rate**) × 3모드(current / outdated / ambiguous) 전 분해.
*   **핵심 주장 = 양방향 교정:** 모델별 상반 편향(GPT류 최신끌림 vs Llama류 역끌림)이 *둘 다* 완화되는지가 파일럿의 "모델별 상반 편향" 발견에 대한 답임.

### 5.3. Exp 3 — 검증 (논문 §6, 순환 공격 방어)
1.  **faithful-wrong LOO** (TV-RAG + TRACE): `<evidence>` 문서를 제거 후 재실행 → 답 변화율. 기존 코드(`10_leave_one_out.py`) 재사용. (진단서 이미 오픈 3모델×A·B에서 **59~76%** 확보.)
2.  **Shortcut probes:** 타임스탬프 (a)제거 (b)셔플(문서-날짜 매칭 파괴) (c)본문 날짜 마스킹 — 의존 프로파일링.
3.  **표면 유사도 ablation:** outdated 청크 패러프레이즈로 유사도 96%→저하 조작, 유사도 구간별 성능 곡선. baseline은 붕괴·TV-RAG는 유지가 목표 그림. **Phase 2에서 파일럿 선행**(TV-RAG도 붕괴 시 내러티브 수정).
4.  reasoning LLM-judge 분석은 보조로 격하하고 LOO를 주 증거로 삼음.

### 5.4. Exp 4 — Ablation (논문 §6)

| Ablation | 검증 질문 |
|---|---|
| **답-앵커 R_citation vs gold-앵커 R_evidence** | **시그니처:** 답-앵커 인용 보상이 mis-attr을 오히려 강화하는가 → 진단·처방 봉합 |
| **비대칭 페널티 vs 단순 곱셈** | 비대칭이 mis-attribution 감소에 기여하는가 (§2.3 결정) |
| SFT-only | GRPO 기여 |
| `<reasoning>` 제거 / `<conflict>` 제거 | 블록·기권 프로토콜 기여 |
| R_answer only | evidence 감독 없이 shortcut 발생? |
| 타임스탬프 제거 | 메타데이터 의존도 |

### 5.5. Exp 5 — 일반화·배포 (논문 §7)
*   6단 사다리 전 평가 + temporal 외 충돌 zero-shot 전이 1건(rag_conflicts misinformation형 — "시간은 가장 깨끗한 무대" 주장 보강).
*   **Plug-in 가드:** TV-RAG 선별 문서만 대형 모델 2~3종에 전달 → 호스트 시간 환각 방어 실증.
*   (선택) 승계 사슬 hard subset(A→B→C 두 번 변경, "사건 E 당시의 X") — 평가 전용 분석 섹션.

### 5.6. 통계·타당성 (전 실험 공통)
*   전 표 부트스트랩 95% CI + 주요 비교 유의성 검정 (특히 B는 n=51이라 필수).
*   GRPO **시드 3개** + 분산 보고, KL 규제.
*   자동 채점(EM/F1/GoldCite) 200건 사람 검증 일치율 부록.
*   CitePrec 구현: 규칙 기반 + 샘플에 NLI 기반 대조(ALCE 정합성). *(규칙 기반 substring 채점은 CONFRAG(ACL 2026)도 비용·재현성 위해 동일 선택 — 선례로 방어.)*

---

## 6. 관련 연구 포지셔닝

### 6.1. 반드시 인용·차별화

| 논문 | 관계 | 차별화 문구 |
|---|---|---|
| **KILT** (NAACL 2021) | GoldCite의 직계 선조 (gold provenance R-precision, KILT score 게이팅) | "KILT의 집합형 gold-앵커를 **생성측 인용 + 시점 조건 충돌**로 확장" |
| **CONFRAG** (ACL 2026) | 실제 웹 다관점 충돌 벤치마크 (논쟁형, 정답 없음, 클러스터링 태스크) | **"정답 없는 충돌 vs 정답 있는 충돌"** 대비 — 논쟁형은 인용 정확성 측정 불가, 시간 충돌만이 유효 답을 결정. 채점(substring)·구축 파이프라인은 선례로 활용 |
| **Bohnet et al. 2022** (Attributed QA) | 답+증거 출력 과제, 채점이 답-앵커(AIS)로 분기한 지점 | 답-앵커가 표준이 된 역사적 분기점으로 서술 |
| **Huang et al.** (ACL 2024, fine-grained rewards) | 인용+정답 보상 RL 선례 — 단 인용 보상이 답-앵커 | "답-앵커 인용 보상은 충돌에서 mis-attr을 강화(우리 ablation)" |
| **CaRR / E-GRPO** (2026) | gold-앵커 세밀 보상의 인접 선점자 | 과제 축 차별화: 멀티홉 증거사슬 vs 시간 충돌 문서 선택 |
| **ConflictRAG** (2026) | 학습형 검출+해결 파이프라인(temporal 하위유형 포함) | 파이프라인 vs 단일 생성기; baseline 재현 시도 |
| **"Don't Ask the LLM to Track Freshness"** (2026) | 결정론 진영 — 우리 전제 정면 반박 | 반격 = 양방향성: recency는 과거-앵커에서 설계상 실패. HoH v3 Date↓ 붕괴(−2.77)로 보강 |
| ERASE (Li et al. 2024) | 코퍼스 편집 진영 + CLARK 출처 | 과거 답이 필요한 질의에선 삭제 불가 |
| Wallat 2025 · Cited-but-Not-Verified 2026 · Verified Misguidance 2026 · Do-Attribution-Metrics-Transfer 2026 | 인용-정확성 괴리·과신 근거 | 진단 뒷받침 (Ding 2025 등 과신 연구 포함) |

### 6.2. gold-앵커가 표준이 못 된 4가지 이유 (서론 소재)
1.  gold 증거 비유일성(복수 유효 문서) → **집합 정의로 우리가 해소.**
2.  장문 생성에서 주장별 gold 주석 비용(ExpertQA: 자동지표–전문가 상관 낮음).
3.  배포 시점엔 gold 부재 → 답-앵커가 산업 표준(RAGAS류)으로.
4.  AIS의 설계 철학("진실성 무관"이 의도).

→ 서사: "충돌 없는 코퍼스에선 답이 맞으면 두 앵커가 ≈일치해 분기가 무해했다. 충돌이 두 앵커를 가르고, 가른 방향이 전부 위험하다."

### 6.3. Related Works 전수 검증 경고 (Phase 0 최우선)
**확정된 합성 오류 2건 (인용 전 수정 필수):**
1.  "HoH가 MRAG·TempRALM을 적용해도 생성기 교란이 방어되지 않음을 실증" — 원문에 없는 실험임(검색 hit-rate 실험과 생성 교란 실험을 잘못 접합). 두 문장으로 분리하여 기술하고, "따라서 생성기 자체의 시간 추론 필요"는 실험이 아닌 논증으로 명시함.
2.  **AMAQA "EM +14점 도약" — 허구.** 실제 AMAQA(2505.13557)는 *메타데이터 QA 데이터셋* 논문으로, 지표는 정확도(GPT-4o 0.50→0.86, 오픈 0.27→0.76)이며 시간 전용도 아님(타임스탬프+채팅명+감정 다차원). **본 계획서에서 AMAQA 인용 제외** (시간 충돌과 무관).

**검증 완료로 정정된 항목:** Zhang 2025 법률 "1810–1881 정확도 0%"는 ✅ 사실이나 **법령(statute)이 아니라 판례 파기(SCOTUS overruling)**임(arXiv:2510.20941, Li Zhang et al., JURIX 2025) → "판례 파기"로 표기. FinTMMBench(arXiv:2503.05185, 금융 시간-멀티모달 RAG 벤치), TRAM·Time-R1·TRACE·RAG-RL·Longpre·Memory-T1 등 전부 원문 확인 완료(§10). **남은 실질 대조 대상 없음** — 인용은 §10 검증 상태를 따름.

---

## 7. 예상 리뷰 공격과 방어 (레드팀)

| 공격 | 방어 |
|---|---|
| "mis-attr 81~100%는 충돌 세팅에서 정의상 당연" | 선제 인정 + 헤드라인을 **랭킹 역전** 및 "따로 보면 우등생, 교차하면 재앙"(EM 0.40×CP 0.98)으로 이동. EM0·CP0 칸이 빈 것 자체가 발견(실패=증거추종, 환각 아님) |
| "GoldCite = KILT 재탕" | 계보 선(先)선언 + 생성측·충돌·시점조건 확장 명시. 집합 정의로 KILT의 set 개념과 정합 |
| "gold 유일성 가정이 비현실" | 집합 E\* + strict/lenient 민감도 분석 |
| "정답 없는 충돌은 어쩌나 / 이 태스크가 유효한가" | 시간 충돌만이 유효 답을 결정 → CONFRAG(논쟁형, 정답 없음)와 명시 대비 (§1.3) |
| "파라메트릭 오염(Kash Patel 등)" | closed-book 기저선 + LOO faithful-wrong |
| "diff 탐지를 학습한 것" | 학습 믹스 2원화 + 유사도 ablation 곡선 + 교차 전이 표 |
| "프롬프트/규칙/에이전트로 충분" | 지시문 민감도 + recency의 outdated mode 실패 + 무학습 ReAct 에이전트(#9) 비교 + HoH v3 Date↓ 역효과 인용 |
| "단일 답 강제가 잘못된 목표" | `<conflict>` 고지 + Abstain + Conflict-report Acc |
| "비대칭 보상 값이 임의적" | 곱셈 vs 비대칭 ablation으로 필요성 실증(§5.4) |
| "emergence 주장 순환" | 감독/유도 분리 기술 + LOO 검증 |
| "baseline이 약함" | TRACE + 시간인식 RAG(#8) + 무학습 에이전트(#9) + ConflictRAG 비교 |
| "n=51 통계력" | CI 전면 + A 확장, B는 야생 검증 역할로 한정 |

---

## 8. 실행 로드맵 (임계 경로 ≈ 3.5~4개월)

원칙: 리스크 큰 검증(논문이 죽을 수 있는 것)을 먼저, 오래 걸리는 학습을 병렬로.

*   **Phase 0 — 정합성·정직성 청소 (즉시, ~1주):** ① Related Works 전수 원문 대조(§6.3, MRAG/TempRALM 문장 교체 포함) ② 수치 불일치 통일(§10) ③ 신규 인용 추가(KILT·CONFRAG·ConflictRAG·CaRR·Don't-Ask·Bohnet·ERASE).
*   **Phase 1 — 설계 확정 (병행, ~1주):** ④ 스토리라인·문제정의(시점-조건부 QA) 고정 ⑤ **E\* 집합 정의로 채점·보상 코드 수정 + 비대칭 보상 구현**(GRPO 전 필수) ⑥ `<conflict>`/Abstain 프로토콜 코드 반영.
*   **Phase 2 — 저렴·치명 검증 4건 (2~3주, 학습 전):** ⑦ closed-book 기저선(전 모델×전 데이터) ⑧ 결정론 recency+프롬프팅을 **outdated mode**에서(존재 이유 선확인) ⑨ 표면 유사도 파일럿 ⑩ **무학습 ReAct 에이전트** 성능·토큰소모 측정(제안 모델 경제성 논리 확보; 깨끗한 케이스만 이기면 잔여 난제로 scope 좁혀 학습 정당화, 전부 이기면 벤치마크 논문 피벗).
*   **Phase 3 — 데이터 구축 (병행 시작, 3~5주):** ⑪ TeCoQA 96K 생성 + E\* 라벨 전파 + 200건 사람 검수 ⑫ StreamingQA 충돌 합성 + Hard Negative Set 구축 ⑬ CLARK-News·Unified Clark 변환, 일반 QA(20%)·rag_conflicts·SituatedQA 포맷 정리.
*   **Phase 4 — 학습 (3~4주, 임계 경로):** ⑭ SFT→GRPO ×시드 3, 체크포인트별 보상 해킹 감사 ⑮ 병렬: baseline 1~5,7 평가 / TRACE·#8·#9 재현 조기 착수(재현 리스크).
*   **Phase 5 — 메인 평가·분석 (3주):** ⑯ 메인 표(9 baseline×4지표×3모드) ⑰ Ablation(답-앵커 vs gold-앵커, 곱셈 vs 비대칭 포함) ⑱ LOO·probe·유사도 본실험·CI·사람검증.
*   **Phase 6 — 일반화·배포 (2주):** ⑲ 6단 사다리 + 전이 1건 ⑳ Plug-in 가드.
*   **Phase 7 — 집필 (Phase 5부터 병행):** 진단 섹션 독립 완결 구조(벤치마크 논문 전환 보험).

**이번 주 3건:** ① Related Works 대조 착수 ② E\* 집합 정의·비대칭 보상 코드 수정 ③ closed-book·recency·무학습 에이전트 baseline 스크립트.

---

## 9. 타깃 학회·플랜 B
*   주 타깃: ACL/EMNLP long, TACL. (벤치마크 확장 시 NeurIPS/ICLR D&B 병행 검토.)
*   플랜 B (GRPO 결과 부진 시): 진단+벤치마크+집합형 GoldCite+baseline 전수평가의 자원 논문으로 피벗. Phase 0~3, Exp 1이 그대로 자산.

---

## 10. 표준 수치·용어 및 정합성 수정 목록

**표준 수치 (본 문서·논문 단일 기준 — v1/v2 원문 불일치 무시):** mis-attribution rate **81~100%**(5모델×A·B) · CitePrec **97~100%** vs GoldCite **44~82%**(괴리 **18~54%p**, 헤드라인) · faithful-wrong **59~76%**(진짜 LOO, 오픈3모델×A·B; 근사값 폐기) · Under-cited **≈0%** · 데이터 깔때기 599→152→**219**(과거180+현재39) · rag_conflicts **51**.

**용어:** TV-RAG(시스템, 잠정·후보 GRACE) · TeCoQA(데이터셋, 구 "TC-RAG") · GoldCite(정답-앵커, 집합 E\*) · mis-attribution rate(=P(CitePrec=1|EM=0)) · E\*(시점-유효 문서 집합) · 시점-조건부 QA(≠freshness).

**정합성 수정 (Phase 0):**
- [ ] 152→219 건수 흐름(기계정제→사람검수 수치 증가 서술 오류) 통일
- [ ] mis-attr 범위 통일(81~100 vs 92~100 vs 97~100 혼재)
- [ ] faithful-wrong 통일(43~81% → 59~76% 진짜 LOO)
- [ ] HoH 명칭 통일(History of History / History of Holders / 원문 대조 후 확정, 약어 "HoH"만 확정 사용)

**참고문헌 검증 상태 (전수 원문 확인 완료):**
- **인용·평가 계보:** KILT(2009.02252) · Bohnet Attributed QA(2212.08037) · ALCE(2305.14627) · AIS(2112.12870) · Huang fine-grained rewards(2402.04315) · Wallat(2412.18004) · Cited-but-Not-Verified(2605.06635) · Verified Misguidance(2605.28565) · Longpre Entity Knowledge Conflicts(2109.05052)
- **충돌·시간 벤치/데이터:** HoH(2503.04800) · CONFRAG(ACL 2026, 2026.acl-long.11) · DRAGged/CONFLICTS(2506.08500) · StreamingQA(2205.11388) · Unified Clark/ERASE·CLARK(2506.07270 / 2410.10584) · TRAM(2310.00835) · FinTMMBench(2503.05185) · Zhang 판례파기(2510.20941) · MADAM-RAG/RAMDocs(2504.13079)
- **RL-RAG·해결 시스템(baseline·경쟁):** TRACE(2505.13258, evidence+citation+정확도 보상) · RAG-RL(2503.12759, R_answer+R_citation, 답-앵커) · CaRR(2601.06021) · ConflictRAG(2605.17301) · Time-R1(2505.13508, RL·parametric) · Memory-T1(2512.20092) · Astute RAG(2410.07176) · VersionRAG(2510.08109) · T-GRAG(2508.01680) · TempRetriever(2502.21024) · Don't-Ask-LLM(2606.01435)
- **제외:** AMAQA(2505.13557) — 주장("EM +14") 허구·시간 무관 (§6.3).
- 과신 근거: Ding(2501.01303) · Si(2310.12558) · Sharma(2402.05880) · Liu(2304.09848). 기타 목록(§6): Schreieder survey(2508.15396), AttributionBench(2402.15089), Search-R1(2503.09516), ConflictBank(2408.12076), SituatedQA 등.
