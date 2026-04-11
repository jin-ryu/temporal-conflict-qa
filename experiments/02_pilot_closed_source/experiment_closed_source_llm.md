# TV-RAG 파일럿 스터디: 두 가지 실험 계획

## 전체 목적

논문의 Section 2 (Motivation/Pilot Study)를 구성하는 두 가지 독립적 주장을 실증한다.

| 주장 | 대응 실험 |
|---|---|
| "웹 검색과 내부 도구를 모두 쓸 수 있는 환경에서도 시간적 충돌은 문제인가?" | 실험 A |
| "닫힌 RAG 환경에서 충돌 컨텍스트는 LLM 성능을 급락시키는가?" | 실험 B |

### 사용 모델 및 환경

| 실험 | 모델 | 접속 방법 | 웹 검색 |
|---|---|---|---|
| 실험 A | Gemini 3 Flash | gemini.google.com | ON (기본값 유지) |
| 실험 B | Gemini 3 Flash | aistudio.google.com | OFF (기본값이 OFF) |

> 실험 B에 AI Studio를 쓰는 이유: Gemini 앱(gemini.google.com)은 웹 검색을 끄는 기능을 제공하지 않음. AI Studio는 검색 그라운딩이 기본값 OFF이며 Gemini 3 Flash 모델을 동일하게 사용 가능.

***

## 공통 샘플 구성 (총 100개)

두 실험 모두 동일한 100개 샘플을 사용한다.

### TC-RAG 데이터셋 전체 모드 분포

| Mode | 전체 건수 | 설명 |
|---|---|---|
| current_raw | 599 | 원본 질문 그대로 + 최신 답변 |
| current | 599 | 시간 표현 명시 질문 + 최신 답변 |
| outdated_0 | 590 | 1단계 이전 시점 질문 + 과거 답변 |
| outdated_1 | 36 | 2단계 이전 |
| outdated_2 | 11 | 3단계 이전 |
| outdated_3 | 8 | 4단계 이전 |
| outdated_4 | 2 | 5단계 이전 |
| outdated_5 | 1 | 6단계 이전 |

### 샘플링 전략

| Mode | 샘플 수 | 선택 이유 |
|---|---|---|
| current_raw | 40개 | 시간 힌트 없음 → 핵심 난이도, 본 연구의 주요 타겟 케이스 |
| current | 30개 | 최신 힌트 있음 → 쉬운 케이스, 대조군 역할 |
| outdated_0 | 20개 | 과거 시점 지칭 → 역방향 추론 케이스 |
| outdated_1 이상 | 10개 | 다단계 충돌 케이스 (outdated_2~5 전수 포함 권장) |
| 합계 | 100개 | |

### 충돌 레이어별 분포

| 레이어 | 설명 | 샘플 수 |
|---|---|---|
| 2-layer | outdated 1개 + current 1개 | 50개 |
| 3-layer | outdated 2개 + current 1개 | 30개 |
| 4-layer 이상 | outdated 3개 이상 (outdated_2~5 전수 포함 권장) | 20개 |

***

## 실험 A: 열린 환경 — gemini.google.com (웹 검색 ON)

### 목적

웹 검색과 내부 도구를 포함한 Gemini 3 Flash의 full capability 상태에서 TC-RAG 질문에 대해 어느 수준의 정답률을 보이는지 측정한다.

교수님의 "제미나이에 직접 물어보니 잘 되던데"라는 방식을 그대로 재현하는 것이 목적이다.

### 실험 방법

1. gemini.google.com 접속
2. 모델 선택에서 Gemini 3 Flash 선택
3. 웹 검색 그라운딩 기본값 유지 (ON 상태)
4. 질문만 단독 입력
5. 매 질문마다 새 대화창 시작 (이전 컨텍스트 오염 방지)

### 프롬프트 템플릿

질문 그대로 입력

### 결과 기록 형식

| idx | mode | 레이어 수 | 정답 | A 답변 | A 정답 |
|---|---|---|---|---|---|
| 0 | current_raw | 2 | Maudiozyma bulderi | ... | O/X |

### 평가 지표

| 지표 | 정의 |
|---|---|
| Answer Accuracy | 정답과 일치하는 응답 비율 |

### 분석 계획

- 전체 100개 Answer Accuracy
- 모드별 분해: current_raw vs current vs outdated
- 레이어 복잡도별 분해: 2-layer vs 3-layer vs 4-layer 이상

### 결과 해석 기준

| 결과 | 해석 | 논문 활용 방향 |
|---|---|---|
| 90% 이상 | 도구 있으면 해결됨 | "도구 없는 닫힌 RAG(실험 B)는 여전히 미해결" 강조 |
| 70~90% | 도구 있어도 완전하지 않음 | "문제의 범위가 열린 환경까지 확장됨" 강조 |
| 70% 미만 | 도구도 취약 | "문제가 예상보다 더 심각함" 강조 |

> 어느 결과가 나와도 논문에 활용 가능.

***

## 실험 B: 닫힌 RAG 환경 — aistudio.google.com (웹 검색 OFF, 충돌 컨텍스트 주입)

### 목적

TV-RAG의 핵심 타겟 환경인 닫힌 RAG에서 current + outdated 청크가 동시에 주어졌을 때 LLM이 올바른 답변과 증거 문서를 선택하는지 검증한다.

### 재현하는 시나리오

기업 내부 문서 DB, 법률 판례 시스템, 의료 가이드라인 등 외부 인터넷 없이 내부 코퍼스만 사용하는 엔터프라이즈 RAG 환경.

### 웹 검색 OFF 설정 방법

1. aistudio.google.com 접속
2. 모델 선택에서 Gemini 3 Flash 선택
3. 우측 설정 패널에서 Tools → Search Grounding이 OFF인지 확인 (기본값 OFF)
4. 매 테스트마다 새 채팅창에서 시작

### 실험 설계

| 조건 | 입력 | 목적 |
|---|---|---|
| no_conflict (통제군) | 질문 + current 청크 + distractor 청크 | 충돌 없는 긴 컨텍스트 → 컨텍스트 길이 효과 분리 |
| conflict (핵심) | 질문 + current 청크 + outdated 청크 + distractor 청크 | 시간적 충돌 상황에서의 성능 저하 실증 |

> no_conflict에서 성능이 괜찮고 conflict에서만 급락하면 "충돌 자체가 원인"임을 증명.

### 프롬프트 템플릿

You are an AI assistant. Use ONLY the provided documents below to answer the question. Do NOT use your own knowledge or search the web.

After answering, you MUST cite the document number you used as your primary evidence (e.g., "Evidence: [Document 2]").

[Document 1] [last_modified: YYYY-MM-DD]
(청크 텍스트)

[Document 2] [last_modified: YYYY-MM-DD]
(청크 텍스트)

[Document 3] [last_modified: YYYY-MM-DD]
(청크 텍스트)

Question: (질문)

> 주의: 청크 순서를 랜덤하게 섞어 제시할 것 (position bias 방지). Evidence 인용을 강제해야 Evidence Accuracy 측정 가능. 답이 맞아도 틀린 문서를 인용하면 Evidence Accuracy = 0으로 처리.

### 결과 기록 형식

| idx | mode | 레이어 수 | 정답 | no_conflict 답변 | no_conflict 정답 | conflict 답변 | conflict 정답 | conflict Evidence | conflict Evidence 정답 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | current_raw | 2 | Maudiozyma bulderi | ... | O/X | ... | O/X | [Doc N] | O/X |

### 평가 지표

| 지표 | 정의 | 적용 조건 |
|---|---|---|
| Answer Accuracy | 정답과 일치하는 응답 비율 | no_conflict, conflict 모두 |
| Evidence Accuracy | ground truth evidence_chunk_id와 인용 문서 일치 비율 | conflict만 |
| Combined Score | Answer Accuracy × Evidence Accuracy | conflict만 |

### 분석 계획

1차: 조건별 Answer Accuracy 비교
- no_conflict vs conflict → 차이가 클수록 충돌 자체가 원인임을 증명

2차: 충돌 레이어 복잡도별 성능 하락
- 2-layer / 3-layer / 4-layer 이상에서 conflict 성능 분해
- 레이어가 높을수록 성능 하락 경향 확인

3차: 쿼리 모드별 성능 비교
- current_raw vs current vs outdated 각각에서 conflict 성능
- current_raw에서 가장 취약할 것으로 예상

### 판단 기준

| 결과 | 해석 | 다음 행동 |
|---|---|---|
| conflict이 no_conflict 대비 15%p 이상 하락 | 문제 실존 확인 | TV-RAG 방향 유지, 실험 2 진행 |
| conflict이 no_conflict 대비 5~15%p 하락 | 문제 약함 | 복잡 레이어 케이스 추가 후 재평가 |
| conflict이 no_conflict 대비 5%p 미만 하락 | 문제 미약 | 주제 방향 재검토 필요 |

***

## 두 실험의 논문 내 활용 구조

실험 A: "Gemini 3 Flash (웹 검색 + 내부 툴 포함)에서도 TC-RAG 질문에 대한 Answer Accuracy는 XX%다"

실험 B: "웹 검색 없이 충돌 컨텍스트가 주어진 닫힌 RAG 환경에서는 Answer Accuracy가 XX%p 급락하고, Evidence Accuracy는 XX%에 불과하다"

결론: "도구가 없는 엔터프라이즈 RAG 환경에서 시간적 충돌은 현재 LLM이 해결하지 못하는 실재하는 문제다"

TV-RAG 제안: "이를 RL 기반 Generator 학습으로 해결한다"

***

## 실험 순서 및 예상 소요 시간

| 순서 | 단계 | 작업 | 예상 시간 |
|---|---|---|---|
| 1 | 준비 | 100개 샘플 추출 및 스프레드시트 구성 | 2~3시간 |
| 2 | 실험 A | gemini.google.com, 100개 질문 입력 | 2~3시간 |
| 3 | — | 실험 A 결과 확인 → 연구 진행 여부 판단 | — |
| 4 | 실험 no_conflict | aistudio.google.com, 충돌 없는 컨텍스트 100개 | 3~4시간 |
| 5 | 실험 conflict | aistudio.google.com, 충돌 컨텍스트 100개 | 3~4시간 |
| 6 | 분석 | 결과 집계 + 지표 계산 + 시각화 | 2~3시간 |
| 합계 | | | 약 2~3일 |

***

## 주의사항

1. 실험 A → B 순서로 진행. A 결과를 보고 B 진행 여부 판단.
2. 실험 B는 반드시 AI Studio에서 진행. Gemini 앱은 웹 검색 비활성화 불가.
3. 동일 샘플에 대해 A / no_conflict / conflict을 별도 대화창에서 각각 테스트.
4. Evidence 인용 없는 응답은 Evidence Accuracy = 0으로 처리.
5. outdated_2~5 케이스(총 22개)는 가능하면 전수 포함.
6. 정답 판정: Exact match 우선, 표현이 다를 경우 연구자 수동 판정. outdated 모드는 target_answer가 과거 정답임에 주의.