# 데이터셋 심층 분석 보고서

> **대상 파일**: `data/qa/hoh_qa_meta-llama-3.1-70b-instruct-awq-int4_0_600.jsonl`
>
> **분석 목적**: 시간적 충돌(Temporal Conflict) QA 연구에 적합한 데이터셋인지 다각도 평가
>
> **분석일**: 2026-03-25

---

## 1. 데이터셋 개요

| 항목 | 값 |
|------|-----|
| 총 레코드 수 | 1,846건 |
| 고유 소스 문서 수 | 599개 (idx 0~599) |
| LLM 모델 | `Meta-Llama-3.1-70B-Instruct-AWQ-INT4` (vllm) |
| Chunk label 종류 | `current`, `outdated`, `distractor` |

### 1.1 모드별 분포

| Mode | 건수 | 설명 |
|------|------|------|
| `current_raw` | 599 | 원본 질문 그대로 + 최신 답변 (베이스라인) |
| `current` | 599 | LLM이 시간 표현을 명시한 질문 + 최신 답변 |
| `outdated_0` | 590 | 1단계 이전 시점 질문 + 과거 답변 |
| `outdated_1` | 36 | 2단계 이전 시점 질문 + 과거 답변 |
| `outdated_2` | 11 | 3단계 이전 |
| `outdated_3` | 8 | 4단계 이전 |
| `outdated_4` | 2 | 5단계 이전 |
| `outdated_5` | 1 | 6단계 이전 |

### 1.2 소스별 모드 조합

| 조합 | 소스 수 |
|------|--------|
| current + current_raw + outdated_0 | 555 |
| current + current_raw + outdated_0~1 | 24 |
| current + current_raw만 (outdated 없음) | 8 |
| current + current_raw + outdated_0~3 | 6 |
| current + current_raw + outdated_0~2 | 3 |
| 기타 (4~5단계) | 3 |

### 1.3 핵심 필드 설명

- **original_question**: 원본 질문
- **new_question**: LLM이 시간적 맥락을 추가하여 재구성한 질문
- **target_answer**: 해당 모드의 정답
- **reasoning_qa**: LLM이 생성한 시간적 추론 근거 (vllm 생성 레코드에만 존재, 1,247건)
- **chunks**: current/outdated/distractor 청크 배열 (각각 `last_modified_time` 포함)

---

## 2. 시간적 충돌 설계 분석

### 2.1 핵심 메커니즘

모든 레코드에 **current 청크**(최신)와 **outdated 청크**(과거)가 함께 포함되어, 모델이 시간적 맥락에 따라 올바른 정보를 선택해야 하는 충돌 상황을 구현한다. 하나의 원본 질문(`original_question`)에서 3가지 모드의 질문이 파생된다:

- **current_raw**: 원본 질문(`original_question`)을 변환 없이 그대로 `new_question`에 사용. 시간적 힌트가 없으므로, current/outdated 청크가 동시에 주어졌을 때 모델이 어떤 답을 선택하는지 테스트하는 충돌 상황용 모드.
- **current**: LLM이 원본 질문을 기반으로, **"최신 시점"을 명시하는 표현을 추가하여 질문을 재구성**한 모드. 예) "~의 인구는?" → "가장 최근 인구조사 기준, ~의 인구는?" 정답은 current 청크의 최신 정보.
- **outdated_0~N**: LLM이 원본 질문을 기반으로, **"특정 과거 시점"을 지칭하는 표현을 추가하여 질문을 재구성**한 모드. 예) "~의 인구는?" → "이전 인구조사 기준, ~의 인구는?" 정답은 해당 outdated 청크의 과거 정보. 숫자가 클수록 더 오래된 시점을 의미.

아래 예시들에서 각 모드별 질문이 어떻게 구성되는지 원문과 번역을 함께 제시한다.

---

#### 예시 1 — 과학/분류학: 효모 학명 변경 (hoh_source_idx: 0)

**원본 질문 (original_question)**:
> "What is the name of the yeast that can be used to ferment gluconolactone to ethanol and carbon dioxide?"
> (글루코노락톤을 에탄올과 이산화탄소로 발효시킬 수 있는 효모의 이름은 무엇인가?)

| 모드 | new_question (영어 원문) | 한글 번역 | 정답 |
|------|--------------------------|-----------|------|
| **current_raw** | "What is the name of the yeast that can be used to ferment gluconolactone to ethanol and carbon dioxide?" | "글루코노락톤을 에탄올과 이산화탄소로 발효시킬 수 있는 효모의 이름은 무엇인가?" | **Maudiozyma bulderi** |
| **current** | "When the latest research on yeast was conducted, what was the name of the yeast that can be used to ferment gluconolactone to ethanol and carbon dioxide?" | "최신 효모 연구가 수행되었을 때, 글루코노락톤을 에탄올과 이산화탄소로 발효시킬 수 있는 효모의 이름은?" | **Maudiozyma bulderi** |
| **outdated_0** | "What type of yeast was used to ferment gluconolactone to ethanol and carbon dioxide before the recent reclassification of yeast species?" | "최근 효모 종의 재분류 이전에, 글루코노락톤을 에탄올과 이산화탄소로 발효시키는 데 사용된 효모는?" | **Saccharomyces bulderi** |

**제공되는 청크 (chunks)**: 총 6개 (current 1개, outdated 1개, distractor 4개)
- **current 청크** (수정일: 2024-07-01): "...the yeast **Maudiozyma bulderi** can ferment gluconolactone..."
- **outdated 청크** (수정일: 2024-06-01): "...the yeast **Saccharomyces bulderi** can ferment gluconolactone..."

→ *학명 재분류라는 실제 과학적 변화를 반영. 동일 문장에서 효모 이름만 바뀌어 있어, 모델이 시간 맥락에 따라 올바른 학명을 선택해야 함.*

---

#### 예시 2 — 경찰서장 교체 (hoh_source_idx: 15)

**원본 질문 (original_question)**:
> "Who is the current chief of the Fullerton Police Department?"
> (현재 풀러턴 경찰서장은 누구인가?)

| 모드 | new_question (영어 원문) | 한글 번역 | 정답 |
|------|--------------------------|-----------|------|
| **current_raw** | "Who is the current chief of the Fullerton Police Department?" | "현재 풀러턴 경찰서장은 누구인가?" | **Jon Radus** |
| **current** | "Who is the current chief of the Fullerton Police Department at the present time?" | "현 시점에서 풀러턴 경찰서장은 누구인가?" | **Jon Radus** |
| **outdated_0** | "Who was the chief of the Fullerton Police Department shortly before the chief stepped down due to health concerns?" | "서장이 건강 문제로 사임하기 직전 풀러턴 경찰서장은 누구였는가?" | **David Hendricks** |

**제공되는 청크**: 총 10개 (current 2개, outdated 1개, distractor 7개)
- **current 청크** (수정일: 2024-12-01): "...The current chief is **Jon Radus**..."
- **outdated 청크** (수정일: 2024-11-01): "...The current chief is **David Hendricks**..."

→ *인사 교체라는 현실적 변화 반영. outdated 질문에서 "건강 문제로 사임하기 직전"이라는 간접적 시간 표현 사용.*

---

#### 예시 3 — 수상 실적 변화 (hoh_source_idx: 534)

**원본 질문 (original_question)**:
> "How many times has The Washington Post won the Pulitzer Prize for its work?"
> (워싱턴 포스트는 퓰리처상을 몇 번 수상했는가?)

| 모드 | new_question (영어 원문) | 한글 번역 | 정답 |
|------|--------------------------|-----------|------|
| **current_raw** | "How many times has The Washington Post won the Pulitzer Prize for its work?" | "워싱턴 포스트는 퓰리처상을 몇 번 수상했는가?" | **76** |
| **current** | "How many times has The Washington Post won the Pulitzer Prize for its work as of the latest available information?" | "최신 정보 기준으로, 워싱턴 포스트는 퓰리처상을 몇 번 수상했는가?" | **76** |
| **outdated_0** | "How many times had The Washington Post won the Pulitzer Prize for its work when the newspaper was still owned by the Graham family?" | "워싱턴 포스트가 아직 그레이엄 가문 소유였을 때, 퓰리처상을 몇 번 수상한 상태였는가?" | **73** |

→ *누적 수상 횟수가 시간에 따라 증가하는 패턴. outdated 질문에서 "그레이엄 가문 소유 시절"이라는 역사적 맥락으로 시점을 특정.*

---

#### 예시 4 — 스포츠 선수 이적: 3단계 시간 레이어 (hoh_source_idx: 569)

**원본 질문 (original_question)**:
> "Which team does Callum Crawford play for in the National Lacrosse League?"
> (Callum Crawford는 내셔널 라크로스 리그(NLL)에서 어느 팀 소속인가?)

| 모드 | new_question (영어 원문) | 한글 번역 | 정답 |
|------|--------------------------|-----------|------|
| **current_raw** | "Which team does Callum Crawford play for in the National Lacrosse League?" | "Callum Crawford는 내셔널 라크로스 리그(NLL)에서 어느 팀 소속인가?" | **Philadelphia Wings** |
| **current** | "Which team does Callum Crawford play for in the National Lacrosse League at the time of the latest update?" | "최신 업데이트 기준, Callum Crawford는 NLL 어느 팀 소속인가?" | **Philadelphia Wings** |
| **outdated_0** | "Which team did Callum Crawford play for in the National Lacrosse League before signing with the Philadelphia Wings?" | "Philadelphia Wings와 계약하기 전, Callum Crawford는 NLL 어느 팀이었는가?" | **Panther City Lacrosse Club** |
| **outdated_1** | "Which team did Callum Crawford play for in the National Lacrosse League shortly before being drafted in the NLL Dispersal Draft?" | "NLL Dispersal Draft에서 드래프트되기 직전, Callum Crawford는 어느 팀이었는가?" | **San Diego Seals** |

**제공되는 청크**: 총 4개 (current 1개, outdated 2개, distractor 1개)
- **current 청크** (2024-12-01): "...plays for the **Philadelphia Wings**..."
- **outdated 청크 0** (2024-09-01): "...plays for **Panther City Lacrosse Club**..."
- **outdated 청크 1** (2024-10-01): "...plays for the **San Diego Seals**..."

→ *3단계 시간 레이어로 선수의 이적 이력을 추적. 각 시점마다 소속팀이 다르며, 모델은 질문의 시간적 맥락에 맞는 청크를 선택해야 함.*

---

#### 예시 5 — 뉴질랜드 축구 대표팀 우승 기록 (hoh_source_idx: 547)

**원본 질문 (original_question)**:
> "How many times has New Zealand won the OFC Nations Cup?"
> (뉴질랜드는 OFC 네이션스컵을 몇 번 우승했는가?)

| 모드 | new_question (영어 원문) | 한글 번역 | 정답 |
|------|--------------------------|-----------|------|
| **current_raw** | "How many times has New Zealand won the OFC Nations Cup?" | "뉴질랜드는 OFC 네이션스컵을 몇 번 우승했는가?" | **6** |
| **current** | "How many times has New Zealand won the OFC Nations Cup as of the latest update?" | "최신 업데이트 기준, 뉴질랜드는 OFC 네이션스컵을 몇 번 우승했는가?" | **6** |
| **outdated_0** | "How many times had New Zealand won the OFC Nations Cup at the time they last participated in the FIFA Confederations Cup before the COVID-19 pandemic?" | "COVID-19 팬데믹 이전에 뉴질랜드가 마지막으로 FIFA 컨페더레이션스컵에 참가했을 시점까지, OFC 네이션스컵 우승 횟수는?" | **5** |

→ *누적 우승 횟수가 시간에 따라 증가. outdated 질문에서 "COVID-19 이전 마지막 컨페더레이션스컵 참가 시점"이라는 복합적 시간 참조 사용.*

---

#### 예시 6 — 축구 대표팀 기록 기준 경기: 4단계 충돌 (hoh_source_idx: 103)

**원본 질문 (original_question)**:
> "What match is used as the reference point for the correctness of caps and goals?"
> (출전 경기 수(caps)와 골 수(goals)의 정확성을 위한 기준 경기는 무엇인가?)

| 모드 | new_question (영어 원문) | 한글 번역 | 정답 |
|------|--------------------------|-----------|------|
| **current_raw** | "What match is used as the reference point for the correctness of caps and goals?" | "출전 경기 수(caps)와 골 수(goals)의 정확성을 위한 기준 경기는 무엇인가?" | **The second match against Kenya.** (케냐와의 두 번째 경기) |
| **current** | "What match is used as the reference point for the correctness of caps and goals at the time of the current squad selection?" | "현재 스쿼드 선발 시점 기준, caps와 goals 정확성의 기준 경기는?" | **The second match against Kenya.** |
| **outdated_0** | "When the policy regarding kit suppliers was being updated, what match was used as the reference point for the correctness of caps and goals?" | "유니폼 공급사 정책이 업데이트되던 시점에, 기준 경기는?" | **The match against Nigeria.** (나이지리아전) |
| **outdated_1** | "What match is used as the reference point for the correctness of caps and goals before the Africa Cup of Nations qualification matches against Cape Verde?" | "아프리카 네이션스컵 카보베르데 예선전 이전에, 기준 경기는?" | **The match against Angola.** (앙골라전) |
| **outdated_2** | "What match is used as the reference point for the correctness of caps and goals at the time of the Africa Cup of Nations qualification matches against Namibia and Zimbabwe?" | "아프리카 네이션스컵 나미비아·짐바브웨 예선전 시점에, 기준 경기는?" | **The match against Zimbabwe.** (짐바브웨전) |

**제공되는 청크**: 총 10개 (current 2개, outdated 3개, distractor 5개)

→ *4단계에 걸쳐 기준 경기가 계속 바뀌는 복잡한 시간적 충돌. 각 outdated 질문이 서로 다른 시점을 이벤트 기반으로 지칭함.*

---

#### 예시 7 — 축구 선수 주장 출전 기록: 5단계 충돌 (hoh_source_idx: 177)

**원본 질문 (original_question)**:
> "How many matches did Jan Oblak start as the team captain?"
> (Jan Oblak은 주장으로서 몇 경기에 선발 출전했는가?)

| 모드 | new_question (영어 원문) | 한글 번역 | 정답 |
|------|--------------------------|-----------|------|
| **current_raw** | "How many matches did Jan Oblak start as the team captain?" | "Jan Oblak은 주장으로서 몇 경기에 선발 출전했는가?" | **50** |
| **current** | "How many matches did Jan Oblak start as the team captain as of the latest update?" | "최신 업데이트 기준, Jan Oblak의 주장 선발 출전 수는?" | **50** |
| **outdated_0** | "How many matches did Jan Oblak start as the team captain at the time the player records were last updated in the summer?" | "여름에 선수 기록이 마지막으로 업데이트되었을 때, 주장 선발 출전 수는?" | **44** |
| **outdated_1** | "As of the summer of the same year the national team played its first match against the Faroe Islands, which team captain started the most matches?" | "대표팀이 페로 제도와 첫 경기를 치른 해 여름 기준, 가장 많이 선발 출전한 주장은?" | **45** |
| **outdated_2** | "How many matches did Jan Oblak start as the team captain around the time the team's recent call-ups were announced?" | "최근 대표팀 소집 명단이 발표될 무렵, 주장 선발 출전 수는?" | **46** |

**제공되는 청크**: 총 10개 (current 2개, outdated 4개, distractor 4개)

→ *5단계의 시간 레이어에서 출전 수가 44→45→46→...→50으로 점진적으로 증가. 모델은 각 시점 표현을 정확히 해석하여 맞는 숫자를 골라야 함.*

### 2.2 답변 변별력

| 변화 유형 | 건수 | 비율 |
|---|---|---|
| 텍스트 변화 (팀명, 인명, 장소 등) | 390 | 66.1% |
| 숫자 변화 (인구, 횟수, 점수 등) | 194 | 32.9% |
| 답변 동일 (충돌 없음) | 6 | 1.0% |

**99%의 레코드에서 current와 outdated_0의 답변이 다르므로** 시간적 충돌 평가에 매우 효과적이다.

동일 답변 6건은 outdated_0에서는 같지만 outdated_1에서 차이가 발생하는 경우(hoh_source_idx: 46, 65, 301, 310)이거나, 실질적 충돌이 없는 경우(hoh_source_idx: 542, 588)이다.

### 2.3 시간 메타데이터 분석

#### 청크 수정일 차이 분포

| current 수정일 | outdated 수정일 | 건수 |
|---|---|---|
| 2024-10 | 2024-09 | 128 |
| 2024-07 | 2024-06 | 103 |
| 2024-11 | 2024-10 | 96 |
| 2024-12 | 2024-11 | 87 |
| 2024-09 | 2024-08 | 79 |
| 2024-08 | 2024-07 | 69 |
| 2개월 이상 차이 | | 17 |

대부분(약 97%) 1개월 차이로 균일하다. `last_modified_time`은 **위키 문서의 편집 시점**을 반영하며, 정보 자체의 실제 유효 기간(예: 2010년 인구조사 vs 2020년 인구조사)과는 괴리가 있다.

### 2.4 키워드 누출(Keyword Leakage) 위험 분석

outdated 모드의 질문은 원본 질문에 시간 표현을 추가하여 재구성한 것이다. 이때 **추가된 키워드가 특정 청크와 직접 매칭**되면, 모델이 시간적 추론 없이 단순 키워드 유사도만으로 정답 청크를 찾는 shortcut이 가능해진다.

#### 누출 유형별 수치 (outdated_0 기준, 590건)

| 누출 유형 | 건수 | 비율 | 설명 |
|---|---|---|---|
| outdated 청크에만 매칭되는 키워드 존재 | 69 | 11.8% | 추가된 키워드가 outdated 청크에만 등장 → 키워드 매칭으로 해당 청크 검색 가능 |
| outdated 질문에 current 정답 포함 | 13 | 2.2% | 질문이 current 답을 직접 언급 → "제외법" shortcut 가능 |
| outdated 질문에 outdated 정답 포함 | 7 | 1.2% | 질문 자체에 정답이 노출 |

#### 시간 표현 유형별 누출 위험도

outdated_0 질문에 추가된 시간 표현을 유형별로 분류하면:

| 유형 | 비율 | 누출 위험 | 설명 |
|---|---|---|---|
| **"at the time of event"** (이벤트 참조) | 65.6% | 낮음 | 외부 이벤트로 시점을 간접 지칭하여 키워드 매칭이 어려움 |
| **"other"** (기타) | 15.9% | 혼합 | 다양한 표현 혼재 |
| **"previous/earlier"** (단순 과거 참조) | 11.7% | 낮음 | "이전 인구조사 기준" 등 일반적 과거 표현. 누출 위험은 낮으나 추론 난이도도 낮음 |
| **"before X"** (current 정보 언급) | 6.3% | **높음** | current 상태를 직접 언급하고 "그 이전"을 묻는 구조 |
| **특정 연도 명시** | 0.5% | 낮음 | 직접적인 연도 지칭 |

#### 누출 위험이 높은 "before X" 패턴 예시

이 유형은 질문에서 **current 정답을 직접 언급**하여, 모델이 "질문에 이미 나온 정보를 제외하고 나머지를 선택"하는 shortcut을 사용할 수 있다.

> **hoh_source_idx: 25**
> - 원문: "What was the nearest airport to Kuppalli **before Shivamogga Airport was built**?"
> - 번역: "**시바모가 공항이 건설되기 전에**, 쿠팔리에서 가장 가까운 공항은?"
> - current 정답(질문에 포함됨): **Shivamogga Airport**
> - outdated 정답(기대 답): **Mangalore International Airport**
> - 문제: 질문이 current 답(시바모가 공항)을 알려주므로, 이를 제외한 나머지를 고르면 됨

> **hoh_source_idx: 58**
> - 원문: "Which team did Grzegorz Kmiecik play for as a striker **before he joined Pogórze Gierałtowice**?"
> - 번역: "Grzegorz Kmiecik이 **Pogórze Gierałtowice에 합류하기 전에** 스트라이커로 뛴 팀은?"
> - current 정답(질문에 포함됨): **Pogórze Gierałtowice**
> - outdated 정답(기대 답): **Puls Broszkowice**

> **hoh_source_idx: 141**
> - 원문: "What was the name of the student-run restaurant at North Hertfordshire College **before The Meadows was introduced**?"
> - 번역: "**The Meadows가 도입되기 전에**, North Hertfordshire College의 학생 운영 레스토랑 이름은?"
> - current 정답(질문에 포함됨): **The Meadows**
> - outdated 정답(기대 답): **Hart Kitchens**

#### 누출 위험이 낮은 "at the time of event" 패턴 예시

이 유형은 **외부 이벤트로 시점을 우회 지칭**하므로, 질문의 키워드만으로는 정답 청크를 특정하기 어렵다.

> **hoh_source_idx: 1**
> - 원문: "What was the median income in Otago compared to the national median income **shortly before the most recent economic downturn**?"
> - 번역: "**가장 최근 경기 침체 직전**, 오타고의 중위 소득은 전국 중위 소득과 비교하여 얼마였는가?"
> - "경기 침체"라는 표현은 어떤 청크에도 직접 등장하지 않음 → 키워드 매칭 불가

> **hoh_source_idx: 103**
> - 원문: "**When the policy regarding kit suppliers was being updated**, what match was used as the reference point for the correctness of caps and goals?"
> - 번역: "**유니폼 공급사 정책이 업데이트되던 시점에**, caps와 goals 정확성의 기준 경기는?"
> - "정책 업데이트"라는 표현은 시점을 간접 지칭 → 시간적 추론 필요

#### 종합 평가

전체 outdated_0 질문의 약 **88%는 키워드 shortcut으로 풀기 어려운 구조**이며, 특히 가장 큰 비중(65.6%)을 차지하는 "at the time of event" 유형은 시간적 추론을 실제로 요구한다. 그러나 **약 12%에서는 키워드 누출 위험이 존재**하며, 특히 "before X" 패턴(6.3%)은 current 정답을 질문에 노출시키는 구조적 문제가 있다.

#### 실제 검색 환경과의 비교를 통한 재검증

키워드 누출이 실질적으로 얼마나 문제가 되는지, 실제 검색 환경의 특성과 대조하여 추가 검증하였다.

**1) current vs outdated 청크 간 텍스트 유사도**

같은 문서의 시간대별 버전(current/outdated)이 얼마나 유사한지 측정하였다.

| 유사도 구간 | 비율 | 의미 |
|---|---|---|
| >0.95 (거의 동일, 숫자/이름만 변경) | 48.5% | 검색 단계에서 둘을 구별 불가능 |
| 0.80~0.95 (일부 문장 추가/삭제) | 19.5% | 검색 점수 유사 |
| <0.80 (상당한 변경) | 32.0% | 검색 점수 차이 발생 가능 |

반면 **distractor 청크와의 유사도는 평균 0.075**로, current/outdated 간 유사도(평균 0.804)와 극명한 차이를 보인다. 이는 실제 위키피디아에서 **같은 문서의 과거 버전과 현재 버전이 거의 동일한 텍스트에 일부만 변경되는 현상과 일치**한다. 즉, retrieval 단계에서는 두 청크가 모두 동일한 relevance로 검색되며, **reader가 시간적 맥락을 추론하여 정답을 선택하는 것이 핵심 과제**가 되는 현실적 구조이다.

**2) 질문-청크 간 word overlap 비교**

각 모드의 질문이 current/outdated 청크 중 어느 쪽과 더 높은 단어 겹침을 보이는지 측정하였다.

| 질문 모드 | outdated 청크 더 높음 | current 청크 더 높음 | 동일 |
|---|---|---|---|
| outdated_0 | 11.7% | 10.7% | **77.6%** |
| current | 7.2% | 12.5% | **80.3%** |
| current_raw (원본) | 4.7% | 4.7% | **90.7%** |

**77~91%에서 두 청크의 word overlap 점수가 동일**하다. 즉 대부분의 경우 단순 키워드 유사도만으로는 정답 청크를 특정할 수 없으며, 시간적 추론이 반드시 필요하다. outdated_0 모드에서 outdated 청크가 더 높은 11.7%는 앞서 분석한 키워드 누출 비율(11.8%)과 정확히 일치하여, 누출이 실제 검색 편향으로 이어지는 범위를 수치적으로 확인할 수 있다.

**3) 시간 구별 근거: 텍스트 내 시간 표현 vs 메타데이터**

모델이 current/outdated 청크를 구별하는 데 사용할 수 있는 단서가 텍스트 내에 있는지, 메타데이터에만 의존해야 하는지 분석하였다.

| 구별 방법 | 비율 | 예시 |
|---|---|---|
| 텍스트 내 시간 표현으로 구별 가능 | 60.5% | "As of the **2020** census" vs "As of the **2010** census" |
| 메타데이터(`last_modified_time`)에만 의존 | 39.5% | 텍스트 동일, 이름/숫자만 변경 (예: 효모 학명 Maudiozyma vs Saccharomyces) |

60.5%의 청크는 텍스트에 연도 등 시간 표현이 포함되어 있어 모델이 텍스트만으로도 시점을 판단할 수 있다. 나머지 39.5%는 `last_modified_time` 메타데이터를 참조해야만 시간 순서를 파악할 수 있으므로, 메타데이터 활용 능력까지 평가하는 더 어려운 케이스에 해당한다.

**4) 종합: 12% 키워드 누출의 수용 가능성**

위 분석을 종합하면, 12%의 키워드 누출은 **실제 검색 환경의 맥락에서 수용 가능**하다고 판단한다.

- **전체 구조가 현실적이다**: 같은 문서의 시간대별 버전이 거의 동일한 텍스트인 것은 실제 위키피디아와 일치한다.
- **77.6%에서 word overlap이 동일**하여, 키워드 shortcut은 소수 케이스에서만 작동한다.
- **"before X" 패턴은 현실적 질문 방식이다**: "새 공항 생기기 전에 가장 가까운 공항은?" 같은 질문은 실제 사용자가 하는 방식이므로, 이를 완전히 제거하면 오히려 비현실적 데이터셋이 된다.
- **다만 논문 작성 시에는** 이 12%의 존재를 limitation으로 명시하고, "before X" 패턴을 제외한 서브셋에서도 평가 결과를 보고하면 더 견고한 실험 설계가 될 수 있다.

### 2.5 Distractor 청크 분포

| Distractor 수 | 소스 수 |
|---|---|
| 0개 | 8 |
| 1~4개 | 79 |
| 5~6개 | 75 |
| 7~8개 | 437 |

평균 약 6.5개의 distractor 청크가 포함되어, 실제 RAG 검색 환경에서의 노이즈를 잘 시뮬레이션한다.

### 2.6 reasoning_qa 필드

vllm이 생성한 1,247건의 레코드에 **시간적 추론 근거**가 포함되어 있다.

> **예시 (hoh_source_idx: 0)**:
> "current 레이블과 최신 수정일 2024-07-01을 가진 증거 청크가 'Maudiozyma bulderi'라는 답을 뒷받침한다. outdated (index 0) 레이블과 수정일 2024-06-01인 청크는 'Saccharomyces bulderi'라는 오래된 답변을 제공할 것이다."

이 필드는 모델의 시간적 추론 과정을 평가하거나 chain-of-thought 학습 데이터로 활용 가능하다.

---

## 3. 질문 다양성 분석

### 3.1 도메인별 분류

| 도메인 | 건수 | 비율 |
|---|---|---|
| 스포츠 (팀, 선수, 경기, 기록) | 141 | 23.5% |
| 인구통계 - 기타 (인구, 비율 등) | 51 | 8.5% |
| 역사/사건 | 39 | 6.5% |
| 미디어/문화 (영화, 음악, 수상) | 39 | 6.5% |
| 교육 (학교, 학생 수, 등록) | 36 | 6.0% |
| 인구통계 - 타운십 | 27 | 4.5% |
| 지리/장소 | 23 | 3.8% |
| 인물/전기 | 20 | 3.3% |
| 정치 | 19 | 3.2% |
| 과학/기술 | 5 | 0.8% |
| 기타 | 199 | 33.2% |

### 3.2 질문 시작 패턴 (상위 15)

| 패턴 | 출현 횟수 |
|---|---|
| "What is the population..." | 33 |
| "In what year did..." | 27 |
| "What is the name..." | 13 |
| "What percentage of people..." | 12 |
| "Who is the current..." | 9 |
| "Who are the opponents..." | 9 |
| "How many players were..." | 8 |
| "In what year was..." | 8 |
| "What percentage of the..." | 7 |
| "What is the student..." | 7 |
| "What match is used..." | 6 |
| "How many times has..." | 6 |
| "How many statistical areas..." | 6 |
| "What is the age..." | 5 |
| "How many players have..." | 4 |

### 3.3 답변 유형

| 유형 | 건수 | 비율 |
|---|---|---|
| 짧은 텍스트 (3단어 이하) | 269 | 44.9% |
| 순수 숫자 | 201 | 33.6% |
| 긴 텍스트 (4단어 이상) | 97 | 16.2% |
| 퍼센트 | 32 | 5.3% |

### 3.4 질문 핵심 키워드 빈도 (상위 15)

| 키워드 | 출현 횟수 |
|---|---|
| year | 70 |
| team | 67 |
| age | 59 |
| play | 50 |
| population | 47 |
| school | 36 |
| township | 27 |
| area | 20 |
| film | 15 |
| city, war | 각 14 |
| win, born | 각 13 |
| university | 12 |
| housing | 10 |
| score | 10 |

---

## 4. 다단계 시간 충돌 분석

이 데이터셋의 차별적 강점 중 하나는 **다단계 시간 레이어**가 존재하는 소스가 있다는 점이다.

### 4.1 최대 outdated 레이어 분포

| 최대 레이어 | 소스 수 |
|---|---|
| outdated 없음 | 8 |
| outdated_0까지 (1단계) | 555 |
| outdated_1까지 (2단계) | 25 |
| outdated_2까지 (3단계) | 3 |
| outdated_3까지 (4단계) | 6 |
| outdated_4까지 (5단계) | 1 |
| outdated_5까지 (6단계) | 1 |

### 4.2 다단계 충돌 상세 예시

**hoh_source_idx: 569 — Callum Crawford 이적 이력 (3단계):**

> | 시점 | 소속팀 |
> |------|--------|
> | current | Philadelphia Wings |
> | outdated_0 | Panther City Lacrosse Club |
> | outdated_1 | San Diego Seals |

**hoh_source_idx: 103 — 축구 대표팀 참조 경기 (4단계):**

> | 시점 | 참조 경기 |
> |------|-----------|
> | current | 케냐와의 두 번째 경기 |
> | outdated_0 | 나이지리아전 |
> | outdated_1 | 앙골라전 |
> | outdated_2 | 짐바브웨전 |

다단계 레코드는 전체의 약 6%(36건)에 해당하며, 모델의 세밀한 시간적 추론 능력을 테스트하는 데 유용하다.

---

## 5. 강점

### 5.1 대규모 + 높은 충돌률
599개 소스, 1,846개 레코드 규모이며 **99%에서 답변이 실제로 다르므로** 시간적 충돌 벤치마크로서 통계적 신뢰도가 높다.

### 5.2 다단계 시간 레이어
최대 6단계(outdated_5)까지의 시간 레이어가 존재하여, 단순 이진 충돌을 넘어 **연속적 정보 변화를 추적**하는 테스트가 가능하다.

### 5.3 reasoning_qa 필드
1,247건에 LLM이 생성한 시간적 추론 근거가 포함되어 있어, 평가 기준 수립이나 chain-of-thought 학습 데이터로 활용 가능하다.

### 5.4 자연스러운 질문 재구성
LLM이 시간 표현을 추가한 질문이 자연스럽다:
- 직접적: "최신 업데이트 기준으로..." / "가장 최근 인구조사에 따르면..."
- 간접적: "미국 인구가 3억 800만을 넘었을 무렵..." / "COVID-19 팬데믹 이전..."

### 5.5 Distractor 청크 구성
평균 6~7개의 distractor 청크가 포함되어 실제 RAG 시스템의 검색 노이즈를 잘 시뮬레이션한다.

### 5.6 다양한 답변 유형
숫자(33.6%), 짧은 텍스트(44.9%), 긴 텍스트(16.2%), 퍼센트(5.3%) 등 다양한 답변 형태를 포함한다.

---

## 6. 약점 및 개선 제안

### 6.1 도메인 편향: 스포츠 과다 — 심각도: Major

스포츠 관련 질문이 **23.5% (141건)**로 가장 큰 비중을 차지하며, 팀/선수/경기 관련 질문이 반복적이다. 특히 "대표팀 스쿼드 선발 기준 경기" 유형이 다수 포함되어 있다.

> **개선안**: 스포츠 비중을 15% 이하로 줄이고, 과학/기술(현재 0.8%), 정치(3.2%), 경제(거의 없음) 도메인을 보강

### 6.2 시간 메타데이터의 인위성 — 심각도: Major

`last_modified_time`이 97% 이상에서 정확히 **1개월 차이**로 설정되어 있다. 실제 정보의 시간적 거리(예: 2010년 vs 2020년 인구조사 = 10년 차이)와 메타데이터 차이(1개월)가 일치하지 않아, 시간 메타데이터 기반의 추론 평가에 한계가 있다.

> **개선안**:
> - `information_valid_date` 같은 별도 필드를 추가하여 정보의 실제 유효 시점 명시
> - 또는 `last_modified_time`을 정보의 실제 시간적 거리에 맞게 조정

### 6.3 시간적 충돌 구조의 강점 보완 참고

모든 레코드의 `chunks` 배열에 current 청크와 outdated 청크가 **이미 동시에 포함**되어 있어, 시간적 충돌 상황이 기본 구조에 내장되어 있다. 특히 `current_raw` 모드는 시간 힌트 없는 원본 질문 + 양쪽 청크 동시 제공으로, 모델이 충돌 상황에서 어떤 답을 선택하는지 직접 테스트할 수 있다.

> - `current_raw`: 시간 지정 없는 질문 + current/outdated 청크 동시 제공 → **충돌 상황 테스트**
> - `current`: 최신 시점 명시 질문 → 모델이 current 청크를 선택해야 함
> - `outdated_0~N`: 과거 시점 명시 질문 → 모델이 해당 outdated 청크를 선택해야 함
>
> 이 3가지 모드 조합으로 **충돌 탐지 + 시간 지정 정확도**를 함께 평가할 수 있는 구조이다.

### 6.4 outdated_0 누락 소스 — 심각도: Minor

9개 소스에서 `outdated_0` 모드가 누락되어 있다. 주로 대표팀 스쿼드, 영화 평점 등의 주제이다.

> 누락 소스: idx 176, 178, 318, 386, 408, 433, 496, 499, 503

### 6.5 질문 난이도 단조로움 — 심각도: Moderate

대부분이 **단순 사실확인형(factoid)** 질문이다. 시간적 추론 능력을 더 깊이 평가하려면 복합적 질문 유형이 필요하다.

> **부족한 질문 유형**:
> - 비교형: "2010년과 2020년 인구 차이는 얼마인가?"
> - 추론형: "인구 변화 추세로 볼 때 증가 추세인가 감소 추세인가?"
> - 다중 시점 종합형: "이 선수가 소속팀을 가장 많이 바꾼 시기는?"
> - 조건부형: "만약 2015년 데이터를 기준으로 한다면 답이 달라지는가?"

### 6.6 동일 답변 사례의 레이블 문제 — 심각도: Minor

6건의 소스에서 current와 outdated_0의 답변이 동일하다. 이 중 일부(hoh_source_idx: 46, 65, 310)는 outdated_1에서야 차이가 발생하므로 outdated_0의 존재 의의가 불명확하다.

> **개선안**: current vs outdated_0 답변이 동일한 경우 outdated_0을 제거하고 outdated_1을 outdated_0으로 재레이블링

---

## 7. 종합 평가

| 평가 항목 | 점수 (5점) | 비고 |
|---|---|---|
| 시간적 충돌 시뮬레이션 | 4 | current/outdated 구조 잘 설계, 다단계 레이어 존재 |
| 답변 변별력 | 5 | 99% 충돌률, 텍스트+숫자 혼합 |
| 질문 다양성 | 3 | 스포츠 편향 있으나 500-600 대비 크게 개선 |
| 도메인 균형 | 3 | 스포츠 과다, 과학/경제 부족 |
| 시간 메타데이터 현실성 | 2 | 1개월 고정 차이, 실제 시간 거리 미반영 |
| 난이도 다양성 | 2 | 대부분 단순 사실확인형 |
| 평가 도구 완성도 | 4 | current_raw/current/outdated 3모드 조합으로 충돌 테스트 가능, reasoning_qa 필드도 강점 |
| 데이터 규모 | 4 | 599 소스, 1,846 레코드로 통계적 유의미 |
| 다단계 시간 레이어 | 3 | 존재하나 36건(6%)으로 소수 |

### 최종 결론

이 데이터셋은 시간적 충돌 QA 연구의 **견고한 기반**을 제공한다. 500-600 슬라이스 대비 규모(6배), 도메인 다양성, 다단계 시간 레이어 측면에서 크게 개선되었다. 특히 reasoning_qa 필드와 99% 충돌률은 벤치마크로서의 실용성을 높인다.

**우선 개선 사항 3가지:**

1. **시간 메타데이터 현실화** — 정보의 실제 유효 시점을 반영하는 별도 필드 추가
2. **도메인 재균형** — 스포츠 비중 축소, 과학/경제/정치 도메인 보강
3. **다단계 시간 레이어 확충** — 현재 6%인 다단계 레코드를 늘려 복잡한 시간 추론 평가 강화
