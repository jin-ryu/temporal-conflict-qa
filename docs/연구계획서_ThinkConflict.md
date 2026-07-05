# 연구 계획서: Lost in Translation from Thought to Text: Investigating Reasoning-Answer Inconsistency in RAG Conflicts

*   **투고 목표:** ACL / EMNLP Main (Long Paper) — 진단 및 기계론적 분석 중심 (Diagnostic & Interpretability Track)
*   **한 줄 요약:** RAG의 외부 문맥 간 충돌(Inter-Context Conflict) 상황에서, Reasoning LLM의 내부 사고(`Thought`)와 최종 답변(`Text`) 간에 발생할 수 있는 **추론-답변 불일치(Reasoning-Answer Inconsistency)**를 체계적으로 진단하고 분석한다.

---

## 1. 서론 (Introduction)

### 1.1. 배경 및 문제 정의
검색 증강 생성(Retrieval-Augmented Generation, RAG)은 외부 지식을 활용해 대형언어모델(LLM)의 신뢰성을 제고하는 핵심 기술이지만, 실제 웹 검색 환경(In-the-wild)에서는 동일한 질의에 대해 상충되는 정보들이 동시에 반환되는 **외부 문맥 간 충돌(Inter-Context Conflict)**이 빈번히 발생합니다. 실제로 웹 검색 기반 RAG 벤치마크 연구에 따르면, 전체 질의의 약 **64.8%**에서 상위 검색 문서들 간에 사실관계 모순, 관점 차이, 시간적 불일치 등 다양한 형태의 지식 충돌이 존재하는 것으로 나타났습니다 (Cattan et al., 2025; Xu et al., 2024). 이러한 복수 문서 간의 모순과 상충은 모델이 신뢰할 수 있는 정보를 분별하지 못하게 하여, RAG 시스템의 답변 오류와 환각을 유발하는 주요 요인으로 작용합니다.

이러한 지식 충돌 문제를 극복하기 위해 기존 학계에서는 일반 LLM에 인위적인 성찰(Self-reflection)이나 프롬프트 기반의 단계적 사고(CoT)를 유도하여 충돌을 제어하려 시도해 왔습니다 (Asai et al., 2023; Hu et al., 2024). 그러나 이러한 선행 연구들의 접근법은 모델이 실제로 문맥 간의 모순을 이해하고 대조하여 결론을 내린 것인지, 아니면 정답을 맞추기 위해 설명 템플릿을 모방한 '사후 합리화(Unfaithful Rationalization)'에 불과한지 확인하기 어렵다는 한계를 지닙니다. 실제로 여러 연구들에 따르면, 일반적인 프롬프트 기반 CoT는 문맥이나 질문에 편향된 힌트를 추가할 경우 모델이 본래의 정답을 뱉기 위해 사고 과정을 그에 맞춰 왜곡하는 등 추론의 충실성(Faithfulness)이 결여될 수 있음이 보고되었습니다 (Turpin et al., 2023; Lyubomenko et al., 2024; Chen et al., 2025).

따라서 LLM이 외부 문맥 간의 불일치를 실제로 어떻게 인지하고 처리하는지 분석하기 위해서는, 프롬프트 기반 CoT 외에 모델 내부의 자발적인 사고 과정을 관측할 필요가 있습니다. 일반 지시조정 모델(Standard Instruct LLMs)은 입력에서 답변으로 곧장 도달하여 내부의 지식 충돌 해소 과정이 블랙박스로 남습니다. 반면, 명시적인 사고 채널(`<think>`)을 거쳐 답변을 생성하는 **Reasoning LLM(예: DeepSeek-R1, Qwen-Thinking 계열)**은 지식의 감지와 대조, 판정 과정을 투명하게 외부화하므로 본 연구의 진단 대상이자 도구로 적합합니다. 우리는 모델의 사고 과정(`Thought`)을 관측하고 최종 답변(`Text`)과의 정합성을 추적함으로써, 모델이 모순된 문맥을 실제로 대조해 내는지, 아니면 사고 과정과 최종 답변 생성 사이에 **추론-답변 불일치(Reasoning-Answer Inconsistency)**가 나타나는지를 분석하는 문제 정의를 제시합니다.

### 1.2. 핵심 질문: 추론-답변 불일치 (Reasoning-Answer Inconsistency)
1.1절에서 제기한 바와 같이, 본 연구는 Reasoning LLM이 외부 문맥 간 충돌 환경에 놓였을 때 내부 사고 과정(`Thought`)에서 최종 표출 답변(`Text`)에 이르는 일련의 추론 과정이 일관되게 유지되는지를 분석하는 데 목적이 있습니다. 나아가, **만약 이러한 일관성이 깨진다면 정보 처리의 어느 단계(인지, 판정, 최종 표출)에서 불일치가 발생하는지 그 기계론적 지점을 체계적으로 진단**하고자 합니다. 우리가 이 진단을 통해 확인하고자 하는 핵심 주제는 **"RAG 지식 충돌 환경에서 모델의 최종 답변 오류가 문서 간 모순을 알아채지 못하는 '인지(Recognition)' 단계에서 기인하는가, 아니면 인지 이후 최종 답변으로 전달되는 '사고-답변 정렬' 단계에서 기인하는가"** 그 경계를 분석하는 것입니다.

이를 위해 우리는 모델의 정보 처리 과정을 **다단계 진단 프로토콜(Multi-stage Diagnostic Protocol)** 관점에서 세 단계로 분해하여 분석합니다. 즉, (1) 상충되는 문서 간의 불일치를 감지하는 단계(Case 1: 인지), (2) 인지 후 논리적 비교를 통해 판정을 내리는 단계(Case 2: 판정), (3) 판정 결과가 최종 답변 생성 단계로 전달되어 표출되는 단계(Case 3: 정렬 및 표출)를 체계적으로 추적합니다. 우리는 이러한 다단계 진단 프로토콜을 바탕으로 모델의 정보 처리 메커니즘을 분석하고 가설을 검증하기 위해 다음 두 가지 연구 질문(RQ)을 설정합니다:
*   **RQ1 (일관성 및 불일치 위치 진단):** Reasoning 모델은 외부 문맥 간 충돌 환경에서 내부 사고 과정과 최종 답변 간의 일관성을 유지하는가? 만약 불일치가 관측된다면, 정보 처리의 어느 단계(인지, 판정, 최종 표출)에서 주로 단절이 발생하는가?
*   **RQ2 (충돌 유형별 영향 및 완화 기법 해부):** 모델의 추론 궤적과 불일치율은 지식 충돌의 유형(사실 vs 의견)과 출처 속성(`date` vs `url`)에 따라 어떻게 상이하게 나타나며, 문헌 기반 RAG 완화 기법(예: CAD, Recency/Authority-Guided Prompting)을 적용할 때 암묵적 도약(Shortcut)이나 사후 합리화(Unfaithful) 같은 기계론적 착시가 어떻게 관찰되는가?


---

## 2. 관련 연구 및 카테고리화 (Related Work)

RAG 환경에서의 지식 충돌 해소와 LLM의 추론 일관성에 관한 연구가 최근 활발히 진행되고 있으므로, 본 연구의 학술적 위치를 세 가지 카테고리로 나누어 기존 문헌들과 명확히 차별화합니다.

```
                     [ RAG 지식 충돌 및 추론 연구 ]
                                 │
         ┌───────────────────────┼───────────────────────┐
         ▼                       ▼                       ▼
 [2.1 RAG 충돌 진단]    [2.2 Inter-Context 해결]   [2.3 LLM 추론 진단]
 - Xu et al. (2024)      - Self-RAG (2024)       - Lee et al. (2025)
 - DRAGged (2025)        - FaithfulRAG (2025)    - Chen et al. (2025)
 - RAMDocs (2024)        - Turpin et al. (2023)  - Lyubomenko (2024)
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 ▼
         [본 연구: Reasoning-Answer Inconsistency 진단]
         - 사후 합리화 프롬프트를 배제한 자발적 내재 추론(`<think>`) 관측
         - 다단계 진단 프로토콜(인지 → 판정 → 표출)을 통한 기계론적 오류 규명
```

### 2.1. RAG에서의 지식 충돌 진단 (Conflict Diagnosis in RAG)
RAG 환경에서 발생하는 지식 충돌을 정의하고 분류 체계를 수립한 기초 연구들입니다.
*   **Xu et al. (EMNLP 2024), *Knowledge Conflicts for LLMs: A Survey*:** LLM이 직면하는 지식 충돌을 종합적으로 분석하여, 모델 내부 파라미터 지식과 외부 검색 문서가 상충하는 **Context-Memory 충돌**, 검색된 다수의 외부 문서들 간에 내용이 엇갈리는 **Inter-Context 충돌**, 모델 내부 지식 간의 모순인 **Intra-Memory 충돌**의 3대 분류 체계를 정립했습니다.
*   **Cattan et al. (ACL 2025), *DRAGged: Investigating RAG in the Wild*:** 실제 웹 검색 환경(In-the-wild)에서 수집된 RAG 충돌 데이터셋을 구축하고 분석한 연구입니다. 실제 질의의 약 64.8%에서 상위 검색 문서 간 불일치가 발생함을 보고하며, 이를 **시간적 충돌(Temporal/Outdated)**, **오정보 충돌(Misinformation)**, **상충되는 의견 충돌(Conflicting Opinions)** 등 원인별 축으로 세분화했습니다.
*   **Xie et al. (2024), *Does RAG Know When Retrieval Is Wrong?* (RAMDocs):** 검색된 노이즈 및 상충 문서가 주어졌을 때 RAG 시스템이 어떻게 오작동하는지 평가하고, 모델의 오류 감지 및 답변 거부(Refusal) 한계를 정량적으로 측정했습니다.
*   **본 연구의 타겟:** 우리는 이 분류 체계 중 실제 웹 검색 기반 RAG에서 가장 빈번하게 나타나는 **Inter-Context 충돌(외부 문서 간 상충)**을 핵심 진단 대상으로 설정하고, DRAGged의 3대 충돌 유형별로 모델의 반응 패턴을 진단합니다.

### 2.2. Inter-Context 충돌 해결 시도와 그 한계 (Mitigating Inter-Context Conflicts)
상충되는 외부 문서들이 공존할 때, 모델이 올바른 정보를 선택하고 충돌을 해소하도록 유도한 공학적 연구들입니다.
*   **Asai et al. (ICLR 2024), *Self-RAG: Learning to Retrieve, Generate, and Critique*:** 일반 지시조정 모델에 반성 토큰(Reflection tokens)을 학습시켜, 검색 문서의 관련성과 생성 답변의 사실성을 모델 스스로 평가하고 제어하도록 한 성찰 기반 RAG 프레임워크입니다.
*   **Hu et al. (ACL 2025), *FaithfulRAG* & *ConflictRAG (2026)*:** 다수의 상충 문서가 반환될 때, 프롬프트 기반의 단계적 사고(Prompted CoT)나 신뢰도 스코어링(Recency/Authority 평가)을 유도하여 답변의 사실성을 높이려 시도한 최신 해결 기법들입니다.
*   **Turpin et al. (2023) & Lanham et al. (2023), *Unfaithful Explanations in CoT Prompting*:** 일반 LLM에 인위적인 프롬프트로 단계적 사고를 유도(Prompted CoT)할 경우, 모델이 실제 내부 계산 과정과 무관하게 정답을 정당화하기 위해 겉치레 설명만 지어내는 **'사후 합리화(Unfaithful Rationalization)'** 현상이 광범위하게 발생함을 밝힌 비판적 연구들입니다.
*   **본 연구의 차별점 1 (자발적 내재 추론의 진단):** 기존 RAG 연구들은 일반 모델에 프롬프트 CoT나 성찰을 억지로 주입하여 해결하려 했으나, 이는 사후 합리화 오염으로 인해 모델이 실제로 충돌을 대조했는지 신뢰할 수 없습니다. 따라서 본 연구는 프롬프트 유도 모델을 배제하고, 자발적으로 명시적 사고 과정(`<think>`)을 전개하는 **본원적 추론 모델(Native Reasoning LLMs, 예: Qwen-Thinking, Olmo-Think)**만을 연구 대상으로 삼아, 사후 합리화를 통제한 순수한 내재적 추론 정책을 진단합니다.

### 2.3. LLM 추론 일관성 및 진단 방법론 (Evaluating LLM Reasoning Inconsistency)
LLM의 사고 과정(CoT)과 최종 답변 간의 정합성(Alignment) 및 일관성을 평가하고 분석하는 연구들입니다.
*   **Lee & Hockenmaier (EMNLP 2025 Findings), *Evaluating Step-by-Step Reasoning Traces*:** LLM의 단계적 추론 궤적을 사실성(Factuality), 타당성(Validity), 일관성(Coherence), 유용성(Utility)의 4가지 축으로 분해하여 평가하는 체계적 진단 프레임워크를 제안했습니다.
*   **Chen et al. (2025), *Do Models Know Why They Changed Their Mind?* & *The Chain Holds, the Answer Folds*:** 최근 Reasoning 모델의 사고 토큰을 분석하여 의사결정의 불변성을 측정하거나, 대화 상대의 반박이라는 사회적 압박(Adversarial Pressure) 하에서 모델이 사고 과정 속에서는 올바른 판단을 내리고도 최종 답변에서 굴복하는 추론-답변 괴리 현상을 보고했습니다.
*   **Lyubomenko et al. (2024), *Investigating Reasoning-Answer Disconnect in Large Language Models*:** 모델이 생성하는 중간 추론 궤적과 최종 출력 간에 발생하는 의미론적 단절(Semantic Disconnect) 및 편위(Drift) 현상을 계량적으로 분석했습니다.
*   **본 연구의 차별점 2 (다단계 진단 프로토콜을 통한 오류 분해):** 기존 일관성 연구들은 주로 일반 수학/QA 도메인의 사회적 압박 하에서 단절을 다루었거나 단일 지표 평가에 그쳤습니다. 본 연구는 RAG 검색 환경의 **'상충 문서 공존이라는 문맥적 압박(Contextual Pressure)'** 속에서, 모델의 처리 과정을 **다단계 진단 프로토콜(Multi-stage Diagnostic Protocol: 인지 → 판정 → 최종 표출)**로 분해하여, 사고 과정이 유효함에도 최종 답변에서 불일치가 발생하는 기계론적 지점을 체계적으로 규명합니다.

---

## 3. 연구 방법론 (Methodology)

### 3.1. 데이터셋 구성 및 충돌 유형 축 (Data Setup & Conflict Taxonomy)
본 연구는 Reasoning LLM이 외부 문맥 간 충돌(Inter-Context Conflict)을 처리하는 내부 메커니즘을 체계적으로 진단하기 위해, 실제 웹 검색 환경의 이질적 출처를 담은 **DRAGged**를 주력 벤치마크로 사용하고, 대규모 상충 문항을 담은 **RAMDocs**를 보강 데이터셋으로 채택합니다.

#### 1. DRAGged — 실제 웹 검색 기반 다원인 충돌 데이터셋
*   **(1) 데이터셋 설명:** Cattan et al.(ACL 2025)이 구축한 실제 웹 검색 환경(In-the-wild) 기반의 RAG 지식 충돌 벤치마크입니다. 인위적으로 단일 토큰을 조작한 합성 데이터와 달리, 실제 구글 검색(Top-10)으로 반환된 웹 문서들을 담고 있어 출처, 날짜, 문맥이 서로 이질적입니다. 이는 Reasoning LLM이 단순 텍스트 스왑이 아닌 실제 출처 비교와 속성 대조(Recency, Authority)를 수행하는 풍부한 사고 트레이스(`<think>`)를 전개하도록 유도하므로 본 진단에 최적입니다.
*   **(2) 데이터셋 충돌 유형과 건수:** 전체 458문항 중 충돌 문항 182건, 대조군 276건으로 구성됩니다.

    | 충돌 유형 (Conflict Taxonomy) | 문항 수 | 특성 및 내용 |
    | :--- | :---: | :--- |
    | **시간적 충돌 (Outdated Information)** | 62건 | 시간 흐름에 따라 과거 지식과 최신 사실이 상충 (예: 2024년 vs 2025년 일정) |
    | **오정보 충돌 (Misinformation)** | 5건 | 신뢰할 수 없는 출처의 잘못된 정보와 공식 출처의 참인 정보가 상충 |
    | **상충되는 의견 (Conflicting Opinions)** | 115건 | 정답이 하나로 정해지지 않고 학술적·사회적 주장이 엇갈리는 관점 충돌 |
    | **상보적 정보 (Complementary Information)** | 115건 | 충돌하지 않고 서로 상보적인 디테일을 제공하는 대조군 |
    | **비충돌 (No Conflict)** | 161건 | 모든 문서가 일관된 정보를 가리키는 일반 RAG 대조군 |
*   **(3) 원본 데이터셋 구조:** 각 문항은 질의(`question`), 충돌 유형(`conflict_type`), 정답(`correct_answer`), 10개의 검색 문서(`search_results`)로 구성됩니다.
    ```jsonc
    {
      "question": "When does this year's Passover start?",
      "conflict_type": "Conflict due to outdated information",
      "correct_answer": "begins at sundown on Saturday, April 12.",
      "search_results": [
        {"date": "2025-01-01", "title": "When Is Passover in 2025...", "url": "...", "text": "Pesach 2025 begins before sundown on Saturday April 12, 2025..."}, // 유효 정보
        {"date": "2024-05-01", "title": "When is Passover 2025?...",   "url": "...", "text": "It starts at dusk on the same day of the Hebrew calendar..."}, // 과거 정보
        /* ... 총 10개의 실제 웹 검색 문서 (출처·날짜 이질적) */
      ]
    }
    ```
*   **(4) 본 연구의 진단에 맞춘 수정 및 전처리 방법:**
    *   **정답 문서(Correct Document) 자동 식별 파이프라인:** DRAGged의 `search_results` 내 개별 문서에는 RAMDocs와 달리 정답 문서 여부에 대한 직접적인 라벨이 박혀있지 않습니다. 따라서 문항 전체의 유효 정답인 `correct_answer` 텍스트가 10개 검색 문서 중 어느 문서의 `text` 또는 `snippet`에 포함되어 있는지를 문자열 일치 및 NLI 기반으로 사전에 매핑하여, 모델이 `<think>` 과정에서 올바른 문서를 지지했는지 추적할 수 있도록 라벨링 파이프라인을 구축합니다.
    *   **문서 셔플링 및 표준 렌더링:** 문서 위치 편향(Position Bias)을 통제하기 위해 매 실험마다 10개 문서의 순서를 무작위 셔플링하고 `[Document 1] ~ [Document 10]` 포맷으로 모델에 주입합니다.
    *   **속성 메타데이터 활성화:** 모델이 다단계 진단 프로토콜의 2단계(판정 단계)에서 최신성이나 출처 권위를 올바르게 비교하는지 추적하기 위해, 원본의 `date`와 `url` 메타데이터를 문서 헤더에 명시적으로 유지하여 제공합니다.

#### 2. RAMDocs — 대규모 다중 문서 모호·오정보 벤치마크
*   **(1) 데이터셋 설명:** Xie et al.(2024)이 제안한 다중 문서 QA 벤치마크(`HanNight/RAMDocs`)의 Test set 500문항 전체입니다. DRAGged는 실제 웹 검색 환경의 다양한 충돌 유형을 포괄하지만, 그중 '의견 충돌'을 제외하고 정답과 오답이 명확히 가려져 정답률 채점이 가능한 '사실 기반 충돌(시간적 충돌 및 오정보 충돌)' 문항은 67건으로 다소 적습니다. 따라서 명확한 정답과 상충 문서가 포함된 RAMDocs Test set 500문항 전체를 보강 데이터로 채택함으로써, 모델의 단계별 오류 진단 및 추론-답변 불일치율(Inconsistency Rate) 측정을 위한 정량적 표본을 보완합니다.
*   **(2) 데이터셋 충돌 유형과 건수:** Test set 전체 **500문항**으로, 문항당 평균 5.5개의 문서가 포함됩니다. 문서별로 정답을 담은 문서(`correct`), 오정보/모순 문서(`misinfo`), 관련 없는 노이즈 문서(`noise`)가 혼재되어 대규모 상충 환경을 제공합니다.
*   **(3) 원본 데이터셋 구조:**
    ```jsonc
    {
      "question": "What is the population of Broken Bow?",
      "documents": [
        {"text": "...total area...", "type": "correct", "answer": "3,559 people"},
        {"text": "...historical...", "type": "misinfo", "answer": "10,000 people"},
        {"text": "...census...",     "type": "noise",   "answer": "..."}
      ],
      "gold_answers": ["3,559 people"],
      "wrong_answers": ["10,000 people"]
    }
    ```
*   **(4) 본 연구의 진단에 맞춘 수정 및 전처리 방법:**
    *   **문서별 매핑 파이프라인 구축:** 원본의 `documents` 배열 내 각 문서가 `gold_answers`에 속하는지 `wrong_answers`에 속하는지 사전에 매핑합니다. 이를 통해 모델이 `<think>` 과정에서 특정 문서를 인용하거나 비교할 때, 해당 문서가 정답 문서인지 오답 문서인지를 자동 판정할 수 있도록 추적 파이프라인을 구축합니다.
    *   **셔플링 및 표준화:** DRAGged와 동일하게 문서 순서를 무작위 셔플링하여 표준 프롬프트 포맷으로 주입합니다.
### 3.2. 다단계 진단 프로토콜 (Multi-stage Diagnostic Protocol)
본 연구는 Reasoning LLM이 내재적 사고 과정(`<think>`)에서 지식 충돌을 처리하고 최종 답변으로 전환하는 과정을 체계적으로 해부하기 위해, **3단계 핵심 인과 프로토콜(3-Stage Causal Protocol)**을 적용합니다. 기존 연구들이 표면적인 충돌 인지 여부나 최종 정답률을 독립적으로 평가한 것과 달리, 본 프로토콜은 충돌 인지에서 해소 판정, 그리고 최종 표출에 이르는 인과적 사슬을 압축적으로 규명합니다.

#### 1. 3단계 핵심 궤적 라벨링 (3-Stage Trajectory Labeling)
각 실험 턴마다 모델의 원시 출력에서 사고 과정(`<think>`)과 최종 답변을 파싱하여 다음과 같이 다차원 평가를 수행합니다:
*   **Level 1 (충돌 및 유형 인지 - Typological Recognition):** 사고 과정에서 검색된 문서들 간의 정보 불일치를 감지하고, 그 원인(시간적 상충, 오정보, 관점 차이 등)까지 식별했는가? $\{correct\_type(\text{유형 인지 성공}), surface\_only(\text{표면적 감지}), unrecognized(\text{인지 실패})\}$
*   **Level 2 (해소 판정 정확도 - Commitment):** 사고 과정에서 여러 문서 중 **정답 지지 문서(Correct Document)가 유효하다고 결론짓는 올바른 추론을 수행했는가?** $\{correct(\text{정답 문서 지지}), wrong(\text{오답 문서 지지}), unresolved(\text{판정 보류/미도달})\}$
*   **Final Action (최종 답변 표출 - Generation):** `<think>` 블록 종료 후 최종 출력 텍스트에서 유효한 정답을 정확하게 표출했는가 (Exact Match)? $\{correct, wrong\}$

#### 2. 진단 지표 체계: 계층적 퍼널 및 전체 전환 행렬 (Funnel & Full Transition Matrix)
본 연구는 모델의 내부 추론 궤적과 최종 디코딩 사이에서 발생하는 비순차적·비규범적 정보 처리(예: 암묵적 도약, 사후 정당화)를 포괄적으로 규명하기 위해, **'엄격한 계층 퍼널 지표'**와 **'전체 상태 전환 행렬(Full Transition Matrix)'**을 결합한 투트랙 분석 체계를 적용합니다.

**(1) 3대 계층적 전환 손실 지표 (Normative Funnel Loss):** 이상적인 순차 추론 사슬 상에서 어느 단계에 병목이 생기는지 요약합니다.
1.  **Level 1 인지 실패율 ($\text{Loss}_{L1}$):** $P(\text{Level 1} = unrecognized)$ — 외부 문서 간의 충돌을 감지하지 못하는 비율.
2.  **Level 2 판정 오류율 ($\text{Loss}_{L2}$):** $P(\text{Level 2} \neq correct \mid \text{Level 1} \neq unrecognized)$ — 충돌을 감지했으나 정답 문서가 유효하다는 판정을 내리지 못하거나 오답에 편향되는 비율.
3.  **추론-답변 불일치율 ($\text{AIR}$, 또는 $\text{Loss}_{FA}$):** $P(\text{Final Action} = wrong \mid \text{Level 2} = correct)$ — 올바른 해소 판정에 도달했음에도 최종 답변 표출 시점에 결론이 유실되는 비율.

**(2) 비순차적 특이 경로 지표 (Non-sequential & Unfaithful Pathways):** 조건부 확률의 제약을 벗어나, 사고 궤적의 도달 상태($L_0 \sim L_2$)와 최종 답변($Correct, Wrong$) 간의 전체 경로를 추적하여 다음의 특이 현상을 독립 규명합니다.
1.  **암묵적 도약율 ($\text{Shortcut Rate}$):** $P(\text{Final Action} = correct \mid \text{Level 1 도달}, \text{Level 2 미도달})$ — 사고 과정에서 명시적인 해소 선언 없이 충돌 인지만으로 올바른 최종 정답에 도달하는 암묵적 처리 비율.
2.  **불성실 추론율 ($\text{Unfaithful Rate}$):** $P(\text{Final Action} = correct \mid \text{Level 2} = wrong)$ — 내부 사고에서는 오답 문서가 유효하다고 결론지었음에도, 최종 표출 시점에 사전학습 지식이나 우연이 개입하여 정답을 맞추는 사후 합리화 비율.

### 3.3. 본 실험 설계 및 프로토콜 (Main Experiment & Evaluation Setup)

본 연구의 핵심 연구 질문(RQ 1·2)을 검증하기 위해, §3.1의 벤치마크에 대해 다단계 진단 프로토콜을 적용하고 RAG 완화 기법의 메커니즘을 해부하는 2대 핵심 실험을 수행합니다.

#### 1. 실험 1 (RQ1·RQ2 검증): 전환 경로 진단 및 RAG 완화 기법의 메커니즘 해부
*   **목적:** 지식 충돌 상황에서 모델이 어느 전환 단계에 병목($\text{Loss}_{L1, L2}, \text{AIR}$) 및 특이 경로(Shortcut, Unfaithful)를 겪는지 규명하고, **기존 문헌에서 제안된 RAG 완화 기법들을 적용했을 때 이 전환 행렬이 어떻게 변화하는지 해부합니다.**
*   **평가 대상 환경 및 기법 (Baselines & Interventions):**
    1.  **Standard RAG (Zero-shot Default):** 기본 검색 문서 주입 대조군 (Xie et al., 2024).
    2.  **Conflict-Aware Prompting (CAD / Instruct-RAG):** 검색 문서 간에 충돌 및 노이즈가 존재할 수 있음을 경고하고 주의 깊은 확인을 유도하는 기법 (Xie et al., 2024; Cuconasu et al., 2024) — $\text{Loss}_{L1}$(인지 실패율) 개선 효과 검증.
    3.  **Recency/Authority-Guided Prompting (최신성·권위 기반 명시 대조):** 문서 충돌 시 작성 날짜(`date`)의 최신성이나 출처(`url`) 권위 메타데이터를 우선적으로 대조하여 결론을 내리도록 유도하는 기법 (Hu et al., 2025; Chen et al., 2024) — $\text{Loss}_{L2}$(판정 오류율) 개선 효과 검증.
    4.  **Self-Reflective RAG (Self-RAG / Reflection):** 생성된 결론과 문서 간의 사실적 부합 여부 및 충돌 해소를 모델 스스로 반추(Reflect)하게 하는 기법 (Asai et al., 2024; Shinn et al., 2023) — 추론-답변 불일치($\text{AIR}$) 완화 효과 검증.
*   **절차:** 위의 4가지 프롬프팅 환경에서 DRAGged 및 RAMDocs 문항(순서 무작위 셔플)을 모델에 주입하여 사고 과정과 최종 답변을 생성한 뒤, 규칙 기반 정규식과 LLM-as-a-Judge를 이중 결합한 하이브리드 파이프라인으로 3단계 도달 여부를 마킹하고 **환경별 전체 경로 전환 행렬(Full Transition Matrix)을 도출 및 대조**합니다.

#### 2. 실험 2 (RQ2 검증): 충돌 유형별 영향 및 속성 민감도 분석
*   **목적:** 충돌 원인에 따라 모델의 내부 추론 궤적과 불일치율(AIR)이 어떻게 달라지는지 비교 분석합니다.
*   **분석 축:**
    *   **사실 기반 충돌 vs 의견 충돌:** 정답률 채점이 가능한 '사실 기반 충돌(시간적 충돌 및 오정보 충돌)' 그룹과 정답이 없는 '의견 충돌' 그룹 간의 Level 1~2 인지 및 대조 패턴 차이를 분석합니다.
    *   **출처 속성 민감도:** 시간적 충돌 문항에서 모델이 최신성(`date`)을 호출하는 비율과, 오정보 충돌 문항에서 출처 권위(`url`)를 호출하는 비율을 측정하고, 해당 속성 대조가 최종 정답 표출로 전환되는 효율을 검증합니다.

#### 3. 대상 모델 및 생성·심사 프로토콜 (Target Models & Evaluation Setup)
*(1) 대상 모델 선정 (Target Reasoning LLMs)*
문헌에서 성능이 검증된 **본원적 추론 모델(Native Reasoning LLMs)**을 2025~2026년 시점 현행 버전 기준으로 3개 주요 계열에 걸쳐 엄선합니다.
*   **본원적 추론 모델 집중:** SFT 증류(Distillation)만 거친 모델은 고유한 자율 추론 탐색 능력이 보존되지 않고 장식적인 출력을 낼 위험이 있습니다. 따라서 본 연구는 증류 모델을 배제하고, 실제 탐색 궤적을 자발적으로 전개하는 20~32B 스위트스팟 규모의 본원적 추론 모델에 집중합니다.
*   **계열 및 구조 통제:** 특정 아키텍처나 구조(Dense vs MoE)에 국한된 현상이 아님을 실증하고, 연구의 완전 개방성(Full Openness)을 담보하기 위해 Qwen, AllenAI, OpenAI 등 3개 독자 계열을 커버합니다.

| 모델 | 계열 | 출시 | 구조 및 규모 | 사고 트레이스 | 선정 근거 및 역할 |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Qwen3.6-27B** | Qwen | 2026-04 | 27B Dense | `<think>` | **주력 분석 모델**. 현행 오픈소스 밀집형 추론 모델의 표준 |
| **Olmo-3.1-32B-Think** | AllenAI | 2025-12 | 32B Dense | `<think>` | **완전 개방형 대조군**. 가중치·데이터·학습 로그 전체 공개로 투명성 담보 |
| **gpt-oss-20b** | OpenAI | 2025-08 | 20B MoE | Harmony¹ | **교차 계열 및 MoE 검증**. OpenAI에서 유일하게 개방한 MoE 아키텍처 추론 모델 |

> ¹ `gpt-oss-20b`는 `<think>` 태그 대신 Harmony 포맷의 Analysis 채널로 사고 과정을 출력하므로, 전용 파서를 통해 동일한 3단계 프로토콜로 라벨링합니다.

*(2) 생성 및 심사 프로토콜 (Generation & Evaluation Setup)*
*   **생성 환경:** 모든 모델은 비양자화 전체 정밀도(bf16)로 구동하여 양자화에 따른 로짓 및 사고 궤적 왜곡을 방지합니다. 디코딩 온도는 각 모델의 권장 설정(예: `temp=0.6, top_p=0.95`)을 따르며, 무작위 시드 3회 반복 생성 후 통계적 유의성 검정(Paired Bootstrap 95% CI)을 수행합니다.
*   **하이브리드 라벨링 파이프라인:** 3단계 진단 라벨링의 객관성을 담보하기 위해 규칙 기반 매칭(문서 인덱스, 날짜/URL 키워드)과 고성능 LLM-as-a-Judge(GPT-4o)를 결합합니다.
*   **심사 신뢰성 검증:** (1) 무작위 200건 샘플에 대한 인간 전문가 2인의 교차 검수 일치도(Cohen's $\kappa$)를 보고하고, (2) 서로 다른 LLM Judge(예: GPT-4o ↔ Claude-3.5-Sonnet) 간의 독립 교차 감사 일치도를 측정하여 평가의 객관성을 확보합니다.

---

## 4. 사고 과정(Thinking Trajectory)의 체계적 진단 기법

본 연구의 3대 진단 기법은 단순히 독립적인 분석들의 나열이 아니라, §3.2에서 확립한 **'3단계 인과 프로토콜(Level 1 인지 $\rightarrow$ Level 2 판정 $\rightarrow$ Final Action 표출)'** 상에서 발생하는 3대 핵심 병목(Shortcut 도약, 해소 판정 실패, 추론-답변 괴리)을 기계론적으로 규명하기 위해 1:1로 매핑된 완벽한 진단 도구셋입니다. 문헌에서 검증된 최신 방법론들을 차용하여 `<think>` 내부의 정보 처리를 세밀하게 해부합니다.

| 진단 목적 및 3단계 병목 매핑 | 담당 진단 기법 (§4) | 문헌 검증 근거 (References) |
| :--- | :--- | :--- |
| **① Level 1 $\rightarrow$ Level 2 병목:**<br>왜 명시적 대조 없이 도약(Shortcut)하거나 앞문서에 고착되는가? | **4.2. RCPD & Lock-in Effect** | Chen et al. (2025); Wu et al. (2024);<br>Snell et al. (ICLR 2025) |
| **② Level 2 내부 판정 오염:**<br>왜 신념이 흔들리거나 오답 문서를 지지($\text{Loss}_{L2}$)하는가? | **4.1. Trajectory Semantic Shift** | Lanham et al. (ACL 2024);<br>Lee & Hockenmaier (EMNLP 2025) |
| **③ Level 2 $\rightarrow$ Final Action 단절:**<br>왜 올바른 판단 후 굴복(AIR)하거나 번복하는가? | **4.2. Overthinking**<br>+ **4.3. Token Entropy Spike** | Kuhn et al. (ICLR 2023);<br>Chen et al. (2025) |

### 4.1. Trajectory Semantic Shift (의미론적 궤적 추적)
CoT 텍스트를 문장 단위($s_1, s_2, \dots, s_T$)로 분할한 뒤, 각 문장의 임베딩 $e(s_t)$를 추출합니다. 본 연구 데이터는 문항당 5~10문서이므로(DRAGged 10 / RAMDocs 평균 5.5), 각 문장 임베딩과 **모든 검색 문서**의 코사인 유사도를 계산해 문서를 **정답 문서군(correct-doc) vs 충돌 문서군(conflicting docs)** 두 군으로 묶어 각 군 대표 중심 임베딩과의 유사도 궤적을 추적합니다 (Lanham et al., 2024; Lee & Hockenmaier, 2025). 이 분석은 정답 채점이 가능한 사실 기반 충돌 그룹에 적용합니다.
*   **목적 및 병목 매핑:** 생각이 진행됨에 따라 모델의 내부 신념(Belief)이 정답 군 vs 충돌 군 어느 쪽으로 기우는지 시계열 궤적(Decision Trajectory)을 정량화하여, **Level 2(해소 판정) 단계에서의 판정 실패($\text{Loss}_{L2}$) 원인을 규명**합니다.
*   **분석 포인트:** 불일치(AIR)가 일어나는 샘플에서 CoT의 마지막 문장까지는 정답 군에 가깝게 유지되다가, `<think>`가 끝나는 임계 영역에서 급격히 충돌 군으로 기우는 현상(Late Shift)을 실증합니다.

### 4.2. Reasoning Completion Point Detection (RCPD) 및 Lock-in Effect 분석
§3.2의 하이브리드 라벨링 파이프라인이 도출한 타임스탬프를 활용하여, 모델이 CoT 상에서 충돌을 감지하는 순간(Level 1)과 해소 결론을 확정하는 순간(Level 2)이 전체 사고 길이 중 어느 지점(Token Index)에서 발생하는지 측정합니다.
*   **Lock-in Effect (초기 고착 효과):** 입력 프롬프트 초기(예: Top-1 위치) 문서가 CoT의 앞쪽 10% 토큰을 지배하고, 이 초기 판단이 전체 사고 경로를 구속하여 최종 답변까지 끌고 가는지 분석합니다 (Wu et al., 2024). 이를 통해 명시적 대조 없이 초기 정보만으로 정답/오답에 도달하는 **암묵적 도약(Shortcut)의 기계론적 원인**을 증명합니다.
*   **Overthinking (과잉 추론 진단):** 충돌 해소 결론(Level 2)이 도출된 이후에도 불필요한 사고 토큰이 길게 이어질 때(Redundant Reasoning), 오히려 논리가 붕괴되며 최종 답변에서 굴복(Capitulation)이 일어날 확률이 높아지는지 조사합니다 (Snell et al., 2025; Chen et al., 2025).

### 4.3. Token-level Entropy & Perplexity Dynamics (토큰 엔트로피 분석)
CoT 생성 전반 및 답변 첫 토큰 생성 시점에 모델이 출력하는 각 토큰의 샤논 엔트로피 $H(X) = -\sum P(x_i) \log P(x_i)$와 Perplexity를 측정합니다 (Kuhn et al., 2023).
*   **목적 및 병목 매핑:** 모델이 겪는 내부적 혼란도(Internal Uncertainty)를 기계론적으로 정량화하여, **Level 2에서 올바른 판단을 내리고도 최종 답변에서 불일치(AIR)가 발생하는 원인을 해부**합니다.
*   **가설:** 추론-답변 불일치(AIR)가 일어나는 샘플은 CoT 상에서 겉으로 올바른 결론(Level 2)을 적었을지라도, 해당 토큰들의 Entropy가 유의미하게 높을 것(Internal Hesitation)이며, 특히 `<think>` 블록이 닫히고(`</think>`) 답변 첫 토큰으로 진입하는 전환 임계 지점(Transition Threshold)에서 Entropy가 스파이크처럼 치솟을 것입니다 (Chen et al., 2025).

### 4.4. 오픈소스 통합 진단 프레임워크 구축 (Integrated Diagnostic Framework)
본 연구는 단순한 일회성 분석을 넘어, §3.2의 3단계 하이브리드 라벨링 파이프라인과 §4.1~4.3의 기계론적 진단 도구셋을 하나의 자동화된 파이프라인으로 결합한 **오픈소스 통합 진단 프레임워크(예: `ThinkConflict-Diagnoser`)**를 구축하고 공개합니다.
*   **프레임워크 아키텍처:**
    1.  **3-Stage Audit Engine:** 모델의 사고 트레이스와 답변을 입력받아 규칙 기반 정규식과 LLM Judge로 Level 1·2 도달 여부를 자동 파싱하고 3x3 전체 전환 행렬(Full Transition Matrix)을 도출합니다.
    2.  **Mechanistic Profiler:** 문장 분리 후 임베딩 유사도 궤적(Semantic Shift)을 시계열로 계산하고, 토큰 인덱스 기반의 초기 고착(Lock-in)/과잉 추론(Overthinking) 비율 및 로짓 엔트로피 스파이크를 자동 추출·시각화합니다.
*   **학술적 기여도:** 후속 연구자들이 임의의 본원적 추론 모델이나 RAG 완화 기법을 손쉽게 플러그인(Plug-in)하여, 모델 사고 궤적 상의 추론-답변 불일치(AIR)와 암묵적 도약(Shortcut)을 클릭 한 번으로 감사(Audit)할 수 있는 표준 툴킷을 생태계에 제공합니다.

---

## 5. 예상 결과 및 한계점 (Expected Results & Limitations)

### 5.1. 주요 가설
*   **가설 1 (위치 및 경로 병목 - RQ1):** 본원적 추론 모델은 문서 간 충돌 상황에서 높은 비율로 충돌을 인지하고 올바른 해소 판단을 내리지만(Level 2), 최종 답변 단계에서 이 판단을 유실하는 추론-답변 불일치율(AIR)이 유의미하게 존재할 것이다. 또한 초기 입력 문서 위치(Top-1)에 구속되어 명시적 해소 없이 정답으로 도약하는 Shortcut 경로나, 과잉 추론(Overthinking)에 의해 판단을 번복하는 굴복 현상이 관찰될 것이다.
*   **가설 2 (RAG 완화 기법의 착시 해부 - RQ2):** 문헌에서 제안된 기존 RAG 충돌 해결 기법(예: CAD, Recency/Authority-Guided Prompting)을 적용할 경우, 표면적 정답률(EM)은 상승하지만 이 중 상당 부분은 명시적 대조 논리 없이 정답으로 도약하는 암묵적 도약(Shortcut)이나 사후 합리화(Unfaithful) 경로에 기인할 것이다. 본 연구의 3단계 프로토콜과 §4의 진단 도구셋(Semantic Shift, Entropy Spike 등)은 이러한 착시와 기계론적 원인을 성공적으로 분해해 낼 것이다.

### 5.2. 연구의 한계점
*   본 연구의 정확성(행동) 평가는 정답이 시간이나 공식 출처에 의해 결정되는 사실 기반 충돌에 국한되며, 주관적 의견 충돌은 인지 및 대조 패턴으로만 분석합니다.
*   **원인별 통계는 사실충돌·의견충돌 두 묶음 수준까지만 주장**합니다. DRAGged 단독 오정보(5건)는 표본이 작아 **정성 사례 연구로만** 제시하며, 오정보 원인의 정량 근거는 RAMDocs(오정보 문서 다수 포함)로 보강합니다.
*   사고 트레이스에 직접 접근할 수 있는 오픈소스 **본원적 추론 모델 3계열**(Qwen: Qwen3.6-27B / AllenAI: Olmo-3.1-32B-Think / OpenAI: gpt-oss-20b)을 중심으로 분석하며, 증류 모델과 폐쇄형 상용 모델(내부 trace 접근 불가)은 제외됩니다.

---

## 6. 로드맵 및 파일럿 게이트 (Roadmap & P1 Gate)

*   **[P0] 데이터 및 인프라 구축:** `DRAGged`(458건) 및 `RAMDocs`(500건)를 셔플 렌더 포맷으로 정비하고, 본원적 사고 출력을 3단계 프로토콜로 파싱하는 하이브리드 파이프라인 완성.
*   **[P1 게이트] 예비 실측 (Go/No-Go):** 주력 **Qwen3.6-27B**를 소량(N≈50)으로 구동하여 실제 불일치율(AIR) 및 Shortcut 경로가 유의미하게 관찰되는지 확인 (불일치율 및 특이 경로 합계 ≥ 30% 시 본격 추진).
*   **[P2] 다모델 스케일 및 RAG 완화 기법 해부:** Olmo-3.1-32B-Think(AllenAI), gpt-oss-20b(OpenAI)로 3계열 확장 및 문헌 기반 RAG 완화 기법(Interventions) 4종에 대한 전체 경로 전환 행렬(Full Transition Matrix) 도출.
*   **[P3] 기계론적 심층 진단 완료:** §4의 3대 진단 기법(Semantic Shift, Lock-in/Overthinking, Token Entropy Spike)을 적용하여 단계별 병목의 기계론적 원인을 입증하고 최종 분석 보고서 완성.
