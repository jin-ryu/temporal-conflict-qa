# 연구 계획서: Lost in Translation from Thought to Text: Investigating Reasoning-Answer Inconsistency in RAG Conflicts

*   **투고 목표:** ACL / EMNLP Main (Long Paper) — 진단 및 해석가능성 중심 (Diagnostic & Interpretability Track)
*   **한 줄 요약:** RAG의 외부 문맥 간 충돌(Inter-Context Conflict) 상황에서, Reasoning LLM의 내부 사고(`Thought`)와 최종 답변(`Text`) 간에 발생하는 **추론-답변 불일치(Reasoning-Answer Inconsistency)**를 (1) 출력 수준의 자기일관성으로 정량 진단하고, (2) 충돌 대조군으로 그 충돌 고유성을 분리하며, (3) 인과 개입으로 그 인과적 지점을 규명한다.

---

## 1. 서론 (Introduction)

### 1.1. 배경 및 문제 정의
검색 증강 생성(Retrieval-Augmented Generation, RAG)은 외부 지식을 활용해 대형언어모델(LLM)의 신뢰성을 제고하는 핵심 기술이지만, 실제 웹 검색 환경(In-the-wild)에서는 동일한 질의에 대해 상충되는 정보들이 동시에 반환되는 **외부 문맥 간 충돌(Inter-Context Conflict)**이 빈번히 발생합니다. 실제 웹 검색 기반 RAG 충돌 벤치마크(CONFLICTS/DRAGged) 연구에 따르면, 전체 458개 질의 중 약 **64.8%(297건)**는 단일 출처만으로 답할 수 없어 여러 출처를 대조·종합해야 하며, 이 가운데 사실관계 모순·관점 차이·시간적 불일치 같은 **명시적 상충은 182건(39.7%)**에 이릅니다 (Cattan et al., 2025, Table 2). 이러한 복수 문서 간의 모순과 상충은 모델이 신뢰할 수 있는 정보를 분별하지 못하게 하여, RAG 시스템의 답변 오류와 환각을 유발하는 주요 요인으로 작용합니다.

**본 연구는 외부 지식 충돌 환경에서 본원적 추론 모델(Native Reasoning LLMs)이 자발적으로 전개하는 내재적 사고 과정(`<think>`) 자체를 핵심 진단 대상으로 설정합니다.** 기존 학계에서는 이러한 지식 충돌 문제를 극복하기 위해 프롬프트 기반의 단계적 사고(CoT)나 성찰(Self-reflection)을 유도하거나 (Asai et al., 2024; Zhou et al., 2023), 외부 분류기·다중 에이전트 토론 등 복잡한 파이프라인을 도입하여 답변 성능을 높이려 시도해 왔습니다 (MADAM-RAG, Wang et al., 2025; ConflictRAG, 2026; FaithfulRAG, Zhang et al., 2025). **우리는 이에 그치지 않고, 이러한 기존 완화 기법(프롬프트 유도·디코딩 개입 등)을 적용할 때에도 해당 기법들이 모델의 내부 사고 궤적을 어떤 경로로 변화시켜 최종 답변에 이르게 하는지 동일한 진단 체계 내에서 함께 비교·분석합니다(§3.3).**

일반 모델에 프롬프트로 단계적 사고(CoT)를 강제하여 내부 과정을 관측하려는 시도도 고려할 수 있으나, 이는 모델이 정답을 정당화하기 위해 사후에 설명을 꾸며내는 **'사후 합리화(Unfaithful Rationalization)'** 오염에 취약합니다 (Turpin et al., 2023; Lanham et al., 2023). **따라서 우리는 인위적인 프롬프트 강제 없이도 모델 스스로 지식의 감지·대조·판정 과정을 투명하게 외부화하는 본원적 추론 모델(Native Reasoning LLMs; 예: Qwen-Thinking, Olmo-Think 계열)을 핵심 연구 대상으로 선택했습니다.** 일반 모델이 입력에서 답변으로 곧장 도달하는 블랙박스라면, 명시적인 사고 채널(`<think>`)을 장착한 추론 모델은 내재적이고 자발적인 사고 과정을 텍스트로 드러냅니다. 우리는 이 본원적 추론 모델의 사고 과정(`Thought`)과 최종 답변(`Text`) 간의 정합성을 추적함으로써, 모델이 모순된 문맥을 실제로 대조해 내는지, 아니면 사고 과정과 최종 답변 생성 사이에 **추론-답변 불일치(Reasoning-Answer Inconsistency)**가 나타나는지를 분석하는 문제 정의를 제시합니다.

> **※ 데이터 재집계 및 대조군 설정 근거 (Cattan et al., 2025 Table 2 기준):** 원 논문은 458건 중 297건(64.8%)을 "conflicting"으로 집계하나, 이는 출처 간 정보가 호환되나 종합이 필요한 상보적 정보(115건)를 포함한 넓은 범주입니다(182 + 115 = 297). 본 연구는 원 논문 Table 2의 범주별 건수와 정확히 일치하는 자체 재집계를 통해, 진짜 모순인 **명시적 상충(182건: 의견 115 + 시간 62 + 오정보 5)**을 핵심 진단 대상으로 삼고, **비상충 대조군(276건: 상보 115 + 비충돌 161)**을 분리하여 충돌 고유 효과(RQ3)를 검증합니다.

### 1.2. 개념 구분: 자기일관성 · 인과적 사용 · 충실성
본 연구의 진단 대상을 정확히 규정하기 위해, 사고-답변 관계에 관한 세 가지 핵심 개념을 명시적으로 구분합니다. 이 구분은 본 연구가 무엇을 "텍스트만으로 측정 가능한 것"으로 주장하고 무엇을 "인과 개입으로만 주장 가능한 것"으로 남기는지를 결정합니다 (Parcalabescu & Frank, 2024).

*   **(L-a) 추론-답변 자기일관성(Reasoning-Answer Self-Consistency):** 모델이 `<think>`에서 내린 결론과 최종 답변(`Text`)의 내용이 논리적으로 일치하는가. 이는 모델에 개입하지 않고 **출력된 텍스트만으로 관측 가능한 속성**입니다. 본 연구는 이를 측정하기 위해 `<think>`에서는 정답을 맞게 추론했으나 최종 답변에서 오답을 내는 **추론-답변 불일치율(AIR)**과, 명시적 결론 없이 정답에 도달하는 **암묵적 도약율(Shortcut)** 지표를 사용합니다. Parcalabescu & Frank (2024)에 따라 본 연구는 AIR을 **출력 수준의 자기일관성 지표로 엄정히 규정**합니다.
*   **(L-b) 인과적 사용(Causal Use):** `<think>`의 사고 과정이 최종 답변을 도출한 **실질적 인과 원동력(Cause)**으로 작동했는가. 이는 텍스트 관측만으로는 증명할 수 없으며, 사고 궤적을 강제로 자르거나(Truncation) 어텐션을 차단(Attention Suppression)하는 등의 **인과 개입 실험**을 통해서만 검증할 수 있습니다 (Lanham et al., 2023; Bogdan et al., 2025). 따라서 본 연구는 **개입 시 정답률 폭락 폭(ΔAccuracy)과 답변 변화율(ΔAIR) 지표**를 통해 인과적 사용 여부를 측정하며, 이에 관한 주장은 §3.3 실험 2(인과 개입)에 한정하여 엄밀하게 제기합니다.
*   **(L-c) 내부 연산에 대한 충실성(Faithfulness to Computation):** 모델이 텍스트로 표출한 생각(`<think>`)이 정답을 고른 '진짜 속마음(내부 연산)'을 정직하게 보여주는가. 예컨대 Turpin et al. (2023)은 질문에 편향된 힌트("나는 A가 답 같아")를 몰래 섞으면 모델이 힌트 때문에 A를 골랐으면서도, 사고 과정에서는 마치 논리적으로 계산해서 A를 고른 것처럼 가짜 변명을 지어냄(충실성 결여)을 증명했습니다. 본 연구는 이러한 '인위적 힌트 편향 실험'을 다루지 않으므로 L-c를 핵심 연구 대상에서 제외합니다. 다만, 지식 충돌 상황에서도 모델이 `<think>`에서는 오답 문서를 지지해 놓고 최종 답변은 뜬금없이 정답을 맞추는 예외적 현상이 발생하므로, 이를 **불성실 추론율(Unfaithful Rate)**이라는 보조 지표(§3.2)로만 부수적으로 관측합니다.

이처럼 본 연구의 핵심 지표인 AIR과 Shortcut은 명시적인 사고 채널(`<think>`)이 답변과 분리된 **Thought→Text 아키텍처(추론 모델)에서만 정의되는 고유한 실패 양식**입니다. 나아가, 관측된 취약성이 사고 채널 자체에서 비롯된 것인지 아니면 기저 모델(Base Model)의 한계를 물려받은 것인지를 명확히 가려내기 위해, §3.3에서는 사고 채널을 강제로 끄고 켜는 **thinking on/off 통제 실험(Regime Control)**을 함께 수행합니다.

### 1.3. 연구 질문 (Research Questions)
1.2절에서 정의한 개념 구분을 바탕으로, 본 연구는 다음 세 가지 핵심 연구 질문(RQ)을 설정하여 실험적 검증을 수행합니다.

*   **RQ1 (진단 및 불일치 위치):** Reasoning 모델은 외부 문맥 간 충돌 환경에서 사고 과정과 최종 답변 간의 자기일관성(L-a)을 유지하는가? 불일치가 관측된다면, 정보 처리의 어느 단계(인지 → 판정 → 최종 표출)에서 주로 단절이 발생하는가?
*   **RQ2 (인과 규명 및 완화 기법 해부):**
    *   **(a)** 텍스트로 관측된 사고 결론은 최종 답변을 **인과적으로 구동(L-b)**하는가, 아니면 답변이 사고와 독립적으로 결정되는가?
    *   **(b)** 문헌 기반 RAG 완화 기법(예: Context-Aware Decoding, Recency/Authority-Guided Prompting, Self-Reflection)을 적용할 때, 표면 정답률(EM) 상승분이 진정한 대조 논리에서 오는가 아니면 암묵적 도약(Shortcut)·사후 합리화(Unfaithful)·맹목 적중(Blind-Hit) 경로에서 오는가?
*   **RQ3 (충돌 고유성 - 대조군 분리):** 관측된 불일치(AIR)와 특이 경로는 **충돌 상황에 고유한 실패**인가, 아니면 단순히 문항 난이도·문서 개수 증가에 따른 부수 효과인가? 동일 모델·동일 질의에 대해 **충돌 vs 비충돌/상보 대조군**을 비교하여 충돌 고유 성분을 분리한다 (Longpre et al., 2021; Xie et al., 2024).

---

## 2. 관련 연구 및 카테고리화 (Related Work)

RAG 환경에서의 지식 충돌 해소와 LLM의 추론 일관성에 관한 연구가 최근 활발히 진행되고 있으므로, 본 연구의 학술적 위치를 네 가지 카테고리로 나누어 기존 문헌들과 명확히 차별화합니다.

```
                     [ RAG 지식 충돌 및 추론 진단 연구 ]
                                 │
          ┌──────────────────────┼──────────────────────┐
          ▼                      ▼                      ▼
 [2.1 충돌 진단]          [2.2 충돌 완화]         [2.3 진단·통제 방법론]
 Xu (2024)               Self-RAG (2024)         Turpin/Lanham/Bogdan/Macar
 Cattan (2025)           Shi/CAD (2024)          Parcalabescu & Frank (2024)
 Wang/RAMDocs (2025)     Zhang/FaithfulRAG       Longpre (2021), Xie (2024)
                         (2025)                  Li et al. (2026), Wu (2024)
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 ▼
         [본 연구: Reasoning-Answer Inconsistency 진단 및 해부]
    - 자발적 내재 추론(`<think>`) 관측 + 출력 수준 자기일관성(AIR) 정량화
    - 기존 완화 기법(CAD, 프롬프트 성찰 등) 적용 시 내부 동작 과정 해부
    - 충돌 vs 비충돌 대조군(RQ3) 및 resampling/steering 인과 개입(RQ2)
```

### 2.1. RAG에서의 지식 충돌 진단 (Conflict Diagnosis in RAG)
최근 RAG 문헌은 LLM이 직면하는 지식 충돌을 단편적 오류가 아닌 구조적 유형(내부 vs 외부, 외부 문서 간 상충)으로 세분화하고, 실제 웹 검색 환경의 다원적 복잡성을 반영한 대규모 벤치마크 구축으로 나아가는 추세입니다.

*   **Xu et al. (EMNLP 2024), *Knowledge Conflicts for LLMs: A Survey* (arXiv:2403.08319):** LLM이 직면하는 지식 충돌을 모델 내부 지식과 외부 문서가 상충하는 **Context-Memory 충돌**, 검색된 외부 문서들 간에 엇갈리는 **Inter-Context 충돌**, 모델 내부 지식 간 모순인 **Intra-Memory 충돌**의 3대 분류로 정립했습니다.
*   **Cattan et al. (2025), *DRAGged into Conflicts: Detecting and Addressing Conflicting Sources in Search-Augmented LLMs* (arXiv:2506.08500):** 실제 구글 검색으로 반환된 웹 문서 기반 RAG 충돌 벤치마크입니다. Table 2 기준 전체 458건을 **No conflict(161)**, **Complementary information(115)**, **Conflicting opinions(115)**, **Freshness/Outdated(62)**, **Misinformation(5)**로 라벨링했습니다. 논문은 이 중 뒤 4개 범주(297건, 64.8%)를 "conflicting"으로 집계하는데, 이는 여러 출처의 종합이 필요한 넓은 범주로 상보적 정보(호환되나 종합 필요)를 포함하며, 진짜 모순인 **명시적 상충은 182건**입니다.
*   **Wang et al. (COLM 2025), *Retrieval-Augmented Generation with Conflicting Evidence* (RAMDocs; arXiv:2504.13079):** 문항당 평균 5.5개의 문서가 정답(`correct`)·오정보(`misinfo`)·노이즈(`noise`)로 혼재된 대규모 다중 문서 QA 벤치마크(500문항 test set)를 제안하고, 모호성·오정보·노이즈가 공존할 때 RAG의 오류 감지와 답변 거부(Refusal) 한계를 정량화했습니다.
*   **본 연구의 타겟:** 우리는 이 분류 중 실제 웹 검색 RAG에서 가장 빈번한 **Inter-Context 충돌(외부 문서 간 상충)**을 핵심 진단 대상으로 설정하고, DRAGged의 충돌 유형별로 모델의 반응 패턴을 진단합니다.

### 2.2. Inter-Context 충돌 해결 시도와 그 한계 (Mitigating Inter-Context Conflicts)
기존의 충돌 해소 연구들은 주로 디코딩 시점의 로짓 제어나 외부 맞춤형 파이프라인(분류기·다중 에이전트 토론)을 도입하는 방향으로 발전해 왔으나, 단일 모델 내부의 자발적 추론 채널(`<think>`)이 충돌을 어떻게 다루는지에 대한 인과적 규명은 부족한 실정입니다. **본 연구는 문헌에서 검증된 이러한 기존 완화 기법들 중 일부(디코딩 제어, 프롬프트 성찰 등)를 본원적 추론 모델(Reasoning LLMs)에 이식했을 때, 내부 사고 궤적과 충돌 해소 메커니즘이 실제로 어떻게 동작하고 변하는지를 해부(Anatomy)하고 분석하는 데 목적이 있습니다.**

*   **Asai et al. (ICLR 2024), *Self-RAG* (arXiv:2310.11511):** 반성 토큰(Reflection tokens)을 학습시켜 검색 문서 관련성과 생성 답변 사실성을 모델 스스로 평가·제어하게 한 성찰 기반 RAG 프레임워크입니다.
*   **디코딩 시점 개입 (CAD, CD2):** 문맥 유무 두 분포의 로짓 차이를 대비 증폭(Contrastive Decoding)하여 생성을 제공 문맥 쪽으로 밀어붙이거나(CAD; Shi et al., 2024, 충돌 적응형 변형 AdaCAD 포함), 문서 간 충돌을 분리해 대비 디코딩(CD2; Jin et al., 2024)하는 디코딩 시점 개입 기법들입니다.
*   **경량 감지 및 맞춤형 파이프라인 (ConflictRAG, 2026; arXiv:2605.17301):** 외부 검색 문서 간의 충돌을 임베딩 기반 MLP 분류기로 감지하고, **Entropy-TOPSIS** 순위 기법으로 출처 신뢰도를 평가해 유형별(오정보·시간·관점 등)로 적합한 문서를 맞춤형 선별·해결하는 파이프라인 기법입니다(프리프린트).
*   **다중 에이전트 토론 (MADAM-RAG; Wang et al., 2025):** §2.1의 RAMDocs 벤치마크와 **동일 논문(arXiv:2504.13079)**에서 함께 제안된 기법으로, 상충되는 다중 외부 문서 각각에 대변인 LLM 에이전트를 할당하고 **에이전트 간 다중 라운드 토론(Debate) 및 중앙 집계**를 통해 모호성과 오정보 충돌을 동시 해소합니다.
*   **사실 수준 충돌 모델링 (FaithfulRAG; Zhang et al., 2025):** 검색 문서의 사실 수준 충돌을 명시적으로 모델링하여 문맥 충실 생성을 유도하는 기법입니다.
*   **Turpin et al. (NeurIPS 2023) & Lanham et al. (2023), *Unfaithful / Faithfulness in CoT*:** 프롬프트로 유도한 CoT가 실제 내부 계산과 무관하게 정답을 정당화하는 **'사후 합리화(Unfaithful Rationalization)'** 현상을 밝히고(Turpin), 답변이 CoT에 실제로 의존하는지를 개입 테스트(early answering, adding mistakes, paraphrasing, filler tokens)로 측정하는 방법을 제시했습니다(Lanham).

### 2.3. 추론 일관성 진단 및 대조군 통제 방법론 (Evaluating Reasoning Inconsistency & Control Paradigms)
최근 본원적 추론 모델(Reasoning LLM)의 등장에 따라 사고 궤적의 일관성을 평가하고 개입하는 방법론이 발전하는 동시에, 문항 난이도와 순수 충돌 효과를 분리하기 위한 대조군 통제 패러다임이 강조되고 있습니다. **본 연구는 수학/QA 도메인의 사회적 반박 압박을 다룬 선행 연구들과 달리, RAG 검색 환경의 '상충 문서 공존이라는 문맥적 압박(Contextual Pressure)' 속에서 처리 과정을 다단계로 분해하고, on-policy 개입과 비충돌 대조군 설계를 결합하여 인과적 규명(RQ2)과 충돌 고유성(RQ3)을 동시에 검증합니다.**

#### (1) 사고 궤적 진단 및 인과 개입 방법론
**추론 모델의 단계별 궤적을 정량화하고, 재생성(Resampling) 및 어텐션 마스킹 등 개입 실험을 통해 특정 추론 단계가 답변에 미치는 인과적 영향력을 측정하는 연구들입니다.**

*   **Lee & Hockenmaier (Findings of EMNLP 2025), *Evaluating Step-by-step Reasoning Traces: A Survey* (arXiv:2502.12289):** 단계적 추론 궤적을 사실성(Factuality)·타당성(Validity)·일관성(Coherence)·유용성(Utility)의 4축으로 분해 평가하는 체계적 진단 프레임워크를 정리했습니다.
*   **Li et al. (2026), *The Chain Holds, the Answer Folds: Trace–Answer Dissociation in Reasoning Models Under Adversarial Pressure* (arXiv:2605.29087, 동시기 프리프린트):** Reasoning 모델이 사고 궤적에서는 사실적으로 올바른 판단을 유지하면서도 반박 압박 하에서 **최종 답변만 오답으로 굴복(unfaithful capitulation)**하는 궤적-답변 해리 현상을 보고했습니다. 본 연구의 AIR과 가장 가까운 선행 현상이나, (1) 사회적 반박 압박이 아닌 **다문서 병존이라는 문맥적 압박**을 다루고, (2) 현상 보고를 넘어 **충돌 대조군 분리(RQ3)와 인과 개입(RQ2)**으로 나아간다는 점에서 차별됩니다. 동시기 프리프린트이므로 포지셔닝 목적으로만 인용하고 그 수치에 의존하지 않습니다.
*   **Bogdan et al. (2025), *Thought Anchors: Which LLM Reasoning Steps Matter?* (arXiv:2506.19143) & Macar et al. (2025), *Thought Branches* (arXiv:2510.27484):** 추론 궤적 내 특정 단계의 인과적 중요도를 **on-policy resampling(선택 지점 이후 재생성)**과 **attention-suppression**으로 측정하는 방법론을 정립했으며, 특히 off-policy 활성 개입(steering)은 resampling 대비 효과가 작고 불안정함을 경고했습니다(본 연구의 개입 설계에 반영).

#### (2) 지표의 개념적 경계와 대조군 통제 설계
**지표가 측정하는 개념(자기일관성 vs 충실성)의 경계를 명확히 규정하고, 문항 난이도나 문서 길이 효과를 제거하여 순수 지식 충돌 효과를 분리하는 통제 실험 패러다임입니다.**

*   **Parcalabescu & Frank (ACL 2024), *On Measuring Faithfulness or Self-Consistency of Natural Language Explanations* (arXiv:2311.07466):** 기존 "충실성 테스트"들이 실제로는 내부 계산에 대한 충실성이 아니라 **출력 수준 자기일관성**을 측정함을 논증했습니다. 본 연구는 이 논지를 수용하여 AIR을 자기일관성 지표로 규정하고(§1.2), 인과 주장은 개입 실험으로 분리합니다.
*   **Longpre et al. (EMNLP 2021), *Entity-Based Knowledge Conflicts in QA*:** 원본(문맥이 정답 지지=비충돌)에서 개체를 치환해 충돌 문항을 파생시키는 **치환 프레임워크**로, 충돌-비충돌 대조를 통해 파라메트릭 과의존을 분리하는 통제 설계의 정본입니다. 본 연구 RQ3 대조군 설계의 근거입니다.
*   **Xie et al. (ICLR 2024), *Adaptive Chameleon or Stubborn Sloth* (arXiv:2305.13300) & Wu et al. (NeurIPS 2024), *ClashEval* (arXiv:2404.10198) & Jin et al. (LREC-COLING 2024), *Tug-of-War Between Knowledge* (arXiv:2402.14409):** 증거 정합성·섭동 강도·다중 문서 상충을 통제하여 충돌 고유 행동(확증 편향, 다수결, 사전지식 과신)을 분리한 연구들로, 본 연구가 난이도가 아닌 충돌 성분을 겨냥함을 정당화합니다.

---

## 3. 연구 방법론 (Methodology)

### 3.1. 데이터셋 구성 및 충돌 유형 축 (Data Setup & Conflict Taxonomy)
본 연구는 Reasoning LLM이 외부 문맥 간 충돌을 처리하는 내부 과정을 진단하기 위해, 실제 웹 검색의 이질적 출처를 담은 **DRAGged**를 주력 벤치마크로, 대규모 상충 문항을 담은 **RAMDocs**를 보강 데이터셋으로 채택합니다. 두 데이터셋은 충돌 정의와 문서 수가 상이하므로 **풀링하지 않고 데이터셋별로 분리 보고**하는 것을 원칙으로 합니다.

#### 1. DRAGged — 실제 웹 검색 기반 다원인 충돌 데이터셋
*   **(1) 데이터셋 설명:** Cattan et al.(2025, arXiv:2506.08500)이 구축한 실제 웹 검색 환경(In-the-wild) 기반 RAG 충돌 벤치마크입니다. 인위적 단일 토큰 조작 합성 데이터와 달리 실제 구글 검색(Top-10) 웹 문서를 담고 있어 출처·날짜·문맥이 이질적이며, Reasoning LLM이 실제 출처 비교와 속성 대조(Recency, Authority)를 수행하는 풍부한 `<think>` 트레이스를 유도하므로 본 진단에 적합합니다.
*   **(2) 데이터셋 충돌 유형과 건수:** 전체 458문항. 아래 분포는 원본 데이터 파일에서 직접 추출한 값입니다.

    | 충돌 유형 (Conflict Taxonomy) | 문항 수 | 특성 및 내용 |
    | :--- | :---: | :--- |
    | **시간적 충돌 (Freshness/Outdated)** | 62건 | 시간 흐름에 따라 과거 지식과 최신 사실이 상충 (예: 2024 vs 2025 일정) |
    | **오정보 충돌 (Misinformation)** | 5건 | 신뢰할 수 없는 출처의 오정보와 공식 출처의 참 정보가 상충 |
    | **상충되는 의견 (Conflicting Opinions)** | 115건 | 정답이 하나로 정해지지 않고 주장이 엇갈리는 관점 충돌 |
    | **상보적 정보 (Complementary Information)** | 115건 | 충돌하지 않고 상보적 디테일을 제공하는 대조군 |
    | **비충돌 (No Conflict)** | 161건 | 모든 문서가 일관된 정보를 가리키는 일반 RAG 대조군 |

    > **※ 데이터 재집계 기준 (§1.1 참조):** 본 연구의 표 수치는 원 논문 Table 2의 범주별 건수(161/115/115/62/5)와 정확히 일치하며, 순수 충돌 효과(RQ3)를 검증하기 위해 명시적 상충(182건)과 비상충 대조군(276건)으로 엄밀하게 구분한 자체 재집계입니다.
*   **(3) 원본 데이터셋 구조:** 각 문항은 질의(`question`), 충돌 유형(`conflict_type`), 정답(`correct_answer`), 10개 검색 문서(`search_results`)로 구성됩니다.
    ```jsonc
    {
      "question": "When does this year's Passover start?",
      "conflict_type": "Conflict due to outdated information",
      "correct_answer": "begins at sundown on Saturday, April 12.",
      "search_results": [
        {"date": "2025-01-01", "title": "When Is Passover in 2025...", "url": "...", "text": "Pesach 2025 begins before sundown on Saturday April 12, 2025..."}, // 유효
        {"date": "2024-05-01", "title": "When is Passover 2025?...",   "url": "...", "text": "It starts at dusk on the same day of the Hebrew calendar..."}, // 과거
        /* ... 총 10개의 실제 웹 검색 문서 (출처·날짜 이질적) */
      ]
    }
    ```
*   **(4) 본 연구의 진단에 맞춘 전처리 (Preprocessing & Setup):**
    *   **정답 문서 자동 식별 파이프라인:** DRAGged는 개별 문서에 정답 라벨이 없으므로, 문항의 `correct_answer` 텍스트가 10개 문서 중 어느 `text`에 포함되는지를 문자열 일치 + NLI 기반으로 사전 매핑하여, 모델이 `<think>`에서 올바른 문서를 지지했는지 추적할 수 있게 라벨링합니다.
    *   **문서 셔플링 및 표준 렌더링:** 위치 편향(Position Bias; Liu et al., 2023)을 통제하기 위해 매 실험마다 10개 문서 순서를 무작위 셔플링하고 `[Document 1] ~ [Document 10]` 포맷으로 주입합니다.
    *   **속성 메타데이터 활성화:** 2단계(판정)에서 최신성·권위 대조를 추적하기 위해 원본 `date`·`url`을 문서 헤더에 명시적으로 유지합니다.

#### 2. RAMDocs — 대규모 다중 문서 모호·오정보 벤치마크
*   **(1) 데이터셋 설명 및 출처:** Wang et al.(COLM 2025)이 제안한 다중 문서 QA 벤치마크(`HanNight/RAMDocs`)의 Test set 500문항입니다. 위키피디아 및 오픈 도메인 QA(NQ, TriviaQA 등)의 정답 문서에 통제된 오정보(Misinformation)와 노이즈를 인위적으로 섞어 넣어 구축되었습니다. DRAGged의 사실 충돌 표본(67건)이 작으므로, 정답/오답이 명확히 채점 가능한 RAMDocs로 정량 표본을 보강합니다.
*   **(2) 데이터셋 충돌 유형과 건수:** Test set 500문항, 문항당 평균 5.53개 문서. 문서는 정답(`correct`, 평균 3.84개), 오정보(`misinfo`, 평균 0.61개), 노이즈(`noise`, 평균 1.08개)로 혼재됩니다.
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
*   **(4) 본 연구의 진단에 맞춘 전처리 (Preprocessing & Setup):**
    *   **정답/오답 문서 자동 매핑:** 각 문서가 `gold_answers`인지 `wrong_answers`인지 사전 매핑하여, 모델이 `<think>` 내에서 인용·비교할 때 정답 문서와 오답 문서를 지지했는지 자동 판정하는 추적 파이프라인을 구축합니다.
    *   **문서 셔플링 및 표준 렌더링:** DRAGged와 동일하게 위치 편향을 통제하기 위해 문서 순서를 무작위 셔플링하고 표준 포맷(`[Document 1] ~ [Document N]`)으로 주입합니다.

### 3.2. 다단계 진단 프로토콜 (Multi-stage Diagnostic Protocol)
본 연구는 Reasoning LLM이 `<think>`에서 충돌을 처리하고 답변으로 전환하는 과정을 해부하기 위해 **3단계 궤적 라벨링**과 **전환 행렬 지표**를 적용합니다. 모든 지표는 §1.2에서 구분한 (L-a) 자기일관성, 즉 **텍스트로 관측 가능한 속성**에 대한 측정임을 명시합니다(인과 주장은 §3.3 실험 2로 분리). 지표의 출처를 명확히 하면, 본 절의 퍼널·특이 경로 지표(AIR·Shortcut·Unfaithful·Blind-Hit·Loss_L1/L2)와 전환 행렬 구성은 **본 연구가 제안하는 신규 지표**이며(자기일관성 개념의 계보는 Parcalabescu & Frank 2024에, 궤적-답변 해리 현상의 관찰은 Li et al. 2026에 둠), §4의 궤적 진단 기법들은 기존 방법론의 차용·각색으로서 각 기법에 출처를 명시합니다.

#### 1. 3단계 궤적 라벨링 (3-Stage Trajectory Labeling)
각 실험 턴마다 `<think>`와 최종 답변을 파싱하여 다차원 라벨링합니다.
*   **Level 1 (충돌·유형 인지):** 문서 간 불일치를 감지하고 그 원인(시간적/오정보/관점)까지 식별했는가? {correct_type, surface_only, unrecognized}
*   **Level 2 (해소 판정):** 여러 문서 중 정답 지지 문서(Correct Document)를 유효하다고 결론짓는 올바른 추론을 수행했는가? {correct, wrong, unresolved}
*   **Final Action (최종 표출):** `<think>` 종료 후 최종 텍스트에서 정답을 정확히 표출했는가(Exact Match)? {correct, wrong}

#### 2. 진단 지표 체계: 계층적 퍼널 및 전체 전환 행렬
**(1) 3대 계층적 전환 손실 지표 (Normative Funnel Loss):**
1.  **Level 1 인지 실패율 (Loss_L1):** P(Level1 = unrecognized).
2.  **Level 2 판정 오류율 (Loss_L2):** P(Level2 ≠ correct | Level1 ≠ unrecognized).
3.  **추론-답변 불일치율 (AIR = Loss_FA):** P(FinalAction = wrong | Level2 = correct). **이것이 본 연구의 핵심 자기일관성 위반 지표**입니다.

**(2) 비순차적 특이 경로 지표 (Non-sequential Pathways):**
1.  **암묵적 도약율 (Shortcut Rate):** P(FinalAction = correct | Level1 도달, Level2 미도달) — 명시적 해소 선언 없이 인지만으로 정답에 도달하는 비율.
2.  **불성실 추론율 (Unfaithful Rate):** P(FinalAction = correct | Level2 = wrong) — 사고에서 오답 문서를 지지했음에도 최종 정답을 맞추는 비율(L-c 개념의 부수 관측).
3.  **맹목 적중률 (Blind-Hit Rate):** P(FinalAction = correct | Level1 = unrecognized) — 충돌을 인지하지 못한 채 정답에 도달하는 비율(파라메트릭 지식 또는 우연). 이로써 정답 경로가 **정상 경로(Level2 = correct → 표출) + Shortcut + Unfaithful + Blind-Hit로 완전 분해**되어, 완화 기법의 EM 상승분 회계(RQ2-b)에 사용됩니다.

**(3) 측정 규약 (Scoring & Reporting Rules):**
1.  **분모 병기:** 조건부 지표는 모델 간 조건부 분모(예: Level 2 = correct 도달률)가 상이하면 서로 다른 부분모집단 위의 비율이 되어 비교가 오도될 수 있으므로, 모든 조건부 지표에 **결합확률(예: P(Level2 = correct ∧ FinalAction = wrong))과 분모 N을 병기**합니다.
2.  **다중 정답 처리(any-gold):** RAMDocs는 문항당 평균 2.2개의 유효 정답을 포함하므로, Level 2·Final Action의 correct는 **정답 집합 중 어느 하나(any-gold)를 지지·표출한 경우**로 정의합니다.
3.  **동치 채점:** Final Action 채점은 표현 차이가 불일치로 오인되지 않도록 **답변 정규화(별칭·수치 형식 통일)와 동치 판정**을 거친 EM으로 수행합니다(채점 노이즈의 AIR 누출 방지).

#### 3. 라벨 타당성 검증 (Label Validity)
AIR의 조건부는 "Level 2 = correct"라는 **텍스트로부터 추론된 내부 상태 라벨**에 의존하므로, 이 라벨의 구성타당도(construct validity)를 확보하는 것이 지표 전체의 신뢰성을 좌우합니다. 다음 세 층의 검증을 사전등록합니다.
*   **(a) 예측 타당성 — think 마스킹 부정 대조(Negative Control):** Level 2 라벨이 실제 정보를 담고 있다면, `<think>`를 제거/마스킹하고 답만 생성했을 때의 정답률 대비 `<think>` 존재 시 Level2→Final 전환에 유의한 예측력이 있어야 합니다. 예측력이 없으면 라벨이 사후적 장식일 수 있음을 시사하므로, 이 대조를 필수 게이트로 둡니다.
*   **(b) 판정자 편향 통제:** LLM-as-a-Judge의 위치 편향·장황함 편향·자기선호 편향을 통제하기 위해(Zheng et al., 2023; Panickssery et al., 2024), 판정자는 **트레이스 생성 모델과 다른 계열**을 사용하고(예: 대상 모델 ≠ 판정자), 옵션 순서를 무작위 스왑하며, 형식-채움(form-filling) 프로토콜(G-Eval; Liu et al., 2023)을 적용합니다. 판정자 구성은 **오픈 가중치 판정자 1종과 상용 판정자 1종을 병행**하여 재현성(오픈)과 판정 품질(상용)을 함께 확보하며, 대상 모델과 동일 계열의 판정자는 해당 모델의 트레이스 판정에서 제외합니다.
*   **(c) 인간·교차 판정 일치도:** 무작위 200건에 대해 인간 전문가 2인의 Cohen's κ와 서로 다른 두 LLM 판정자 간 교차 일치도를 보고합니다. 단, κ(라벨러 간 일치)는 (a)의 예측 타당성(라벨이 내부 상태를 가리키는가)을 대체하지 않음을 명시합니다.

### 3.3. 본 실험 설계 (Main Experiments)
본 연구는 RQ1~3을 검증하기 위해 3대 핵심 실험과 1개 경량 처방(proof-of-concept)을 수행합니다.

#### 실험 1 (RQ1 · RQ2-b): 전환 경로 진단 및 완화 기법 해부
*   **목적:** 모델이 어느 전환 단계에 병목(Loss_L1, Loss_L2, AIR)과 특이 경로(Shortcut, Unfaithful)를 겪는지 규명하고, 완화 기법 적용 시 전환 행렬이 어떻게 변하는지 대조합니다.
*   **평가 대상 환경 (Baselines & Interventions):** 아래 완화 기법 다수는 본래 비-thinking instruct 모델에서 개발·검증되었으므로, 본 연구는 이들을 사고 모델로 **이식(transplant)했을 때의 거동을 해부**하는 것임을 명시하고, 기법별로 아키텍처 무관 여부를 구분합니다.
    1.  **Standard RAG (Zero-shot):** 기본 검색 문서 주입 대조군.
    2.  **Context-Aware Decoding (CAD; Shi et al., 2024) — 아키텍처 무관(디코딩):** 문맥 유무 로짓 차이를 대비 증폭하는 디코딩 기법으로, 사고 모델 포함 임의의 자기회귀 모델에 그대로 적용됩니다 — Loss_L1/L2 개선 효과 검증. (충돌 시에만 적응 개입하는 AdaCAD, Wang et al., 2024/NAACL 2025 병행.) 사고 모델에서는 대비 디코딩의 적용 구간이 결과 해석을 좌우하므로, **(i) 사고+답변 전 구간 적용 vs (ii) `</think>` 이후 답변 구간 한정 적용**을 절제(ablation)로 구분 비교합니다.
    3.  **Recency/Authority-Guided Prompting — 전략 이식(프롬프트):** 충돌 시 작성 날짜(`date`) 최신성·출처(`url`) 권위를 우선 대조하도록 유도하는 프롬프트 — Loss_L2 개선 효과 검증.
    4.  **Reflection-style Prompting (Self-RAG의 반추 전략 차용; Asai et al., 2024) — 전략 이식:** Self-RAG 원본은 Llama 기반으로 학습된 별도 모델이라 사고 모델에 직접 이식할 수 없으므로, 본 연구는 그 **반추(reflection) 전략을 프롬프트로 차용**하여 결론과 문서의 사실 부합·충돌 해소를 스스로 재검토하게 합니다 — AIR 완화 효과 검증.
*   **절차:** 4개 환경에서 DRAGged·RAMDocs 문항(순서 셔플)을 주입해 사고·답변을 생성하고, 규칙 기반 정규식 + LLM-as-a-Judge 하이브리드로 3단계를 마킹하여 **환경별 전체 전환 행렬(Full Transition Matrix)**을 도출·대조합니다.

#### 실험 2 (RQ2-a): 인과 개입 — 사고→답변의 인과적 사용 규명
텍스트 상관을 넘어 **사고 결론이 최종 답변을 실제로 구동하는지(L-b)**를 개입으로 검증합니다. Bogdan et al.(2025)/Macar et al.(2025)의 경고에 따라 **on-policy resampling을 1차 방법**으로 삼고, off-policy steering은 탐색적으로만 사용합니다.
*   **(1) Truncation / Early-Answering (1차, Lanham et al., 2023):** `</think>` 직전 등 여러 절단 지점에서 사고를 끊고 즉시 답변을 강제 생성해, 어느 지점부터 답변이 정답으로 고정되는지(=사고 후반부가 답변에 인과적으로 기여하는지) 측정.
*   **(2) On-policy Resampling (1차, Bogdan et al., 2025; Macar et al., 2025):** 해소 판정 문장(Level 2) 직후 지점에서 이후 궤적을 K회 재생성하여, 해당 판정이 최종 답변 분포를 얼마나 좌우하는지 인과 기여도를 추정. AIR 샘플에서 이 기여도가 낮다면 "사고 결론이 답변에 인과적으로 반영되지 않음"을 실증. 모델이 자신의 중간 추론을 과소 사용한다는 인과 매개 분석의 게재 근거(Paul et al., 2024/FRODO, Findings of EMNLP 2024)가 이 개입의 이론적 배경을 뒷받침합니다.
*   **(3) Attention-Suppression (2차, Bogdan et al., 2025):** 정답 지지 문장에 대한 어텐션을 마스킹했을 때 답변이 오답으로 무너지는지로 그 문장의 인과 역할을 검증.
*   **(4) Activation Steering (탐색적, Rimsky et al., 2024/CAA; Turner et al., 2023/ActAdd):** `</think>` 전환 지점 잔차 스트림에 "문맥 충실" 대비 벡터(게재된 CAA, ACL 2024)를 주입해 최종 답변이 정답 쪽으로 이동하는지 관찰하며, 문맥 충실 유도의 보완 기법으로 게재된 context-faithful prompting(Zhou et al., 2023, Findings of EMNLP 2023)을 병행 비교합니다. **단, off-policy 개입은 resampling 대비 효과가 작고 불안정하다는 점(Macar et al., 2025)을 사전등록하고 보조 증거로만 취급**합니다.

#### 실험 3 (RQ2-b): 진단 기반 경량 처방 — AIR-Gate (Proof-of-Concept)
진단이 처방으로 이어짐을 보이기 위해, §4.3의 엔트로피 스파이크 발견을 트리거로 하는 **학습 불필요·디코딩 시점 경량 개입**을 시제품으로 제시합니다(전면적 방법 기여가 아닌 개념 증명으로 스코프 한정).
*   **트리거:** `</think>`→답변 첫 토큰 전환부에서 토큰 엔트로피(또는 단일 패스 Semantic Entropy Probe; Kossen et al., 2024) 스파이크 감지. 단, §4.3의 스파이크 신호가 약하거나 판별력이 낮게 실측될 경우에는 `<think>` 내 Level 2 판정 결론과 답변 후보 간 불일치를 텍스트 수준에서 감지하는 **대체 트리거로 전환**하여, 본 실험이 §4.3 가설의 성립 여부에 단일 의존하지 않도록 설계합니다.
*   **개입:** 스파이크 시 (i) 문맥 재가중 디코딩(CAD/AdaCAD; Shi et al., 2024; Wang et al., 2024) 또는 (ii) 정답 후보 문서 강제 재대조(re-read) 1회를 발동. 관련 선행으로 토큰 엔트로피로 충돌 토큰을 감지해 문맥을 우선하는 COIECD(Findings of ACL 2024, arXiv:2402.11893)가 있습니다.
*   **평가:** AIR 감소 폭과 EM 변화, 그리고 **개입이 새로운 Shortcut/Unfaithful 경로를 만들지 않는지**를 전환 행렬로 재확인.

#### 실험 4 (RQ3): 충돌 고유성 대조군 분리
*   **목적:** AIR·Shortcut·Unfaithful이 충돌 고유 현상인지, 난이도·문서 수의 부수 효과인지 분리합니다(Longpre et al., 2021; Xie et al., 2024의 통제 패러다임 적용).
*   **대조 설계:**
    *   **DRAGged 내부 대조:** 채점 가능한 사실 기반 충돌군(시간적 62 + 오정보 5 = 67) vs 비충돌 대조군(상보적 115 + 비충돌 161)에서 AIR·특이 경로율의 차이(Δ)를 측정하여 충돌 성분만 분리합니다. 의견 충돌은 정답 채점이 불가하므로 AIR 비교에서 제외하고 Level 1 인지·대조 패턴 비교에만 사용합니다. DRAGged 사실 충돌 표본이 작은 점(67건)은 아래 RAMDocs 대조로 보강합니다.
    *   **RAMDocs 문서 구성 대조:** 오정보 문서 포함(충돌) vs 노이즈만 포함(비충돌·동일 문서 수) 서브셋을 매칭하여, 문서 개수를 통제한 채 충돌 유무의 순효과를 추정.
    *   **통계:** 문서 수·질의 유형을 공변량으로 하는 혼합효과 로지스틱 회귀로 "충돌" 항의 유의성을 검정(Paired Bootstrap 95% CI 병행).

#### 대상 모델 및 생성·심사 프로토콜
*(1) 대상 모델:* 문헌에서 검증된 **본원적 추론 모델**을 3개 계열에서 엄선합니다. SFT 증류(Distillation)만 거친 모델은 자율 탐색 능력이 보존되지 않고 장식적 출력을 낼 위험이 있어 배제하고, 실제 탐색 궤적을 자발적으로 전개하는 20~32B 규모에 집중합니다.

| 모델 | 계열 | 출시 | 구조 및 규모 | 사고 트레이스 | 선정 근거 및 역할 |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Qwen3.6-27B** | Qwen | 2026-04 | 27B Dense | `<think>` | **주력 분석 모델**. 현행 오픈소스 밀집형 추론 모델 표준 |
| **Olmo-3.1-32B-Think** | AllenAI | 2025-12 | 32B Dense | `<think>` | **완전 개방형 대조군**. 가중치·데이터·학습 로그 전체 공개 |
| **gpt-oss-20b** | OpenAI | 2025-08 | 20B MoE | Harmony¹ | **교차 계열·MoE 검증**. OpenAI 개방 MoE 추론 모델 |

> ¹ `gpt-oss-20b`는 `<think>` 대신 Harmony 포맷의 Analysis 채널로 사고를 출력하므로 전용 파서로 동일한 3단계 프로토콜로 라벨링합니다. 단, 파싱 규약이 다른 점은 계열 간 비교 시 교란 변인으로 명시하고, 모델별 결과를 분리 보고합니다.

*(2) 생성·심사:* 모든 모델은 비양자화(bf16)로 구동해 로짓·궤적 왜곡을 방지하고, 권장 디코딩(예: `temp=0.6, top_p=0.95`)으로 무작위 시드 3회 반복 후 Paired Bootstrap 95% CI로 유의성을 검정합니다. 라벨링은 규칙 기반(문서 인덱스, 날짜/URL 키워드) + 고성능 LLM-as-a-Judge 하이브리드로 수행하되, §3.2(3)의 라벨 타당성 검증(부정 대조·판정자 편향 통제·인간 κ)을 필수로 병행합니다.

*(3) Thinking vs Non-thinking 통제 (Regime Control):* "관측된 취약성이 사고 채널 고유의 것인가, 아니면 기저 모델에서 물려받은 것인가"라는 귀속 반론을 선제적으로 통제합니다. AIR·Shortcut은 분리된 사고 결론을 전제하므로 비-thinking 모델에서는 정의되지 않지만, **충돌 해소 정확도(EM)와 완화 기법의 EM 이득**은 두 레짐에서 공통으로 비교 가능합니다.
*   **동일 가중치 토글(주력):** 가중치를 고정한 채 사고 채널만 켜고 끕니다(예: 하이브리드 추론 모델의 thinking on/off 플래그, `gpt-oss`의 reasoning effort 레벨). 이는 아키텍처·데이터를 완전히 통제한 가장 깨끗한 대조입니다.
*   **Matched 비-thinking 형제(보강):** 가능한 경우 동일 베이스의 instruct(비-thinking) 형제 모델을 병행하여 토글 미지원 계열을 보완합니다.
*   **귀속 판정:** 사고 채널 도입이 (i) 충돌 해소 정확도를 순증시키는지, (ii) 그럼에도 비-thinking에 없던 새로운 실패 양식(AIR·Shortcut·굴복)을 발생시키는지를 대조하여, 관측 현상을 사고 채널에 귀속합니다. 이 설계는 동시기 연구의 think/no-think 매칭 통제(Li et al., 2026)와 정합적입니다.

---

## 4. 사고 궤적(Thinking Trajectory)의 진단 기법

§4.1~4.3은 텍스트·로짓에서 관측되는 **상관적(observational) 궤적 지표**로, 병목의 위치와 상관 패턴을 규명합니다. **인과 주장은 §3.3 실험 2의 개입 배터리(truncation/resampling/steering)로만** 수행하며, 본 절의 지표는 그 개입 대상 지점을 특정하는 데 사용됩니다. 즉 4절은 "어디를 개입할지"를 찾고, 실험 2가 "그 지점이 인과적인지"를 검증하는 상호보완 구조입니다.

| 진단 목적 (3단계 병목) | 담당 기법 | 성격 | 문헌 근거 |
| :--- | :--- | :--- | :--- |
| **① L1→L2 병목:** 명시적 대조 없이 도약(Shortcut)·초기 문서 고착 | 4.2 Lock-in / RCPD | 상관 | Liu et al. (2023); Bogdan et al. (2025) |
| **② L2 판정 오염:** 신념 흔들림·오답 지지(Loss_L2) | 4.1 Trajectory Semantic Shift | 상관 | Lanham et al. (2023); Lee & Hockenmaier (2025) |
| **③ L2→FA 단절:** 올바른 판단 후 굴복(AIR) | 4.2 Overthinking + 4.3 Entropy Spike | 상관 | Kuhn et al. (2023); Li et al. (2026) |
| **①~③ 인과 검증** | §3.3 실험 2 개입 배터리 | **인과** | Lanham (2023); Bogdan (2025); Macar (2025) |

### 4.1. Trajectory Semantic Shift (의미론적 궤적 추적)
CoT를 문장 단위($s_1,\dots,s_T$)로 분할하고 각 문장 임베딩 $e(s_t)$와 **모든 검색 문서**의 코사인 유사도를 계산해, 문서를 정답 문서군(correct-doc) vs 충돌 문서군(conflicting docs)으로 묶고 각 군 중심과의 유사도 궤적을 추적합니다 (Lanham et al., 2023; Lee & Hockenmaier, 2025). 정답 채점이 가능한 사실 기반 충돌 그룹에 적용합니다.
*   **목적:** 사고가 진행됨에 따라 내부 신념이 어느 군으로 기우는지 시계열로 정량화하여 Loss_L2의 상관 패턴을 규명.
*   **분석 포인트:** AIR 샘플에서 마지막 문장까지 정답 군에 가깝다가 `<think>` 종료 임계 영역에서 급격히 충돌 군으로 기우는 **Late Shift**를 관측하고, 해당 지점을 실험 2의 개입 대상으로 지정.

### 4.2. Reasoning Completion Point Detection (RCPD): Lock-in 및 Overthinking 분석
사고 완료 지점 탐지(Reasoning Completion Point Detection, RCPD)로, 하이브리드 라벨링의 타임스탬프를 활용해 충돌 감지(L1)와 해소 확정(L2)이 전체 사고 길이 중 어느 토큰 인덱스에서 발생하는지 측정합니다.
*   **Lock-in Effect (초기 고착):** 입력 초기(Top-1) 문서가 앞쪽 10% 토큰을 지배하고 이 초기 판단이 최종까지 이어지는지 분석하여, 명시적 대조 없이 도달하는 **Shortcut의 상관 원인**을 규명합니다. 위치 편향의 기저는 Liu et al. (2023, *Lost in the Middle*)의 U자형 primacy/recency 곡선을 근거로 삼습니다.
*   **Overthinking (과잉 추론):** L2 결론 이후에도 불필요한 사고 토큰이 길게 이어질 때 오히려 최종 답변에서 굴복(Capitulation)이 늘어나는지 조사합니다. 긴 사고가 정확도를 해칠 수 있다(추론 길이와 정확도의 역U자 관계)는 근거로는 게재된 효율적 추론 서베이 *Stop Overthinking: A Survey on Efficient Reasoning for LLMs* (TMLR 2025)를 인용합니다.

### 4.3. Token-level Entropy & Perplexity Dynamics
CoT 전반 및 답변 첫 토큰 시점의 샤논 엔트로피 $H(X)=-\sum P(x_i)\log P(x_i)$와 Perplexity를 측정합니다 (Kuhn et al., 2023).
*   **목적:** 내부 혼란도(Internal Uncertainty)를 정량화하여, L2에서 올바른 판단을 내리고도 AIR이 발생하는 샘플의 상관 특성을 해부하고, §3.3 실험 3(AIR-Gate)의 트리거 신호를 설계.
*   **가설:** AIR 샘플은 L2에서 올바른 결론을 적었더라도 해당 토큰 엔트로피가 유의하게 높고(Internal Hesitation), 특히 `</think>`→답변 첫 토큰 전환 임계 지점에서 엔트로피가 스파이크로 치솟을 것입니다. 이 전환부 굴복 현상 자체는 Li et al.(2026)이 보고한 궤적-답변 해리와 정합적이며, 본 연구는 그 기저 신호로 엔트로피 동역학을 새로 제시합니다.

### 4.4. 오픈소스 통합 진단 프레임워크 (Integrated Diagnostic Framework)
§3.2의 3단계 하이브리드 라벨링, §4.1~4.3의 상관 진단, §3.3 실험 2의 인과 개입 배터리를 하나의 자동화 파이프라인으로 결합한 **오픈소스 통합 진단 프레임워크(`ThinkConflict-Diagnoser`)**를 구축·공개합니다.
*   **아키텍처:** (1) **3-Stage Audit Engine** — 사고·답변에서 Level 1·2 도달 여부를 자동 파싱해 전체 전환 행렬 도출; (2) **Trajectory Profiler (상관 진단)** — Semantic Shift 시계열, Lock-in/Overthinking 비율, 엔트로피 스파이크를 추출·시각화; (3) **Causal Prober (인과 개입)** — truncation/resampling/attention-suppression을 자동 실행해 지점별 인과 기여도를 리포트.
*   **기여:** 후속 연구자가 임의의 본원적 추론 모델·완화 기법을 플러그인하여 AIR·Shortcut을 감사(Audit)하고, 상관 관측을 인과 개입으로 즉시 검증할 수 있는 표준 툴킷을 제공합니다.
*   **공개물:** 논문 단계에서 (i) 신규/차용 구분을 명시한 **지표 레지스트리**, (ii) 새 모델 플러그인에 필요한 **최소 어댑터(사고 채널 파서) 명세** — gpt-oss용 Harmony 파서를 실증 사례로 동봉, (iii) 인간 검증 골드 라벨 200건(§3.2 라벨 타당성 검증 데이터)을 **프레임워크 테스트셋**으로 함께 공개합니다.

---

## 5. 예상 결과 및 한계점 (Expected Results & Limitations)

### 5.1. 주요 가설
*   **가설 1 (위치·경로 병목 — RQ1):** 본원적 추론 모델은 높은 비율로 충돌을 인지하고 올바른 해소 판단(L2)을 내리지만, 최종 답변에서 이를 유실하는 AIR이 유의하게 존재하며, 초기 문서 위치 고착(Shortcut)·과잉 추론에 의한 굴복이 관찰될 것이다.
*   **가설 2 (인과적 유실 — RQ2-a):** AIR 샘플에서 해소 판정 문장의 on-policy 인과 기여도가 비-AIR 대비 낮아, 사고 결론이 답변에 인과적으로 반영되지 않는 **진짜 유실**임이 실증될 것이다(단순 표기 오류가 아님).
*   **가설 3 (완화 기법의 착시 — RQ2-b):** 기존 완화 기법 적용 시 EM은 상승하나 그 상당분이 Shortcut·Unfaithful·Blind-Hit 경로에 기인하며, AIR-Gate 같은 진단 기반 처방은 새 특이 경로를 만들지 않고 AIR을 줄일 것이다.
*   **가설 4 (충돌 고유성 — RQ3):** AIR·특이 경로율은 비충돌/상보 대조군 대비 충돌군에서 유의하게 높아, 문서 수·난이도로 환원되지 않는 **충돌 고유 성분**이 존재할 것이다.

### 5.2. 연구의 한계점
*   행동(정확도) 평가는 사실 기반 충돌에 국한되며, 주관적 의견 충돌은 인지·대조 패턴으로만 분석합니다.
*   원인별 통계는 사실충돌·의견충돌 두 묶음 수준까지만 주장합니다. DRAGged 단독 오정보(5건)는 표본이 작아 정성 사례로만 제시하고, 오정보의 정량 근거는 RAMDocs로 보강합니다.
*   **라벨 타당성:** Level 2 라벨은 텍스트로부터의 추론이므로, §3.2(3)의 부정 대조·판정자 편향 통제·인간 κ로 구성타당도를 확보하되, 잔여 불확실성을 명시합니다.
*   **인과 개입의 한계:** off-policy activation steering은 on-policy resampling 대비 불안정하므로(Macar et al., 2025) 보조 증거로만 사용하며, 인과 주장의 주력은 truncation/resampling에 둡니다.
*   본원적 추론 모델 3계열(Qwen3.6-27B / Olmo-3.1-32B-Think / gpt-oss-20b)을 중심으로 분석하며, 증류 모델과 내부 trace 접근이 불가한 폐쇄형 상용 모델은 제외됩니다. gpt-oss의 Harmony 파싱 상이는 계열 간 비교의 교란 변인으로 명시합니다.
*   **스코프와 귀속:** AIR·Shortcut은 Thought→Text 아키텍처에서만 정의되는 실패 양식이므로 "thinking 모델 고유 현상"임은 결함이 아니라 명시적 스코프입니다. 다만 이 취약성이 사고 채널에서 비롯되는지 기저 모델에서 유래하는지는 §3.3(3)의 thinking on/off 통제로 귀속하며, 완화 기법은 대부분 비-thinking 모델 기원임을 밝히고 "사고 모델로의 이식 거동"을 해부하는 것으로 프레이밍합니다. 비-thinking 모델에 대한 AIR 일반화는 주장하지 않습니다.
*   **인용 등급(2026-07 arXiv 메타데이터 확인):** 하중 근거는 모두 게재 확정 문헌으로 떠받칩니다 — 인과 개입의 이론적 배경은 Lanham(2023, 표준 인용)·FRODO(Findings of EMNLP 2024)·CAA(ACL 2024)·context-faithful prompting(Findings of EMNLP 2023), 완화/오버싱킹 근거는 COIECD(Findings of ACL 2024)·FaithfulRAG(ACL 2025)·*Stop Overthinking*(TMLR 2025)·Lee & Hockenmaier(Findings of EMNLP 2025). 의도적으로 유지한 프리프린트는 두 부류로 한정됩니다: (1) **직접 사용하는 데이터셋 산출물** DRAGged(2506.08500) — 속성을 자체 검증하며 게재된 RAMDocs(COLM 2025)로 이중화, (2) **resampling 방법 출처** Thought Anchors(2506.19143)·Thought Branches(2510.27484) — 게재된 Lanham·FRODO로 백본을 대체 가능하게 설계. 동시기 프리프린트 Li et al.(2026)은 포지셔닝 전용으로 수치 비의존입니다. 제출 시점에 이들 프리프린트의 게재 상태를 재확인해 등급을 갱신합니다.

---

## 6. 로드맵 및 파일럿 게이트 (Roadmap & Gates)

*   **[P0] 데이터·인프라:** DRAGged(458)·RAMDocs(500)를 셔플 렌더 포맷으로 정비하고, 3단계 파싱 하이브리드 파이프라인 완성. **라벨 타당성 부정 대조(think 마스킹)를 P0 게이트로 선검증.**
*   **[P1 게이트] 예비 실측 (Go/No-Go):** 주력 Qwen3.6-27B를 소량(N≈50) 구동하여 AIR 및 Shortcut 경로가 유의하게 관찰되는지 확인(불일치율 + 특이 경로 합계 ≥ 30% 시 본격 추진).
*   **[P2] 다모델 스케일 + 완화 기법 해부 (실험 1):** Olmo-3.1-32B-Think, gpt-oss-20b로 3계열 확장, 완화 기법 4종의 전체 전환 행렬 도출.
*   **[P3] 충돌 고유성 + 레짐 대조 (실험 4 + Regime Control):** 충돌 vs 비충돌/상보 대조군의 AIR·특이 경로 Δ를 혼합효과 회귀로 검정하고, thinking on/off 토글로 충돌 해소 정확도·완화 EM 이득의 레짐 차이를 측정해 사고 채널 귀속을 확정.
*   **[P4] 상관 궤적 진단 (§4.1~4.3):** Semantic Shift·Lock-in/Overthinking·Entropy Spike로 병목 지점 특정.
*   **[P5] 인과 개입 (실험 2):** truncation/resampling/attention-suppression으로 사고→답변 인과적 사용을 검증하고, steering은 탐색적 보조 증거로 병행.
*   **[P6] 경량 처방 PoC (실험 3):** AIR-Gate(엔트로피 트리거 + CAD/재대조)로 AIR 감소·부작용 부재를 확인하고 최종 분석 보고서 완성.

---

## 부록 A. 인용 정본 목록 (검증 완료)

*   Xu et al. (2024). *Knowledge Conflicts for LLMs: A Survey.* EMNLP 2024. arXiv:2403.08319.
*   Cattan et al. (2025). *DRAGged into Conflicts: Detecting and Addressing Conflicting Sources in Search-Augmented LLMs.* arXiv:2506.08500. (프리프린트 — 직접 사용하는 주력 데이터셋, RAMDocs로 이중화)
*   Wang, Prasad, Stengel-Eskin, Bansal (2025). *Retrieval-Augmented Generation with Conflicting Evidence (RAMDocs + MADAM-RAG).* COLM 2025. arXiv:2504.13079. (RAMDocs 벤치마크와 MADAM-RAG 기법이 동일 논문)
*   Asai et al. (2024). *Self-RAG.* ICLR 2024. arXiv:2310.11511.
*   Shi et al. (2024). *Trusting Your Evidence: Hallucinate Less with Context-Aware Decoding (CAD).* NAACL 2024. arXiv:2305.14739.
*   Wang et al. (2024). *AdaCAD: Adaptively Decoding to Balance Conflicts between Contextual and Parametric Knowledge.* NAACL 2025. arXiv:2409.07394.
*   Zhang et al. (2025). *FaithfulRAG: Fact-Level Conflict Modeling for Context-Faithful RAG.* ACL 2025. arXiv:2506.08938.
*   *ConflictRAG: Detecting and Resolving Knowledge Conflicts in Retrieval-Augmented Generation.* 2026. arXiv:2605.17301. (MLP 충돌 감지 + Entropy-TOPSIS; 프리프린트)
*   Cuconasu et al. (2024). *The Power of Noise: Redefining Retrieval for RAG Systems.* SIGIR 2024. arXiv:2401.14887.
*   Zhou et al. (2023). *Context-faithful Prompting for Large Language Models.* Findings of EMNLP 2023. arXiv:2303.11315.
*   Turpin et al. (2023). *Language Models Don't Always Say What They Think: Unfaithful Explanations in CoT Prompting.* NeurIPS 2023. arXiv:2305.04388.
*   Lanham et al. (2023). *Measuring Faithfulness in Chain-of-Thought Reasoning.* Anthropic 프리프린트. arXiv:2307.13702.
*   Atanasova et al. (2023). *Faithfulness Tests for Natural Language Explanations.* ACL 2023. arXiv:2305.18029.
*   Parcalabescu & Frank (2024). *On Measuring Faithfulness or Self-Consistency of Natural Language Explanations.* ACL 2024. arXiv:2311.07466.
*   Lee & Hockenmaier (2025). *Evaluating Step-by-step Reasoning Traces: A Survey.* Findings of EMNLP 2025. arXiv:2502.12289.
*   Li, Krishnan, Padman (2026). *The Chain Holds, the Answer Folds: Trace–Answer Dissociation in Reasoning Models Under Adversarial Pressure.* arXiv:2605.29087. (동시기 프리프린트 — 포지셔닝 전용, 수치 비의존)
*   Bogdan et al. (2025). *Thought Anchors: Which LLM Reasoning Steps Matter?* arXiv:2506.19143.
*   Macar et al. (2025). *Thought Branches: Interpreting LLM Reasoning Requires Resampling.* arXiv:2510.27484.
*   Turner et al. (2023). *Activation Addition: Steering Language Models Without Optimization (ActAdd).* arXiv:2308.10248.
*   Rimsky et al. (2024). *Steering Llama 2 via Contrastive Activation Addition (CAA).* ACL 2024. arXiv:2312.06681.
*   Paul et al. (2024). *Making Reasoning Matter: Measuring and Improving Faithfulness of CoT (FRODO).* Findings of EMNLP 2024. arXiv:2402.13950.
*   *Stop Overthinking: A Survey on Efficient Reasoning for Large Language Models.* TMLR 2025. (게재 — 오버싱킹/추론길이-정확도 역U자 근거)
*   Kuhn, Gal, Farquhar (2023). *Semantic Uncertainty.* ICLR 2023. arXiv:2302.09664.
*   Kossen et al. (2024). *Semantic Entropy Probes.* arXiv:2406.15927.
*   Liu et al. (2023). *Lost in the Middle: How Language Models Use Long Contexts.* TACL 2024. arXiv:2307.03172.
*   Longpre et al. (2021). *Entity-Based Knowledge Conflicts in Question Answering.* EMNLP 2021. arXiv:2109.05052.
*   Xie et al. (2024). *Adaptive Chameleon or Stubborn Sloth: Revealing the Behavior of LLMs in Knowledge Conflicts.* ICLR 2024. arXiv:2305.13300.
*   Wu et al. (2024). *ClashEval: Quantifying the Tug-of-War Between an LLM's Internal Prior and External Evidence.* NeurIPS 2024 D&B. arXiv:2404.10198.
*   Jin et al. (2024). *Tug-of-War Between Knowledge: Resolving Knowledge Conflicts in RAG.* LREC-COLING 2024. arXiv:2402.14409.
*   Zheng et al. (2023). *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.* NeurIPS 2023. arXiv:2306.05685.
*   Liu et al. (2023). *G-Eval: NLG Evaluation Using GPT-4 with Better Human Alignment.* EMNLP 2023. arXiv:2303.16634.
*   Panickssery et al. (2024). *LLM Evaluators Recognize and Favor Their Own Generations.* NeurIPS 2024. arXiv:2404.13076.
*   Yuan et al. (2024). *Discerning and Resolving Knowledge Conflicts through Adaptive Decoding with Contextual Information-Entropy Constraint (COIECD).* Findings of ACL 2024. arXiv:2402.11893.
