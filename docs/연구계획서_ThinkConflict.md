# 연구 계획서: Lost in Translation from Thought to Text: Investigating Reasoning-Answer Inconsistency in RAG Conflicts

*   **투고 목표:** ACL / EMNLP Main (Long Paper) — 진단 및 해석가능성 중심 (Diagnostic & Interpretability Track)
*   **한 줄 요약:** RAG의 외부 문맥 간 충돌(Inter-Context Conflict) 상황에서 추론 모델(Reasoning LLM)의 **사고→답변 과정을 인지→해소→표출 3단계 전환 행렬로 분해·정량화하는 재현 가능한 진단 프레임워크**를 제안하고, 이를 통해 **추론-답변 불일치(AIR)**를 비롯한 실패가 **어느 단계에서 발생하는지 측정**하며, 충돌 대조군으로 충돌 고유성을, 인과 개입으로 인과적 지점을 규명한다.

---

## 1. 서론 (Introduction)

### 1.1. 배경 및 문제 정의
검색 증강 생성(Retrieval-Augmented Generation, RAG)의 외부 문맥 간 충돌(Inter-Context Conflict) — 동일 질의에 상충 문서가 함께 반환되는 상황 — 은 지식 충돌의 주요 유형으로 정립되었으며 (Xu et al., 2024), 실제 웹 검색 벤치마크에서 질의의 상당수가 여러 출처의 대조·종합을 요구합니다 (Cattan et al., 2025). 이는 개방형 웹 검색에 국한되지 않아, 위키피디아 같은 큐레이션 코퍼스와 기업 내부 지식베이스에서도 문서 간 모순이 실재합니다 (WikiContradict, Hou et al., 2024; EnterpriseRAG-Bench, Sun et al., 2026). 이런 충돌에서 모델을 평가하는 표준은 정답률(EM)이지만, EM은 정작 **모델이 왜 맞히고 틀리는지**를 가리지 못합니다. 같은 정답이라도 문서를 올바로 대조한 결과와 우연히 도달한 결과가 뒤섞여 있고, 오답 역시 충돌을 아예 못 본 것·보고도 잘못 판단한 것·옳게 판단하고도 최종 답변에서 뒤집은 것이 서로 다른 실패이기 때문입니다.

추론 모델(Reasoning LLM)은 이 물음에 접근할 창을 엽니다. 프롬프트로 강제한 단계적 사고(CoT)가 정답을 사후에 정당화하는 **'사후 합리화(Unfaithful Rationalization)'**에 오염되는 것과 달리 (Turpin et al., 2023; Lanham et al., 2023), 추론 모델은 사고 과정(`<think>`)을 답변과 분리된 채 자발적으로 외부화합니다. 사고 결론과 최종 답변이 어긋나는 현상 자체는 동시기 연구가 추론 모델에서 관찰했으나 (Li et al., 2026), 이들은 현상을 **보고**하는 데 머물렀을 뿐, 그 어긋남이 인지·해소·표출 중 어느 단계에서 비롯되는지, 성공한 정답이 진짜 추론의 산물인지 우연인지를 **단계별로 분해·귀속하지 않았습니다.**

본 연구는 충돌 하 추론 모델의 사고→답변 과정을 **인지·해소·표출 3단계로 분해**하고, 하나의 전환 행렬로 모든 결과를 **정상·Shortcut·Discordant Hit·Blind-Hit 경로에 귀속(attribute)**하는 진단 프레임워크를 제안합니다. 그 안에서 '올바로 해소하고도 답변에서 유실'하는 자기일관성 위반을 **추론-답변 불일치율(AIR)**로 정식화합니다(자기일관성·인과·충실성의 개념적 경계는 §1.2). 다만 네이티브 트레이스라 해서 내부 연산에 충실하다고 전제하지 않으며, 우리가 주장하는 것은 출력 수준의 자기일관성에 한정됩니다. 이 귀속은 정답을 맞혔는지뿐 아니라 **어떤 경로로 맞히고 틀렸는지**를 드러내어, 정답률만으로는 보이지 않던 실패 지형과 완화 기법의 작동 방식을 해부하는 토대가 됩니다.

**본 연구의 기여는 다음과 같습니다.**
1.  **진단 프레임워크:** 충돌 하 사고→답변 과정을 **인지·해소·표출 3단계 전환 행렬**로 분해하고 모든 정답을 **정상·Shortcut·Discordant Hit·Blind-Hit 네 경로로 상호배타·전수 귀속**하는 재현 가능한 프레임워크를 정립하며, 자기일관성 위반을 **추론-답변 불일치율(AIR)**로 정식화합니다.
2.  **실패 지형과 완화 분해 (발견 1, RQ1·RQ2):** 3계열 추론 모델·5개 완화 환경에서 전환 행렬을 실측해 실패가 **어느 단계에 몰리는지** 지도화하고, 완화 전후를 **문항별로 짝지어 경로 이동을 추적**함으로써 (i) EM 상승 중 진짜 대조 추론에서 온 비율(**정당 이득 비율**), (ii) 순 정확도가 가린 **숨은 퇴행**, (iii) 취약 경로 정답이 섭동에 더 잘 깨진다는 **경로별 취약성**을 드러냅니다 — 정답률만 보면 상쇄되어 보이지 않던 이 셋이 "정확도 상승 ≠ 추론 개선"을 정량·인과적으로 밝힙니다.
3.  **충돌 고유성과 인과 (발견 2, RQ3·RQ4):** 비충돌 대조군으로 관측 현상이 난이도가 아닌 **충돌 고유 성분**임을 분리하고, truncation·resampling 개입으로 사고 결론이 답변을 실제로 구동하는지(L-b)를 **인과적으로** 규명합니다.

### 1.2. 개념 구분: 자기일관성 · 인과적 사용 · 충실성
본 연구의 진단 대상을 정확히 규정하기 위해, 사고-답변 관계에 관한 세 가지 핵심 개념을 명시적으로 구분합니다. 이 구분은 본 연구의 **주장 경계**를 정합니다: **사고와 답변이 어긋났다(AIR)**는 출력 텍스트만으로 판정하지만, **그 사고가 답변을 실제로 유발했다(인과)**는 텍스트로는 입증할 수 없어 개입 실험(§3.3)을 통해서만 검증하고 그 주장을 해당 실험에 한정합니다.

*   **(L-a) 추론-답변 자기일관성(Reasoning-Answer Self-Consistency):** 모델이 `<think>`에서 내린 결론과 최종 답변(`Text`)의 내용이 논리적으로 일치하는가. 이는 모델에 개입하지 않고 **출력된 텍스트만으로 관측 가능한 속성**입니다. 본 연구의 핵심 지표 **추론-답변 불일치율(AIR)**은 `<think>`에서 정답을 맞게 추론(결론 존재)했으나 최종 답변에서 오답을 내는 경우로, 결론과 답변의 **불일치**를 직접 잽니다. Parcalabescu & Frank (2024)에 따라 본 연구는 AIR을 **출력 수준의 자기일관성 지표로 엄정히 규정**합니다. 한편 **암묵적 도약율(Shortcut)** — `<think>`에서 어느 문서가 옳은지 **명시적 결론을 내리지 않은 채** 곧장 정답을 표출한 경우 — 은 성격이 다릅니다. AIR은 '적어둔 결론'과 답변을 맞대어 불일치를 재지만, **Shortcut은 애초에 맞대어 볼 결론이 없습니다.** 따라서 이는 자기일관성 위반이 아니라, **모델이 해소 과정을 생략한 채 정답에 도달했다는 사실(근거가 약한 정답)**을 포착하는 인접 지표로 함께 사용합니다. AIR과 **방향이 반대**인 현상 — `<think>`에서는 오답 문서를 지지해 놓고 최종 답변은 정답을 맞추는 경우 — 도 결론과 답변의 불일치라는 점에서 동일하게 텍스트로 관측되며, §3.2 전환 행렬에서 **불협 적중률(Discordant Hit)**로 함께 기록합니다. 다만 이는 어디까지나 출력 수준 불일치의 관측일 뿐, 트레이스가 내부 연산에 불충실함(L-c)을 주장하는 것은 아닙니다.
*   **(L-b) 인과적 사용(Causal Use):** `<think>`의 사고 과정이 최종 답변을 도출한 **실질적 인과 원동력(Cause)**으로 작동했는가. 이는 텍스트 관측만으로는 증명할 수 없으며, 사고 궤적을 강제로 자르거나(Truncation) 판정 지점 이후를 재생성(on-policy Resampling)하는 등의 **인과 개입 실험**을 통해서만 검증할 수 있습니다 (Lanham et al., 2023; Bogdan et al., 2025). 따라서 본 연구는 **개입 시 정답률 폭락 폭(ΔAccuracy)과 개입 전후 답변 변화율** 지표를 통해 인과적 사용 여부를 측정하며, 이에 관한 주장은 §3.3 실험 3(인과 개입)에 한정하여 엄밀하게 제기합니다.
*   **(L-c) 내부 연산에 대한 충실성(Faithfulness to Computation):** 모델이 텍스트로 표출한 생각(`<think>`)이 정답을 고른 '진짜 속마음(내부 연산)'을 정직하게 반영하는가. 예컨대 Turpin et al. (2023)은 질문에 편향된 힌트를 몰래 섞으면 모델이 힌트 때문에 답을 골랐으면서도 사고 과정에서는 논리적으로 계산한 척 가짜 근거를 지어냄(충실성 결여)을 보였고, 이러한 불충실성은 프롬프트 강제 CoT뿐 아니라 자연스러운 설정의 추론 모델 트레이스에서도 관측됩니다 (Arcuschin et al., 2025). **본 연구는 L-c를 진단 대상에서 제외합니다.** 두 가지 이유에서입니다: (i) 충실성은 텍스트만으로 판정할 수 없고 힌트 편향 조작이나 내부(기계적) 접근을 요구하는데, 이는 본 연구의 텍스트+생성개입 방법론의 범위 밖이며 그 자체로 활발한 별도 연구 영역입니다 (Arcuschin et al., 2025; Young, 2026); (ii) 본 연구의 질문은 자기일관성(L-a, 측정)과 인과적 사용(L-b, 개입)으로 답할 수 있어, 트레이스가 내부 연산을 반영하는지를 판정하지 않아도 성립합니다. 따라서 본 연구는 트레이스를 '충실한 속마음의 거울'로 전제하지 않으며, 최근 논의를 따라 이를 **불완전하지만 유용한 관측 신호(monitorable-but-fragile signal)**로 취급합니다 (Korbak et al., 2025) — AIR은 바로 이 신호가 충돌 상황에서 답변과 어긋나 신뢰를 잃는 지점을 정량화합니다.

정리하면, 본 연구는 자기일관성(L-a)은 텍스트로 측정하고, 인과적 사용(L-b)은 개입으로 검증하며, 충실성(L-c)은 판정하지 않습니다. 이 세 경계가 이후 지표 정의(§3.2)와 실험 설계(§3.3)의 준거가 됩니다.

### 1.3. 연구 질문 (Research Questions)
1.2절에서 정의한 개념 구분을 바탕으로, 본 연구는 다음 네 가지 핵심 연구 질문(RQ)을 설정하여 실험적 검증을 수행합니다.

*   **RQ1 (진단 및 불일치 위치):** 추론 모델은 외부 문맥 간 충돌 환경에서 사고 과정과 최종 답변 간의 자기일관성(L-a)을 유지하는가? 불일치가 관측된다면, 정보 처리의 어느 단계(인지 → 판정 → 최종 표출)에서 주로 단절이 발생하는가?
*   **RQ2 (완화 기법 해부):** 문헌 기반 RAG 완화 기법(예: Context-Aware Decoding, Recency/Authority-Guided Prompting, Self-Reflection)을 적용할 때, 표면 정답률(EM) 상승분이 진정한 대조 논리에서 오는가, 아니면 암묵적 도약(Shortcut)·불협 적중(Discordant Hit)·맹목 적중(Blind-Hit) 경로에서 오는가?
*   **RQ3 (충돌 고유성 - 대조군 분리):** 관측된 불일치(AIR)와 특이 경로는 **충돌 상황에 고유한 실패**인가, 아니면 단순히 문항 난이도·문서 개수 증가에 따른 부수 효과인가? 동일 모델·동일 질의에 대해 **충돌 vs 비충돌/상보 대조군**을 비교하여 충돌 고유 성분을 분리합니다 (Longpre et al., 2021; Xie et al., 2024).
*   **RQ4 (인과 규명):** 텍스트로 관측된 사고 결론은 최종 답변을 **인과적으로 구동(L-b)**하는가, 아니면 답변이 사고와 독립적으로 결정되는가? 사고 궤적 개입(truncation·resampling)으로 검증합니다.

---

## 2. 관련 연구 및 카테고리화 (Related Work)

RAG 환경에서의 지식 충돌 해소와 LLM의 추론 일관성에 관한 연구가 최근 활발히 진행되고 있으므로, 본 연구의 학술적 위치를 세 가지 카테고리로 나누어 기존 문헌들과 명확히 차별화합니다.

```
                     [ RAG 지식 충돌 및 추론 진단 연구 ]
                                 │
          ┌──────────────────────┼──────────────────────┐
          ▼                      ▼                      ▼
 [2.1 충돌 진단]          [2.2 충돌 완화]         [2.3 진단·통제 방법론]
 Xu (2024)               Self-RAG (2024)         Turpin/Lanham/Bogdan/Macar
 Cattan/DRAGged (2025)   Shi/CAD, Jin/CD2        Parcalabescu&Frank/Arcuschin/Korbak
 Wang/RAMDocs (2025)     MADAM-RAG, FaithfulRAG  Li/Young (충돌×트레이스 교차 선행)
                         ConflictRAG (2026)      Longpre/Xie/Wu/Jin (대조군 통제)
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 ▼
         [본 연구: 충돌 하 사고→답변 실패를 단계별로 측정·귀속하는 진단 프레임워크]
    - 3단계 전환 행렬로 결과를 4경로(정상·Shortcut·Discordant Hit·Blind-Hit)에 귀속 (신규 지표 AIR)
    - 기존 완화 기법(CAD, 프롬프트 성찰 등) 이식 시 EM 상승의 경로별 출처 분해 (RQ2)
    - 충돌 vs 비충돌 대조군(RQ3) 및 truncation/resampling 인과 개입(RQ4)
```

### 2.1. RAG에서의 지식 충돌 진단 (Conflict Diagnosis in RAG)
최근 RAG 문헌은 LLM이 직면하는 지식 충돌을 단편적 오류가 아닌 구조적 유형(내부 vs 외부, 외부 문서 간 상충)으로 세분화하고, 실제 웹 검색 환경의 다원적 복잡성을 반영한 대규모 벤치마크 구축으로 나아가는 추세입니다.

*   **Xu et al. (EMNLP 2024), *Knowledge Conflicts for LLMs: A Survey* (arXiv:2403.08319):** LLM이 직면하는 지식 충돌을 모델 내부 지식과 외부 문서가 상충하는 **Context-Memory 충돌**, 검색된 외부 문서들 간에 엇갈리는 **Inter-Context 충돌**, 모델 내부 지식 간 모순인 **Intra-Memory 충돌**의 3대 분류로 정립했습니다.
*   **Cattan et al. (2025), *DRAGged into Conflicts: Detecting and Addressing Conflicting Sources in Search-Augmented LLMs* (arXiv:2506.08500):** 실제 구글 검색으로 반환된 웹 문서 기반 RAG 충돌 벤치마크입니다. Table 2 기준 전체 458건을 **No conflict(161)**, **Complementary information(115)**, **Conflicting opinions(115)**, **Freshness/Outdated(62)**, **Misinformation(5)**로 라벨링했습니다. 논문은 이 중 뒤 4개 범주(297건, 64.8%)를 "conflicting"으로 집계하는데, 이는 여러 출처의 종합이 필요한 넓은 범주로 상보적 정보(호환되나 종합 필요)를 포함하며, 진짜 모순인 **명시적 상충은 182건**입니다.
*   **Wang et al. (COLM 2025), *Retrieval-Augmented Generation with Conflicting Evidence* (RAMDocs; arXiv:2504.13079):** 문항당 평균 5.5개의 문서가 정답(`correct`)·오정보(`misinfo`)·노이즈(`noise`)로 혼재된 대규모 다중 문서 QA 벤치마크(500문항 test set)를 제안하고, 모호성·오정보·노이즈가 공존할 때 RAG의 오류 감지와 답변 거부(Refusal) 한계를 정량화했습니다.
*   **본 연구의 타겟:** 우리는 이 분류 중 실제 웹 검색 RAG에서 가장 빈번한 **Inter-Context 충돌(외부 문서 간 상충)**을 핵심 진단 대상으로 설정하고, DRAGged의 충돌 유형별로 모델의 반응 패턴을 진단합니다.

### 2.2. Inter-Context 충돌 해결 시도와 그 한계 (Mitigating Inter-Context Conflicts)
기존의 충돌 해소 연구들은 주로 디코딩 시점의 로짓 제어나 외부 맞춤형 파이프라인(분류기·다중 에이전트 토론)을 도입하는 방향으로 발전해 왔으나, 단일 모델 내부의 자발적 추론 채널(`<think>`)이 충돌을 어떻게 다루는지에 대한 인과적 규명은 부족한 실정입니다. **본 연구는 문헌에서 검증된 이러한 기존 완화 기법들 중 일부(디코딩 제어, 프롬프트 성찰 등)를 추론 모델(Reasoning LLMs)에 이식했을 때, 내부 사고 궤적과 충돌 해소 메커니즘이 실제로 어떻게 동작하고 변하는지를 해부(Anatomy)하고 분석하는 데 목적이 있습니다.**

*   **Asai et al. (ICLR 2024), *Self-RAG* (arXiv:2310.11511):** 반성 토큰(Reflection tokens)을 학습시켜 검색 문서 관련성과 생성 답변 사실성을 모델 스스로 평가·제어하게 한 성찰 기반 RAG 프레임워크입니다.
*   **디코딩 시점 개입 (CAD, CD2):** 문맥 유무 두 분포의 로짓 차이를 대비 증폭(Contrastive Decoding)하여 생성을 제공 문맥 쪽으로 밀어붙이거나(CAD; Shi et al., 2024, 충돌 적응형 변형 AdaCAD 포함), 문서 간 충돌을 분리해 대비 디코딩(CD2; Jin et al., 2024)하는 디코딩 시점 개입 기법들입니다.
*   **경량 감지 및 맞춤형 파이프라인 (ConflictRAG, 2026; arXiv:2605.17301):** 외부 검색 문서 간의 충돌을 임베딩 기반 MLP 분류기로 감지하고, **Entropy-TOPSIS** 순위 기법으로 출처 신뢰도를 평가해 유형별(오정보·시간·관점 등)로 적합한 문서를 맞춤형 선별·해결하는 파이프라인 기법입니다(프리프린트).
*   **다중 에이전트 토론 (MADAM-RAG; Wang et al., 2025):** §2.1의 RAMDocs 벤치마크와 **동일 논문(arXiv:2504.13079)**에서 함께 제안된 기법으로, 상충되는 다중 외부 문서 각각에 대변인 LLM 에이전트를 할당하고 **에이전트 간 다중 라운드 토론(Debate) 및 중앙 집계**를 통해 모호성과 오정보 충돌을 동시 해소합니다.
*   **사실 수준 충돌 모델링 (FaithfulRAG; Zhang et al., 2025):** 검색 문서의 사실 수준 충돌을 명시적으로 모델링하여 문맥 충실 생성을 유도하는 기법입니다.

### 2.3. 추론 일관성 진단 및 대조군 통제 방법론 (Evaluating Reasoning Inconsistency & Control Paradigms)
최근 추론 모델(Reasoning LLM)의 등장에 따라 사고 궤적의 일관성을 평가하고 개입하는 방법론이 발전하는 동시에, 문항 난이도와 순수 충돌 효과를 분리하기 위한 대조군 통제 패러다임이 강조되고 있습니다. **본 연구는 수학/QA 도메인의 사회적 반박 압박을 다룬 선행 연구들과 달리, RAG 검색 환경의 '상충 문서 공존이라는 문맥적 압박(Contextual Pressure)' 속에서 처리 과정을 다단계로 분해하고, on-policy 개입과 비충돌 대조군 설계를 결합하여 인과적 규명(RQ4)과 충돌 고유성(RQ3)을 동시에 검증합니다.**

#### 2.3.1. 사고-답변 괴리 현상과 궤적 개입 방법론
**추론 모델에서 사고와 답변이 어긋나는 현상을 보고한 선행 연구들과, 단계별 궤적을 정량화하고 재생성(Resampling)·궤적 절단(Truncation) 등 개입 실험으로 특정 추론 단계의 인과적 영향력을 측정하는 방법론을 함께 다룹니다.**

*   **Lee & Hockenmaier (Findings of EMNLP 2025), *Evaluating Step-by-step Reasoning Traces: A Survey* (arXiv:2502.12289):** 단계적 추론 궤적을 사실성(Factuality)·타당성(Validity)·일관성(Coherence)·유용성(Utility)의 4축으로 분해 평가하는 체계적 진단 프레임워크를 정리했습니다.
*   **Li et al. (2026), *The Chain Holds, the Answer Folds: Trace–Answer Dissociation in Reasoning Models Under Adversarial Pressure* (arXiv:2605.29087, 동시기 프리프린트):** Reasoning 모델이 사고 궤적에서는 사실적으로 올바른 판단을 유지하면서도 반박 압박 하에서 **최종 답변만 오답으로 굴복(unfaithful capitulation)**하는 궤적-답변 해리 현상을 보고했습니다. 본 연구의 AIR과 가장 가까운 선행 현상이나, (1) 사회적 반박 압박이 아닌 **다문서 병존이라는 문맥적 압박**을 다루고, (2) 현상 보고를 넘어 **충돌 대조군 분리(RQ3)와 인과 개입(RQ4)**으로 나아간다는 점에서 차별됩니다. 동시기 프리프린트이므로 포지셔닝 목적으로만 인용하고 그 수치에 의존하지 않습니다.
*   **Young (2026), *Why Models Know But Don't Say: Chain-of-Thought Faithfulness Divergence Between Thinking Tokens and Answers in Open-Weight Reasoning Models* (arXiv:2603.26410, 프리프린트):** 오픈 가중치 추론 모델 12종에서 사고 토큰에는 담긴 정보가 최종 답변에서 누락되는 **사고-답변 해리(thinking-answer divergence)**를 대규모 측정했습니다(힌트 추종 사례의 55.4%). 본 연구가 다루는 사고-답변 불일치(AIR·Discordant Hit)와 현상이 인접하나, 이 연구는 **인위적 오도 힌트(hint-bias, L-c 패러다임)**로 트레이스 투명성을 측정하는 반면, 본 연구는 **inter-context 충돌(L-a)**에서 정의되는 자기일관성 위반을 다루고 인과 개입(RQ4)·충돌 대조군(RQ3)으로 나아간다는 점에서 층위가 다릅니다.
*   **답변 굴복의 행동적 배경 (Sycophancy·Overthinking):** 사고와 답변이 어긋나며 답변이 뒤집히는 현상의 후보 기제로, 모델이 진실보다 사용자·외부 신념에 영합해 답을 바꾸는 **아부(sycophancy)** (Sharma et al., 2023)와, 사고가 길어질수록 정확도가 오히려 떨어지는 **과잉 추론(overthinking)** (*Stop Overthinking*, TMLR 2025)이 있습니다. 다만 이들은 사용자 압박·추론 길이를 다루는 반면, 본 연구는 **상충 문서 공존이라는 문맥적 압박**에서의 굴복을 겨냥한다는 점에서 다릅니다.
*   **Lanham et al. (2023), *Measuring Faithfulness in Chain-of-Thought Reasoning* (arXiv:2307.13702):** 최종 답변이 사고 과정에 실제로 의존하는지를 **개입 테스트(early answering, adding mistakes, paraphrasing, filler tokens)**로 측정하는 방법을 정립했습니다. 본 연구의 truncation 개입(early answering의 일반화)의 방법론적 근거입니다.
*   **Bogdan et al. (2025), *Thought Anchors: Which LLM Reasoning Steps Matter?* (arXiv:2506.19143) & Macar et al. (2025), *Thought Branches* (arXiv:2510.27484):** 추론 궤적 내 특정 단계의 인과적 중요도를 **on-policy resampling(선택 지점 이후 재생성)**과 **attention-suppression**으로 측정하는 방법론을 정립했습니다. 본 연구는 이 중 내부 접근이 필요 없는 resampling과 truncation을 주력 개입으로 채택합니다.

#### 2.3.2. 지표의 개념적 경계 (Metric Conceptual Boundaries)
**지표가 측정하는 개념 — 내부 연산에 대한 충실성(L-c)인가, 출력 수준 자기일관성(L-a)인가 — 의 경계를 규정하는 연구들입니다.**

*   **Turpin et al. (NeurIPS 2023), *Language Models Don't Always Say What They Think* (arXiv:2305.04388):** 프롬프트로 유도한 CoT가 내부 계산과 무관하게 정답을 정당화하는 **'사후 합리화(Unfaithful Rationalization)'**를 밝혀, 텍스트 관측만으로는 충실성을 주장할 수 없음을 보였습니다. 본 연구가 L-c(충실성)를 제외하고 L-a(자기일관성)로 스코프를 한정하는 근거입니다.
*   **Parcalabescu & Frank (ACL 2024), *On Measuring Faithfulness or Self-Consistency of Natural Language Explanations* (arXiv:2311.07466):** 기존 "충실성 테스트"들이 실제로는 내부 계산에 대한 충실성이 아니라 **출력 수준 자기일관성**을 측정함을 논증했습니다. 본 연구는 이 논지를 수용하여 AIR을 자기일관성 지표로 규정하고(§1.2), 인과 주장은 개입 실험으로 분리합니다.
*   **Arcuschin et al. (ICML 2026), *Chain-of-Thought Reasoning In The Wild Is Not Always Faithful* (arXiv:2503.08679) & Korbak et al. (2025), *Chain of Thought Monitorability: A New and Fragile Opportunity for AI Safety* (arXiv:2507.11473, 다기관 포지션 페이퍼):** 추론 모델의 CoT가 자연스러운 설정에서도 불충실할 수 있으며(Arcuschin), 따라서 트레이스를 계산의 충실한 거울이 아니라 **불완전하고 취약하지만 유용한 감시 신호(monitorable-but-fragile signal)**로 다뤄야 함을(Korbak) 논증한 연구들입니다. 본 연구는 이 관점을 채택하여 트레이스의 충실성(L-c)을 전제하지 않고, AIR을 **충돌 상황에서 이 신호가 답변과 어긋나 신뢰를 잃는 지점의 정량화**로 규정합니다(§1.2).
*   **귀속·근거성 지표와의 구분 (Attribution/Grounding; Gao et al., 2023):** LLM 답변이 인용한 근거가 그 답변을 실제로 지지하는지를 재는 귀속 지표 계열은 **답변↔근거(answer↔evidence)**의 정합을 봅니다. 반면 본 연구의 AIR·자기일관성은 **트레이스 결론↔답변(thought↔answer)**의 정합을 봅니다. 사고 채널을 보지 않는 귀속 지표는 답변이 자기가 인용한 근거에 잘 지지되더라도(근거성 만점) 그 답변이 **트레이스가 옳게 내린 결론을 버렸는지**는 원리적으로 포착하지 못하므로, 두 계열은 서로 다른 축을 측정합니다.

#### 2.3.3. 대조군 통제 설계 (Control Paradigms for Conflict Isolation)
**문항 난이도나 문서 길이 효과를 제거하여 순수 지식 충돌 효과를 분리하는 통제 실험 패러다임으로, 본 연구 RQ3(충돌 고유성) 대조군 설계의 근거입니다.**

*   **Longpre et al. (EMNLP 2021), *Entity-Based Knowledge Conflicts in QA*:** 원본(문맥이 정답 지지=비충돌)에서 개체를 치환해 충돌 문항을 파생시키는 **치환 프레임워크**로, 충돌-비충돌 대조를 통해 파라메트릭 과의존을 분리하는 통제 설계의 정본입니다.
*   **Xie et al. (ICLR 2024), *Adaptive Chameleon or Stubborn Sloth* (arXiv:2305.13300) & Wu et al. (NeurIPS 2024), *ClashEval* (arXiv:2404.10198) & Jin et al. (LREC-COLING 2024), *Tug-of-War Between Knowledge* (arXiv:2402.14409):** 증거 정합성·섭동 강도·다중 문서 상충을 통제하여 충돌 고유 행동(확증 편향, 다수결, 사전지식 과신)을 분리한 연구들로, 본 연구가 난이도가 아닌 충돌 성분을 겨냥함을 정당화합니다.

---

## 3. 연구 방법론 (Methodology)

### 3.1. 데이터셋 구성 및 충돌 유형 축 (Data Setup & Conflict Taxonomy)
본 연구는 Reasoning LLM이 외부 문맥 간 충돌을 처리하는 내부 과정을 진단하기 위해, 실제 웹 검색의 이질적 출처를 담은 **DRAGged**를 주력 벤치마크로, 대규모 상충 문항을 담은 **RAMDocs**를 보강 데이터셋으로 채택합니다. 두 데이터셋은 충돌 정의와 문서 수가 상이하므로 **풀링하지 않고 데이터셋별로 분리 보고**하는 것을 원칙으로 합니다.

#### 3.1.1. DRAGged — 실제 웹 검색 기반 다원인 충돌 데이터셋
*   **(1) 데이터셋 설명:** Cattan et al.(2025, arXiv:2506.08500)이 구축한 실제 웹 검색 환경(In-the-wild) 기반 RAG 충돌 벤치마크입니다. 인위적 단일 토큰 조작 합성 데이터와 달리 실제 구글 검색(Top-10) 웹 문서를 담고 있어 출처·날짜·문맥이 이질적이며, Reasoning LLM이 실제 출처 비교와 속성 대조(Recency, Authority)를 수행하는 풍부한 `<think>` 트레이스를 유도하므로 본 진단에 적합합니다.
*   **(2) 데이터셋 충돌 유형과 건수:** 전체 458문항. 아래 분포는 원본 데이터 파일에서 직접 추출한 값입니다.

    | 충돌 유형 (Conflict Taxonomy) | 문항 수 | 특성 및 내용 |
    | :--- | :---: | :--- |
    | **시간적 충돌 (Freshness/Outdated)** | 62건 | 시간 흐름에 따라 과거 지식과 최신 사실이 상충 (예: 2024 vs 2025 일정) |
    | **오정보 충돌 (Misinformation)** | 5건 | 신뢰할 수 없는 출처의 오정보와 공식 출처의 참 정보가 상충 |
    | **상충되는 의견 (Conflicting Opinions)** | 115건 | 정답이 하나로 정해지지 않고 주장이 엇갈리는 관점 충돌 |
    | **상보적 정보 (Complementary Information)** | 115건 | 충돌하지 않고 상보적 디테일을 제공하는 대조군 |
    | **비충돌 (No Conflict)** | 161건 | 모든 문서가 일관된 정보를 가리키는 일반 RAG 대조군 |

    > **※ 데이터 재집계와 진단 범위 (Cattan et al., 2025 Table 2 기준):** 위 표 수치는 원 논문 Table 2의 범주별 건수(161/115/115/62/5)와 정확히 일치하는 자체 재집계입니다. 원 논문은 458건 중 297건(64.8%)을 "conflicting"으로 집계하나, 이는 호환되나 종합이 필요한 상보적 정보(115건)를 포함한 넓은 범주입니다(182 + 115 = 297). 본 연구가 겨냥하는 충돌의 규모는 다음 **세 단계로 좁혀집니다**(점점 작은 부분집합): (i) 종합 필요 넓은 범주 **297건(64.8%)** → (ii) 그중 진짜 상충인 **명시적 상충 182건(39.7%: 의견 115 + 시간 62 + 오정보 5)** → (iii) 그중 정답 채점이 가능해 정확도·AIR을 직접 측정할 수 있는 **사실 기반 충돌 67건**(시간 62 + 오정보 5). 나머지 의견 충돌(115건)은 정답이 하나로 정해지지 않으므로 **행동(정확도) 지표에서는 제외**하되, 정답이 필요 없는 **자기일관성(트레이스 입장↔답변)·인지·대조 패턴 분석에는 사용**합니다(§3.2 이중 트랙). 순수 충돌 효과(RQ3)도 이 이중 트랙을 따릅니다: **행동(정확도·AIR) 비교는 채점 가능한 사실 충돌 67건 vs 비충돌 대조군**(DRAGged 내부는 문항 간 대조라 회귀 보정, 매칭 대조는 RAMDocs가 담당; §3.3.3)으로, **자기일관성·인지 패턴 비교는 명시적 상충 182건 vs 비상충 276건(상보 115 + 비충돌 161)**으로 수행합니다. 사실 충돌 표본(67)이 얇은 점은 정량 채점이 명확한 RAMDocs에서 **별도로 재현 확인**합니다(풀링이 아니라 데이터셋별 분리 보고; §3.1.2).
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
    *   **정답 문서 자동 식별 파이프라인 (전수 인간 검증):** DRAGged는 개별 문서에 정답 라벨이 없으므로, 문항의 `correct_answer` 텍스트가 10개 문서 중 어느 `text`에 포함되는지를 문자열 일치 + NLI 기반으로 사전 매핑합니다. 다만 이 매핑은 Level 2 라벨 전체의 기반이므로, **행동(AIR·정확도) 주장이 걸린 사실 충돌 67건은 자동 매핑을 1차 초안으로만 쓰고 전수 인간 검증·교정한 골드 매핑을 사용**합니다(N이 작아 전수 검증이 가능). 어느 문서에도 매칭되지 않거나 복수 문서에 매칭되어 정답 문서가 모호한 문항은 플래그하여 Level-2 의존 지표에서 제외하고 그 건수를 보고합니다. (비충돌 161건은 문서가 일치해 매핑이 자명하고, RAMDocs는 `type`·`gold_answers`가 원본에 라벨돼 있어 이 매핑이 불필요합니다.)
    *   **문서 셔플링 및 표준 렌더링:** 위치 편향(Position Bias; Liu et al., 2023)을 통제하기 위해 매 실험마다 10개 문서 순서를 무작위 셔플링하고 `[Document 1] ~ [Document 10]` 포맷으로 주입합니다.
    *   **속성 메타데이터 활성화:** 2단계(판정)에서 최신성·권위 대조를 추적하기 위해 원본 `date`·`url`을 문서 헤더에 명시적으로 유지합니다.

#### 3.1.2. RAMDocs — 대규모 다중 문서 모호·오정보 벤치마크
*   **(1) 데이터셋 설명 및 출처:** Wang et al.(COLM 2025)이 제안한 다중 문서 QA 벤치마크(`HanNight/RAMDocs`)의 Test set 500문항입니다. 위키피디아 및 오픈 도메인 QA(NQ, TriviaQA 등)의 정답 문서에 통제된 오정보(Misinformation)와 노이즈를 인위적으로 섞어 넣어 구축되었습니다. DRAGged의 사실 충돌 표본(67건)이 작으므로, 정답/오답이 명확히 채점 가능한 RAMDocs에서 주요 관측을 **별도로 재현 확인**합니다(두 데이터셋을 합산·풀링하지 않고 분리 보고). 다만 RAMDocs가 보강하는 것은 **오정보형 충돌**이며, 시간적 충돌은 DRAGged(62건)에 한정됨을 명시합니다.
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
    *   **트레이스의 문서 지지 추적:** RAMDocs는 문서별 `type`(correct/misinfo/noise)과 `gold_answers`/`wrong_answers`가 **원본에 라벨돼 있어 정답 문서 식별이 불필요**합니다(DRAGged와 달리 매핑을 유도하지 않음). 따라서 파이프라인은 gold를 그대로 받아, 모델이 `<think>` 내에서 인용·비교할 때 **정답 문서와 오답 문서 중 어느 쪽을 지지했는지**를 판정·추적하는 역할만 합니다.
    *   **다중 정답 채점(any-gold):** RAMDocs는 문항당 평균 **2.2개의 유효 정답**을 포함하므로, Level 2·Final Action의 `correct`는 **정답 집합 중 어느 하나(any-gold)를 지지·표출한 경우**로 정의합니다.
    *   **문서 셔플링 및 표준 렌더링:** DRAGged와 동일하게 위치 편향을 통제하기 위해 문서 순서를 무작위 셔플링하고 표준 포맷(`[Document 1] ~ [Document N]`)으로 주입합니다.

### 3.2. 다단계 진단 프로토콜 (Multi-stage Diagnostic Protocol)
본 연구는 Reasoning LLM이 `<think>`에서 충돌을 처리하고 답변으로 전환하는 과정을 해부하기 위해 **3단계 궤적 라벨링**과 **전환 행렬 지표**를 적용합니다. 모든 지표는 §1.2에서 구분한 (L-a) 자기일관성 — 즉 사고와 답변이 **내용상 일치하는가**라는, **개입 없이 텍스트만으로 판정되는** 속성 — 을 측정하며, **사고가 답변을 실제로 유발했는가(인과)는 텍스트만으로는 알 수 없어** §3.3 실험 3(개입)로 분리합니다. 지표의 출처를 명확히 하면, 본 절의 단계별 오류·특이 경로 지표(AIR·Shortcut·Discordant Hit·Blind-Hit·Loss_L1/L2)를 하나의 전환 행렬로 묶어 **충돌×트레이스 처리에 대해 정식화한 것이 본 연구의 신규 기여**입니다(단계별 오류 분해라는 일반 아이디어가 아니라, 이 정식화와 경로 귀속이 신규입니다). 자기일관성 개념의 계보는 Parcalabescu & Frank(2024)에, 궤적-답변 해리 현상의 관찰은 Li et al.(2026)에 두며, 부록 B의 궤적 분석 기법들은 기존 방법론의 차용·각색으로서 각 기법에 출처를 명시합니다.

#### 3.2.1. 3단계 궤적 라벨링 (3-Stage Trajectory Labeling)
각 실험 턴마다 `<think>`와 최종 답변을 파싱하여, 모델의 충돌 처리를 **① 탐지 → ② 해소 → ③ 표출** 세 단계로 라벨링하고 **단계별로 오류를 분해**합니다(stage-wise error decomposition). 이 순서는 이상적 규범 경로일 뿐 강제되지 않으며, 순서를 벗어나 정답에 이르는 경우는 특이 경로 지표(Shortcut·Blind-Hit)로 따로 기록합니다.

| 단계 | 진단 질문 | 라벨 |
| :--- | :--- | :--- |
| **① Level 1 — 탐지 (Detection)** | 문서 간 불일치를 감지했는가? | {detected, unrecognized} |
| **② Level 2 — 해소 (Resolution)** | 여러 문서 중 정답 지지 문서를 유효하다고 결론짓는 올바른 추론을 수행했는가? | {correct, wrong, unresolved} |
| **③ Final Action — 표출 (Expression)** | `<think>` 종료 후 최종 텍스트에서 정답을 정확히 표출했는가(Exact Match)? | {correct, wrong, abstain} |

*   **동치 채점:** Final Action의 `correct`(EM)는 표현 차이가 불일치로 오인되지 않도록 **답변 정규화(별칭·수치 형식 통일)와 동치 판정**을 거쳐 판정합니다(표기 노이즈가 AIR로 새는 것을 방지).
*   **기권 분리:** 확정 답을 내지 않은 **`abstain`은 `wrong`과 구분하여 AIR·오답률 분모에서 제외하고 별도 기권율로 보고**합니다(기권을 오답으로 밀어넣어 AIR이 부풀려지는 것을 방지). 이는 **두 데이터셋 공통 라벨링 규칙**이며, 기권이 드문 DRAGged보다 모호·오정보를 다루는 RAMDocs에서 빈도가 높습니다. 따라서 AIR·특이 경로 지표는 확정 답변을 낸 응답 위에서 정의됩니다.

*   **(부가 축) 유형 인지(Typological Recognition):** Level 1에서 충돌을 감지한 뒤 그 **원인 유형**(시간적/오정보/관점)까지 짚었는가? {`correct_type`, `surface_only`}. 탐지의 **깊이 축**으로 별도 하위 지표(유형 인지율)로 보고합니다. 유형에 따라 Level 2 해소에서 **비교하는 속성**이 달라집니다 — 시간적 충돌이면 작성 날짜의 최신성(`recency`)으로, 오정보 충돌이면 출처의 권위(`authority`)로 유효한 문서를 가립니다. (관점/의견 충돌은 단일 정답이 없어 Level 2 정답 판정 대상이 아니며, 탐지·대조 패턴 분석에만 쓰입니다; §3.2 이중 트랙.)

*   **라벨 신뢰도 (요약):** 라벨링은 규칙 기반(문서 인덱스·날짜/URL 키워드) + **LLM-as-a-Judge** 하이브리드로 수행하며, 신뢰도는 **판정자 편향 통제**(대상 모델과 다른 계열의 판정자 사용)와 **인간 2인 검토 κ**(무작위 200건)로 확보합니다. 특히 가장 주관적인 **Level 2(올바른 해소) 라벨은 판정자·인간 κ를 별도 보고**하고, **κ가 임계(예: 0.6) 미만이거나 두 판정자가 불일치하는 문항은 인간 전문가가 판정(adjudication)**하여 최종 라벨을 확정합니다. 다만 부재 조건 지표(Loss_L1·Shortcut·Blind-Hit)는 라벨이 '언어화된' 상태라 **'알고도 안 적음(침묵)'과 '진짜 못 함(무능)'이 섞일 수 있으므로**, 후속 유도 프로브로 두 원인을 분리하고 Loss_L1을 '**언어화된** 인지 실패율'로 명명합니다. **핵심 지표 AIR은 존재 조건(L2 = correct) 위에서 정의되어 이 갭에 강건**합니다. (세부 프로토콜은 부록 A.)

#### 3.2.2. 진단 지표 체계: 단계별 오류 분해 및 전체 전환 행렬
아래 지표들(정상 경로를 제외한 6개 진단 지표: Loss_L1·Loss_L2·AIR·Shortcut·Discordant Hit·Blind-Hit)은 6개의 독립 측정이 아니라, 한 번의 3단계 라벨링이 만드는 **하나의 전환 행렬에서 뽑은 셀**들입니다 — 각각 다른 진단 질문에 답하며, 결합하면 임의의 결과가 **'어디서 왔는지'를 귀속**할 수 있습니다. 각 지표가 **'왜·어디서'** 그렇게 됐는지는 부록 B(궤적 분석층)에서 상관적으로 들여다봅니다.

**(1) 3대 단계별 전환 손실 지표 (Stage-wise Error Decomposition):** 각 지표는 **직전 단계를 통과했다는 조건 아래 그 단계에서 실패할 확률**입니다(통과 상태: Level 1 = detected, Level 2 = correct). 조건은 항상 통과 상태를 `=`로 표기하며, 실패 사건은 단일 라벨이면 `=`, Level 2처럼 복수(wrong·unresolved)이면 `≠ correct`로 씁니다. 조건부 지표는 모델 간 분모(예: Level 2 = correct 도달률)가 다르면 서로 다른 부분모집단 위의 비율이 되어 비교가 오도될 수 있으므로, **결합확률(예: P(Level2 = correct ∧ FinalAction = wrong))과 분모 N을 함께 보고**합니다.

| 지표 | 수식 | 해석 |
| :--- | :--- | :--- |
| **Loss_L1 (인지 실패율)** | P(Level1 = unrecognized) | 충돌을 **아예 못 본** 인지 실패 |
| **Loss_L2 (판정 오류율)** | P(Level2 ≠ correct \| Level1 = detected) | 보고도 **잘못 판단한** 판정 실패 |
| **AIR (추론-답변 불일치율, = Loss_FA)** | P(FinalAction = wrong \| Level1 = detected, Level2 = correct) | 올바른 해소에 도달하고도 답변에서 유실하는 **'사고→답변 유실'** — 본 연구의 **핵심 관측** |

**측정 기준점 (트레이스↔답변):** 위 지표들이 재는 '불일치'는 모두 **트레이스가 내린 결론과 최종 답변 사이**(thought↔answer)의 정합이며, 답변이 인용한 근거와 답변 사이(answer↔evidence)를 보는 귀속·근거성 지표(§2.3.2)와는 기준점이 다릅니다. 이 구분에는 실용적 함의가 있습니다 — 트레이스↔답변 정합은 **정답의 존재를 요구하지 않으므로**, 단일 정답이 없는 의견·상보 충돌에서도 "트레이스가 택한 입장을 답변이 뒤집는가"라는 **자기일관성은 측정할 수 있습니다**(다만 그 입장이 '옳았는가'는 판정 불가). 따라서 본 연구는 **자기일관성 계열 관측은 명시적 상충 전체(182건)**로 넓히되, **정답이 결부된 AIR(Level 2 = correct 조건)은 채점 가능한 사실 충돌(67건)**에 한정해 보고하는 **이중 트랙**을 취합니다.

**(2) 비순차적 특이 경로 지표 (Non-sequential Pathways):** 모델이 **정답을 맞혔다면, 그 정답은 아래 네 경로(정상·Shortcut·Discordant Hit·Blind-Hit) 중 딱 하나를 거쳐** 나온 것입니다 — 네 경로가 서로 겹치지 않으면서 정답이 나오는 모든 경우를 빠짐없이 덮기 때문입니다(상호배타·전수 분해). 이들을 갈라주는 기준이 바로 Level 1·Level 2 라벨입니다. 이렇게 정답을 경로별로 쪼개 두면, 나중에 완화 기법이 정답률(EM)을 올렸을 때 **늘어난 정답이 어느 경로에서 왔는지** — 진짜 추론 덕인지, 아니면 도약·우연 덕인지 — 를 가려낼 수 있습니다(RQ2).

| 경로 (지표) | Level 1 | Level 2 | 수식 | 해석 |
| :--- | :---: | :---: | :--- | :--- |
| **정상 경로 (Legitimate)** | detected | correct | P(FA = correct \| L1 = detected, L2 = correct) | 인지·해소를 제대로 거쳐 정답 표출 |
| **Shortcut (암묵적 도약)** | detected | unresolved | P(FA = correct \| L1 = detected, L2 = unresolved) | 명시적 해소 없이 도약한 정답 — 근거가 약한 취약한 정답 |
| **Discordant Hit (불협 적중)** | detected | wrong | P(FA = correct \| L1 = detected, L2 = wrong) | 오답 문서를 지지해 놓고 최종 정답 표출 (AIR과 방향이 반대) |
| **Blind-Hit (맹목 적중)** | unrecognized | — | P(FA = correct \| L1 = unrecognized) | 충돌 인지조차 없이 정답 도달 (파라메트릭·우연) |

### 3.3. 본 실험 설계 (Main Experiments)
본 연구는 아래 공통 설정 위에서, RQ1~4를 검증하는 **3대 실험**을 수행합니다: 전환 경로 진단(실험 1) → 귀속 통제(실험 2) → 인과 개입(실험 3).

#### 3.3.1. 실험 공통 설정: 대상 모델 및 생성·심사 프로토콜
*(1) 대상 모델:* 문헌에서 검증된 **추론 모델**을 3개 계열에서 엄선합니다. SFT 증류(Distillation)만 거친 모델은 자율 탐색 능력이 보존되지 않고 장식적 출력을 낼 위험이 있어 배제하고, 실제 탐색 궤적을 자발적으로 전개하는 20~32B 규모에 집중합니다.

| 모델 | 계열 | 출시 | 구조 및 규모 | 사고 트레이스 | 선정 근거 및 역할 |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Qwen3.6-27B** | Qwen | 2026-04 | 27B Dense | `<think>` | **주력 분석 모델**. 추론·코딩 겸비 최신 밀집형 추론 모델 |
| **Olmo-3.1-32B-Think** | AllenAI | 2025-12 | 32B Dense | `<think>` | **완전 개방형 대조군**. 가중치·데이터·학습 로그 전체 공개 |
| **gpt-oss-20b** | OpenAI | 2025-08 | MoE (총 21B / 활성 3.6B) | Harmony¹ | **교차 계열·MoE 검증**. OpenAI 개방 MoE 추론 모델 |

> ¹ `gpt-oss-20b`는 `<think>` 대신 Harmony 포맷의 Analysis 채널로 사고를 출력하므로 전용 파서로 동일한 3단계 프로토콜로 라벨링합니다. 단, 파싱 규약이 다른 점은 계열 간 비교 시 교란 변인으로 명시하고, 모델별 결과를 분리 보고합니다.

*(2) 생성·심사:* 모든 모델은 비양자화(bf16)로 구동해 로짓·궤적 왜곡을 방지하고, 권장 디코딩(예: `temp=0.6, top_p=0.95`)으로 무작위 시드를 반복(사실 충돌 표본이 얇아 검정력 확보를 위해 최소 5회)한 뒤 Paired Bootstrap 95% CI로 유의성을 검정합니다. 라벨링은 규칙 기반(문서 인덱스, 날짜/URL 키워드) + 고성능 LLM-as-a-Judge 하이브리드로 수행하되, 라벨 타당성 검증(판정자 편향 통제·인간 κ·언어화 갭 보강; 상세는 부록 A)을 필수로 병행합니다.

#### 3.3.2. 실험 1 (RQ1·RQ2): 전환 경로 진단 및 완화 기법 해부
*   **목적:** 모델이 어느 전환 단계에 병목(Loss_L1, Loss_L2, AIR)과 특이 경로(Shortcut, Discordant Hit)를 겪는지 규명하고, 완화 기법 적용 시 전환 행렬이 어떻게 변하는지 대조합니다.
*   **평가 대상 환경 (Baselines & Interventions):** 서론(§1.1)·관련 연구(§2.2)에서 정리한 완화 기법 중 **단일 추론 모델에 이식 가능한 경량 기법(디코딩·프롬프트)**을 대상으로 하며, 이들 다수가 본래 비-thinking instruct 모델에서 개발·검증되었으므로 **사고 모델로 이식(transplant)했을 때의 거동을 해부**하는 것임을 명시합니다. 여기서 얻는 완화 기법별 전환 행렬 변화가 RQ2(완화 기법의 EM 상승분이 진짜 추론에서 오는가)의 핵심 근거입니다.

| # | 평가 환경 | 분류 | 내용 | 예상 주 영향 지점 |
| :---: | :--- | :--- | :--- | :---: |
| ① | **Standard RAG (Zero-shot)** | 대조군 | 기본 검색 문서 주입 | 기저 성능 |
| ② | **CAD** (Shi et al., 2024, +AdaCAD) | 디코딩(아키텍처 무관) | 문맥 유무 로짓 차이를 대비 증폭해 생성을 제공 문맥 쪽으로 미는 기법. **(i) 사고+답변 전 구간 vs (ii) `</think>` 이후 답변 구간 한정** 적용을 절제(ablation)로 비교 | Loss_L1/L2 |
| ③ | **CD2** (Jin et al., 2024) | 디코딩(아키텍처 무관) | 상충 문서 지식을 분리해 대비 디코딩 — CAD의 '문맥 vs 파라메트릭'과 달리 **문서 간(inter-context) 충돌 자체**를 겨냥해 본 진단과 더 직접 맞물림 | Loss_L2 |
| ④ | **Recency/Authority-Guided Prompting** | 프롬프트(전략 이식) | 충돌 시 작성 날짜(`date`) 최신성·출처(`url`) 권위를 우선 대조하도록 유도하는 프롬프트 | Loss_L2 |
| ⑤ | **Reflection-style Prompting** (Self-RAG 반추 전략 차용; Asai et al., 2024) | 프롬프트(전략 이식) | Self-RAG 원본은 학습된 별도 모델이라 직접 이식 불가 → 반추(reflection) 전략만 프롬프트로 차용해 결론과 문서의 사실 부합·충돌 해소를 스스로 재검토 | AIR |

*   **스코프 밖(명시):** ConflictRAG(외부 MLP 분류기 + Entropy-TOPSIS 선별)·MADAM-RAG(다중 에이전트 토론)는 단일 모델을 외부 시스템으로 대체하므로, "단일 추론 모델이 충돌을 내부에서 어떻게 처리하는가"라는 본 연구 목표와 층위가 달라 baseline에서 제외하고 향후 과제로 둡니다.
*   **절차:** 위 **5개 환경**에서 DRAGged·RAMDocs 문항(순서 셔플)을 주입해 사고·답변을 생성하고, 규칙 기반(문서 인덱스·날짜/URL 키워드) + LLM-as-a-Judge 하이브리드로 3단계를 마킹하여 **환경별 전체 전환 행렬(Full Transition Matrix)**을 도출합니다. 다만 환경 간 행렬을 **집계 수준에서 빼는 것(net 대조)에 그치지 않고**, 아래 문항별 흐름 분석으로 나아갑니다.
*   **문항별 흐름 분석 (Item-level Flow, RQ2의 핵심):** baseline과 각 완화 환경은 **동일 문항**에 대해 수행되므로, `question_id`로 짝지어 **완화 전 경로 → 완화 후 경로**의 이동을 문항 단위로 추적합니다(before→after 흐름 행렬). 집계 대조로는 상쇄되어 보이지 않는 **개선·퇴행의 동시 발생**을 드러내는 것이 목적이며, 다음 두 지표를 산출합니다.
    *   **정당 이득 비율(Legitimate Gain Ratio, LGR):** 완화로 새로 정답이 된 문항 중 **정상 경로(Legitimate)로 도달한 비율**. 낮을수록 EM 상승이 취약 경로(Shortcut·Discordant Hit·Blind-Hit)에 기인함을 뜻합니다. 기법·모델별로 보고해 "정확도 상승이 곧 대조 추론 개선은 아님"을 정량화합니다.
    *   **숨은 퇴행율(Hidden Regression):** 완화 전 **정상 경로 정답이던 문항이 완화 후 AIR·오답으로 이동한 비율**. 순 EM 상승 뒤에 가려진 파괴를 드러내어 "정확도는 불완전할 뿐 아니라 *가리는* 신호"임을 보입니다.
    *   흐름 지표는 문항별 다수결 경로(시드 ≥5) 위에서 정의하고, 문항·시드 부트스트랩으로 CI를 병기합니다.

#### 3.3.3. 실험 2 (RQ3): 귀속 통제 — 충돌 고유성 & 사고 채널 귀속
관측된 AIR·특이 경로가 **진짜 현상인지, 아니면 부수 효과인지**를 두 축으로 통제합니다: (a) **충돌에 고유한가**(난이도·문서 수의 부산물이 아닌가), (b) **사고 채널에 고유한가**(기저 모델에서 물려받은 것이 아닌가).

**(a) 충돌 고유성 대조** (Longpre et al., 2021; Xie et al., 2024의 통제 패러다임):

| 대조 | 충돌 조건 | 비충돌(대조) 조건 | 통제·추정 |
| :--- | :--- | :--- | :--- |
| **DRAGged 내부** | 사실 기반 충돌군(시간적 62 + 오정보 5 = 67) | 상보적 115 + 비충돌 161 | AIR·특이 경로율의 차이(Δ)로 충돌 성분 분리. 의견 충돌은 채점 불가라 Level 1 인지·대조 패턴 비교에만 사용. 표본 작은 점(67)은 RAMDocs로 보강 |
| **RAMDocs 문서 구성** | 오정보 문서 포함 | 노이즈만 포함(동일 문서 수) | 문서 개수를 통제한 채 충돌 유무의 순효과 추정 |

두 대조는 통제 강도가 다릅니다: **RAMDocs는 같은 문항에서 misinfo↔noise만 바꾼 매칭(within-item) 대조**로 충돌의 순효과를 직접 격리하는 반면, **DRAGged는 서로 다른 문항 간(between-item) 대조**라 충돌 성분이 질의 차이와 교락될 수 있어 문서 수·질의 유형을 공변량으로 하는 혼합효과 로지스틱 회귀로 보정합니다("충돌" 항의 유의성 검정, Paired Bootstrap 95% CI 병행). 따라서 **매칭 통제의 무게는 RAMDocs가, 생태적 타당성(실제 웹 충돌)은 DRAGged가** 담당하며 두 근거를 교차 확인합니다. 다만 정확도 결부 비교의 표본이 얇으므로(사실 충돌 67), 정확도 기반 결론은 효과 크기에 민감하고 **자기일관성 축(명시적 상충 182; §3.2 이중 트랙)으로 검정력을 보완**합니다.

**(b) 사고 채널 귀속 — Thinking vs Non-thinking (Regime Control):** "관측된 취약성이 사고 채널 고유의 것인가, 아니면 기저 모델에서 물려받은 것인가"를 통제합니다. AIR·Shortcut은 분리된 사고 결론을 전제하므로 비-thinking 모델에서는 정의되지 않지만, **충돌 해소 정확도(EM)와 완화 기법의 EM 이득**은 두 레짐에서 공통으로 비교 가능합니다.
*   **동일 가중치 토글(주력):** 가중치를 고정한 채 사고 채널만 켜고 끕니다(예: 하이브리드 추론 모델의 thinking on/off 플래그, `gpt-oss`의 reasoning effort 레벨 — 이진 토글이 아니라 최저·최고 레벨로 근사). 아키텍처·데이터를 완전히 통제한 가장 깨끗한 대조입니다. 그 극단으로 `<think>` 전체를 마스킹하고 답만 생성했을 때 답 분포가 달라지지 않으면 사고 채널이 답과 무관한 장식이라는 뜻이므로, 이 **전면 마스킹을 사고-답변 진단 성립 여부의 go/no-go**로 함께 확인합니다.
*   **Matched 비-thinking 형제(보강):** 가능한 경우 동일 베이스의 instruct(비-thinking) 형제 모델을 병행하여 토글 미지원 계열을 보완합니다.
*   **귀속 판정:** 사고 채널 도입이 (i) 충돌 해소 정확도를 순증시키는지, (ii) 그럼에도 비-thinking에 없던 새로운 실패 양식(AIR·Shortcut 등 사고-답변 괴리)을 발생시키는지를 대조하여, 관측 현상을 사고 채널에 귀속합니다. 이 설계는 동시기 연구의 think/no-think 매칭 통제(Li et al., 2026)와 정합적입니다.

#### 3.3.4. 실험 3 (RQ4): 인과 개입 — 사고→답변의 인과적 사용 규명
텍스트 상관을 넘어 **사고 결론이 최종 답변을 실제로 구동하는지(L-b)**를 개입으로 검증합니다. 두 개입 모두 **모델 내부 접근 없이 생성 조작만으로 구현**되며, 서로 다른 방식(절단 vs 재생성)으로 같은 결론을 삼각 검증합니다.

*   **경로별 취약성 검증 (RQ2 연계):** 위 개입을 AIR 샘플뿐 아니라 **정답을 낸 각 경로(정상·Shortcut·Blind-Hit) 표본에 동일하게 적용**해, 개입 후 정답이 뒤집히는 **flip률**을 경로별로 비교합니다. 취약 경로(Shortcut·Blind-Hit) 정답이 정상 경로 정답보다 유의하게 높은 flip률을 보이면, "취약 경로"가 라벨상의 분류가 아니라 **섭동에 실제로 무너지는 인과적 성질**임이 입증됩니다 — 이로써 RQ2의 "완화가 올린 정답의 상당분이 유리처럼 깨지는 정답"이라는 주장이 인과적으로 뒷받침됩니다.

| 개입 | 우선순위 | 무엇을 하나 | 근거·비고 |
| :--- | :---: | :--- | :--- |
| **Truncation / Early-Answering** | 1차 (주력) | `</think>` 직전 등 여러 절단 지점에서 사고를 끊고 즉시 답변을 강제 생성 → 어느 지점부터 답변이 정답으로 고정되는지(사고 후반부의 인과 기여) 측정 | Lanham et al., 2023. 입력 문자열 조작만으로 구현. 실험 2(b)의 전면 마스킹(go/no-go)은 이 개입의 가장 거친 특수 사례이며 본 실험이 절단 지점·표본 수준으로 정밀화 |
| **On-policy Resampling** | 1차 (주력) | 해소 판정 문장(Level 2) 직후에서 이후 궤적을 K회 재생성 → 그 판정이 최종 답변 분포를 얼마나 좌우하는지 인과 기여도 추정. AIR 샘플서 낮으면 "사고 결론이 답변에 반영 안 됨" 실증 | Bogdan et al., 2025; Macar et al., 2025. 프리픽스 고정 샘플링으로 구현. **재생성 지점(L2 문장) 국소화가 판정자 의존이므로, 국소화 잡음을 완충하도록 인접 문장 다지점에서 함께 재생성**. 인과 매개 배경: Paul et al., 2024/FRODO (Findings of EMNLP 2024) |

---

## 4. 실험 결과 (Experimental Results)

> *(실험 수행 완료 후 실제 결과로 작성한다. 현재는 무게중심과 보고 계층, 실험별 예상 결과/가설을 기재하며, 실측치가 나오면 대체한다.)*

**결과의 무게중심 (1차 주력):** 충돌 하에서 정답률(EM)만으로는 추론의 질을 알 수 없다 — 본 연구의 전환 행렬은 정답을 **4경로(정상·Shortcut·Discordant Hit·Blind-Hit)로 분해**하고, 완화 전후를 **문항별로 짝지어 경로 이동을 추적(흐름 분석)**하여 다음 세 지표로 회계한다:
> *   **LGR(정당 이득 비율):** 완화로 새로 정답이 된 문항 중 정상 경로 비율 — **낮으면** EM 상승이 취약 경로(Shortcut·Discordant Hit·Blind-Hit)에서 온 것.
> *   **숨은 퇴행율:** 완화 전 정상 경로 정답이 완화 후 **AIR·오답으로 이동**한 비율 — 순 EM 상승이 가린 파괴.
> *   **경로별 flip률:** 각 경로 정답을 섭동했을 때 뒤집히는 비율 — **취약 경로 > 정상 경로**이면 "취약"이 인과적으로 입증됨.

이 회계 능력이 본 논문의 핵심 결과이며, 회계의 결론이 어느 쪽이든(이득이 진짜든 취약하든) 기여는 성립한다 — 특히 **숨은 퇴행율**과 **경로별 flip률**은 완화가 잘 작동하더라도 독립적으로 관측되므로, 발견이 밋밋해질 위험을 구조적으로 낮춘다. **AIR**은 그 안에서 '올바른 해소가 답변에서 유실되는' 지점을 짚는 서명 지표다.

**보고 계층:** 과적재를 피하기 위해 결과를 세 계층으로 위계화하여 제시한다.
*   **① 주력 — 완화 회계 (RQ1·RQ2):** 모델·환경별 **전체 전환 행렬**과, 완화 기법별 **EM 상승분의 경로 귀속**(정상 vs 취약). 이 절이 논문의 중심이다.
*   **② 검증 — 현상의 실재성 (RQ3·RQ4):** 관측이 난이도의 부산물이 아니라 **충돌 고유**임을(대조군 Δ), 그리고 사고 결론이 답변을 **인과적으로 구동/유실**함을(개입) 확인해 주력 결과의 타당성을 뒷받침한다.
*   **③ 보강 — 강건성·기제 (레짐 통제·유형 인지·부록 B):** thinking on/off 귀속, 유형별 인지 깊이, 궤적 상관 분석으로 견고성과 기저 신호를 보인다(부차적).

**예상 결과 (가설):**
*   **① 주력 (RQ1·RQ2):** 추론 모델은 높은 비율로 충돌을 인지·해소(L2 correct)하면서도 최종 답변에서 유실하는 AIR이 유의하게 존재하고, 초기 문서 고착(Shortcut) 등 취약 경로가 관찰될 것이다. **핵심적으로, 완화 기법이 올린 EM의 상당분이 정상 경로가 아니라 Shortcut·Discordant Hit·Blind-Hit에서 오는 경우가 있어, 표면 정답률 상승이 곧 대조 추론 개선은 아님을 회계로 드러낼 것이다.**
*   **② 검증 (RQ3·RQ4):** AIR·특이 경로율은 비충돌/상보 대조군 대비 충돌군에서 유의하게 높아 **충돌 고유 성분**이 존재하고(RQ3), AIR 샘플에서 해소 판정 문장의 on-policy 인과 기여도가 비-AIR 대비 낮아 **사고 결론이 답변에 인과적으로 반영되지 않는 진짜 유실**임이 실증될 것이다(RQ4, 단순 표기 오류 아님).
*   **③ 보강:** 사고 채널을 끄면(비-thinking) AIR·Shortcut이 정의되지 않거나 사라져 관측 취약성이 사고 채널에 귀속되고(레짐 통제), Semantic Shift·엔트로피 스파이크 등 궤적 신호가 AIR 지점과 상관될 것이다(부록 B).

---

## 5. 한계 (Limitations)
*   **채점 스코프(이중 트랙):** 정답이 결부된 **정확도·AIR 평가는 사실 기반 충돌(67건)**에 국한됩니다. 다만 단일 정답이 없는 의견·상보 충돌에서도 **트레이스 입장↔답변의 자기일관성은 측정 가능**하므로, 자기일관성 계열 관측은 명시적 상충 전체(182건)로 넓혀 보고합니다(§3.2 이중 트랙).
*   **검정력:** 사실 충돌 표본(67)이 얇아 **정확도 기반 결론은 효과 크기에 민감**합니다. 시드 반복(≥5)·Paired Bootstrap CI로 완화하고, 검정력이 상대적으로 높은 **자기일관성 축(182)으로 보완**하되, 작은 효과의 유의성 확보에는 한계가 있음을 명시합니다.
*   **대조군 매칭:** 충돌 고유성(RQ3)의 **DRAGged 대조는 서로 다른 문항 간(between-item) 비교라 회귀 보정에 의존**하며 잔여 교락 가능성이 있습니다. 같은 문항에서 misinfo↔noise만 바꾼 **매칭 대조는 RAMDocs가 담당**하여 두 근거를 교차 확인합니다.
*   **시간 충돌 표본:** 사실 충돌 중 시간적 충돌은 **DRAGged 62건에 한정**되며, RAMDocs는 오정보형만 보강하므로 시간 충돌 표본은 넓히지 못합니다.
*   원인별 통계는 사실충돌·의견충돌 두 묶음 수준까지만 주장합니다. DRAGged 단독 오정보(5건)는 표본이 작아 정성 사례로만 제시하고, 오정보의 정량 근거는 RAMDocs로 보강합니다.
*   **라벨 타당성:** Level 2 라벨은 텍스트로부터의 추론이므로, §3.2의 판정자 편향 통제·인간 κ로 판정 신뢰도를 확보하되, 잔여 불확실성을 명시합니다.
*   **언어화 갭:** 모든 라벨은 "언어화된" 인지·해소이므로, 부재 조건 지표(Loss_L1·Shortcut·Blind-Hit)에는 "속으로 했으나 안 적음"이 섞일 수 있습니다. §3.2의 언어화 갭 보강(유도 프로브·암묵적 기울기 백스톱·다중 시드)으로 침묵과 무능을 분리하되, 완전 제거는 불가함을 명시합니다. 또한 모델별 언어화 습관(gpt-oss Harmony vs Qwen `<think>`) 차이가 Loss_L1에 섞이는 교란 변인이므로 모델별 분리 보고합니다. **핵심 지표 AIR은 존재 조건(L2 = correct) 위에서 정의되어 이 갭에 강건합니다.**
*   **인과 개입의 범위:** 인과 개입은 **모델 내부 접근 없이 생성 조작만으로 구현되는 truncation·resampling** 두 방법으로 한정합니다(어텐션 마스킹·activation steering 등 내부 개입은 본 연구 범위 밖).
*   **검증 도메인:** 본 연구는 개방형 웹 검색(DRAGged)과 합성 다중 문서(RAMDocs)에서 검증하며, §1.1에서 동기로 든 기업 내부 지식베이스의 버전·시간 충돌은 직접 다루지 않습니다. 다만 제안하는 3단계 전환 행렬·경로별 분해는 **문서 출처·색인 환경에 독립적인 도메인 불문 진단 도구**이므로, 기업 KB(WikiContradict·EnterpriseRAG-Bench류)로의 이식은 동일 프로토콜로 가능하며 향후 과제로 둡니다.
*   추론 모델 3계열(Qwen3.6-27B / Olmo-3.1-32B-Think / gpt-oss-20b)을 중심으로 분석하며, 증류 모델과 내부 trace 접근이 불가한 폐쇄형 상용 모델은 제외됩니다. gpt-oss의 Harmony 파싱 상이는 계열 간 비교의 교란 변인으로 명시합니다.
*   **스코프와 귀속:** AIR(자기일관성 실패)·Shortcut(취약한 성공)·Discordant Hit은 Thought→Text 아키텍처에서만 정의되는 양식이므로 "thinking 모델 고유 현상"임은 결함이 아니라 명시적 스코프입니다. 다만 이 취약성이 사고 채널에서 비롯되는지 기저 모델에서 유래하는지는 §3.3의 통제 분석(Regime Control)으로 귀속하며, 완화 기법은 대부분 비-thinking 모델 기원임을 밝히고 "사고 모델로의 이식 거동"을 해부하는 것으로 프레이밍합니다. 비-thinking 모델에 대한 AIR 일반화는 주장하지 않습니다.
*   **인용 등급(2026-07 arXiv 메타데이터 확인):** 하중 근거는 모두 게재 확정 문헌으로 떠받칩니다 — 인과 개입의 이론적 배경은 Lanham(2023, 표준 인용)·FRODO(Findings of EMNLP 2024), 완화/오버싱킹 근거는 FaithfulRAG(ACL 2025)·*Stop Overthinking*(TMLR 2025)·Lee & Hockenmaier(Findings of EMNLP 2025), CoT 불충실성·귀속 지표 대비는 Arcuschin(ICML 2026)·ALCE(Gao et al., EMNLP 2023). 의도적으로 유지한 프리프린트는 두 부류로 한정됩니다: (1) **직접 사용하는 데이터셋 산출물** DRAGged(2506.08500) — 속성을 자체 검증하며 게재된 RAMDocs(COLM 2025)로 이중화, (2) **resampling 방법 출처** Thought Anchors(2506.19143)·Thought Branches(2510.27484) — 게재된 Lanham·FRODO로 백본을 대체 가능하게 설계. 그 밖에 새로 인용한 프리프린트(monitorability 프레이밍의 Korbak 2025, 포지셔닝 대상 Young 2026, 굴복 배경 Sharma 2023, 동기 예시 WikiContradict·EnterpriseRAG-Bench, 동시기 Li et al. 2026)는 모두 **동기·포지셔닝·개념 프레이밍 용도의 비하중(non-load-bearing)이며 수치에 의존하지 않습니다.** 제출 시점에 이들 프리프린트의 게재 상태를 재확인해 갱신합니다.

---

## 6. 향후 연구 (Future Work)

본 연구는 **진단**에 집중하며, 처방(완화 설계)은 진단이 개입 지점을 특정한 뒤에야 근거를 갖습니다. 아래는 실측이 나오기 전이라 **예상 결과에 따라 분기하는 향후 방향**으로 제시합니다 — 진단 프레임워크가 무엇을 가리키느냐에 따라 뒤따르는 연구가 달라집니다.

### 6.1. 진단 결과에 따라 분기하는 처방 연구
| 예비 진단 지표가 이러면 | 뒤따르는 향후 연구 |
| :--- | :--- |
| **AIR ↑** (+ 숨은 퇴행율 ↑, 엔트로피 스파이크가 `</think>`→답변 전환부에 집중) | 그 지점만 겨냥한 **경량 개입**(엔트로피 트리거 재디코딩·표출 직전 검증 스텝)을 처방으로 설계·검증 — 진단이 개입 지점을 국소화했음을 활용 |
| **LGR ↓** (완화 EM 상승분이 취약 경로 Shortcut·Discordant Hit·Blind-Hit에 집중) | 정상 경로를 **선택적으로 강화**하거나, 문항의 병목 유형을 보고 완화를 고르는 **경로-인지형 선택기** 설계 |
| **Shortcut ↑** (+ Lock-in: 초기 위치 문서 고착 관측) | 위치 편향 완화·**답변 전 명시적 해소 강제**(프롬프트/경량 학습) |
| **Discordant Hit ↑** (오답 문서 지지 후 정답 표출) | 파라메트릭 지식이 트레이스를 덮어쓰는 **메모리-문맥 상호작용** 규명 |
| **레짐 통제: thinking-off 시 AIR·Shortcut 소멸/정의 불가** | 사고→답변 **핸드오프 자체를 겨냥한 정합성 정렬**(학습 개입) |

이때 처방은 **"SOTA 완화"가 아니라 "진단이 실행 가능함을 보이는 타깃 시연"**으로 자리매김하여, 개선폭이 작더라도 진단 기여의 실용성을 입증하는 역할로 한정합니다.

### 6.2. 프레임워크 확장
*   **도메인 이식:** §5에서 스코프 밖으로 둔 **기업 내부 지식베이스**(버전·시간 충돌; WikiContradict·EnterpriseRAG-Bench류)에 동일 프로토콜을 적용해 도메인 불문성을 실증합니다.
*   **충돌 유형 확장:** Inter-Context를 넘어 **Context-Memory 충돌**(내부 지식 vs 외부 문서)로 3단계 라벨링을 확장합니다.
*   **모델 범위:** 로짓 접근이 불가한 **폐쇄형 API 모델**에는 엔트로피 신호(부록 B.3)를 제외하고 행동 라벨·개입만으로 이식하며, 증류 모델과의 대조로 "자율 탐색 궤적" 가정을 검증합니다.
*   **상관 → 기계적 심화:** 부록 B의 상관 신호를 **내부(활성화) 수준 개입**(어텐션·activation steering)으로 끌어올려, 본 연구가 범위 밖으로 둔 기제적 인과까지 규명합니다.

### 6.3. 예측적 활용
경로 서명(예: Shortcut·Blind-Hit·높은 전환부 엔트로피)이 **추론 시점에 취약한 정답을 사전 식별**하는 신호가 되는지 검증하여, **신뢰도 보정·선택적 기권**의 트리거로 활용하는 방향을 탐색합니다.

---

## References

*   Xu et al. (2024). *Knowledge Conflicts for LLMs: A Survey.* EMNLP 2024. arXiv:2403.08319.
*   Cattan et al. (2025). *DRAGged into Conflicts: Detecting and Addressing Conflicting Sources in Search-Augmented LLMs.* arXiv:2506.08500. (프리프린트 — 직접 사용하는 주력 데이터셋, RAMDocs로 이중화)
*   Wang, Prasad, Stengel-Eskin, Bansal (2025). *Retrieval-Augmented Generation with Conflicting Evidence (RAMDocs + MADAM-RAG).* COLM 2025. arXiv:2504.13079. (RAMDocs 벤치마크와 MADAM-RAG 기법이 동일 논문)
*   Hou et al. (2024). *WikiContradict: A Benchmark for Evaluating LLMs on Real-World Knowledge Conflicts from Wikipedia.* arXiv:2406.13805. (프리프린트 — 큐레이션 코퍼스 내 실재 모순 근거)
*   Sun et al. (2026). *EnterpriseRAG-Bench: A RAG Benchmark for Company Internal Knowledge.* arXiv:2605.05253. (프리프린트 — 기업 내부 코퍼스의 버전·시간 충돌 근거)
*   Asai et al. (2024). *Self-RAG.* ICLR 2024. arXiv:2310.11511.
*   Shi et al. (2024). *Trusting Your Evidence: Hallucinate Less with Context-Aware Decoding (CAD).* NAACL 2024. arXiv:2305.14739.
*   Wang et al. (2024). *AdaCAD: Adaptively Decoding to Balance Conflicts between Contextual and Parametric Knowledge.* NAACL 2025. arXiv:2409.07394.
*   Zhang et al. (2025). *FaithfulRAG: Fact-Level Conflict Modeling for Context-Faithful RAG.* ACL 2025. arXiv:2506.08938.
*   *ConflictRAG: Detecting and Resolving Knowledge Conflicts in Retrieval-Augmented Generation.* 2026. arXiv:2605.17301. (MLP 충돌 감지 + Entropy-TOPSIS; 프리프린트)
*   Cuconasu et al. (2024). *The Power of Noise: Redefining Retrieval for RAG Systems.* SIGIR 2024. arXiv:2401.14887.
*   Zhou et al. (2023). *Context-faithful Prompting for Large Language Models.* Findings of EMNLP 2023. arXiv:2303.11315.
*   Turpin et al. (2023). *Language Models Don't Always Say What They Think: Unfaithful Explanations in CoT Prompting.* NeurIPS 2023. arXiv:2305.04388.
*   Lanham et al. (2023). *Measuring Faithfulness in Chain-of-Thought Reasoning.* Anthropic 프리프린트. arXiv:2307.13702.
*   Sharma et al. (2023). *Towards Understanding Sycophancy in Language Models.* arXiv:2310.13548.
*   Arcuschin et al. (2025). *Chain-of-Thought Reasoning In The Wild Is Not Always Faithful.* ICML 2026. arXiv:2503.08679.
*   Korbak et al. (2025). *Chain of Thought Monitorability: A New and Fragile Opportunity for AI Safety.* 다기관 포지션 페이퍼(프리프린트). arXiv:2507.11473.
*   Young (2026). *Why Models Know But Don't Say: Chain-of-Thought Faithfulness Divergence Between Thinking Tokens and Answers in Open-Weight Reasoning Models.* arXiv:2603.26410. (동시기 프리프린트 — 힌트 편향 기반, 포지셔닝 대상)
*   Atanasova et al. (2023). *Faithfulness Tests for Natural Language Explanations.* ACL 2023. arXiv:2305.18029.
*   Parcalabescu & Frank (2024). *On Measuring Faithfulness or Self-Consistency of Natural Language Explanations.* ACL 2024. arXiv:2311.07466.
*   Lee & Hockenmaier (2025). *Evaluating Step-by-step Reasoning Traces: A Survey.* Findings of EMNLP 2025. arXiv:2502.12289.
*   Li, Krishnan, Padman (2026). *The Chain Holds, the Answer Folds: Trace–Answer Dissociation in Reasoning Models Under Adversarial Pressure.* arXiv:2605.29087. (동시기 프리프린트 — 포지셔닝 전용, 수치 비의존)
*   Bogdan et al. (2025). *Thought Anchors: Which LLM Reasoning Steps Matter?* arXiv:2506.19143.
*   Macar et al. (2025). *Thought Branches: Interpreting LLM Reasoning Requires Resampling.* arXiv:2510.27484.
*   Paul et al. (2024). *Making Reasoning Matter: Measuring and Improving Faithfulness of CoT (FRODO).* Findings of EMNLP 2024. arXiv:2402.13950.
*   *Stop Overthinking: A Survey on Efficient Reasoning for Large Language Models.* TMLR 2025. (게재 — 오버싱킹/추론길이-정확도 역U자 근거)
*   Kuhn, Gal, Farquhar (2023). *Semantic Uncertainty.* ICLR 2023. arXiv:2302.09664.
*   Liu et al. (2023). *Lost in the Middle: How Language Models Use Long Contexts.* TACL 2024. arXiv:2307.03172.
*   Longpre et al. (2021). *Entity-Based Knowledge Conflicts in Question Answering.* EMNLP 2021. arXiv:2109.05052.
*   Xie et al. (2024). *Adaptive Chameleon or Stubborn Sloth: Revealing the Behavior of LLMs in Knowledge Conflicts.* ICLR 2024. arXiv:2305.13300.
*   Wu et al. (2024). *ClashEval: Quantifying the Tug-of-War Between an LLM's Internal Prior and External Evidence.* NeurIPS 2024 D&B. arXiv:2404.10198.
*   Jin et al. (2024). *Tug-of-War Between Knowledge: Resolving Knowledge Conflicts in RAG.* LREC-COLING 2024. arXiv:2402.14409.
*   Zheng et al. (2023). *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.* NeurIPS 2023. arXiv:2306.05685.
*   Liu et al. (2023). *G-Eval: NLG Evaluation Using GPT-4 with Better Human Alignment.* EMNLP 2023. arXiv:2303.16634.
*   Gao et al. (2023). *Enabling Large Language Models to Generate Text with Citations (ALCE).* EMNLP 2023. arXiv:2305.14627.
*   Panickssery et al. (2024). *LLM Evaluators Recognize and Favor Their Own Generations.* NeurIPS 2024. arXiv:2404.13076.


---

## 부록 A. 라벨 타당성 검증 프로토콜 (Label Validity)

§3.2의 3단계 라벨은 **텍스트로부터 관측된 '언어화된(verbalized)' 인지·해소**이며 "언어화하지 않음 ≠ 내부적으로 하지 않음"입니다. 이 비대칭은 지표별로 다르게 작용합니다: **AIR은 L2 = correct(올바른 해소를 *적은* 경우)라는 존재(presence) 조건 위에서 정의되므로 언어화 갭에 강건**하고(가려진 침묵은 이 버킷의 coverage만 줄일 뿐 값의 타당성은 훼손하지 않음), 취약한 것은 **부재(absence)를 조건으로 하는 Loss_L1·Shortcut·Blind-Hit**입니다. 아래에서 검증하는 것은 **판정 라벨이 텍스트를 정확히 읽었는가**(측정 도구의 신뢰도)이며, "사고가 답변을 실제로 구동했는가"(L-b)는 별개의 질문으로 실험 3이 담당합니다. 다음 세 층을 사전등록합니다.

*   **(a) 판정자 편향 통제:** LLM-as-a-Judge의 위치 편향·장황함 편향·자기선호 편향을 통제하기 위해(Zheng et al., 2023; Panickssery et al., 2024), 판정자는 **트레이스 생성 모델과 다른 계열**을 사용하고(예: 대상 모델 ≠ 판정자), 옵션 순서를 무작위 스왑하며, 형식-채움(form-filling) 프로토콜(G-Eval; Liu et al., 2023)을 적용합니다. 판정자 구성은 **오픈 가중치 판정자 1종과 상용 판정자 1종을 병행**하여 재현성(오픈)과 판정 품질(상용)을 함께 확보하며, 대상 모델과 동일 계열의 판정자는 해당 모델의 트레이스 판정에서 제외합니다.
*   **(b) 인간·교차 판정 일치도:** 트레이스 3단계 라벨의 신뢰도를 위해 무작위 200건에 대해 인간 전문가 2인의 Cohen's κ와 서로 다른 두 LLM 판정자 간 교차 일치도를 보고합니다(이는 §3.1.1의 정답 문서 매핑 검증 — 사실 충돌 67건 전수 — 과는 별개의 검증입니다). 특히 가장 주관적인 **Level 2(올바른 해소) 라벨은 κ를 별도 보고하고, κ가 임계(예: 0.6) 미만이거나 두 판정자가 불일치하는 문항은 인간 전문가가 adjudication하여 최종 라벨을 확정**합니다(§3.2). 단, κ(라벨러 간 일치)는 판정 신뢰도를 담보할 뿐, 사고 채널이 답변에 행동적으로 유의미한지(실험 2(b) 마스킹 게이트)나 인과적으로 쓰였는지(실험 3)를 대체하지 않습니다.
*   **(c) 부재 조건 지표의 언어화 갭 보강 — "안 적음"과 "못 함"의 분리:** Shortcut·Blind-Hit·Loss_L1은 "언어화 부재"를 조건으로 하므로, 다음 네 장치로 *침묵(알고도 안 적음)*과 *무능(진짜 못 함)*을 분리합니다.
    1.  **유도 프로브(Elicitation Probe):** unresolved/unrecognized로 분류된 샘플에 후속으로 직접 질의(*"충돌이 있는가? 어느 문서가 유효한가?"*) — 물으니 맞히면 언어화 갭(침묵), 물어도 못 하면 진짜 처리 실패. 두 원인의 비율을 보고. **단, 이 프로브 결과는 Loss_L1 값 자체를 바꾸지 않습니다 — Loss_L1은 원 답변 시점에 언어화하지 않은 비율(행동)로 고정하고, 프로브(능력 상한)는 그 옆에 "물어도 못 함(무능 하한) / 물으면 됨(침묵 상한)" 분해로 참고용으로만 병기**합니다. 명시적으로 물어 맞힌 것이 원 실행에서 실제로 처리했음을 뜻하지는 않으므로, 능력 정보를 행동 지표에 합치지 않는 것입니다.
    2.  **암묵적 기울기 백스톱:** 명시적 해소 선언이 없어도 부록 B.1 Semantic Shift가 정답 문서군 쪽 드리프트를 연속 신호로 포착하므로, 판정자에게 **암묵적 leaning도 인정**하도록 지시하여 라벨 recall을 높입니다.
    3.  **다중 시드 빈도:** 다중 시드(≥5) 중 충돌 인지/해소가 표출되는 빈도를 소프트 신호로 사용해, 우연한 1회 미표출을 "미인지"로 오판하는 것을 줄입니다.
    4.  **정직한 명명·교란 명시:** Loss_L1을 **"언어화된 인지 실패율"**로 규정하고, 모델별 언어화 습관(예: gpt-oss Harmony vs Qwen `<think>`) 차이가 이 비율에 섞이는 교란 변인임을 §5(한계)에 명시합니다.

---

## 부록 B. 궤적 분석층: 상관 분석 방법 (Trajectory Analysis Layer)

§3.2의 지표가 **무슨 일이 얼마나** 일어났는지를 이산 라벨로 센다면, 본 분석층은 그렇게 라벨된 샘플(특히 AIR·Shortcut)을 입력으로 받아 **그 뒤에 깔린 연속 궤적**을 들여다보며 '**왜·어디서** 그렇게 됐나'를 설명합니다. 따라서 새 지표를 정의하는 것이 아니라 §3.2 지표에 종속되며(라벨이 먼저 있어야 분석 대상이 정해짐), 세 기법(B.1~B.3)은 모두 텍스트·로짓에서 관측되는 **상관적(observational)** 신호로 **인과 주장은 하지 않습니다**(인과는 §3.3 실험 3의 개입 배터리 전담). 실질 역할은 그 **개입 대상 지점(실험 3)을 특정**하는 것이며, AIR·특이 경로가 예비 실측에서 유의하게 나타날 때 해당 경로에 선택 적용합니다. 실험 결과가 산출되면 그 분석 결과는 §4(실험 결과)에 함께 제시합니다.

| 진단 목적 (3단계 병목) | 담당 기법 | 성격 | 문헌 근거 |
| :--- | :--- | :--- | :--- |
| **① L1→L2 병목:** 명시적 대조 없이 도약(Shortcut)·초기 문서 고착 | B.2 Lock-in / RCPD | 상관 | Liu et al. (2023); Bogdan et al. (2025) |
| **② L2 판정 오염:** 신념 흔들림·오답 지지(Loss_L2) | B.1 Trajectory Semantic Shift | 상관 | Lanham et al. (2023); Lee & Hockenmaier (2025) |
| **③ L2→FA 단절:** 올바른 판단 후 굴복(AIR) | B.2 Overthinking + B.3 Entropy Spike | 상관 | Kuhn et al. (2023); Li et al. (2026) |
| **①~③ 인과 검증** | §3.3 실험 3 개입 배터리 | **인과** | Lanham (2023); Bogdan (2025); Macar (2025) |

각 기법의 배정은 **주 담당**일 뿐이며 여러 병목에 걸칠 수 있습니다 — 예: B.1 Semantic Shift는 ②(Loss_L2)뿐 아니라 ③(AIR의 Late Shift)도 함께 조명합니다.

### B.1. Trajectory Semantic Shift (의미론적 궤적 추적)
CoT를 문장 단위($s_1,\dots,s_T$)로 분할하고 각 문장 임베딩 $e(s_t)$와 **모든 검색 문서**의 코사인 유사도를 계산해, 문서를 정답 문서군(correct-doc) vs 충돌 문서군(conflicting docs)으로 묶고 각 군 중심과의 유사도 궤적을 추적합니다 (Lanham et al., 2023; Lee & Hockenmaier, 2025). 정답 채점이 가능한 사실 기반 충돌 그룹에 적용합니다.
*   **목적:** 사고가 진행됨에 따라 내부 신념이 어느 군으로 기우는지 시계열로 정량화하여 Loss_L2의 상관 패턴을 규명.
*   **분석 포인트:** AIR 샘플에서 마지막 문장까지 정답 군에 가깝다가 `<think>` 종료 임계 영역에서 급격히 충돌 군으로 기우는 **Late Shift**가 나타나는지 확인하고, 관측될 경우 해당 지점을 실험 3의 개입 대상으로 지정.

### B.2. Reasoning Completion Point Detection (RCPD): Lock-in 및 Overthinking 분석
사고 완료 지점 탐지(Reasoning Completion Point Detection, RCPD)로, 하이브리드 라벨링의 타임스탬프를 활용해 충돌 감지(L1)와 해소 확정(L2)이 전체 사고 길이 중 어느 토큰 인덱스에서 발생하는지 측정합니다.
*   **Lock-in Effect (초기 고착):** 입력 초기 위치의 문서가 앞쪽 10% 토큰을 지배하고 이 초기 판단이 최종까지 이어지는지 분석하여, 명시적 대조 없이 도달하는 **Shortcut의 상관 원인**을 규명합니다. §3.1의 문서 셔플링이 위치와 내용을 분리해 주므로, 셔플 전반에서 **초기 위치 문서가 지배한다면 이는 내용이 아닌 위치에 기인한 고착(positional lock-in)**임을 가려낼 수 있습니다. 위치 편향의 기저는 Liu et al. (2023, *Lost in the Middle*)의 U자형 primacy/recency 곡선을 근거로 삼습니다.
*   **Overthinking (과잉 추론):** L2 결론 이후에도 불필요한 사고 토큰이 길게 이어질 때 오히려 최종 답변에서 굴복(=AIR, 올바른 해소 후 오답)이 늘어나는지 조사합니다. 긴 사고가 정확도를 해칠 수 있다(추론 길이와 정확도의 역U자 관계)는 근거로는 게재된 효율적 추론 서베이 *Stop Overthinking: A Survey on Efficient Reasoning for LLMs* (TMLR 2025)를 인용합니다.

### B.3. Token-level Entropy & Perplexity Dynamics
CoT 전반 및 답변 첫 토큰 시점의 샤논 엔트로피 $H(X)=-\sum P(x_i)\log P(x_i)$와 Perplexity를 측정합니다 (Kuhn et al., 2023).
*   **목적:** 내부 혼란도(Internal Uncertainty)를 정량화하여, L2에서 올바른 판단을 내리고도 AIR이 발생하는 샘플의 상관 특성을 해부합니다(향후 처방 설계 시 트리거 신호 후보).
*   **가설:** AIR 샘플은 L2에서 올바른 결론을 적었더라도 해당 토큰 엔트로피가 유의하게 높고(Internal Hesitation), 특히 `</think>`→답변 첫 토큰 전환 임계 지점에서 엔트로피가 스파이크로 치솟을 것입니다. 이 전환부 굴복 현상 자체는 Li et al.(2026)이 보고한 궤적-답변 해리와 정합적이며, 본 연구는 엔트로피 동역학을 **AIR의 전환 지점에 새로 연결**하여 그 기저 신호로 살핍니다.

### B.4. 통합 진단 파이프라인 (Integrated Diagnostic Pipeline)
§3.2의 3단계 하이브리드 라벨링, B.1~B.3의 상관 진단, §3.3 실험 3의 인과 개입 배터리를 하나의 자동화 파이프라인으로 결합하여, 모델·완화 환경 전반의 진단을 **일관된 프로토콜**로 수행합니다.
*   **아키텍처:** (1) **3-Stage Audit Engine** — 사고·답변에서 Level 1·2 도달 여부를 자동 파싱해 전체 전환 행렬 도출; (1b) **Flow Diff 모듈** — `question_id`로 환경 간 결과를 짝지어 before→after 흐름 행렬을 만들고 LGR·숨은 퇴행율을 자동 산출(집계 대조가 가리는 개선·퇴행 동시 발생 포착); (2) **Trajectory Profiler (상관 진단)** — Semantic Shift 시계열, Lock-in/Overthinking 비율, 엔트로피 스파이크를 추출·시각화; (3) **Causal Prober (인과 개입)** — truncation·resampling을 자동 실행해 지점별 인과 기여도를 리포트하며, **정답을 낸 경로별 표본에 개입을 적용해 경로별 flip률(취약성)까지 산출**.
*   **모듈성:** 새 추론 모델은 **최소 어댑터(사고 채널 파서)**만 추가하면 동일 프로토콜로 진단되며(gpt-oss용 Harmony 파서를 실증 사례로 포함), 신규/차용을 구분한 **지표 레지스트리**로 측정의 재현성을 확보합니다. 인간 검증 골드 라벨 200건(부록 A)은 파이프라인 라벨 품질의 내부 검증셋으로 사용합니다.

