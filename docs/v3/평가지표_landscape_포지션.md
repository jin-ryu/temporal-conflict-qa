# RAG 평가 지표 지형과 본 연구의 포지션

작성: 2026-07-05 · 검색·인용·생성 3단 평가 지표 정리 + GoldCite 포지셔닝

## 0. 핵심 관찰 (이 문서의 출발점)

**검색 평가는 "정답 근거가 검색됐나"(gold 기준)를 보는데, 인용 평가는 "인용이 *모델 답*을 뒷받침하나"(답 기준)를 본다.** 즉 파이프라인 3단계 중 검색·생성은 gold 기준인데 **인용만 답 기준(순환적)**이다. 이 비대칭이 본 연구의 출발점이다.

| 단계 | 무엇을 재나 | 표준 지표 | **기준점(anchor)** |
|---|---|---|---|
| **검색 (Retrieval)** | 정답 근거가 top-k에 왔나 | Recall@k, Hit@k, MRR, nDCG, Context Recall/Precision(RAGAS) | **gold 근거** |
| **인용 (Citation)** | 인용이 답을 뒷받침하나 | CitePrec, CiteRecall(ALCE), AIS | **모델 답** ← 비대칭 |
| **생성 (Generation)** | 답이 맞나 | EM, Token-F1, Faithfulness(RAGAS) | **gold 답** |

→ 인용을 검색·생성처럼 **gold 기준**으로 바꾸면 어떻게 되나? 그게 GoldCite다. **단, 이 아이디어 자체는 이미 존재한다(아래 §2).** 우리 novelty는 *어디에 적용하느냐*에 있다(§4).

---

## 1. 단계별 표준 지표

### 1.1. 검색 평가 (gold 기준)

* **Recall@k / Hit@k** — 정답 근거(gold)가 상위 k개에 포함된 비율/여부. RAG 검색의 1순위 지표.
* **Precision@k** — 상위 k개 중 관련 문서 비율.
* **MRR (Mean Reciprocal Rank)** — 첫 정답 문서의 순위 역수 평균 (얼마나 앞에 왔나).
* **nDCG@k** (Järvelin & Kekäläinen, 2000) — 등급별 관련성 + 위치 할인.
* **Context Recall / Context Precision** (RAGAS; Es et al., 2024) — 답에 필요한 정보가 검색 청크에 담긴 비율 / 검색 청크의 관련 비율. 정답 없이도 측정 가능.
* 최근: "Redefining Retrieval Evaluation in the Era of LLMs" (arXiv:2510.21440) — LLM 시대 검색 평가 재정의 논의.

→ **공통점: 전부 gold(정답/관련 근거) 대비로 채점.** 검색은 "옳은 근거를 가져왔나"를 당연히 gold로 잰다.

### 1.2. 인용 평가 (답 기준 — 비대칭의 핵심)

* **ALCE** (Gao et al., EMNLP 2023, arXiv:2305.14627):
  - **CiteRecall** = 인용 문서들이 *모델 자신의 문장*을 NLI로 함의하면 1.
  - **CitePrec** = 각 인용이 *모델 문장*을 함의하는지 (불필요 인용 감점).
  - 채점 방향 = **인용 → 모델 답**. gold 근거와 대조하지 않음.
* **AIS** (Rashkin et al., Comput. Linguistics 2023, arXiv:2112.12870): "According to source P, s" — 출력 s가 *제시된 출처*에서 검증되나. **correctness와 의도적 분리**(출처에 충실해도 사실은 틀릴 수 있음).
* **CiteEval** (arXiv:2506.01829), **Correctness≠Faithfulness** (arXiv:2412.18004) — 인용 충실성과 정확성을 계속 분리해서 봐야 한다는 최근 논의.

→ **왜 답 기준인가 (설계 근거, §2 상술):** 열린 생성엔 단일 gold가 없고, 유효 근거가 복수이며, correctness와 분리하려 했고, gold 근거 주석이 비싸기 때문. NLI-to-답이 이 넷을 우회한다.

### 1.3. 생성 평가 (gold 기준)

* **EM / Token-F1** (Rajpurkar et al., SQuAD 2016) — gold 답 대비.
* **Faithfulness / Answer Relevancy** (RAGAS) — 답이 검색 문맥에 근거하나 / 질의와 관련되나.

---

## 2. "gold 기준 인용"은 이미 있다 — KILT 계열 (novelty 경계)

당신의 관찰("인용도 정답 근거를 인용했나 보면 되잖아")은 **이미 명명된 지표 계열로 존재한다.** 이걸 모르고 "우리가 발명"이라 하면 반박당한다.

| 지표 | 출처 | 무엇 | GoldCite와의 관계 |
|---|---|---|---|
| **KILT R-precision** | Petroni et al., NAACL 2021 (2009.02252) | 쓴 근거를 **gold provenance 집합**과 대조. 집합 기반·유효집합 max | **가장 가까운 조상.** "복수 유효 근거" 문제를 집합-max로 이미 해결 |
| **Citation-overlap P/R** | HAGRID eval framework (2409.08014) | 생성 인용 ∩ **gold 인용** | gold 기준 인용 P/R (이름까지 동일) |
| **GaRAGe Attribution P/R/F1** | Sorodoc et al., ACL Findings 2025 (2506.07671) | 인용을 **참조(gold) 응답의 인용과 대조** | gold 기준, 명시적 |
| **Citation Accuracy** | Shaier et al. 2024 | 생성 인용 문자열 ∩ gold 인용 문자열 | gold 기준 |
| **ERASER rationale-match** | DeYoung et al., ACL 2020 | 예측 근거 vs **gold rationale** (IOU·token-F1) | 문장/토큰 단위 사촌 |
| **HotpotQA supporting-facts F1 / FEVER evidence-F1 / 2WikiMultihop / MuSiQue** | 각 | 선택 근거 vs gold 근거 | "gold 근거 맞혔나" 계열 |

→ **결론: "gold 기준 인용 채점"은 KILT→HAGRID→GaRAGe 계보로 확립돼 있다.** 2025 attribution 서베이(2508.15396)도 *"citation retrieval metrics는 gold standard를 쓴다"*고 답-기준(attribution)과 명시적으로 구분한다. **mechanism은 novel이 아니다.**

### 왜 인용 평가가 답 기준이 됐나 (설계 근거)

ALCE/AIS가 gold 대비를 *의도적으로 피한* 이유:
1. **열린 생성엔 단일 gold 없음** — 요약·에세이의 각 문장이 어느 gold 근거에 대응하는지 사전 주석 불가.
2. **유효 근거가 복수** — 한 gold 문서만 인정하면 동등 유효 문서 인용을 부당하게 감점.
3. **correctness와 분리 의도** (AIS의 창립 원칙).
4. **gold 주석이 비쌈** — 문장 단위 근거 라벨링 비용.

→ NLI-to-답이 이 넷을 우회. 하지만 **시간 충돌 설정에선 이 넷이 다 무너진다**(§4).

---

## 3. 시간/충돌 설정 — 아무도 결합 안 한 지점

| 논문 | 시간축 있나 | gold 기준 인용 있나 | 결합? |
|---|---|---|---|
| **HoH** (2503.04800) | ✅ (outdated 영향) | ❌ (답 정확도 벤치) | 인용 지표 없음 |
| **GaRAGe** (2506.07671) | ✅ (OUTDATED 라벨) | ✅ (Attribution P/R/F1) | **❌ 둘을 안 합침** ← 결정적 |
| **Evidence-Force** (2605.28044) | ✅ (temporal 축) | ❌ (답-기준 warrant) | 답 기준이라 다른 축 |
| **CONFLICTS/DRAGged** (2506.08500) | ✅ (temporal 유형) | ❌ (충돌 해소/답) | 인용-gold 지표 없음 |
| **KILT** (2009.02252) | ❌ (정적 스냅샷) | ✅ (provenance) | 시간 무관 |

→ **정확한 빈칸:** *충돌 상황에서, 인용 단계에서, 인용 문서가 gold 근거 집합 $E^*$의 **시점-유효 버전**인지* 채점하는 지표는 **없다.** GaRAGe는 재료(OUTDATED 라벨 + gold 인용 지표)를 나란히 갖고도 **결합 안 함.** GoldCite = 그 결합.

---

## 4. 본 연구의 포지션

### 4.1. novelty 좌표 (정직하게)

* **mechanism(gold 기준 인용 채점) = novel 아님** → KILT provenance 계보. *발명이라 주장하지 않는다.*
* **delta = 시점 유효성 축에 적용** → gold 근거의 어느 버전이 유효한지가 *질문 시점에 따라 바뀌는* 설정에서, 인용이 그 유효 버전을 담는지 채점. 이 교집합은 비어 있다.

**정확한 위치:**
- **vs KILT:** KILT provenance는 시간 무관(정적 스냅샷, "주제상 gold 페이지인가"). GoldCite는 provenance에 **버전/시점 유효성 차원**을 추가.
- **vs ALCE:** ALCE는 답 기준이라 *outdated 문서가 유창한 오답을 완벽히 함의하면 통과*하는 구조적 맹점. GoldCite는 gold 기준이라 이를 포착.
- **vs GaRAGe:** OUTDATED 라벨 + gold 인용 지표를 **결합**한 것이 GoldCite.

### 4.2. 우리가 취할 3단 포지션 (파이프라인 국소화)

당신의 통찰(검색은 gold로 재는데 인용은 왜 안?)을 **실패 국소화 프레임**으로 확장한다:

| 단계 | 지표 | 실패 시 진단 |
|---|---|---|
| 1. 검색 | **$E^*$-Recall@k** (=KILT식 gold 근거 검색) | 검색 실패 |
| 2. 인용/근거 선택 | **GoldCite** (= $E^*$ 검색됨 \| 인용이 유효 버전 담나) | mis-attribution (근거 선택 실패) |
| 3. 시점 추론 | **EM** (\| GoldCite=1: 유효 문서 인용하고도 틀림) | 시점 추론 실패 |

→ 세 단계 모두 **gold($E^*$) 기준으로 통일**하면 "오답+높은 인용"이 검색/선택/추론 중 어디서 깨졌는지 국소화된다. **인용만 답 기준이라 이 국소화가 지금까지 불가능했다** = 우리 기여의 정확한 자리.

### 4.3. 반드시 선점할 반박 3종

1. **"KILT 재탕 아니냐"** → *"gold 기준 인용은 KILT provenance 계보임을 인정한다. 우리 delta는 provenance에 없던 **시점 유효성** 축이다(정적 스냅샷 → 버전 충돌)."* KILT를 **먼저 인용**해 정직성 + "복수 유효 근거" 반박 동시 차단(집합-max = 우리 $E^*$).
2. **"correctness를 attribution에 섞는다"(가장 강함)** → *"GoldCite는 답-기준 attribution의 **대체가 아니라 보완 진단 축**이다. outdated 문서가 유창한 오답을 함의하면 ALCE-NLI는 통과하지만 인용은 시점-무효 — GoldCite는 ALCE가 구조적으로 못 잡는 그 실패만 포착한다."* 정답 없이 **타임스탬프만으로** 판정된다는 점도 명시(correctness 아님).
3. **"gold 주석 비싸잖아"** → *"열린 생성에선 맞다(ALCE가 피한 이유). 그러나 시간 충돌에선 유효 근거 집합 $E^*$가 충돌 구성에 필요한 **타임스탬프/버전 메타데이터로 기계적으로 도출**된다 → 오히려 우리 설정의 강점."* (단점을 장점으로.)

### 4.4. 한계 (정직)

* $E^*$(시점-유효 gold 근거) 주석이 **결정론적 정답·유효 시간창이 있는 충돌(시간·사실 대체)에서만 깨끗이 정의**됨. 의견·모호성 충돌엔 적용 불가.
* 문서(집합-멤버십) 단위라 span 단위보다 거침 → ERASER식 token-F1이 정밀화 경로(필요 시).
* 깨끗한 대규모 데이터 희소(HoH + 야생 rag_conflicts 의존).

---

## 5. 참고문헌 (본 문서 인용)

**검색 평가:** RAGAS (Es et al., 2024, arXiv:2309.15217) · Redefining Retrieval Eval (2510.21440) · nDCG (Järvelin & Kekäläinen 2000)
**인용/귀속:** ALCE (2305.14627, EMNLP'23) · AIS (2112.12870, CL'23) · CiteEval (2506.01829) · Correctness≠Faithfulness (2412.18004) · Attribution survey (2508.15396)
**gold 기준 인용 계보:** KILT (2009.02252, NAACL'21) · HAGRID eval (2409.08014) · GaRAGe (2506.07671, ACL'25 Findings) · ERASER (ACL'20) · HotpotQA/FEVER/2WikiMultihop/MuSiQue
**시간/충돌:** HoH (2503.04800, ACL'25) · Evidence-Force (2605.28044) · CONFLICTS/DRAGged (2506.08500)
**생성:** SQuAD/EM-F1 (1606.05250)

> 주의(할루시네이션 방지): HAGRID의 "citation-overlap P/R"은 원논문(2307.16883)이 아니라 후속 eval framework(2409.08014)에서 정의됨 — 후자를 인용. 2025~2026 arXiv ID(2506.x, 2605.x 등)는 투고 전 원문 재확인 필요. 확실한 gold-기준 앵커는 **KILT·GaRAGe** 두 개.
