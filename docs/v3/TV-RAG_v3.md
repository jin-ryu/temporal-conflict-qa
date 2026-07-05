# 연구 계획서 v3

## 시간 충돌 검색 증강 생성에서의 근거 오귀속: 진단과 시점-유효 인용을 통한 완화

**Evidence Mis-attribution in Temporal-Conflict RAG: Diagnosis and Mitigation via Time-Valid Citation**

* **투고 목표:** ACL / EMNLP Main (Long Paper) 또는 Findings — 진단 · 평가 지표 · 경량 완화 (시스템 논문이 아님)
* **핵심 주장:** 검색 문맥에 상이한 시점의 문서가 혼재할 때, 대형 언어 모델(LLM)은 질문 시점에 유효하지 않은 문서를 근거로 삼아 시점상 낡은 답을 생성하면서도 해당 문서를 인용으로 제시한다. 표준 인용 평가(`CitePrec`, `AIS`)는 인용을 모델 자신의 답을 기준으로 채점하기 때문에 이러한 오답을 통과시킨다. 본 연구는 이 현상을 근거 오귀속(evidence mis-attribution)으로 규명·정량화하고, 인용의 채점 기준을 시점 유효성으로 전환함으로써 이를 드러내고(지표) 완화한다(경량 개입).

---

## 초록 (Abstract)

검색 증강 생성(RAG) 시스템은 답변의 검증 가능성을 높이기 위해 인용(citation)을 제공한다. 그러나 검색된 문맥에 서로 다른 시점에 작성된 문서가 공존하는 시간 충돌(temporal conflict) 환경에서는, 모델이 질문 시점에 유효하지 않은 문서를 근거로 낡은 답을 생성하면서도 그 문서를 인용으로 제시하는 근거 오귀속 현상이 발생한다. 표준 인용 평가 지표(`CitePrec`(ALCE), `AIS`)는 인용의 타당성을 모델 자신의 답을 기준으로 판정하므로, 오답을 담은 문서의 인용을 정상으로 통과시킨다. 5개 모델과 2개 데이터셋을 대상으로 한 진단 결과, 오답 중 표준 인용 평가를 통과하는 비율은 81~100%에 이르며, 표준 지표와 시점 유효성 기준 인용 정확도 사이에는 최대 54%p의 괴리가 관찰된다. Leave-One-Out 반사실 검증을 통해 이 오류의 59~76%가 사후 인용이 아니라 오류 문서에 대한 실질적 의존의 결과임을 확인한다.

본 연구는 이 현상의 원인을 인용 평가의 기준점(anchor)이 모델의 답이라는 순환적 구조에 있다고 본다. 이에 채점 기준을 시점 유효성으로 전환한 지표(GoldCite)를 제안하여 오귀속을 정량화하고, 동일한 원리를 개입 신호로 사용하여 이를 완화한다. 구체적으로, 검색된 문서에서 시점별 근거를 추출해 출처를 보존한 시간축으로 재구성하는 경량 개입만으로도 오귀속이 감소함을 보인다. 나아가 이 현상이 문서 간(명시적) 충돌에 특유함을 경계 실험으로 확인한다. 즉 단일 누적 문서 내의 시점 중의성(암시적)에서는 인용이 판별력을 잃어 오귀속 개념 자체가 성립하지 않는다.

---

## 1. 서론

### 1.1. 배경과 문제 정의

검색 증강 생성에서 인용은 "출처가 제시되었으므로 신뢰할 수 있다"는 신뢰 신호로 기능한다. 그러나 실제 검색 환경에서는 동일한 사실에 대해 서로 다른 시점에 작성된 문서가 함께 검색된다. 예컨대 "현재 FBI 국장은 누구인가"라는 질의에 대해 2016년 문서(Comey)와 2025년 문서(Patel)가 동시에 반환될 수 있다. 이때 모델이 2016년 문서를 근거로 "Comey"라는 오답을 생성하고 해당 문서를 인용하면, 표준 인용 평가는 "인용 문서가 답을 뒷받침하는가"를 물어 이를 통과시킨다. 그 결과 시점상 잘못된 답변이 정상적인 근거를 갖춘 것으로 오인되어 배포된다. 이는 판례 파기나 임상 지침 변경과 같은 고위험 도메인에서 특히 치명적이다(Zhang et al., 2025, arXiv:2510.20941; Facts Fade Fast, 2025, arXiv:2509.04304).

이 문제의 본질은 모델의 오류가 아니라 평가 방법의 사각지대에 있다. 인용 품질이 우수하다고 보고된 시스템이 시간 충돌 환경에서는 오답까지 포함해 부풀려진 점수를 받는다.

### 1.2. 기존 인용 평가의 한계: 기준점의 순환성

표준 인용 정밀도(ALCE의 `CitePrec`, `AIS`)는 인용 문서가 모델이 생성한 답을 뒷받침하는지를 측정한다. 채점의 기준점이 모델 자신의 출력이라는 점이 문제의 핵심이다. 모델은 애초에 잘못된 문서로부터 답을 도출하였으므로, 자신이 선택한 문서로 자신을 채점하는 구조에서는 항상 높은 점수를 받는다. 본 연구는 이를 순환의 함정(circular trap)이라 부른다.

### 1.3. 제안과 기여

본 연구는 인용 평가의 기준점을 모델의 답에서 시점 유효성으로 전환한다. 즉 "인용 문서가 질문 시점에 유효한 정답을 담는가"를 묻는다. 이 전환은 두 가지 역할을 동시에 수행한다. 지표로서는 `CitePrec`이 은폐한 오귀속을 드러내며, 두 지표의 격차가 곧 은폐된 오귀속의 크기다. 개입 신호로서는 학습 또는 프롬프트가 유효한 출처를 지향하도록 유도하여 오귀속을 완화한다.

본 연구의 기여는 다음과 같다.

1. **근거 오귀속의 규명과 정량화.** 시간 충돌에서 표준 인용 평가가 시점상 잘못된 문서의 인용을 81~100% 통과시킴을 5개 모델·2개 데이터셋·반사실 검증으로 입증한다. 인용 충실성의 일반적 문제는 Wallat et al.(2025)이 규명하였으며, 본 연구의 기여는 이를 결정론적으로 검증 가능한 시점 충돌 하위 부류로 좁혀 정량화한 데 있다.
2. **시점-유효 인용 지표(GoldCite).** 인용 평가의 기준점을 시점 유효성으로 전환한 지표를 제안한다. `CitePrec`과의 격차로 오귀속을 정량화한다.
3. **경량 완화.** 출처를 보존한 시간축 재구성으로 유효 출처의 인용을 유도하여 오귀속을 감소시킨다. 측정에 그치지 않고 개선 가능성을 실증한다.
4. **경계 규명.** 문서 간(명시적) 충돌과 문서 내(암시적) 시점 중의성을 대비하여, 오귀속과 GoldCite의 유효 범위가 문서 간 충돌에 국한됨을 밝힌다.

---

## 2. 관련 연구

### 2.1. 인용·귀속 평가와 그 한계

`AIS`(Rashkin et al., 2023)와 ALCE(Gao et al., 2023)는 귀속(attribution, 인용이 답을 뒷받침하는가)을 정확성(correctness, 답이 옳은가)과 의도적으로 분리하여 측정한다. 이 분리 자체는 타당하며 본 연구는 이를 부정하지 않는다. 본 연구의 주장은 보다 한정적이다. 즉 `CitePrec`을 단독 인용 품질 지표로 사용할 경우 충돌 환경에서 체계적으로 오도된다는 것이다.

그 메커니즘은 충돌이 "투명한 실패"를 "오귀속"으로 전환한다는 데 있다. 충돌이 없는 깨끗한 검색에서는 오답을 뒷받침하는 문서가 존재하지 않으므로 `CitePrec`이 낮게 나타나고, 오답은 인용 실패(투명한 실패)로 지표에 포착된다. 반면 충돌 환경에서는 오답을 뒷받침하는 시점상 잘못된 문서가 존재하므로, 모델이 이를 인용하면 `CitePrec`이 통과되어 오답이 지표를 벗어난다. 오귀속이 성립하려면 주제상 관련되면서 오답을 담은 문서가 필요하며, 이것이 곧 충돌이다. 주제와 무관한 방해 문서(distractor)로는 오귀속이 발생하지 않는다(실측: 오답 시 방해 문서 인용은 8~15%에 불과한 반면 충돌 문서 인용은 67~81%). 따라서 오귀속은 "`CitePrec`이 정확성을 보지 않는다"는 자명한 사실이 아니라, 충돌이 `CitePrec`을 체계적으로 부풀리는 조건부 현상이다. 이 부풀림은 모델 간 신뢰도 순위 역전(취약한 모델이 인용을 더 잘하는 것으로 오인됨)을 초래하므로, 정확성 지표(EM)를 별도로 보고하더라도 오도된 인용 품질 순위 자체는 교정되지 않는다. 결과적으로 충돌 평가에서는 인용을 시점 유효성으로 조건화하거나 오귀속률(EM×CitePrec 교차)을 함께 보고해야 한다. 부기하면, 인용 문서가 질문 시점에 유효한가는 문서의 타임스탬프만으로 판정 가능하여 정답을 요구하지 않으므로, 이는 정확성이 아니라 귀속의 확장이다.

가장 근접한 선행 연구는 Evidence-Force(Qian et al., 2026, arXiv:2605.28044)로, 주제적 관련성이 주장을 정당화하지 못하는 현상(citation laundering)을 관계·양상·범위·시점 유효성·수치 정밀성의 다섯 축에 대한 일반적 보정으로 다룬다. 본 연구와의 차별점은 (i) 충돌 조건에 특화되고, (ii) 시점-유효 기준으로 인용을 채점하며, (iii) 오귀속률을 정량화한다는 점이다. GaRAGe(2506.07671)는 근거에 "Outdated" 라벨을 부여하나 이는 문서 관련성 수준의 주석이며 인용 지표가 아니다.

### 2.2. 지식 충돌과 시간 인식 QA

Xu et al.(2024, EMNLP)의 지식 충돌 분류에서 본 연구의 명시적(문서 간) 충돌은 문맥 간 충돌(inter-context conflict)에 정확히 대응하는 표준 범주다. 반면 문서 내 시점 중의성(암시적)은 이 분류에 명명된 범주가 없으며, 가장 근접한 축은 WikiContradict(Hou et al., 2024)의 명시적·암시적 충돌 구분이다. 시간 인식 QA(SituatedQA, 2021; TimeQA, 2021; TempAmbiQA, 2024)는 답의 시점 의존성을 다루나 문서 간 인용 선택 문제로는 접근하지 않는다.

---

## 3. 문제 정의

질문 시점을 $t_q$, 실제 정답을 $a^*$, 모델 답을 $\hat a$, 인용 문서 집합을 $S$, 검색 문맥을 $\mathcal C$로 표기한다.

**정의 1 (표준 인용 정밀도, 답-기준).** $\text{CitePrec}=\mathbf 1[\exists c\in S:\ c\models \hat a]$. 인용이 모델의 답을 뒷받침하는지를 판정한다.

**정의 2 (시점-유효 근거 집합).** $E^*(t_q)=\{c\in\mathcal C:\ a^*\in c\ \wedge\ t_q\in[t_c^{\text{start}},t_c^{\text{end}})\}$. 정답을 담고 그 값이 질문 시점에 유효한 문서의 집합이다.

**정의 3 (시점-유효 인용, GoldCite).** $\text{GoldCite}=\mathbf 1[S\cap E^*\neq\varnothing]$. 인용이 시점-유효 근거를 담는지를 판정한다.

**정의 4 (근거 오귀속률).** 조건부 $P(\text{CitePrec}{=}1\mid \text{EM}{=}0)$(현상의 편재성 진단)과 절대 $|\text{EM}{=}0\wedge\text{CitePrec}{=}1|/N$(학습 전후 및 모델 간 비교의 주 지표)의 두 형태로 보고한다.

**충돌 유형의 정의.** 명시적 충돌(explicit, 문서 간)은 서로 다른 문서가 배타적으로 상충하는 답을 담아 어느 문서가 유효한지가 정오를 결정하는 경우다. 암시적 시점 중의성(implicit, 문서 내)은 단일 누적 문서가 여러 시점의 값을 함께 담아, 한 문서 안에서 유효한 값을 선택해야 하는 경우다. 후자는 문서가 내부적으로 일관되므로 엄밀한 의미의 충돌이 아니며, 인용의 관점에서 단일 문서를 인용하면 항상 통과되어 $\text{GoldCite}\to\text{CitePrec}\to 1$로 판별력을 상실한다. 따라서 본 연구의 인용 기여는 명시적 충돌에 국한되며, 암시적 유형은 경계 대조로만 사용한다.

---

## 4. 진단 연구

### 4.1. 데이터셋

세 데이터셋을 사용하며, 규모는 검증 완료분을 기준으로 하되 기여에 필요한 최소선을 상회하도록 설정한다. 명시적 충돌 2종(통제·야생)과 암시적 경계 1종으로 구성한다.

**(1) HoH — 명시적·통제·고유사 (진단 평가 및 완화 학습 겸용).** 위키피디아 편집 이력에서 파생하며, 각 항목은 현재/과거 두 답과 시점 라벨이 부여된 현재(R)·과거(O)·방해(D) 문서를 제공한다. 검증 완료 219건을 진단 기준으로 하되, 로컬 보유 진성 항목(약 760건)에서 인간 검증을 통해 약 500건까지 확장한다. 완화 학습이 필요할 경우 원본(96K QA)에서 약 2~3K를 파생한다.

```jsonc
// 청크 10개(R/O/D), 라벨 배타적. 이미 평가 포맷.
{ "new_question": "As of October 2024, how many players have won the PL with two clubs?",
  "target_answer": "Eight", "target_side": "outdated", "evidence_chunk_id": 38,
  "chunks": [ {"chunk_id":38,"label":"outdated","last_modified_time":"2024-09","text":"...eight players..."},  // 과거=정답
              {"chunk_id":1, "label":"current", "last_modified_time":"2024-10","text":"...seven players..."},  // 현재=오답 유발
              /* + 방해 문서 8개 */ ] }
// 가공: 완화 학습 시 source_idx 기준 8:2 분할(누수 차단)만 추가.
```

**(2) rag_conflicts — 명시적·야생·저유사 (진단 평가 전용, 51건).** 실제 구글 웹 검색 결과(DRAGged, Cattan et al., 2025)의 시점 충돌 유형을 변형 없이 사용한다. 표본이 작은 것은 깨끗한 야생 명시적 시점 충돌 데이터의 원천적 희소성을 반영하며, 이를 생태적 타당성 검증용으로만 활용한다.

```jsonc
{ "new_question":"When does this year's Passover start?", "target_answer":"...April 12",
  "chunks":[{"label":"current_dup","last_modified_time":"2025-01-01","text":"..."},
            {"label":"outdated","last_modified_time":"2024-05-01","text":"..."}, /* 총 10, 실제 매체 */] }
```

**(3) TQA — 암시적 경계 대조 (266건).** 원본은 동일 문서의 두 시점 스냅샷(2010·2014)을 제공하며, 최신 스냅샷이 누적적이어서 두 시점의 답을 모두 담는다(실측 90%). 이 중 단일 누적 문서에 두 답이 모두 포함된 266건을 선별하여, 누적 문서 하나만 제시하고 과거 시점을 묻는 문서 내 시점 선택 과제로 구성한다.

```jsonc
{ "new_question":"Who was the head of the Maori Party in 2010?", "target_answer":"Tariana Turia",  // 과거 값
  "chunks":[ {"chunk_id":1,"label":"current","last_modified_time":"2020",
              "text":"...Tariana Turia (2010)... Debbie Ngarewa-Packer (2020)..."} ] }  // 한 문서에 두 값 공존
```

| 유형 | 데이터 | 규모 | 역할 | 유사도 |
|---|---|---|---|---|
| 명시적·통제 | HoH | 진단 500 / 학습 2~3K | 주 진단 및 완화 학습 | 0.96 |
| 명시적·야생 | rag_conflicts | 51 | 생태적 타당성 검증 | 0.05 |
| 암시적·경계 | TQA(누적) | 266 | 경계 대조 | — |

### 4.2. 충돌 유형 자동 분류

경계 대조를 위해 각 항목을 검색 집합 단위로 분류한다(충돌 유형은 항목 내재 속성이 아니라 검색된 집합에 상대적이므로 이를 명시한다). 선행 연구(Contradiction Detection in RAG, 2504.00180)가 충돌 탐지·유형 예측·분할을 수행한 바 있다.

```python
def classify(passages, a_old, a_new):        # a_old, a_new: 두 시점 답 (TimeQA류가 제공)
    old_in = [p for p in passages if contains(p, a_old)]   # 별칭 정규화·개체 연결
    new_in = [p for p in passages if contains(p, a_new)]
    if any(p in old_in for p in new_in):      # 한 문서에 두 값 공존 → 암시적
        return "implicit"                      # + 날짜 정규화(SUTime)로 두 값이 상이한 시간 표현에 결부됨을 확인
    if old_in and new_in:                      # 서로 다른 문서에 분리 → 명시적
        return "explicit"                      # + NLI 백스톱: (과거 문서, 현재 문서) 쌍의 contradiction 확인
    return "mixed"                             # 다값·암시적 언급 → 인간 검수
```

구조적 신호(공존 대 분리)는 고정밀이며 NLI와 날짜 정규화를 백스톱으로 사용한다. 무작위 200건을 인간 검수하여 일치도($\kappa$)를 보고한다. 실패 모드(별칭 변이, 암시적 언급, 검색 의존성, 다값 항목, 시점 문장에 대한 NLI 취약성)를 한계로 명시한다.

### 4.3. 평가 프레임워크

정확성(EM)과 표준 인용 통과(`CitePrec`)를 교차한 2×2 분해를 분석 렌즈로 사용한다.

| | CitePrec=1 | CitePrec=0 |
|---|---|---|
| **EM=1** | Grounded (정합) | Under-cited (드묾) |
| **EM=0** | Mis-attribution (지표 과대평가, 위험) | Transparent-fail (지표가 정상 포착) |

### 4.4. 예상 결과

> 진단 실험 재사용: HoH·rag_conflicts에 대한 5개 모델(GPT-5.5, Gemini 3.1 Pro, Qwen3-32B, Qwen3-8B, Mistral-Small-24B) 진단이 완료되어 있어 재사용한다.
> `# TODO(추가 실험): (a) HoH 검증셋을 500건으로 확장, (b) 암시적 경계(TQA) 진단, (c) 새 백본(§6) 재실행, (d) 완화 개입 전후 비교.`
> 아래는 신규 실험으로 간주하여 예상 결과를 기술한다.

**결과 1 (현상의 편재성).** 오답이면서 표준 인용 평가를 통과하는 비율은 HoH에서 92~100%, rag_conflicts에서 81~93%에 이른다. 인용 실패로 정직하게 포착되는 오답(EM=0, CitePrec=0)은 모델당 0~5건에 불과하다. 즉 충돌 환경의 실패는 무근거 환각이 아니라 눈앞의 잘못된 근거에 대한 과도한 충실성에서 비롯된다.

**결과 2 (표준 지표의 사각지대와 순위 역전).** `CitePrec`은 만점에 가까운 반면(HoH 0.97~1.00, rag_conflicts 0.86~0.94) GoldCite는 급락한다(HoH 0.44~0.82, rag_conflicts 0.69~0.78). 최대 괴리는 54%p이며, 이로 인해 취약한 모델이 인용을 더 잘하는 것으로 오인되는 모델 간 순위 역전이 발생한다.

**결과 3 (행동적 충실성).** 오답을 뒷받침한 문서를 문맥에서 제거하고 재추론하는 Leave-One-Out 검증에서, 오픈소스 3개 모델·2개 데이터셋 기준 59~76%의 사례에서 답이 변한다. 이는 인용이 사후적 장식이 아니라 오류 문서에 대한 실질적 의존의 결과임을 보인다(Wallat et al., 2025의 사후 합리화와 구별).

**결과 4 (충돌이 원인임, 통제).** 단일 문서 조건(`current_only`/`outdated_only`, 충돌 제거; LOO 데이터 재사용)에서는 오답이 대부분 투명한 실패(`CitePrec` 낮음)로 이동하여 오귀속이 급감할 것으로 예상된다. 이는 오귀속이 `CitePrec`의 결함이 아니라 충돌이 유발하는 조건부 현상임을 대조로 확증한다(§2.1의 실증판).

**결과 5 (경계, 암시적).** TQA 누적 문서에서 `CitePrec`과 GoldCite의 격차는 5%p 미만이며 양자 모두 높게 나타나 오귀속 개념이 붕괴한다. 다만 EM은 여전히 낮아(문서 내 시점 선택 실패) 오귀속이 문서 간 검색에 특유한 현상임을 확증한다.

---

## 5. 완화: 인용 기준점의 전환

기준점 전환은 지표이자 개입 신호다. §4가 이를 지표로 사용하였다면, 본 절은 동일 원리를 개입 신호로 사용한다. 풀 강화학습 시스템이 아니라 기준점을 시점 유효성으로 전환할 때 오귀속이 감소함을 보이는 경량 개입에 국한한다.

### 5.1. 개입 단계

**단계 1 (시점-유효 재정렬, 학습 없음).** 생성 이전에 검색 문서를 타임스탬프와 시점 유효성으로 재정렬하여 유효 문서를 상위에 배치한다.

**단계 2 (출처 보존 시간축 재구성, 프롬프트).** 각 문서에서 (값, 유효 시점/구간, 출처 식별자)를 추출하여 명시적 시간축으로 재구성한 뒤, 질문 시점에 유효한 값을 선택하고 원 출처를 인용하게 한다. 예: `doc3:[Milan, 2000-2010]`, `doc1:[Inter, 2014~]`에서 $t_q=2010$이면 "Milan [doc3]"을 산출한다. 모델이 명시적 충돌에서 실패하는 원인은 여러 문서에 흩어진 시점 구조를 조립하지 못하는 데 있으며, 이 구조를 명시화하면 유효 출처를 선택하게 된다. 문서를 병합하지 않으므로 인용이 유지되고 GoldCite가 개선된다. 이는 깨끗한 타임스탬프를 전제한 결정론적 접근(2606.01435)을 내용 유래 유효 구간으로 확장한 것이다.

**단계 3 (경량 학습, 선택).** GoldCite를 선호/보상 신호로 삼아 DPO 또는 소규모 GRPO로 유효 출처 인용을 강화한다. HoH 학습 분할(2~3K)을 사용한다. 풀 시스템이나 최신 성능 경신을 목표로 하지 않으며, 기준점 신호가 오귀속을 감소시킴을 실증하는 데 국한한다.

### 5.2. 예상 결과

> `# TODO: 단계 1·2는 학습이 없어 즉시 실행 가능. 단계 3은 HoH 학습 분할과 백본 1종으로 수행.`

| 개입 | 학습 | 오귀속(절대율, HoH) | GoldCite | 비고 |
|---|---|---|---|---|
| 무개입 (Standard RAG) | — | 높음(기저) | 0.44~0.82 | 진단값 |
| 단계 1 (재정렬) | 없음 | 중간 감소 | 상승 | 값싼 상한 |
| 단계 2 (시간축 재구성) | 없음 | 큰 감소 | 큰 상승 | 핵심(프롬프트만) |
| 단계 3 (경량 학습) | 소규모 | 최대 감소 | 최고 | 선택 |
| Oracle (유효 문서만) | — | ~0 | ~1.0 | 상한 |

학습이 없는 단계 2만으로 오귀속이 유의하게 감소하면 완화가 성립한다. rag_conflicts에 대한 제로샷 전이로 일반화를 확인한다.

---

## 6. 실험 설정

### 6.1. 백본 모델

최근(2025~2026) 인용·충돌·시간 인식 RAG 연구의 사용 경향을 반영한다. 주 백본으로 Qwen3-8B(`Qwen/Qwen3-8B`)를 사용한다. 이는 현 시점 표준이며 사고(thinking) 모드 전환을 통해 추론의 효과를 별도 비용 없이 분리할 수 있고 verl/TRL의 GRPO·DPO 지원이 성숙하다. 선행 수치와의 직접 비교를 위해 재현 기준으로 Qwen2.5-7B-Instruct를 함께 사용한다(RAG-RL, FaithfulRAG, VeriCite 등에서 최다 사용). 계열 일반성 확인을 위해 Llama-3.1-8B-Instruct를 교차 백본으로 사용하며, 규모 확장 1점으로 Qwen2.5-32B-Instruct를 선택적으로 포함한다. 참조 상한으로 프론티어 모델(GPT-5.5, Gemini 3.1 Pro)의 제로샷 결과를 병기한다.

### 6.2. 평가 지표

| 지표 | 정의 | 근거 |
|---|---|---|
| EM / Token-F1 | $\mathbf 1[\hat a=a^*]$ / 토큰 중첩 $F_1$ | Rajpurkar et al. (2016) |
| CitePrec (답-기준) | 인용이 모델 답을 뒷받침 | Gao et al. (2023), ALCE |
| GoldCite (시점-유효) | $\mathbf 1[S\cap E^*\neq\varnothing]$ | 본 연구; Petroni et al. (2021), KILT |
| 오귀속률 | $P(\text{CP}{=}1\mid\text{EM}{=}0)$ 및 절대율 | 헤드라인 |
| 참조: CiteRecall, AIS | 학습 미사용 외부 귀속 지표 | ALCE; Rashkin et al. (2023) |

### 6.3. 통제

문서 순서를 문항마다 무작위로 섞어 위치 편향을 통제하고, 별도의 위치 절제 실험으로 위치 효과를 분리한다. HoH 학습·평가 분할은 원본 인덱스 기준으로 분리하여 누수를 차단하며, 사전학습 차단일 이후 슬라이스를 병기하여 오염을 통제한다. 충돌 유형 분류는 200건을 인간 검수하여 $\kappa$를 보고한다. 모든 표에 paired bootstrap 95% 신뢰구간을 병기하고, 강화학습은 3개 시드로 독립 수행한다.

---

## 7. 기여 및 한계

### 7.1. 기여

1. 시간 충돌에서의 근거 오귀속을 규명하고 정량화한다(오답 통과율 81~100%, 반사실 검증 59~76%).
2. 인용 평가의 기준점을 시점 유효성으로 전환한 지표(GoldCite)를 제안하여 귀속 평가에 결여된 시점 차원을 보완한다.
3. 출처 보존 시간축 재구성이라는 경량 개입으로 오귀속을 감소시켜, 측정을 넘어 개선 가능성을 실증한다.
4. 명시적 충돌과 암시적 시점 중의성의 대비를 통해 오귀속과 GoldCite의 유효 범위가 문서 간 충돌에 국한됨을 규명한다.

### 7.2. 한계

본 연구의 학습 코어는 위키피디아 계열(HoH)이며 일반화 검증은 뉴스·웹(rag_conflicts)까지의 제로샷 전이로 한정된다. 서론에서 인용하는 법률·의료 고위험 도메인의 실패 수치는 문제의 이해관계를 예시하기 위한 인용이며, 해당 도메인 데이터의 공개가 미비하여 실험 대상에는 포함하지 못하였다. 따라서 고위험 도메인으로의 일반화는 주장하지 않는다. 야생 표본(rag_conflicts n=51)의 규모는 깨끗한 야생 명시적 시점 충돌 데이터의 원천적 희소성을 반영하며 이를 한계로 명기한다. 자동 충돌 분류는 별칭 변이·암시적 언급·검색 의존성·시점 문장에 대한 NLI 취약성으로 인해 불완전하며 인간 검수로 보강한다. 완화의 범위는 경량 개입(재정렬·프롬프트·경량 학습)에 국한되며, 다중 턴 재검색과 풀 강화학습은 향후 과제로 남긴다.

---

## 8. 연구 일정

```
[P0 정합성·코드]
  → [P1 새 백본으로 명시적 진단 재현 + 암시적 경계 대조]
  → [P2 무학습 개입(단계 1·2) 전후 비교]
  → (개선 뚜렷) [P3 경량 학습(단계 3) + 일반화 검증] → [P4 집필]
  → (개선 미미) 진단·GoldCite·경계 규명으로 완결(분석/자원 논문)
```

학습(단계 3) 이전에 무학습 개입(단계 1·2)으로 완화 성립 여부를 먼저 확인하여 위험을 최소화한다. 투고 이전에 Evidence-Force(2605.28044)의 전문을 대조하여 차별점 문단을 확정한다.

---

## 참고문헌 (주요 인용 문헌)

표기 형식: 저자 (연도). 제목. *발표처.* arXiv 식별자.
2025~2026년 문헌 중 일부는 저자·서지 정보가 확정되지 않아 제목과 식별자로 표기하며, 투고 전 원문 대조로 확정한다(†).

**인용·귀속 평가 (Citation & Attribution Evaluation)**
* Gao, T. et al. (2023). Enabling Large Language Models to Generate Text with Citations. *EMNLP 2023.* arXiv:2305.14627. — ALCE, `CitePrec`. 본 연구가 대비하는 답-기준 지표.
* Rashkin, H. et al. (2023). Measuring Attribution in Natural Language Generation Models. *Computational Linguistics 49(4).* arXiv:2112.12870. — AIS. 귀속과 진실성의 의도적 분리.
* Petroni, F. et al. (2021). KILT: A Benchmark for Knowledge Intensive Language Tasks. *NAACL 2021.* arXiv:2009.02252. — 집합형 근거 채점의 계보.
* Wallat, J. et al. (2025). Correctness is not Faithfulness in RAG Attributions. *ICTIR 2025.* arXiv:2412.18004. — 인용 충실성 문제의 일반적 규명. 본 연구는 시점 충돌 하위 부류로 특화.
* Relevant Is Not Warranted: Evidence-Force Calibration for Cited RAG (2026). arXiv:2605.28044.† — 최근접 경쟁작(일반 5축 보정). 차별점: 충돌 조건부·시점-유효 기준·오귀속률 정량화.
* GaRAGe: A Benchmark with Grounding Annotations for RAG (2025). arXiv:2506.07671.† — "Outdated" 근거 라벨(문서 관련성 수준).

**지식 충돌 (Knowledge Conflict)**
* Xu, R. et al. (2024). Knowledge Conflicts for LLMs: A Survey. *EMNLP 2024.* arXiv:2403.08319. — 문맥 간(inter-context) 충돌 = 본 연구의 명시적 충돌.
* Hou, Y. et al. (2024). WikiContradict: Evaluating LLMs on Real-World Knowledge Conflicts from Wikipedia. *NeurIPS 2024 D&B.* arXiv:2406.13805. — 명시적·암시적 충돌 구분(암시적 유형의 최근접 선례).
* Cattan, A. et al. (2025). DRAGged into Conflicts: Detecting and Addressing Conflicting Sources in Search-Augmented LLMs. arXiv:2506.08500.† — 실제 웹 충돌 분류·prevalence, `rag_conflicts` 출처.
* Contradiction Detection in RAG Systems (2025). arXiv:2504.00180.† — 충돌 탐지·유형 예측·분할(자동 분류 선례).

**시간 인식 QA 및 시점 충돌 (Temporal QA & Conflict)**
* Chen, W. et al. (2021). A Dataset for Answering Time-Sensitive Questions (TimeQA). *NeurIPS 2021 D&B.* arXiv:2108.06314. — 암시적 경계 대조 데이터 원천.
* Zhang, M. J. Q. & Choi, E. (2021). SituatedQA: Incorporating Extra-Linguistic Contexts into QA. *EMNLP 2021.* arXiv:2109.06157. — 시점 조건부 답.
* Detecting Temporal Ambiguity in Questions (TempAmbiQA) (2024). *Findings of EMNLP 2024.* arXiv:2409.17046.† — 시점 중의성 자동 판정.
* Don't Ask the LLM to Track Freshness: A Deterministic Recipe for Memory Conflict Resolution (2026). arXiv:2606.01435.† — 추출-후-규칙의 골격(깨끗한 타임스탬프 한정). 본 연구는 내용 유래 유효 구간으로 확장.
* Ho, H. et al. / HoH: A Dynamic Benchmark for Evaluating the Impact of Outdated Information on RAG (2025). arXiv:2503.04800.† — 명시적·통제 데이터 원천.

**고위험 도메인 동기 (High-stakes Motivation, 인용 전용)**
* Do LLMs Truly Understand When a Precedent Is Overruled? (2025). *JURIX 2025.* arXiv:2510.20941.† — 법률 판례 파기.
* Facts Fade Fast: Evaluating LLMs on Outdated Medical Knowledge (2025). arXiv:2509.04304.† — 폐기된 임상 지침.

**방법·백본 (Method & Backbone)**
* Qwen Team (2025). Qwen3 Technical Report. arXiv:2505.09388. — 주 백본 Qwen3-8B.
* RAG-RL: Reinforcement Learning for Answer Generation and Citation (2025). arXiv:2503.12759.† — Qwen2.5-7B 기반 GRPO 선례.
* FaithfulRAG: Fact-Level Conflict Modeling for RAG (2025). *ACL 2025.* arXiv:2506.08938.† — 8B 백본 관행.
* VeriCite (2025). arXiv:2510.11394.† — 인용 검증, 다중 백본 관행.

**표준 지표·자원 (Standard Metrics)**
* Rajpurkar, P. et al. (2016). SQuAD: 100,000+ Questions for Machine Comprehension of Text. *EMNLP 2016.* arXiv:1606.05250. — EM / Token-F1.
