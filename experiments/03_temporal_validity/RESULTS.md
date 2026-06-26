# Main Experiment Results — Temporal Validity of RAG Citations

작성: 2026-06-25 · 5 models × 219 entities (as-of-past n=180, current control n=39) · 무료 결정론 채점(proxy, ALCE-NLI 근사)

---

## 1. Setup
- **Task**: 검색 컨텍스트에 같은 사실의 *옛/새 버전*이 공존(conflict). *as-of-past* 질문에 *시점-유효* 문서를 인용·사용하는지 측정.
- **Conditions** (per item): `conflict`(옛+새+distractor) · `outdated_only`(옛만) · `current_only`(새만).
- **Prompt**: 각 문서에 `[modified: τ]` 타임스탬프 + 질문에 `As of {월 년}` + 시스템지시("시점-유효 문서 골라 인용") *모두 제공* → 과제는 사람-풀이 가능·공정.
- **Data**: HoH(ACL 2025) 파생, 무결성+구성(genuine 시간변화 vs 정정) 수동 필터 219 entity.
- **Models**: GPT-5.5, Gemini 3.1 Pro (frontier) / Mistral-Small-24B, Qwen3-32B, Qwen3-8B (open, vLLM). *Llama 계열은 데이터 생성자라 제외.*

## 2. Metrics
- **EM**: conflict 답 == 시점-유효 정답 (정규화).
- **CitePrec** (foil, ALCE식): 인용 문서가 답을 뒷받침하나 — *시점 무관*.
- **TV_cite** (신규): 시점-유효 문서(evidence chunk)를 인용했나.
- **TV_behav** (신규, counterfactual): conflict 답이 행동상 어느 시점 문서에 좌우되나 (옛/새 단일조건 비교).
- **★ blind-spot β** = P(CitePrec=1 | TV_cite=0): 틀린시점 인용 중 표준 통과 비율.
- **Conflict-drop Δ** = Acc(outdated_only) − Acc(conflict): 충돌이 유발한 순수 성능 하락.

## 3. Headline Table (as-of-past, n=180)

| Model | Class | outdated_only | conflict | **Δ drop** | wrong_time | CitePrec | **β (invisible)** |
|---|---|---|---|---|---|---|---|
| GPT-5.5 | frontier | 90.6% | 80.6% | −10.0pp | 0.183 | 1.000 | **100%** |
| Gemini 3.1 Pro | frontier | 85.6% | 67.2% | −18.3pp | 0.306 | 0.972 | 90.9% |
| Mistral-Small-24B | open | 84.4% | 66.1% | −18.3pp | 0.267 | 0.972 | **100%** |
| Qwen3-32B | open | 87.8% | 50.0% | −37.8pp | 0.478 | 0.994 | 98.8% |
| Qwen3-8B | open | 81.1% | 40.0% | −41.1pp | 0.561 | 0.983 | 98.0% |

## 4. Findings

### F1 — ★ blind-spot은 보편적이고 강력하다 (핵심)
- 5개 모델 전부 **CitePrec ∈ [0.97, 1.00]** (표준 인용평가는 만점에 가까움).
- 그러나 틀린시점 인용의 **β ∈ [91%, 100%]** 가 표준평가에 *비가시*.
- **GPT-5.5·Mistral은 β=100%** — 틀린시점 인용을 표준평가가 *단 하나도* 못 잡음.
- → `CitePrec = 1 ⇏ TV_cite = 1`. temporal validity는 correctness와 *직교*.

### F2 — 충돌이 멀쩡한 모델을 망가뜨린다 (무능 아님)
- 모든 모델이 `outdated_only`(시점-유효 문서만)에선 **81~91%** 정답 → *지식·능력 문제 아님*.
- conflict에서 떨어짐(Δ −10 ~ −41pp) = *충돌이 유발한 실패*.
- 타임스탬프·시점지시를 *전부 쥐고도* 실패 → recency-pull이 명시적 시점지시를 이김.

### F3 — 충돌 견딤은 능력·크기·open여부로 예측 안 되는 *모델별 축*
- **Mistral-24B(open) ≈ Gemini(frontier)** (Δ 둘 다 −18.3pp, β 100% vs 91%) → *"open이라 나쁘다" 반증*.
- 같은 크기여도 Qwen3-32B(−38pp) ≫ Mistral-24B(−18pp) → *모델 고유 특성*.
- 대체로 소형일수록 큰 낙폭(8B −41pp)이나 단조 아님.

### F4 — 대조군(current, n=39)
- 현재 시점이 정답인 경우 wrong_time이 현저히 낮음(frontier ~0.1) → 실패가 *과거 시점에 특정* = recency성 시간 grounding 실패(랜덤 노이즈 아님).

### F5 — 파일럿(n=48) → 본실험(n=180) 안정성
- 핵심 지표(wrong_time, CitePrec, β) 모두 ±2~3pp 이내 유지 → 신호는 우연 아님, 신뢰구간 축소.

## 4b. Position Ablation — 충돌 실패의 메커니즘 (n=180)
시점-유효(evidence) 청크를 컨텍스트 **맨앞(first) / 맨뒤(last)** 에 강제 배치해 위치 효과 측정. Δ = TV_cite(first) − TV_cite(last). 클수록 위치 의존(모델이 *맨뒤 문서*를 고름 = in-context recency).

| Model | Δ n=48 (pilot) | **Δ n=180** | 위치 편향 |
|---|---|---|---|
| Mistral-24B | −0.42 | **−0.41** | 강함 |
| Qwen3-32B | −0.33 | **−0.18** | 중간 (pilot 과대평가) |
| Qwen3-8B | −0.04 | **−0.07** | 거의 없음 |

- **위치 편향은 모델별 스펙트럼** — 보편적 recency 아님. (Mistral 강 / Qwen32B 중 / 8B 무).
- **충돌 실패는 위치로 환원 불가**: Qwen3-32B는 conflict-drop −38pp인데 위치효과 −18%, Qwen3-8B는 drop −41pp인데 위치효과 −7% → *낙폭의 상당 부분은 위치 아닌 시간추론 실패*.
- → **메커니즘 = in-context position bias(일부 모델) + temporal-reasoning failure(공통)의 혼합.**
- **해결법 함의**: 위치 기반 리랭킹(evidence를 맨뒤로)은 모델특정·부분적 → *시점-유효 청크 selection(틀린시점 제거)이 robust*.

## 4c. Faithfulness 분류 — Wallat(2025)과 구별되는 실패 범주 (핵심)
틀린시점 인용(TV_cite=0) 항목을, **문서 제거 counterfactual**(conflict 답 vs outdated_only/current_only 답, 추가 호출 0)로 분류:
- **faithful-wrong-time**: conflict 답이 *새 문서만* 조건 답과 같고 *옛 문서만* 과 다름 → 모델이 새(틀린시점) 문서에 *행동 의존* = 진짜 썼는데 시점 틀림.
- **post-rationalization 의심**(Wallat): 새 답이 어느 단일조건과도 불일치 → 문서 없이 그 답 = parametric/사후인용.
- **invariant**: 두 단일조건 답 동일 → 반사실 판정 불가(데이터 한계).

| Model | wrong_time | **faithful-wrong-time** | post-rat | invariant |
|---|---|---|---|---|
| GPT-5.5 | 33 | **76%** | 0% | 24% |
| Gemini | 55 | **64%** | 0% | 35% |
| Mistral-24B | 48 | **71%** | 12% | 15% |
| Qwen3-32B | 86 | **81%** | 1% | 17% |
| Qwen3-8B | 101 | **74%** | 7% | 15% |

- **5모델 전부 64~81%가 faithful-wrong-time** — 모델이 틀린시점 문서를 *실제로 사용*(제거 시 답 변화). 인용이 장식 아님.
- **post-rationalization은 0~12%** — 우리 실패의 주범 아님 → **Wallat의 범주와 명확히 구별.**
- → **핵심 주장**: 본 실패는 *correct(답 뒷받침)·faithful(진짜 사용)인데 temporal만 틀림*. 표준 correctness(ALCE)도, Wallat의 faithfulness도 못 잡는 **제3의 독립 축**. (Wallat이 future work로 남긴 "내용변경 counterfactual"을 시간충돌로 충족.)

## 5. Position vs Prior Work
- **temporal validity = 인용품질의 제3축** — correctness(ALCE, Gao 2023) · faithfulness(Wallat 2025, ICTIR)와 직교.
- 본 실패는 *correct·faithful해 보이는데(TV_behav 0.42~0.75로 문서를 실제 사용)* **시점만 틀림** → Wallat의 post-rationalization으로 설명되지 않는 *별도 범주*.
- HoH(ACL 2025)는 답 정확도 하락만 측정, 인용평가 맹점은 미다룸.
- 시간-인식 RAG(TimeR4·T-GRAG·VersionRAG 등)는 *해결법*(검색·리랭킹) — 본 연구는 *측정/진단*에 집중.

## 6. Cost & Reproducibility
- frontier 누적: GPT $10.0 / Gemini $4.6 (resume로 파일럿 재사용). open·채점(proxy)은 무료.
- 채점은 **유료 LLM judge 없이** 결정론적 포함검사(ALCE-NLI 근사) → 완전 재현 가능.

## 7. TODO
- [x] **위치 ablation 재실행 (n=48 → 180)** — open 3모델 완료(§4b): 위치 편향 모델별, 충돌 실패는 위치로 환원 불가.
- [x] **faithfulness 분류 분석** — 완료(§4c): 64~81% faithful-wrong-time, post-rat 0~12% → Wallat과 구별되는 범주 입증.
- [ ] 다른 attribution 지표(citation-recall, AutoAIS/TRUE NLI)도 동일 맹점 보이나 — ★ 일반화.
- [ ] 데이터 300~500 확장(필요시 LLM 분류기 + 골드 검증).

> 포지셔닝·수식: `../../docs/방향성_보고_TemporalValidity.md` · 지표 상세: `지표_설명.md`
