# Exp04 — Real-World Conflict Results (rag_conflicts)

작성: 2026-06-29 · 현실 웹 검색 충돌, 현재지향(자연 질문), as-of-past 변형 없음

---

## 1. 동기 — exp03(HoH)의 두 약점을 보완
| 약점 (exp03 HoH) | exp04가 푸는 방식 |
|---|---|
| **as-of-past가 마이너** ("2024년 6월 기준…" 인위적 질문) | rag_conflicts는 **자연스러운 현재 질문**("who is the director of the FBI?") 그대로 사용 |
| **난이도가 인위적** (옛/새 청크가 거의 동일한 긴 위키 문단) | **진짜 웹 검색결과** — 다양한 출처·날짜, 자연스러운 충돌 |

→ exp03이 "통제된 증명"이면, exp04는 "현실에서도 ★가 실재하나"의 검증.

## 2. 데이터
- **출처**: Google **rag_conflicts** (Cattan et al., "DRAGged into CONFLICTS", 2025), `conflict_type == "Conflict due to outdated information"` 62개.
- **변환**(`convert_rag_conflicts.py`, 변형 없음): 청크=search_results(short_text+date), 질문=원래(현재지향), 정답=correct_answer(최신). → 모델이 *outdated 문서*를 인용/사용해 틀리면 ★.
- **필터**: 정답 담은 문서 ≥1 + outdated(정답 안 담은) 문서 ≥1 (충돌 존재) → **51개** 채택. (정답문서 없음 6, 충돌날짜 부족 3, outdated 없음 2 제외.)
- mode=current(현재지향). 청크 평균 9.2개.

## 3. 결과 (n=51, 무료 결정론 채점)

**채점**(`scripts/evaluate_rc.py`): exp04는 정답(최신값)이 *여러 문서*에 담길 수 있고 모델이 인용을 1개만 뽑으므로, evidence_chunk_id(단일 지정) 대신 *내용 기준*으로 시점유효성 판정:
- **TV_cite** = 1[인용 청크가 정답(최신값)을 담음] = 맞는 시점(최신) 문서 인용
- **CitePrec** = 1[인용 청크가 모델 *답*을 뒷받침] (시점 무관, 표준)
- **★** = TV_cite=0 & CitePrec=1, **β** = P(CitePrec=1 | TV_cite=0)

| 모델 | 분류 | TV_cite | wrong_time | CitePrec | **β (비가시)** | ★ 셀 |
|---|---|---|---|---|---|---|
| GPT-5.5 | frontier | 0.784 | 0.216 | 0.922 | **0.727** | 8 |
| Gemini 3.1 Pro | frontier | 0.725 | 0.275 | 0.902 | **0.818** | 9 |
| Qwen3-32B | open | 0.745 | 0.255 | 0.902 | **0.769** | 10 |
| Mistral-Small-24B | open | 0.686 | 0.314 | 0.863 | **0.625** | 10 |
| Qwen3-8B | open | 0.706 | 0.294 | 0.941 | **0.929** | 13 |

> **채점 교정 이력**: 초기엔 evidence 단일문서 일치(TV_cite)로 채점 → 정답이 여러 문서에 있을 때 *다른 정답문서 인용*을 틀림 처리해 wrong_time 과대(0.73). 정답 *내용* 기준으로 교정(인용문서가 정답 담았나) → wrong_time 0.26~0.31로 정확화. **교정 후에도 ★ 10~13건, β 62~93% 유지** (과장 없이 실재).
> 데이터: 초기 깨끗필터(정답문서≤3) 22건 → "정답이 여러 문서에 분산"은 충돌 무효가 아님을 반영해 필터 완화(outdated 문서 1개+ 존재) → **51건**.

## 4. 핵심 발견

### F1 — 현실 웹에서도 ★ 실재, 프론티어 포함 (인위적 난이도 아님)
- 틀린시점 인용의 **β=62~93%**가 표준평가에 비가시 (**5모델** — frontier GPT·Gemini + open Qwen·Mistral) — exp03(β 91~100%)과 동급.
- **GPT-5.5(최강)조차 wrong_time 22%, ★ 8건, β 73%** — 현실 웹 충돌서 옛 문서 인용 + 표준평가 비가시.
- 현실 웹·현재 질문에서도 wrong_time 22~31%, 그 대부분이 표준평가 통과(★ 8~13건).
- → "HoH의 ★는 인위적으로 어려운 셋 때문 아니냐"는 의심 **반박**. + 노이즈(CAPTCHA·에러 페이지 등 실제 검색결과 그대로 포함)까지 있는 현실 조건.

### F2 — 양방향 실패 (recency 편향의 대칭)
- exp03(as-of-past): 과거 질문 → *최신*으로 끌림.
- exp04(현재지향): 현재 질문 → *outdated 문서에 속음* (예: "current CIA director"에 전임자, "this year's Ramadan"에 작년 날짜).
- → 충돌은 *두 방향 모두* 인용 실패를 부르고, **표준 인용평가(CitePrec 86~91%)는 둘 다 못 잡음**.

### F3 — 모델 규모·패밀리
- wrong_time이 모델별로 26~31%로 비슷, ★ 10~13건 일관. β는 8B(0.93) > 32B(0.77) > Mistral(0.62) — 표준평가 비가시율은 모델별 차이 있으나 모두 높음.

## 5. exp03 대비 위치
| | exp03 HoH | exp04 rag_conflicts |
|---|---|---|
| 성격 | 통제·as-of-past | 현실 웹·현재지향 |
| 난이도 | (인위적 우려) | 자연스러움 |
| β (Qwen3-8B) | 98% | 93% |
| 역할 | 통제된 증명 + 깨끗한 counterfactual | **ecological validity** (현실서도 ★) |

→ **두 셋·두 방향·여러 모델에서 ★ 일관** = 인용평가 시점 맹점이 *현실적·보편적* 문제임을 입증.

## 6. 한계 / TODO
- n=51. 정제·확대 여지.
- 5모델 완료(open 3 + frontier 2).
- [x] frontier(GPT·Gemini) 완료 — 둘 다 ★ 실재(β 73~82%).
- [ ] faithfulness 분류·지표 일반성(03의 05·06)을 exp04에도 적용.

> 측정 스크립트: `scripts/evaluate_rc.py` · 변환: `scripts/convert_rag_conflicts.py` · exp03: `../03_temporal_validity/RESULTS.md`
