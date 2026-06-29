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
- **변환**(`convert_rag_conflicts.py`, 변형 없음): 청크=search_results(short_text+date), 질문=원래(현재지향), 정답=correct_answer(최신), evidence=정답 담은 *가장 최신* 문서. → 모델이 *outdated 문서*를 인용/사용해 틀리면 ★.
- **필터**: evidence 명확(정답문서 ≤3 + outdated 문서 존재) → **22개** 채택.
- mode=current(현재지향). 청크 평균 9.2개.

## 3. 결과 (n=22, 무료 결정론 채점)

**지표는 exp03과 동일** (`03_temporal_validity/scripts/03_evaluate.py` 그대로 사용 — 별도 채점기 없음).
β = blind_spot_rate = P(CitePrec=1 | TV_cite=0) = 틀린시점 인용 중 표준평가가 '정상' 통과시킨 비율.

| 모델 | TV_cite | wrong_time | CitePrec | **β (비가시)** | ★ 셀 |
|---|---|---|---|---|---|
| Qwen3-8B | 0.273 | 0.727 | 0.864 | **0.813** | 13 |
| Qwen3-32B | 0.500 | 0.500 | 0.909 | **0.824** | 9 |
| Mistral-Small-24B | 0.227 | 0.773 | 0.909 | **0.882** | 15 |

> 이전 버전에서 보고한 "★ 59/41/68%"는 ★/n(전체 분모) 표기였음. exp03과 통일하기 위해 **β(틀린시점 인용 분모)**로 재보고 — 결론(★ 실재·표준평가 비가시)은 동일.

## 4. 핵심 발견

### F1 — 현실 웹에서도 ★ 실재 (인위적 난이도 아님)
- 틀린시점 인용의 **β=81~88%**가 표준평가에 비가시 (3 open 모델, Qwen·Mistral 패밀리 불문) — exp03(β 91~100%)과 동급.
- → "HoH의 ★는 인위적으로 어려운 셋(거의 동일한 긴 문단) 때문 아니냐"는 의심 **반박**. 진짜 웹에서도 틀린시점 인용이 다수(wrong_time 0.50~0.77)이고 그 대부분이 비가시.

### F2 — 양방향 실패 (recency 편향의 대칭)
- exp03(as-of-past): 과거 질문 → *최신*으로 끌림.
- exp04(현재지향): 현재 질문 → *outdated 문서에 속음* (예: "current CIA director"에 전임자, "this year's Ramadan"에 작년 날짜).
- → 충돌은 *두 방향 모두* 인용 실패를 부르고, **표준 인용평가(CitePrec 86~91%)는 둘 다 못 잡음**.

### F3 — 모델 규모
- 32B가 8B보다 robust(EM 23→45%, ★ 59→41%)이나, **큰 모델도 ★ 41%** = 여전히 큼.

## 5. exp03 대비 위치
| | exp03 HoH | exp04 rag_conflicts |
|---|---|---|
| 성격 | 통제·as-of-past | 현실 웹·현재지향 |
| 난이도 | (인위적 우려) | 자연스러움 |
| ★ (Qwen3-8B) | β 98% | 59% |
| 역할 | 통제된 증명 + 깨끗한 counterfactual | **ecological validity** (현실서도 ★) |

→ **두 셋·두 방향·여러 모델에서 ★ 일관** = 인용평가 시점 맹점이 *현실적·보편적* 문제임을 입증.

## 6. 한계 / TODO
- n=22 (모호한 evidence 31개 제외). 정제·확대 여지.
- EM 낮음(23~45%) — 일부는 evidence 매핑 모호(정답이 여러 문서 분산) 영향 가능 → 매핑 정밀화 검토.
- [ ] Mistral / frontier(GPT·Gemini) 추가 — "프론티어도 현실서 ★" 확인.
- [ ] faithfulness 분류·지표 일반성(03의 05·06)을 exp04에도 적용.

> 측정 스크립트: `scripts/eval_star.py` · 변환: `scripts/convert_rag_conflicts.py` · exp03: `../03_temporal_validity/RESULTS.md`
