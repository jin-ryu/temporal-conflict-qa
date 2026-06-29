# Exp05 — Conflict-Type Generalization (ConflictBank)

작성: 2026-06-29 · 보조 실험 — "mis-attribution이 *시간 외* 충돌에서도 생기나"

---

## 1. 동기
- exp03·04는 모두 **시간(temporal) 충돌**. → reviewer: *"틀린답에 좋은인용(mis-attribution)이 시간 충돌만의 문제냐?"*
- exp05는 **시간 외 충돌 유형(오정보·의미)에서도** mis-attribution이 발생함을 보여 *(broad) "충돌 일반의 문제"* 주장을 뒷받침.

## 2. 데이터
- **출처**: [ConflictBank](https://arxiv.org/abs/2408.12076) (NeurIPS 2024 D&B). [HF: Warrieryes/CB_qa](https://huggingface.co/datasets/Warrieryes/CB_qa)
- 한 항목 = 정답 evidence + **3유형 충돌 evidence**(misinformation/temporal/semantic) + 정답(correct_option).
- **변환**(`convert_conflictbank.py`): 유형별로 [정답 evidence] + [해당 유형 충돌 evidence] + [distractor 2개] → 충돌 컨텍스트. 유형당 40건 = **120건**.
- ⚠️ **합성**: evidence가 LLM 생성(소설체), temporal이 미래 날짜(2026~) 등 → *통제·일반화 보조용*. 현실성은 exp03/04가 담당.

## 3. 결과 (오픈 3종, frontier 스킵 — 합성·보조라 불필요)

mis-attribution = P(CitePrec=1 | EM=0) = 답 틀렸는데 표준 인용평가가 통과시킨 비율. (exp03/04와 동일 지표)

| 모델 | misinformation | temporal | semantic |
|---|---|---|---|
| Qwen3-8B | 100% (★8) | 100% (★20) | 100% (★16) |
| Qwen3-32B | 100% (★5) | 100% (★20) | 100% (★18) |
| Mistral-24B | 100% (★9) | 100% (★31) | 100% (★20) |

(EM정답률은 유형·모델별 22~88%. CitePrec은 전부 100% — 합성이라 충돌 evidence가 답을 명확히 담음.)

## 4. 핵심 발견
- **3모델 × 3유형 = 9개 조합 모두 mis-attribution 100%**: 틀린 답을 했을 때 표준 인용평가(CitePrec)가 *예외 없이* "잘 인용"이라 통과.
- → **mis-attribution은 시간 충돌 특유가 아니라 *충돌 일반*의 문제** (오정보·의미 충돌서도 동일).
- 단 CitePrec 100%는 *합성 데이터 특성*(충돌 evidence가 답을 명확히 담음) → 현실 수치는 exp03/04(81~100%)가 더 신뢰. exp05는 *"현상 존재(시간 외에도)"* 증명용.

## 5. 위치
| 실험 | 충돌 | 데이터 | 역할 |
|---|---|---|---|
| exp03 HoH | 시간 | 진짜 위키·통제 | 통제된 증명 |
| exp04 rag_conflicts | 시간 | 진짜 웹·현실 | ecological validity |
| **exp05 ConflictBank** | **시간+오정보+의미** | 합성·대규모 | **충돌유형 일반화** |

→ 시간 깊이(exp03/04) + 유형 넓이(exp05) = mis-attribution이 *통제·현실·다유형* 전반에서 발생.

## 6. TODO
- [x] open 3종(Qwen3-8B·32B·Mistral) 완료 — 9개 조합 모두 mis-attr 100%. frontier는 스킵(합성·보조).

> 채점: `scripts/eval_misattr.py` · 변환: `scripts/convert_conflictbank.py`
