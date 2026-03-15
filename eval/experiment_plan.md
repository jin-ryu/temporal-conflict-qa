# LLM 평가 실험 계획

## 연구 질문

최신 LLM이 시간적 충돌(temporal conflict)이 있는 RAG context에서 올바른 문서를 선택하고 정답을 생성할 수 있는가?

---

## 실험 조건 (3가지)

| 조건 | 키워드 | 시간 충돌 | 질문 시간 힌트 | 대상 mode |
|------|--------|----------|--------------|-----------|
| **충돌 없음** | `no_conflict` | 없음 (outdated 청크 제거) | 있음 | current, outdated_* |
| **충돌 있음** | `conflict` | 있음 (전체 청크) | 있음 | current, outdated_* |
| **충돌 + 모호** | `ambiguous` | 있음 (전체 청크) | 없음 | current_raw |

### 조건 설계 의도

- **no_conflict → conflict**: 충돌 정보(outdated 청크) 존재 여부가 성능에 미치는 영향
- **conflict → ambiguous**: 질문의 시간 힌트 유무가 성능에 미치는 영향
- **no_conflict vs ambiguous**: 충돌 + 모호성의 복합 효과

---

## 가설

- **H1**: no_conflict→conflict 성능 하락 (충돌 정보가 혼란 유발)
- **H2**: conflict→ambiguous 추가 하락 (시간 힌트 없으면 더 어려움)
- **H3**: evidence accuracy가 answer accuracy보다 더 크게 하락

---

## 모델

- **GPT-4.1-mini** (주력)
- **GPT-4.1** (예산 허용 시)

---

## 평가 지표

| 지표 | 설명 |
|------|------|
| Answer EM | exact match (정규화 후 완전 일치) |
| Answer Token F1 | 토큰 단위 precision/recall F1 |
| Evidence Accuracy | `<relevance>` 블록의 chunk id가 ground truth와 일치 |
| Combined Score | answer EM × evidence accuracy |

---

## 데이터 규모

- 입력 파일: `data/qa/hoh_qa_gpt_0_50.jsonl` — 총 151건
  - `current` / `outdated_*`: 101건 → no_conflict, conflict 조건 대상
  - `current_raw`: 50건 → ambiguous 조건 대상

---

## 파일 컨벤션

```
data/eval/eval_{model}_{condition}_{qa_stem}.jsonl       # 건별 결과 (resume 지원)
data/eval_summary/summary_{model}_{condition}_{qa_stem}.json  # 조건별 집계
```

### 실행 예시

```bash
python eval/evaluate_llm.py --input data/qa/hoh_qa_gpt_0_50.jsonl --condition no_conflict
python eval/evaluate_llm.py --input data/qa/hoh_qa_gpt_0_50.jsonl --condition conflict
python eval/evaluate_llm.py --input data/qa/hoh_qa_gpt_0_50.jsonl --condition ambiguous

python eval/summarize_eval.py  # 전체 집계 (파일별 summary 생성)
```

---

## 프롬프트 구조

LLM에게 주어지는 입력:
```
[Query] {question}

[Document 1] [modified: {timestamp}]
{chunk_text}

[Document 2] [modified: {timestamp}]
{chunk_text}
...
```

기대 출력:
```xml
<thought> 시간적 추론 과정 </thought>
<relevance> 문서 번호 (예: 3) </relevance>
<answer> 답변 </answer>
```
