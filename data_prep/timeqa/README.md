# TimeQA → 충돌 평가셋 변환 (`data_prep/timeqa`)

실험03(Temporal Validity)의 **충돌-grounding 평가셋**을 만들기 위해 **TimeQA**를
재조립한다. **LLM 생성 없음** — 질문·정답·근거가 전부 TimeQA에서 오고, 우리는
"옛/새 두 시점을 한 컨텍스트에 놓는 충돌 구조"로 **재구성**만 한다.
(대조: `data_prep/hoh`는 LLM으로 질문을 *생성*했음. 여기는 *재조립*뿐.)

## 출처 · 라이선스 · 받는 법
- **TimeQA** (Chen, Wang, Wang, *A Dataset for Answering Time-Sensitive Questions*, **NeurIPS 2021 Datasets&Benchmarks**) — github.com/wenhuchen/Time-Sensitive-QA
- 받기: 레포 clone 후 `dataset/annotated_*.json` 과 `relations.json` 을
  `data/timeqa/source/` 로 복사 (해당 폴더는 `.gitignore`).

## 입력 포맷 (`annotated_*.json`)
엔티티당 1 레코드:
```
{ index, type(Wikidata 관계 Pxx), link,
  paras:    [문단 문자열, ...],                       # 위키 페이지 문단들
  questions:[ [ [time_start, time_end],               # 시점 구간
                [ {para, from, end, answer}, ... ] ]  # 정답 + gold 근거 문단 idx + 글자 span
              , ... ] }
```
- `questions`의 각 항목 = **한 시점 구간의 정답 + 그 근거 문단**(crowd-worker 라벨).
- `relations.json`: 관계 Pxx → 질문 템플릿 (예 P54 → `"Which team did $1 play for $4?"`,
  `$1`=주체, `$2/$4`=시점).

## 변환 절차 (`timeqa_to_qa.py`, 결정론적·무생성)
1. **유효 시점 추출** — 단일 비공백 답 + 시작연도가 있는 시점만 사용.
2. **옛/새 쌍 선택** — 새 = 가장 늦은 시점, 옛 = *답이 다른* 가장 이른 시점(실제 변화 보장).
3. **청크 구성** — 각 시점의 **gold 문단**(`paras[para]`)을 청크로. 옛=`outdated_0`,
   새=`current`. *근거가 사람-라벨이라 추론·게이트 불필요.*
4. **질문** — `relations.json` 템플릿 + 엔티티명 + `"in {시작연도}"` (**명시적 연도형**).
5. **정답·라벨** — `target_answer` = TimeQA 주석 답, `evidence_chunk_id` = 그 시점 청크.

엔티티당 **2 레코드**(`outdated_0` 과거지향 + `current` 현재지향 대조군)를 만들고,
둘은 **같은 충돌 청크쌍**[옛=outdated_0, 새=current]을 공유한다.

## 품질 필터 (제외 사유 카운트로 출력)
| 사유 | 의미 |
|---|---|
| `no_drift_pair` | 답이 다른 시점쌍이 없음(실제 변화 X → 충돌 아님) |
| `team_ambiguous` | 스포츠 클럽↔국가대표/유스 혼합("which team" 단일정답 모호) |
| `evidence_mismatch` | 안전망 — 답이 근거 문단에 없음(주석 노이즈 방어, 사람-라벨이라 거의 0) |
| `bad_para` / `empty_evidence` | 문단 인덱스 범위 밖 / 빈 문단 |
| `dup_entity` | 분할(train/dev/test) 간 동일 엔티티 중복 |

## 출력 (`data/timeqa/qa_timeqa.jsonl`)
`01_sample_eval_set.py`가 소비하는 스키마:
```json
{"id","source_idx","mode":"outdated_0|current","new_question","target_answer",
 "evidence_chunk_id","chunks":[{"chunk_id","label":"outdated_0|current","text","last_modified_time"}]}
```
규모: **4180 엔티티 → 8360 레코드** (사람 검수 통과 ~96%).

## 우리가 더한 유일한 "판단"
질문의 시점을 구간 **시작연도**로 고정(`in YYYY`). 그 연도에 그 답이 유효하므로
결정론적이다. 그 외 질문 형식·정답·gold 근거는 **전부 TimeQA에서 유래**.

## 실행
```bash
python3 data_prep/timeqa/timeqa_to_qa.py                 # 전체(annotated_*.json)
python3 data_prep/timeqa/timeqa_to_qa.py --split dev      # dev만
python3 data_prep/timeqa/timeqa_to_qa.py --max 200        # 200 엔티티(테스트)
```

## 인용
질문·정답·근거의 원천이므로 **TimeQA(Chen et al., NeurIPS 2021)** 를 반드시 인용.
우리 기여는 *시간-유효 근거가 라벨된 사실로부터 충돌 컨텍스트를 구성*한 것.
