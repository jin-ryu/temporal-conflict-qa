# TQA(Temporal Wiki) → 충돌 평가셋 변환 (`data_prep/tqa`)

실험03의 **robustness 재현용 보조** 데이터셋. 주력은 `data_prep/timeqa`(peer-review),
이쪽은 **TQA/Temporal Wiki**로 같은 ★결과를 한 번 더 재현하는 용도다.
TimeQA와 마찬가지로 **LLM 생성 없음** — 옛/새 시점 문단을 충돌 구조로 재조립.

> ⚠️ **신빙성 주의**: TQA는 **arXiv preprint**(미peer-review)이고 레포가 *데이터 덤프 + 3줄
> README*뿐이라(가공 코드 비공개), 메인 베이스가 아니라 **보조**로만 쓴다.

## 출처 · 라이선스 · 받는 법
- **TQA / Temporal Wiki** (Özer & Yıldız, *Question Answering under Temporal Conflict*,
  **arXiv 2506.07270**, 2025, **MIT**) — github.com/atahanoezer/TQA
- 받기: 레포 clone 후 `full_data_filtered/`(878개 엔티티 JSON)를 `data/tqa/source/`로
  복사 (해당 폴더는 `.gitignore`).

## 입력 포맷 (`source/*.json`, 엔티티당 1파일)
```
{ event_id,
  incidents: { "<연도>": { question, answer:[{wikidata_id,name}],
                           dump:{body_par, infobox}, url(revision permalink) }, ... } }
```
- `incidents[연도]` = 그 시점의 연도형 질문 + 결정론적 답 + **그 시점 위키 revision 문단**(`body_par`).
- TimeQA와 달리 **gold 근거 문단의 명시적 라벨이 없다** → 답이 문단에 실재하는지 우리가 *추론*해야 함(아래 게이트).

## 변환 절차 (`tqa_to_qa.py`, 결정론적·무생성)
1. **옛/새 쌍 선택** — 새 = 가장 최근 연도, 옛 = 답이 다른 가장 이른 연도.
2. **청크 구성** — 각 연도 `body_par`를 정리(인라인 CSS `.mw-parser-output{...}` 제거)해 청크로.
   옛=`outdated_0`, 새=`current`, `last_modified_time`=연도.
3. **질문·라벨** — 질문은 **TQA 원본**(명시적 연도형 "…in 2010?"), 정답=TQA 답,
   `evidence_chunk_id`=그 연도 청크.

## 품질 필터 (제외 사유)
| 사유 | 의미 |
|---|---|
| `no_drift_pair` | 답이 다른 연도쌍 없음(변화 X) |
| `team_ambiguous` | 스포츠 클럽↔국가대표/유스 혼합 |
| `evidence_mismatch` | **답이 근거 문단에 없음** — TQA엔 근거 라벨이 없어 *변별 토큰 매칭*으로 추론(정식명↔약칭 흡수, generic 역할단어 제외). **이 게이트가 ~30% 제거**(TimeQA는 사람-라벨이라 거의 0). |

## 출력 (`data/tqa/qa_tqa.jsonl`)
스키마는 TimeQA와 동일(→ `01_sample_eval_set.py` 공용):
```json
{"id","source_idx","mode":"outdated_0|current","new_question","target_answer",
 "evidence_chunk_id","chunks":[{"chunk_id","label","text","last_modified_time"}]}
```
규모: **454 엔티티 → 908 레코드**.

## TimeQA와의 차이 (요약)
| | TimeQA (주력) | TQA (보조) |
|---|---|---|
| 게재 | NeurIPS'21 peer-review | arXiv preprint |
| 근거 시점 라벨 | **사람-라벨**(para idx+span) | 없음 → 변별토큰으로 **추론**(게이트) |
| 규모 | 8360 레코드 | 908 레코드 |
| 질문 | relations.json 템플릿 | TQA 원본 질문 |

## 실행
```bash
python3 data_prep/tqa/tqa_to_qa.py            # 전체
python3 data_prep/tqa/tqa_to_qa.py --max 100   # 100 엔티티(테스트)
```

## 인용
질문·정답·근거의 원천이므로 **TQA/Temporal Wiki(Özer & Yıldız, arXiv 2506.07270)** 인용.
