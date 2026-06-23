# 파일럿 실험 계획 — 근거 시점 유효성(Temporal Validity)

연계 문서: `연구_포지셔닝_TemporalValidity.md`
본 문서만으로 구현 가능하도록 데이터·프롬프트·지표·산출 스키마를 모두 명시한다.

---

## 1. 목적

세 주장을 최소 비용으로 실증한다(모두 측정·진단, 개선 방법론은 범위 밖).

| # | 주장 |
|---|---|
| **C1** | 최신 LLM조차 시간 충돌 시 *틀린 시점의 근거*를 인용해 답한다 |
| **C2** | 분야 표준 citation 평가(정답률·citation precision)는 이 오류를 *구조적으로 포착하지 못한다* |
| **C3** | 새 지표 **Temporal Validity(TV)** 를 도입하면 그 오류를 포착할 수 있다 |

> **초점 — as-of-past**: 본 파일럿은 *과거 시점이 정답 근거가 되는* 케이스를 주력으로 한다. 이것이 선행연구(특히 GaRAGe, Findings of ACL 2025)가 비워둔 유일한 칸이기 때문이다(포지셔닝 §5.3). 현재-지향은 *recency-bias 대조군*으로만 포함한다(§3.4).

### 실험 한눈에

- **무엇을**: 시간 충돌에서 모델이 *틀린 시점 근거를 인용*하는지(C1), 표준 지표가 이를 놓치는지(C2), 신규 지표 TV가 잡는지(C3)를 측정.
- **모델**: 파일럿 최소 = **서로 다른 lab의 frontier 2종** (예: **Gemini 3 Pro + Claude Opus 4.8** — 둘 다 frontier·저렴). 경계 결과 시 **GPT-5.5** 추가(3종). *(오픈 Llama-3.3-70B는 선택)*
- **데이터**: **동일한 50문항**(as-of-past 40 + current 대조 10)을 모델마다 × **3조건**(conflict / current-only / outdated-only)으로 실행.
- **규모·비용**: 모델당 50×3 = **150콜**, 3종 ≈ 450콜 + judge ≈ **총 $5.5**.
- **본다**: 모델별 `wrong_time_cite_rate`(as-of-past) + 2×2 ★셀(TV=FAIL & CitePrec=PASS).

---

## 2. 가설 (사전 등록)

- **H1**: 충돌 조건에서 프론티어 모델의 *틀린 시점 인용 비율*이 유의하게 높다(인용 기준·행동 기준 양쪽).
- **H2**: 모델이 틀린 시점 문서를 인용한 사례 중 상당수를 표준 citation precision이 "정확한 인용(pass)"으로 판정한다(= 맹점).
- **H3**: TV는 H2의 맹점 사례를 0점으로 식별하며, 충돌 조건에서 정답(EM)과의 정합성이 citation precision보다 높다.

---

## 3. 데이터셋

### 3.1 요구 조건
(a) 현재·과거 버전이 한 컨텍스트에 **공존**(충돌), (b) **목표 시점 결정론적**(단일 정답), (c) **시점-유효 근거 청크가 라벨로 지정**(핵심 병목), (d) 확장 가능 규모.

### 3.2 기성 데이터셋 조사

| 데이터셋 (ID) | 충돌 (a) | 시점-유효 근거 라벨 (c) | 단일 정답 (b) | 규모/도메인 | 적합 |
|---|---|---|---|---|---|
| WikiContradict (NeurIPS'24, 2406.13805) | ○ | ✕ (기대=충돌 드러내기) | ✕ | 253, 위키 | 현실성 참조 |
| TempRAGEval | △ | △ (gold evidence는 있으나 버전 라벨 아님) | ○ | 시간민감 QA | 보조 |
| ChronoQA (2508.12282) | △ | ✕ | ○ | 5,176, 뉴스 | 답 중심 |
| ConflictBank (2408.12076) | ○(합성) | △ | ○ | 합성 | 합성 한계 |
| DRAGged CONFLICTS (2506.08500) | ○ | ✕ | △ | temporal 62, 웹 | 소규모 |
| HoH(2503.04800) + 자체 가공 | ○ | △ (위키 편집 diff — 정정/오류 혼입) | ○ | 위키 | legacy(보류) |
| TQA / Temporal Wiki(2506.07270) MIT | ○ | ○ (연도별 revision 문단+답) | ○ | 878 엔티티, 위키 | 보조(robustness·preprint) |
| **★ TimeQA (Chen+ NeurIPS'21 D&B)** | **○** | **○ (사람-라벨 gold 문단+span)** | **○** | **20K·5.5K facts, 위키** | **★ 주력 채택(peer-review)** |

→ **데이터 출처 변경 (HoH → TimeQA 주력 + TQA 보조)**: HoH는 시점-유효 근거 라벨이 있으나 그 "outdated"가 위키 *편집 diff*라 **세상의 변화와 오류정정이 섞인다**(파일럿 검수 ~50% 결함, as-of-past 아님). → **TimeQA**(NeurIPS'21 peer-review)는 Wikidata 시변사실을 위키에 정렬하고 **시점별 정답 + gold 근거 문단을 사람이 라벨**해 제공 → 진짜 세상변화 + *시점-유효 근거가 이미 라벨됨*(요구 c 직접 충족, 추론 게이트 불필요). 검수 통과 ~96%. **신빙성**: TimeQA는 peer-review·처리코드 공개라 메인 베이스로 적합. **TQA**(Özer 2025, preprint·pre-paired 문단)는 같은 결과를 한 번 더 재현하는 **robustness 보조**로 둔다. 현실성(다출처)은 WikiContradict로 보강.

### 3.3 TimeQA 기반 변환 (주력) · TQA 보조

**TimeQA annotated** = 엔티티마다 위키 페이지 문단(`paras`) + 시점 구간별 {정답 + gold 근거 문단 인덱스(`para`)+span}을 **사람이 라벨**. 변환(`data_prep/timeqa/timeqa_to_qa.py`, **LLM 없음·무료**):

- **Step 1 — 옛/새 시점 선택**: 단일 비공백 답을 가진 시점들 중 새=가장 늦은, 옛=답이 다른 가장 이른 시점. 각 시점의 **gold 문단(`paras[para]`)** 을 청크로 — 옛=`outdated_0`, 새=`current`. `last_modified_time`=시작연도.
- **Step 2 — 질의·라벨**: 질문은 `relations.json` 템플릿(관계 Pxx + 엔티티 + "in YYYY", **명시적 연도형**). target=해당 시점 답, `evidence_chunk_id`=그 시점 청크. *근거가 사람-라벨이라 시점-유효 근거가 보장됨.*
- **Step 3 — 게이트**: 스포츠 클럽/국대 모호 제외, 근거-답 안전망 검사. *(보조 TQA: `data_prep/tqa/tqa_to_qa.py` — `incidents[연도].body_par`를 옛/새 청크로, 근거 변별토큰 게이트로 필터. 상세 동일.)*
- **Step 3 — 검증**: (자동) **근거 게이트** — 답의 변별 토큰이 그 시점 문단에 실재해야 통과(정식명↔약칭 흡수, generic 역할단어 제외; 미통과 시 제외). **(수동) 표본을 사람이 확인**: 시점 결정성 + 자연스러움 + 스포츠 클럽/국대 모호·가공인물 배제(파일럿 검수 기준 ~90% 통과).
- **레코드 스키마**:
  ```json
  {"id","source_idx","mode":"current|outdated_0","new_question","target_answer",
   "evidence_chunk_id","chunks":[{"chunk_id","label":"current|outdated_0","text","last_modified_time"}]}
  ```

### 3.4 파일럿 서브셋 (확정 — 경향성 파악 최소)

**총 50문항 = `outdated_i`(과거 지향) 40 + `current`(대조군) 10.** mode별 개수는 `01_sample_eval_set.py --quota`로 지정한다(기본 `outdated=40,current=10`; 세분화 예 `outdated_0=20,outdated_1=15,current=10`).

**mode별 역할:**

- **`outdated_0` (과거 지향) — 주력(기여)**: GaRAGe가 비운 칸(포지셔닝 §5.3). wrong-time 인용 비율 + 2×2 ★셀의 신호원(wrong-time ~30%여도 TV=FAIL ≈12건). *(TimeQA·TQA 모두 엔티티당 옛/새 1쌍 = 단일층 outdated_0)*
- **`current` (현재 지향) — recency-bias 대조군** *(기여가 아니라 대조)*:
  - ① **실패가 과거-지향 특유임을 격리** — current는 TV 높고 past는 낮으면 "grounding 능력은 있는데 *과거일 때만* 무너진다"가 증명됨(일반적 무능 아님).
  - ② **최신 편향 입증** — current는 최신=정답이라 잘 맞히고, past에서 그 편향이 *오답으로* 드러남 → 체계적 recency bias.
  - ③ **sanity check**. 소수(10)면 충분(통계 파워가 아니라 방향성 대비용).
- **`current_raw` (원질문) — 제외**: 시간 신호가 없어 현재/과거 둘 다 정답이 되어 모호.

각 항목: `current` 청크 ≥1 + `outdated` 청크 ≥1 **공존**, 목표 시점 결정론적, `source_idx`(엔티티) 중복 금지(leakage 방지). 경계 결과 시 100+로 확장(TimeQA 변환 풀 **~4180 엔티티**, TQA ~454).

### 3.5 데이터 변환 실행 (LLM 없음 · 무료)

질문은 **TQA 원본**에서 나오므로 생성 모델이 필요 없다(`data_prep/tqa/tqa_to_qa.py`는 순수 변환). 편향 분리 대상은 **judge ≠ 테스트** 둘뿐이다.

**역할 배정:**

| 역할 | 주체 | 이유 |
|---|---|---|
| 질문(원천) | **TimeQA**(위키 기반) | 모델 생성 아님 → 생성 편향·비용 0 |
| judge | **Gemini 3.1 Pro** | 저렴($2/$12), 테스트와 분리 |
| 테스트(답변) | **GPT-5.5 + Claude Opus 4.8** | 헤드라인 frontier 2종 → "최고 모델조차 실패" 강함 |

**변환 → 샘플 → 테스트:**
```bash
# 0) TimeQA annotated → 충돌 QA (루트에서, 무료)
python3 data_prep/timeqa/timeqa_to_qa.py      # data/timeqa/source/annotated_*.json → data/timeqa/qa_timeqa.jsonl
# 1~4) 샘플(검수) → 테스트 → 채점
cd experiments/03_temporal_validity/scripts
python3 01_sample_eval_set.py --input ../../../data/timeqa/qa_timeqa.jsonl   # → validation_sheet 검수
python3 02_run_models.py --model gpt   ; python3 02_run_models.py --model claude
python3 03_evaluate.py --model gpt --judge gemini ; python3 03_evaluate.py --model claude --judge gemini
```
*(보조: 같은 파이프라인을 `data_prep/tqa/tqa_to_qa.py` → `data/tqa/qa_tqa.jsonl`로 돌려 robustness 재현.)*

---

## 4. 모델 (확정, 2026-06 기준)

경향성 파악용 *최소* 구성 = **블랙박스 프론티어 2종이 코어**, 오픈 1종은 선택.

### 4.1 블랙박스 (최신 프론티어) — 코어
- **GPT-5.5** (`gpt-5.5`) — OpenAI 현 flagship(2026-04 API). *(GPT-5.6은 6월 말 예정·미출시)*
- **Claude Opus 4.8** (`claude-opus-4-8`) — Anthropic 최상위 추론 Opus(2026-05). 가격 **$5 / $25** per 1M tok(in/out).
- **Gemini 3 Pro** (`gemini-3.1-pro-preview`, Google) — `--model gemini`로 동일 실행(OpenAI 호환 엔드포인트).
- (선택) Claude Fable 5(`claude-fable-5`, 2026-06 신규 상위 tier).
- **파일럿 최소 = 서로 다른 lab의 frontier 2종**(권장: Gemini 3 Pro + Claude Opus 4.8). "약한 모델만/같은 lab만"은 C1 반박 여지 → *cross-lab frontier*가 가장 강함. 경계 시 GPT-5.5 추가.

### 4.2 오픈 LLM (단일 H100 80GB) — 선택
- 질문이 모델 생성이 아니라 **TQA(위키) 원본**이라 자가생성 편향 제약이 없다 → 오픈 모델도 자유롭게 테스트 가능.
- 권장 오픈 모델: **Qwen3 35B-A3B** 또는 **Gemma 4 31B** (단일 H100 여유).
- 서빙 vLLM(`vllm serve … --quantization awq`). "프론티어조차 + 오픈도" 프레이밍 강화용 — **최소 파일럿엔 생략 가능**.

> **데이터 출처 메모**: 평가 질문·답·근거는 **TimeQA**(NeurIPS'21, Wikidata 시변사실 + 시점별 위키 문단을 사람이 라벨; 보조 TQA)에서 직접 유래(모델 생성 아님). 따라서 자가생성 편향이 원천적으로 없고, judge(Gemini)·테스트(GPT/Claude)와도 무관하다. 단 §3.3 Step 3 수동 검증(시점 결정성·자연스러움)은 유지.

---

## 5. 근거(grounding) 측정 방법 — 표준 citation 방식 채택

### 5.1 인용 추출 = inline self-citation (ALCE 표준)
- 현대 RAG의 근거 표시는 "가장 관련 문서 1개 고르기"가 아니라 **답의 주장에 근거 문서를 inline citation으로 부착**(ALCE, Gao et al. 2023). 프로덕션도 동일(Claude Citations API, OpenAI file_search annotations).
- **비교 가능성**을 위해 모든 모델에 *동일한 prompted inline-citation*을 적용(프로바이더별 native API는 형식이 달라 비교성 저해).
- 단, Claude native Citations는 robustness 보조 확인용으로 선택 사용 가능.

- 답은 짧은 factoid이므로 단일 주장 = 단일(또는 소수) 인용.
- 모델 출력: 짧은 답 + 근거 문서 번호 `[k]`(필요 시 `[k][m]`).

### 5.2 인용을 그대로 믿지 않는다 — 반사실 보강
- ALCE도 self-citation이 "실제 컨텍스트 사용을 충실히 반영 못 함"을 지적(포지셔닝 §4.2).
- → 인용(선언) 외에 **세 컨텍스트 조건**으로 답이 *실제로* 어느 문서에 좌우되는지를 행동으로 측정한다.

| 조건 | 제시 청크 | 목적 |
|---|---|---|
| **Conflict (주)** | current + outdated + distractor 전부 | 충돌 상황의 답·인용 |
| **Current-only** | current + distractor | 현재만 있을 때 답 |
| **Outdated-only** | outdated + distractor | 과거만 있을 때 답 |

Conflict 답이 **Outdated-only와 일치 & Current-only와 불일치** → 모델은 *행동적으로* outdated에 의존(자기 보고 무관).

---

## 6. 절차 및 프롬프트 (전문 명시)

### 6.1 시스템 프롬프트
```
You are given a set of documents, each with an index and a modification timestamp.
Use the timestamps to identify the document whose information is valid for the time
frame the question refers to, and base your answer on that document.
Cite the supporting document inline using its index in square brackets.
Output EXACTLY these two tags and nothing else:
<reasoning> brief reasoning about which document is time-appropriate </reasoning>
<answer> short answer [index] </answer>
Example: <answer> Saccharomyces bulderi [2] </answer>
```

### 6.2 컨텍스트 구성
- 조건별 청크를 **무작위 순서**로 섞고(위치 편향 방지) (순서 index → chunk_id) 매핑 저장.
- 각 청크 렌더링:
  ```
  [Document {k}] [modified: {last_modified_time}]
  {text}
  ```
- 사용자 메시지 = `"[Query] {new_question}\n\n"` + 문서 블록들.

### 6.3 호출
각 (모델 × 조건 × 항목) 독립 호출(이전 대화 없음), `temperature=0`. 호출 수 = 모델수 × 3조건 × 50.

### 6.4 파싱
- `<reasoning>`, `<answer>` 추출. `<answer>` 내 `[k]`(들) → 인용 문서 번호 → 매핑으로 `chunk_id` 집합 `C_cite`.
- 답 문자열 = `<answer>`에서 `[..]` 제거분.
- 인용 누락/형식 오류 시 `C_cite=∅`로 별도 집계(citation 형식 실패율 보고 — ALCE 알려진 이슈).

---

## 7. 측정 지표 (정의·수식)

`ans_*`는 정규화 답(소문자·구두점·관사 제거). Conflict 조건 기준.

**기본**
- `EM = 1[ans_conflict == target_answer]`, `F1` 토큰 단위.
- `C_cite` = 인용된 청크 집합.

**기존 표준 foil — Citation Precision (ALCE)**
- `CitePrec = 1[∃ c ∈ C_cite: text(c) 가 ans_conflict 를 함의(entail)]`
  - NLI 모델 또는 LLM-판정(§12). **시점 무관, 함의만** 본다. 이것이 분야 표준 근거 신뢰 지표다.

**신규 — Temporal Validity (인용 기준)**
- `TV_cite = 1[evidence_chunk_id ∈ C_cite]` (시점-유효 청크를 인용했는가)

**신규 — Temporal Validity (행동 기준, §5.2 반사실)**
- `behav = outdated` if `ans_conflict == ans_outdated_only` and `≠ ans_current_only`
- `behav = current`  if `ans_conflict == ans_current_only` and `≠ ans_outdated_only`
- `behav = other` 그 외
- `TV_behav = 1[behav가 목표 시점 쪽]`

**파생율**
- 틀린 시점 인용 비율(선언) = `1 − mean(TV_cite)`
- 틀린 시점 의존 비율(행동) = `mean(behav가 틀린 시점 쪽)`
- 인용↔행동 일치율 = `mean(1[TV_cite == TV_behav])` (Q2 진단: 인용의 신뢰도)
- **맹점 비율** = `P(CitePrec = 1 | TV_cite = 0)` (틀린 시점 인용인데 표준 지표가 통과)

---

## 8. 분석

> 개념 근거(TV vs 정답률, 인용의 신뢰성에 대한 Q1·Q2)는 포지셔닝 문서 §4.1·§4.2 참조. 본 절은 측정 결과의 분석만 다룬다.

**C1**: 모델별 *틀린 시점 인용/의존 비율*(선언·행동)을 모드별 보고.
**C2**: 충돌 항목 한정 **2×2 분할표**(핵심 산출물)

| | CitePrec = PASS | CitePrec = FAIL |
|---|---|---|
| **TV = PASS**(시점-유효 인용) | 이상적 | 드묾 |
| **TV = FAIL**(틀린 시점 인용) | **★ 맹점: 표준 지표는 통과시키나 시점 틀림** | 양쪽 포착 |

★ 셀이 유의하게 비어 있지 않으면 C2 성립. 보조: 충돌에서 `corr(CitePrec, EM)≈0` vs `corr(TV, EM) 높음`.
**C3**: TV(인용·행동 양쪽)가 ★ 셀을 0점으로 식별함을 제시.

---

## 9. 판정 기준 (사전 등록)

| 결과 | 해석 | 다음 |
|---|---|---|
| 최고 모델 **틀린 시점 인용 비율 > 25%** & ★ 셀 유의 | C1·C2·C3 성립 | 진행 — 발견 논문(+후속 method) |
| **TV > 85%** 안정, ★ 셀 거의 빔 | 프론티어 이미 해결 | 중단/재검토 — step-out |
| 그 사이 | Findings급은 되나 메인은 무리 | 규모 확대 + method 결합 |

---

## 10. 한계
- 소규모 N, 단일 출처(위키)·동질 충돌 → 외부 타당성은 본 논문 보강.
- 근거는 inline citation(선언) + 반사실(행동)으로 근사 — 완전한 인과 attribution은 본 논문 과제.
- CitePrec foil은 NLI/LLM-판정 기반 잡음 → 표본 사람 검수.
- 행동 기준 TV는 답 문자열 일치 기반 → 정규화·동의어 처리 필요.

---

## 11. 산출물 · 비용 · 일정
- **산출물**: ① 모델별 지표표(EM/TV_cite/TV_behav/CitePrec/틀린시점비율/인용-행동 일치율/citation 형식 실패율), ② §8 2×2 표, ③ 대표 사례 정성 분석.
- **비용**: 모델 2–4 × 3조건 × 60 → 수백 콜, 수십 달러.
- **일정**: 1–2일.

---

## 12. 구현 명세 (문서만으로 작성 가능)

**입력**: §3.3 스키마의 가공 레코드 50개(§3.4 선별).

**파이프라인**
1. 각 항목 × {Conflict, Current-only, Outdated-only} 컨텍스트 생성(§6.2). 셔플 + (index→chunk_id) 매핑 저장.
2. 각 모델로 §6.1 프롬프트 호출(temp 0).
3. 파싱(§6.4): reasoning/answer, answer 내 `[k]`→`C_cite`.
4. 지표(§7): EM, CitePrec, TV_cite, TV_behav.
5. 집계(§8): 율 + 2×2 표.

**Citation Precision 판정(LLM-judge / NLI)** — ALCE식 함의:
```
Passage: "{text(c)}"   # c ∈ C_cite, 하나라도 yes면 CitePrec=1
Answer: "{ans_conflict}"
Does the passage state or directly support the Answer? Reply only "yes" or "no".
```

**EM 정규화**: 소문자 → 구두점·관사(a/an/the) 제거 → 공백 정규화 → 완전일치. F1은 토큰 집합.

**항목별 출력 스키마**
```json
{"id","model","mode",
 "conflict":{"reasoning","C_cite":[...],"answer","EM","F1","TV_cite","CitePrec"},
 "current_only":{"answer"}, "outdated_only":{"answer"},
 "behav","TV_behav"}
```
**집계 출력**: 모델별 {mean EM, TV_cite, TV_behav, CitePrec, 틀린시점비율(선언/행동), 인용-행동 일치율, citation 형식 실패율, 맹점비율 P(CitePrec=1|TV_cite=0)} + 2×2 카운트.

---

### 참고 (방법 표준)
ALCE citation precision/recall (Gao et al. 2023) · Claude Citations API(2025) · OpenAI file_search annotations.
