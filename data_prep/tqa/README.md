# TQA(Temporal Wiki) → 충돌 평가셋 변환 (`data_prep/tqa`)

실험03(Temporal Validity)의 **주력 베이스**. temporal 데이터셋 24종 전수조사 결과,
"**옛/새 *실제 버전* 문서 + 깨끗한 답 + 시점-유효 라벨**이 한 문맥에 공존"하는 구조는
TQA에만 존재한다(다른 건 단일문서거나 freshness거나 답만 있음). → TQA가 유일 적합.

**LLM 생성 없음** — 질문·답·근거가 전부 TQA에서 오고, 우리는 옛/새 두 시점을
충돌 구조로 **재조립 + 품질 필터**만 한다. (대조: HoH는 LLM으로 질문을 *창작* → 위험.
여기는 *재조립*뿐.)

> ⚠️ TQA는 arXiv **preprint**(미peer-review)다. 신빙성은 향후 *TQA의 레시피를 우리가 직접
> 재현*(Wikidata 시간한정자 + 위키 revision)해 보강 가능 — 그게 곧 TQA가 한 일이다(아래).

---

## 원천 데이터 — TQA가 만든 것
- **TQA / Temporal Wiki** (Özer & Yıldız, *Question Answering under Temporal Conflict*,
  **arXiv 2506.07270**, 2025, **MIT**) — github.com/atahanoezer/TQA
- TQA의 구성: **TempLAMA**(Wikidata 시간한정 트리플)에서 연도형 **질문**을 템플릿 생성 →
  **답** = Wikidata 객체값 → **근거** = 각 연도의 실제 위키 **revision 스냅샷**(`body_par`).
- 받기: 레포 clone 후 `full_data_filtered/`(878 엔티티 JSON)를 `data/tqa/source/`로 복사
  (해당 폴더는 `.gitignore`).

## 입력 구조 (`source/<entity>.json`, 엔티티당 1파일)
```json
{
  "event_id": "...",
  "incidents": {
    "2010": {
      "question": "Question: Which team did Dida play for in 2010?",
      "answer":   [{"wikidata_id": "Q1543", "name": "Associazione Calcio Milan"}],
      "dump":     {"body_par": "<2010 revision 인트로 본문>", "infobox": "..."},
      "url":      "https://en.wikipedia.org/w/index.php?title=...&oldid=409749899"
    },
    "2014": { "...다른 팀(바뀐 답)..." }
  }
}
```
- `incidents[연도]` = 그 시점의 {질문, 답(Wikidata), **그 시점 위키 revision 본문**, oldid 영구링크}.

## 출력 구조 (`data/tqa/qa_tqa.jsonl`) — `01_sample_eval_set.py`가 소비
엔티티당 **2 레코드**(`outdated_0` + `current`)가 **같은 충돌 청크쌍**을 공유한다:
```json
{
  "id": "tqa_<eid>_outdated_0",
  "source_idx": "<eid>",                 // leakage 방지 dedup 키(=엔티티)
  "mode": "outdated_0",                  // 과거지향(주력) | "current"=현재지향(대조군)
  "new_question": "Which team did Dida play for in 2010?",
  "target_answer": "Associazione Calcio Milan",
  "evidence_chunk_id": 0,                // 이 질문의 *시점-유효* 청크 id
  "chunks": [
    {"chunk_id": 0, "label": "outdated_0", "text": "<2010 revision>", "last_modified_time": "2010"},
    {"chunk_id": 1, "label": "current",    "text": "<2014 revision>", "last_modified_time": "2014"}
  ]
}
```
- **outdated_0 레코드**: 질문=옛 연도, 정답=옛 답, `evidence_chunk_id`=0(옛 청크)
- **current 레코드**: 질문=새 연도, 정답=새 답, `evidence_chunk_id`=1(새 청크)
- 두 레코드가 **동일한 `chunks`(옛+새 공존)** 를 공유 → 이게 충돌 컨텍스트.

---

## 변환 스텝 (`tqa_to_qa.py`, 결정론적·무생성)

### Step 1 — 옛/새 시점 쌍 선택 (`pick_pair`)
- **유효 연도** = 숫자 연도 + 답 있음 + `body_par` 있음.
- **새(current)** = 가장 늦은 유효 연도.
- **옛(outdated)** = 아래 3조건을 *모두* 만족하는 **가장 이른** 연도:
  1. 답이 새와 **다름** (실제 변화)
  2. **near-dup 아님** — 옛/새 답이 부분문자열이거나 토큰 60%+ 겹치면 약한 충돌 → 제외
     (예: "MP" vs "MP for Woolwich East")
  3. `body_par`가 새와 **다름** — 인트로가 같으면 내용 충돌이 없음 → 제외

### Step 2 — 청크 구성
- 각 연도 `body_par`를 `clean_body`로 정리: 인라인 CSS(`.mw-parser-output{...}`) 제거,
  공백 정규화, **2000자 cap** → 청크 텍스트.
- 옛=`outdated_0`, 새=`current`, `last_modified_time`=연도.

### Step 3 — 질문·라벨
- 질문 = **TQA 원본 incident question**("Question:" 접두 제거, *명시적 연도형*).
- 정답 = 해당 연도 답(Wikidata name), `evidence_chunk_id` = 그 시점 청크.

### Step 4 — 품질 게이트 (제외 사유 카운트로 출력)
| 사유 | 의미 |
|---|---|
| `no_drift_pair` | Step1 3조건을 만족하는 쌍 없음(변화 없음 / near-dup / 같은 본문 포함) |
| `team_ambiguous` | 스포츠 클럽↔국가대표/유스 혼합("which team" 단일정답 모호) |
| `near_identical_evidence` | 옛/새 청크가 사실상 같은 문서(앞 200자 동일 or 토큰 자카드 ≥0.9) → 충돌 아님 |
| `stale_old_snapshot` | 옛 청크가 *라벨연도+2 이후* 미래 연도를 언급 = 늦은 fallback(진짜 옛 문서가 아님). TQA가 어린 문서에서 "옛" 스냅샷을 최신으로 대체하는 현상 차단 |
| `evidence_mismatch` | 답의 변별 토큰이 근거 문단에 없음(시점-유효 근거 보장 실패; 정식명↔약칭 흡수, generic 역할단어 제외) |

**현재 결과: 878 엔티티 → 309 변환 (618 레코드 = outdated 309 + current 309).**
제외: `no_drift_pair` 241 · `evidence_mismatch` 197 · `near_identical_evidence` 59 · `team_ambiguous` 45 · `stale_old_snapshot` 27.

> ⚠️ **TQA 자체 한계**(메모): `last_modified_time`=질문 연도지만 실제 문서는 "그 연도 이후 가장 이른 스냅샷". 어린/빈약한 문서는 "옛" 스냅샷이 최신으로 fallback돼 비일관(옛 문서가 이미 새 정보를 담음). 위 두 필터로 ~10% 제거. 근본 해결은 *Wikidata+위키 revision 직접 빌드*(스냅샷 선택을 우리가 통제).

---

## 실행
```bash
python3 data_prep/tqa/tqa_to_qa.py             # 전체
python3 data_prep/tqa/tqa_to_qa.py --max 100    # 100 엔티티(테스트)
```

## 데이터 검증 (테스트)
산출 데이터의 무결성을 자동 점검(스키마·충돌공존·비동일청크·근거지지·near-dup 등):
```bash
python3 data_prep/validate_qa.py data/tqa/qa_tqa.jsonl
```
(같은 스크립트로 `data/timeqa/qa_timeqa.jsonl`도 검증 가능.)

## 인용
질문·답·근거의 원천이므로 **TQA/Temporal Wiki(Özer & Yıldız, arXiv 2506.07270)** 인용 필수.
