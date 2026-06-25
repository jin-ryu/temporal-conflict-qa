# 구성 필터 판단 기준 (genuine 시간변화 vs 정정)

> HoH 충돌 항목을 **genuine 시간변화(g, 유지)** vs **정적사실 정정(c, 버림)** 으로 가르는 라벨링 가이드.
> 용도: ① 사람 검수 기준 ② **LLM 분류기 rubric**(few-shot) ③ 논문 annotation guideline.
> 사용처: `make_review_sheet.py`(시트 생성) → 이 기준으로 verdict(g/c) 기입 → `apply_review.py`(genuine만 추출).

---

## 왜 가르나 (construct validity)
실험 과제 = *"as-of-past 질문에 그 시점 유효 문서를 고르나"*. 따라서 **옛 답이 *옛 시점엔 진짜로 유효*했어야** 측정이 성립.
- **g(genuine)**: 옛 답이 옛 시점의 *진짜 정답* → "옛 문서 골라야 정답" → **측정 가능.**
- **c(correction)**: "과거 유효 답"이 *없음*(그냥 틀린 걸 고친 것/시간불변 사실) → 옛 답을 정답 처리하면 **채점 오염** → 버림.

---

## 단일 테스트 (3단계)
> **"옛 답이 *옛 시점엔 진짜로 맞았을 수* 있나?"**

1. **무슨 속성인가?** (예: 감독의 소속 클럽 / 고대 전투 결과 / 종(species) 개수)
2. **그 속성이 시간 따라 *변할 수 있나?*** (클럽=변함 / 1258년 전투=불변 / 개수=대개 고정·재집계)
3. **이 ~1개월 창에서 *진짜* 변했을 법한가?** (실제 이적=그럴듯 / 카운트 +2 점프=수상)

→ **변하는 속성 + 진짜 변화 그럴듯 = g** / 그 외 = **c**. **애매하면 보수적으로 c.**

---

## 분류 기준표

### ✅ g (유지) — 세상이 시간 따라 *진짜* 변하는 것
- **현직자**: 시장·MP·장관·secretary·경찰서장·교장·학장·코치·감독·race director·의장·vice mayor
- **소속/역할**: 선수의 현 소속팀, 코치/감독의 현 클럽, assistant coach 소속
- **소유/모회사**: 회사 인수·소유권 변경(parent company)
- **챔피언/타이틀**: 현 챔피언, defending champion, 현 타이틀 홀더
- **개명(rename)**: 기관·경기장·고속도로·역·도시·구단 명칭 변경 (예: Staples Center→Crypto.com Arena, Clingmans Dome→Kuwohi)
- **승강/리그 이동**: 승격·강등, 컨퍼런스 realignment
- **선거 결과**: 최근 선거 당선자, 의석
- **스폰서/브랜드**: 리그·구단 스폰서명 변경 (예: Ligue 1 Uber Eats→McDonald's)
- **"최신/최근/현재 X"**: 최신 회원, 가장 최근 수상자, caps-and-goals 기준 경기(매 경기 갱신)

### ❌ c (버림) — 시간 불변 / 그냥 정정
- **고대·역사 고정사실**: 1258년 전투, 왕조, 역사적 사건의 결과
- **전기 고정사실**: 출생지·출생연도·국적·인종·종교 (특히 고인)
- **학명/분류**: 종 재명명, taxonomic 재분류 (예: Accipiter→Tachyspiza)
- **신화·종교 서사**: 신화 속 인물·사건
- **소설/영화 줄거리**: 등장인물·플롯 디테일
- **집계 오류(카운트)**: 종 개수·수상 횟수·층수·연도 등 *재집계/오타 정정* (철자 숫자 "four→three" 포함)
- **명백한 반달리즘/오타**: 엉뚱한 값, 유명인 이름 끼워넣기

### ⚠️ 경계선 → 보수적으로 c
- 카운트 증가(앨범 6→8, 수상 4→5)는 *진짜 추가*일 수도 있으나 *재집계*와 구분 어려움 → **기본 c** (단, 2024 올림픽 메달수처럼 *명확한 이벤트*면 g)
- 행정구역 재분류·연대 정밀화·동의어 교체 → 대개 c

---

## 워크드 예시 (LLM few-shot용)
| 질문 | 옛 → 새 | 판정 | 이유 |
|---|---|---|---|
| Liam Rosenior 감독 클럽 | Hull City → Strasbourg | **g** | 2024-07엔 진짜 Hull 감독, 8월 이적 = 세상이 변함 |
| 몽골이 Siege of Baghdad서 격파한 상대 | Ayyubids → Abbasids | **c** | 1258년 고정사. 진실은 원래 Abbasids, Ayyubids는 그냥 오답 |
| Clingmans Dome 트레일 교차점 | Clingmans Dome → Kuwohi | **g** | 2024 공식 개명 |
| 프랑스 어순위 most-spoken | Fifth → Sixth | **c** | 순위 통계, 재집계성 |
| EU 농업 집행위원 | Wojciechowski → Christophe Hansen | **g** | 2024 새 집행부 현직 |
| Café del Mar 바 개수 | Eight → Seven | **c** | 카운트 정정 의심 |
| London Broncos 리그 tier | Super League → Championship | **g** | 2024 강등 |
| David Livingstone 국적 | British → Scottish | **c** | 고인 전기 고정사실(정정) |

---

## 한계 & 검증 (논문용)
- **1인·세계지식 기반 판단** — 외부 팩트체크 없이 plausibility 의존 → fallible.
- **보수적 편향**(애매=c) → keep(g) 정밀도↑, genuine 일부 놓침(recall↓).
- **권장**: 2nd annotator 스팟체크 + inter-annotator agreement(IAA) 보고. LLM 분류기 쓸 땐 *사람 골드라벨에 검증* 후 사용.
