"""
HoH-QAs 원본 데이터셋 → hoh_chunks.jsonl

russwest404/HoH-QAs 를 로드하여 Wikipedia 히스토리 revision을 fetch하고,
슬라이딩 윈도우 청킹 + 라벨링 후 hoh_chunks.jsonl 로 저장한다.

chunks_to_qa.py 의 입력으로 사용된다.

Wikipedia fetch 전략
---------------------
- current answer  : current last_modified_time 기준 revision fetch
- outdated_infos  : 각 outdated last_modified_time 기준 revision fetch
- MediaWiki API   : rvstart=DATE&rvdir=older&rvlimit=1

청크 라벨링
-----------
- current snapshot  → evidence 포함 청크 : "current"
                    → 나머지              : "distractor"  (last_modified_time = current_time)
- outdated snapshot → evidence 포함 청크 1개만 : "outdated"  (distractor 중복 방지)

Record 구조
-----------
  id      : "hoh_NNNNN"
  question: 원본 질문
  answers : [current] + [outdated_0, outdated_1, ...]
  chunks  : current snapshot 전체 + outdated evidence 청크 1개씩

Resume
------
출력 파일에 이미 기록된 id를 읽어 처리 완료 여부를 판단한다.
별도 checkpoint 파일 없이 출력 파일 자체로 중단 재개 가능.

분할 실행 예시
--------------
  python hoh_to_chunks.py --start 0   --end 500
  python hoh_to_chunks.py --start 500 --end 1000
  python merge_shards.py --step 1
"""

import argparse
import json
import os
import re
import time
import unicodedata
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mwparserfromhell
import requests
from datasets import load_dataset
from dotenv import load_dotenv

from config import (
    CHUNK_SIZE, STRIDE, MAX_CHUNKS,
    DIR_CHUNKS, CHUNKS_PATH,
    WIKI_API, WIKI_SLEEP, WIKI_USER_AGENT,
    setup_logging,
)

load_dotenv()
logger = setup_logging("hoh_to_chunks")

_hf_token = os.environ.get("HF_TOKEN")
if _hf_token:
    from huggingface_hub import login
    login(token=_hf_token, add_to_git_credential=False)


# ---------------------------------------------------------------------------
# Wikipedia helpers
# ---------------------------------------------------------------------------

def _to_wiki_ts(date_str: str) -> str:
    date_part = str(date_str).strip().split(" ")[0]
    return f"{date_part}T23:59:59Z"


def _fetch_revision_wikitext(title: str, wiki_ts: str) -> str | None:
    params = {
        "action": "query",
        "titles": title,
        "prop": "revisions",
        "rvprop": "content|timestamp",
        "rvlimit": 1,
        "rvstart": wiki_ts,
        "rvdir": "older",
        "format": "json",
        "redirects": 1,
    }
    headers = {"User-Agent": WIKI_USER_AGENT}
    try:
        resp = requests.get(WIKI_API, params=params, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            revisions = page.get("revisions")
            if revisions:
                return revisions[0].get("*", "")
    except Exception as e:
        logger.warning("wiki error: title='%s' ts=%s: %s", title, wiki_ts, e)
    return None


def _wikitext_to_plain(wikitext: str) -> str:
    return mwparserfromhell.parse(wikitext).strip_code()


def normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text).strip()


def split_sentences(text: str) -> list[str]:
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in raw if s.strip()]


def fetch_sentences(title: str, date_str: str) -> list[str]:
    wiki_ts  = _to_wiki_ts(date_str)
    wikitext = _fetch_revision_wikitext(title, wiki_ts)
    time.sleep(WIKI_SLEEP)
    if not wikitext:
        return []
    return split_sentences(_wikitext_to_plain(wikitext))


# ---------------------------------------------------------------------------
# Chunk helpers
# ---------------------------------------------------------------------------

def find_sentence_index(sentences: list[str], evidence: str) -> int | None:
    evidence_n = normalize(evidence)
    for i, s in enumerate(sentences):
        if normalize(s) == evidence_n:
            return i
    best_i, best_len = None, 0
    ev_words = set(evidence_n.split())
    for i, s in enumerate(sentences):
        overlap = len(ev_words & set(normalize(s).split()))
        if overlap > best_len:
            best_len, best_i = overlap, i
    return best_i if best_len > 0 else None


def make_chunks(sentences: list[str]) -> list[dict]:
    chunks, n, start, cid = [], len(sentences), 0, 0
    while start < n:
        end = min(start + CHUNK_SIZE, n)
        chunks.append({
            "chunk_id": cid,
            "sentence_indices": list(range(start, end)),
            "text": " ".join(sentences[start:end]),
        })
        cid += 1
        if end == n:
            break
        start += STRIDE
    return chunks


def _trim_to_max_chunks(
    cur_labeled: list[dict],
    outdated_chunks: list[dict],
) -> list[dict]:
    """총 청크가 MAX_CHUNKS 초과 시 distractor를 균등 샘플링으로 줄인다."""
    all_chunks = cur_labeled + outdated_chunks
    if len(all_chunks) <= MAX_CHUNKS:
        return all_chunks

    keep       = [ch for ch in all_chunks if ch["label"] != "distractor"]
    distractors = [ch for ch in cur_labeled if ch["label"] == "distractor"]
    n_keep = len(keep)
    if n_keep >= MAX_CHUNKS:
        return keep

    n_distractor = MAX_CHUNKS - n_keep
    step     = max(1, len(distractors) // n_distractor)
    sampled  = distractors[::step][:n_distractor]
    return sorted(keep + sampled, key=lambda c: c["chunk_id"])


# ---------------------------------------------------------------------------
# Resume: 출력 파일에서 기처리 id 읽기
# ---------------------------------------------------------------------------

def load_done_ids(output_path: Path) -> set[str]:
    done: set[str] = set()
    if not output_path.exists():
        return done
    with output_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    done.add(json.loads(line)["id"])
                except Exception:
                    pass
    return done


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def hoh_to_chunks(
    output_path: Path = CHUNKS_PATH,
    start: int = 0,
    end: int | None = None,
) -> None:
    range_tag = f"[{start}:{end if end is not None else 'end'}]"
    logger.info("=== hoh_to_chunks: %s %s ===", output_path, range_tag)

    ds         = load_dataset("russwest404/HoH-QAs", split="train")
    total      = len(ds)
    actual_end = end if end is not None else total
    logger.info("Dataset: %d samples / processing %d~%d", total, start, actual_end - 1)

    done_ids = load_done_ids(output_path)
    if done_ids:
        logger.info("resume: %d records already written", len(done_ids))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("a", encoding="utf-8") as fout:
        for raw_idx, sample in enumerate(ds):
            if raw_idx < start:
                continue
            if raw_idx >= actual_end:
                break

            base_id = f"hoh_{raw_idx:06d}"
            if base_id in done_ids:
                continue

            question       = sample.get("question", "")
            current_answer = sample.get("answer", "")
            current_time   = str(sample.get("last_modified_time", ""))
            current_ev     = sample.get("evidence", "")
            outdated_infos = sample.get("outdated_infos", []) or []
            title          = (sample.get("document") or {}).get("title", "")

            if not title:
                logger.warning("[%s] no title, skipping", base_id)
                continue

            # ── current snapshot ──────────────────────────────────────
            cur_sentences = fetch_sentences(title, current_time)
            if not cur_sentences:
                logger.warning("[%s] failed to fetch current snapshot, skipping", base_id)
                continue

            cur_ev_idx     = find_sentence_index(cur_sentences, current_ev)
            cur_chunks_raw = make_chunks(cur_sentences)

            cur_labeled: list[dict] = []
            for ch in cur_chunks_raw:
                s_set = set(ch["sentence_indices"])
                if cur_ev_idx is not None and cur_ev_idx in s_set:
                    label, lmt = "current", current_time
                else:
                    label, lmt = "distractor", current_time
                cur_labeled.append({
                    "chunk_id": ch["chunk_id"],
                    "label": label,
                    "text": ch["text"],
                    "last_modified_time": lmt,
                })

            # ── outdated snapshots ─────────────────────────────────────
            answers = [{
                "label": "current",
                "answer": current_answer,
                "last_modified_time": current_time,
            }]

            next_chunk_id   = max((c["chunk_id"] for c in cur_labeled), default=-1) + 1
            outdated_chunks: list[dict] = []

            for i, oi in enumerate(outdated_infos):
                oi_answer = oi.get("answer", "")
                oi_ev     = oi.get("evidence", "")
                oi_time   = oi.get("last_modified_time", "")

                answers.append({
                    "label": "outdated",
                    "answer": oi_answer,
                    "last_modified_time": oi_time,
                    "outdated_index": i,
                })

                if not oi_time:
                    logger.warning("[%s] outdated[%d] has no timestamp, skipping chunk", base_id, i)
                    continue

                oi_sentences = fetch_sentences(title, oi_time)
                if not oi_sentences:
                    logger.warning("[%s] failed to fetch outdated[%d] snapshot, skipping chunk", base_id, i)
                    continue

                oi_ev_idx     = find_sentence_index(oi_sentences, oi_ev)
                oi_chunks_raw = make_chunks(oi_sentences)

                for ch in oi_chunks_raw:
                    if oi_ev_idx is not None and oi_ev_idx in set(ch["sentence_indices"]):
                        outdated_chunks.append({
                            "chunk_id": next_chunk_id,
                            "label": "outdated",
                            "text": ch["text"],
                            "last_modified_time": oi_time,
                            "outdated_index": i,
                        })
                        next_chunk_id += 1
                        break

            all_chunks = _trim_to_max_chunks(cur_labeled, outdated_chunks)

            record = {
                "id": base_id,
                "hoh_source_idx": raw_idx,
                "question": question,
                "answers": answers,
                "chunks": all_chunks,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()
            done_ids.add(base_id)

            if len(done_ids) % 50 == 0:
                logger.info("%d records written (last: %s)", len(done_ids), base_id)

    logger.info("Done → %s  (%d total records)", output_path, len(done_ids))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HoH-QAs → hoh_chunks.jsonl")
    parser.add_argument("--start", type=int, default=0,    help="시작 인덱스 (inclusive)")
    parser.add_argument("--end",   type=int, default=None, help="종료 인덱스 (exclusive), 생략 시 전체")
    args = parser.parse_args()

    if args.end is not None:
        out = DIR_CHUNKS / f"hoh_chunks_{args.start}_{args.end}.jsonl"
    else:
        out = CHUNKS_PATH

    hoh_to_chunks(out, start=args.start, end=args.end)
