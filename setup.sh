#!/usr/bin/env bash
# 가상환경 생성 및 의존성 설치
set -e

VENV_DIR=".venv"

if [ ! -d "$VENV_DIR" ]; then
    echo ">>> 가상환경 생성 중: $VENV_DIR"
    python3 -m venv "$VENV_DIR"
else
    echo ">>> 가상환경 이미 존재: $VENV_DIR"
fi

echo ">>> 의존성 설치 중 ..."
"$VENV_DIR/bin/pip" install --upgrade pip -q
"$VENV_DIR/bin/pip" install -r requirements.txt -q

if [ ! -f ".env" ]; then
    echo ">>> .env 파일 생성 중 (.env.example 복사)"
    cp .env.example .env
    echo ">>> .env 파일에 GEMINI_API_KEY를 입력하세요."
else
    echo ">>> .env 파일 이미 존재"
fi

echo ""
echo "=== 준비 완료 ==="
echo "가상환경 활성화: source $VENV_DIR/bin/activate"
echo ""
echo "--- Step 1: Wikipedia 청크 생성 ---"
echo "  (전체, 샤드 없이)"
echo "  python build_dataset.py"
echo ""
echo "  (4개 샤드로 분할 시 각각 실행)"
echo "  python build_dataset.py --shard 0 --total-shards 4"
echo "  python build_dataset.py --shard 1 --total-shards 4"
echo "  python build_dataset.py --shard 2 --total-shards 4"
echo "  python build_dataset.py --shard 3 --total-shards 4"
echo ""
echo "--- Step 1 병합 (샤드 사용 시) ---"
echo "  python merge_shards.py --step 1"
echo ""
echo "--- Step 2: Gemini로 LLM pair 생성 ---"
echo "  (전체)"
echo "  python build_llm_pairs_gemini.py"
echo ""
echo "  (범위별 Step 1 결과 파일을 입력으로 지정)"
echo "  python build_llm_pairs_gemini.py --input data/chunks/hoh_chunks_0_50.jsonl"
echo "  python build_llm_pairs_gemini.py --input data/chunks/hoh_chunks_50_100.jsonl"
echo "  ..."
echo ""
echo "--- Step 2 병합 (샤드 사용 시) ---"
echo "  python merge_shards.py --step 2"
