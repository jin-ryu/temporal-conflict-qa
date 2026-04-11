import json
import random

# experiments/01_pilot_initial_eval/scripts/evaluate_llm.py 에서 가져온 원본 시스템 프롬프트
EVAL_SYSTEM_PROMPT = (
    "You are an assistant that answers questions using the provided documents. "
    "Each document has a modification timestamp. "
    "Use the timestamps to identify the most temporally relevant document for the question.\n\n"
    "You MUST respond using ONLY the following XML tags, in this exact order:\n\n"
    "<thought>\n"
    "Step-by-step reasoning about which document is most relevant based on the timestamps.\n"
    "</thought>\n"
    "<relevance>\n"
    "The number of the most relevant document (e.g. 3)\n"
    "</relevance>\n"
    "<answer>\n"
    "ONLY the exact short answer (a name, number, date, or short phrase). "
    "Do NOT include explanations, full sentences, or extra context.\n"
    "</answer>\n\n"
    "Do not include any text outside these tags.\n\n"
    "Example:\n"
    "<thought>\nThe question asks about the name before the mid-2024 update. "
    "Document 3 (modified: 2024-07-01) states 'Maudiozyma bulderi' and Document 5 (modified: 2024-06-01) states 'Saccharomyces bulderi'. "
    "Since the question asks about the name before the update, Document 5 predates the change and is more relevant.\n</thought>\n"
    "<relevance>\n5\n</relevance>\n"
    "<answer>\nSaccharomyces bulderi\n</answer>"
)

def build_combined_prompt(question, chunks):
    """01_pilot_initial_eval의 build_eval_prompt와 동일한 형식으로 구성"""
    lines = []
    lines.append(f"[Query] {question}")
    lines.append("")
    for i, ch in enumerate(chunks, 1):
        lmt = ch.get("last_modified_time") or "N/A"
        lines.append(f"[Document {i}] [modified: {lmt}]")
        lines.append(ch["text"])
        lines.append("")
    
    # 시스템 프롬프트를 앞에 붙여서 하나의 텍스트로 완성 (수동 입력용)
    return f"{EVAL_SYSTEM_PROMPT}\n\n---\n\n" + "\n".join(lines)

def create_results_templates():
    with open("experiments/02_pilot_closed_source/data/sample_100.jsonl", "r") as f:
        samples = [json.loads(line) for line in f]
    
    metadata = {
        "experiment_name": "TV-RAG Pilot Study",
        "total_samples": len(samples)
    }
    
    results_a = {"metadata": {**metadata, "model": "Gemini 3 Flash (Web ON)"}, "records": []}
    results_b = {"metadata": {**metadata, "model": "Gemini 3 Flash (Web OFF)"}, "records": []}
    
    for s in samples:
        target_chunk = next((c for c in s["chunks"] if c["chunk_id"] == s["evidence_chunk_id"]), None)
        target_time = target_chunk["last_modified_time"] if target_chunk else "Unknown"
        
        # --- Exp A (No changes needed for A, as it is simple Q&A) ---
        results_a["records"].append({
            "id": s["id"],
            "mode": s["mode"],
            "layers": s["layers"],
            "ground_truth": {"answer": s["target_answer"], "target_time": target_time},
            "question": s["question"],
            "response": "",
            "is_correct": None
        })

        # --- Exp B (Conflict & No-Conflict) ---
        # 1. Conflict (Current + Outdated + Distractors)
        shuffled_conflict = list(s["chunks"])
        random.shuffle(shuffled_conflict)
        correct_doc_label = ""
        for i, c in enumerate(shuffled_conflict, 1):
            if c["chunk_id"] == s["evidence_chunk_id"]:
                correct_doc_label = str(i)

        # 2. No-Conflict (Current + Distractors only)
        no_conflict_pool = [c for c in s["chunks"] if c["label"] in ["current", "distractor"]]
        random.shuffle(no_conflict_pool)
        correct_doc_label_nc = ""
        for i, c in enumerate(no_conflict_pool, 1):
            if c["chunk_id"] == s["evidence_chunk_id"]:
                correct_doc_label_nc = str(i)

        results_b["records"].append({
            "id": s["id"],
            "mode": s["mode"],
            "layers": s["layers"],
            "ground_truth": {
                "answer": s["target_answer"],
                "target_time": target_time,
                "correct_relevance": correct_doc_label,
                "no_conflict_correct_relevance": correct_doc_label_nc
            },
            "no_conflict": {
                "prompt": build_combined_prompt(s["question"], no_conflict_pool),
                "raw_response": "",
                "parsed_thought": "",
                "parsed_relevance": "",
                "parsed_answer": "",
                "is_correct": None
            },
            "conflict": {
                "prompt": build_combined_prompt(s["question"], shuffled_conflict),
                "raw_response": "",
                "parsed_thought": "",
                "parsed_relevance": "",
                "parsed_answer": "",
                "is_answer_correct": None,
                "is_evidence_correct": None
            }
        })
        
    with open("experiments/02_pilot_closed_source/results/results_exp_a.json", "w") as f:
        json.dump(results_a, f, indent=2)
    with open("experiments/02_pilot_closed_source/results/results_exp_b.json", "w") as f:
        json.dump(results_b, f, indent=2)
    
    print("Created templates with IDENTICAL prompts to automated evaluation!")

if __name__ == "__main__":
    create_results_templates()
