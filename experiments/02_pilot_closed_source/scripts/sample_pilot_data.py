import json
import random
from collections import defaultdict

def sample_pilot_from_qa():
    with open("data/qa/qa_llama3_1-70b-awq_0_600.jsonl", "r") as f:
        all_qa = [json.loads(line) for line in f]

    # Group by mode
    by_mode = defaultdict(list)
    for qa in all_qa:
        mode = qa["mode"]
        if mode.startswith("outdated_") and mode != "outdated_0":
            by_mode["outdated_1_plus"].append(qa)
        else:
            by_mode[mode].append(qa)

    # Function to get layer count from chunks
    def get_layers(chunks):
        times = set()
        for c in chunks:
            if c["label"] in ["current", "outdated"]:
                times.add(c["last_modified_time"])
        return len(times)

    samples = []
    
    # Target distribution
    targets = {
        "current_raw": 40,
        "current": 30,
        "outdated_0": 20,
        "outdated_1_plus": 10
    }

    selected_ids = set()

    for mode, count in targets.items():
        pool = by_mode[mode]
        # Sort by layers (descending) to prioritize complex cases if possible
        pool.sort(key=lambda x: get_layers(x["chunks"]), reverse=True)
        
        # Select top 3-layer+ cases first, then random
        complex_cases = [q for q in pool if get_layers(q["chunks"]) >= 3]
        simple_cases = [q for q in pool if get_layers(q["chunks"]) < 3]
        
        selected = complex_cases[:count]
        if len(selected) < count:
            needed = count - len(selected)
            selected += random.sample(simple_cases, min(needed, len(simple_cases)))
            
        for q in selected:
            samples.append({
                "id": q["id"],
                "hoh_source_idx": q["hoh_source_idx"],
                "mode": q["mode"],
                "layers": get_layers(q["chunks"]),
                "question": q["new_question"],
                "target_answer": q["target_answer"],
                "evidence_chunk_id": q["evidence_chunk_id"],
                "chunks": q["chunks"] # Store for Exp B prompt generation
            })

    random.shuffle(samples)

    with open("experiments/02_pilot_closed_source/data/sample_100.jsonl", "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    print(f"Re-sampled 100 cases from QA dataset to experiments/02_pilot_closed_source/data/sample_100.jsonl")

if __name__ == "__main__":
    sample_pilot_from_qa()
