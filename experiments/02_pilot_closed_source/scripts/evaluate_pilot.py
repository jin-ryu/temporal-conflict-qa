import json
from collections import defaultdict

def evaluate_pilot_study():
    # Load separate files
    try:
        with open("experiments/02_pilot_closed_source/results/results_exp_a.json", "r") as f:
            data_a = json.load(f)
        with open("experiments/02_pilot_closed_source/results/results_exp_b.json", "r") as f:
            data_b = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: Could not find result files. {e}")
        return
        
    records_a = {r["id"]: r for r in data_a["records"]}
    records_b = {r["id"]: r for r in data_b["records"]}
    
    # Combined set of IDs
    all_ids = set(records_a.keys()) | set(records_b.keys())
    
    # Accuracy statistics
    mode_stats = defaultdict(lambda: {"total": 0, "exp_a_correct": 0, "no_conflict_correct": 0, "conflict_ans_correct": 0, "conflict_evid_correct": 0})
    layer_stats = defaultdict(lambda: {"total": 0, "exp_a_correct": 0, "no_conflict_correct": 0, "conflict_ans_correct": 0, "conflict_evid_correct": 0})
    overall = {"total": 0, "exp_a_correct": 0, "no_conflict_correct": 0, "conflict_ans_correct": 0, "conflict_evid_correct": 0}

    for rid in all_ids:
        ra = records_a.get(rid)
        rb = records_b.get(rid)
        
        mode = (ra or rb)["mode"]
        layers = (ra or rb)["layers"]
        
        mode_group = "outdated" if mode.startswith("outdated_") else mode
        layer_group = "4+" if layers >= 4 else str(layers)
            
        def update_stats(stats):
            stats["total"] += 1
            if ra and ra.get("is_correct"): stats["exp_a_correct"] += 1
            if rb:
                if rb.get("no_conflict", {}).get("is_correct"): stats["no_conflict_correct"] += 1
                if rb.get("conflict", {}).get("is_answer_correct"): stats["conflict_ans_correct"] += 1
                if rb.get("conflict", {}).get("is_evidence_correct"): stats["conflict_evid_correct"] += 1

        update_stats(mode_stats[mode_group])
        update_stats(layer_stats[layer_group])
        update_stats(overall)

    summary = {
        "overall": {k: (v / overall["total"] if k.endswith("correct") else v) for k, v in overall.items()},
        "by_mode": {m: {k: (v / mode_stats[m]["total"] if k.endswith("correct") else v) for k, v in mode_stats[m].items()} for m in mode_stats},
        "by_layer": {l: {k: (v / layer_stats[l]["total"] if k.endswith("correct") else v) for k, v in layer_stats[l].items()} for l in layer_stats}
    }

    with open("experiments/02_pilot_closed_source/metrics/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("=== Pilot Study Evaluation Summary (Combined) ===")
    print(f"Total Samples: {overall['total']}")
    print(f"Exp A (Web ON) Accuracy: {summary['overall']['exp_a_correct']:.2%}")
    print(f"No Conflict Accuracy: {summary['overall']['no_conflict_correct']:.2%}")
    print(f"Conflict Answer Accuracy: {summary['overall']['conflict_ans_correct']:.2%}")
    print(f"Conflict Evidence Accuracy: {summary['overall']['conflict_evid_correct']:.2%}")

if __name__ == "__main__":
    evaluate_pilot_study()
