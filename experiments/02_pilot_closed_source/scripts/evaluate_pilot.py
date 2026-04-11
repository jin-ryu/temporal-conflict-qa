import json
import re
import string
from collections import defaultdict

def _normalize(s: str) -> str:
    """소문자화, 관사 제거, 구두점 제거, 공백 정규화 (evaluate_llm.py와 동일)"""
    if not s: return ""
    s = str(s).lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    return " ".join(s.split())

def parse_xml_tag(text: str, tag: str) -> str:
    """XML 태그 내 내용 추출"""
    m = re.search(f"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return m.group(1).strip() if m else ""

def evaluate_pilot_study():
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
    all_ids = set(records_a.keys()) | set(records_b.keys())
    
    stats_template = lambda: {"total": 0, "exp_a_correct": 0, "no_conflict_correct": 0, "conflict_ans_correct": 0, "conflict_evid_correct": 0}
    mode_stats = defaultdict(stats_template)
    layer_stats = defaultdict(stats_template)
    overall = stats_template()

    for rid in all_ids:
        ra = records_a.get(rid)
        rb = records_b.get(rid)
        
        meta = ra or rb
        mode_group = "outdated" if meta["mode"].startswith("outdated_") else meta["mode"]
        layer_group = "4+" if meta["layers"] >= 4 else str(meta["layers"])
            
        def update_stats(stats):
            stats["total"] += 1
            
            # --- Exp A Auto-grading ---
            if ra and ra.get("response"):
                # 수동 체크값이 있으면 그것을 쓰고, 없으면 자동 비교
                if ra.get("is_correct") is not None:
                    if ra["is_correct"]: stats["exp_a_correct"] += 1
                else:
                    pred = _normalize(ra["response"])
                    gt = _normalize(ra["ground_truth"]["answer"])
                    if pred == gt or gt in pred: # 포함 관계도 정답으로 간주 (완화된 EM)
                        stats["exp_a_correct"] += 1
            
            # --- Exp B Auto-grading ---
            if rb:
                # No-Conflict
                nc = rb.get("no_conflict", {})
                if nc.get("raw_response"):
                    if nc.get("is_correct") is not None:
                        if nc["is_correct"]: stats["no_conflict_correct"] += 1
                    else:
                        ans = _normalize(parse_xml_tag(nc["raw_response"], "answer") or nc["raw_response"])
                        gt = _normalize(rb["ground_truth"]["answer"])
                        if ans == gt or gt in ans: stats["no_conflict_correct"] += 1
                
                # Conflict
                cf = rb.get("conflict", {})
                if cf.get("raw_response"):
                    # Answer Check
                    if cf.get("is_answer_correct") is not None:
                        if cf["is_answer_correct"]: stats["conflict_ans_correct"] += 1
                    else:
                        ans = _normalize(parse_xml_tag(cf["raw_response"], "answer") or cf["raw_response"])
                        gt = _normalize(rb["ground_truth"]["answer"])
                        if ans == gt or gt in ans: stats["conflict_ans_correct"] += 1
                    
                    # Evidence Check
                    if cf.get("is_evidence_correct") is not None:
                        if cf["is_evidence_correct"]: stats["conflict_evid_correct"] += 1
                    else:
                        rel = parse_xml_tag(cf["raw_response"], "relevance")
                        gt_rel = rb["ground_truth"].get("correct_relevance")
                        if rel == gt_rel: stats["conflict_evid_correct"] += 1

        update_stats(mode_stats[mode_group])
        update_stats(layer_stats[layer_group])
        update_stats(overall)

    # Metrics calculation and Summary output
    def calc_metrics_exp_a(s):
        return {
            "total": s["total"],
            "exp_a_accuracy": s["exp_a_correct"] / s["total"] if s["total"] > 0 else 0.0
        }

    def calc_metrics_exp_b(s):
        return {
            "total": s["total"],
            "no_conflict_accuracy": s["no_conflict_correct"] / s["total"] if s["total"] > 0 else 0.0,
            "conflict_answer_accuracy": s["conflict_ans_correct"] / s["total"] if s["total"] > 0 else 0.0,
            "conflict_evidence_accuracy": s["conflict_evid_correct"] / s["total"] if s["total"] > 0 else 0.0
        }

    summary_a = {
        "overall": calc_metrics_exp_a(overall),
        "by_mode": {m: calc_metrics_exp_a(mode_stats[m]) for m in mode_stats},
        "by_layer": {l: calc_metrics_exp_a(layer_stats[l]) for l in layer_stats}
    }

    summary_b = {
        "overall": calc_metrics_exp_b(overall),
        "by_mode": {m: calc_metrics_exp_b(mode_stats[m]) for m in mode_stats},
        "by_layer": {l: calc_metrics_exp_b(layer_stats[l]) for l in layer_stats}
    }

    with open("experiments/02_pilot_closed_source/metrics/summary_exp_a.json", "w") as f:
        json.dump(summary_a, f, indent=2)
    with open("experiments/02_pilot_closed_source/metrics/summary_exp_b.json", "w") as f:
        json.dump(summary_b, f, indent=2)

    print("\n=== Pilot Study Evaluation Summary (Auto-graded) ===")
    print(f"Total Samples: {overall['total']}")
    print("-" * 50)
    print(f"Exp A (Web ON) Accuracy:         {summary_a['overall']['exp_a_accuracy']:.2%}")
    print(f"No Conflict Accuracy:            {summary_b['overall']['no_conflict_accuracy']:.2%}")
    print(f"Conflict Answer Accuracy:        {summary_b['overall']['conflict_answer_accuracy']:.2%}")
    print(f"Conflict Evidence Accuracy:      {summary_b['overall']['conflict_evidence_accuracy']:.2%}")
    print("-" * 50)
    print("Note: If 'is_correct' fields are null, results are auto-calculated from 'response'.")

if __name__ == "__main__":
    evaluate_pilot_study()
