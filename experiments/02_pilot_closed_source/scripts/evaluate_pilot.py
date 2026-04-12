import json
import re
import string
import argparse
from pathlib import Path
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

def calc_metrics(stats):
    if stats["total"] == 0:
        return {k: 0.0 for k in stats if k.endswith("correct")}
    return {k: (v / stats["total"] if k.endswith("correct") else v) for k, v in stats.items()}

def evaluate_pilot_study():
    parser = argparse.ArgumentParser(description="TV-RAG Pilot Study Evaluation Tool")
    parser.add_argument("--mode", choices=["a", "b", "all"], default="all", 
                        help="실험 모드 선택: a (Web ON), b (Closed RAG), all (전체)")
    parser.add_argument("--save", action="store_true", default=True,
                        help="자동 채점 결과를 원본 JSON 파일에 저장 (기본값: True)")
    args = parser.parse_args()

    path_a = Path("experiments/02_pilot_closed_source/results/results_exp_a.json")
    path_b = Path("experiments/02_pilot_closed_source/results/results_exp_b.json")
    
    data_a, data_b = None, None
    
    if args.mode in ["a", "all"] and path_a.exists():
        with open(path_a, "r") as f:
            data_a = json.load(f)
    if args.mode in ["b", "all"] and path_b.exists():
        with open(path_b, "r") as f:
            data_b = json.load(f)

    if not data_a and not data_b:
        print("Error: 처리할 결과 데이터가 없습니다.")
        return

    stats_a = lambda: {"total": 0, "exp_a_correct": 0}
    stats_b = lambda: {"total": 0, "no_conflict_correct": 0, "conflict_ans_correct": 0, "conflict_evid_correct": 0}
    
    mode_stats_a = defaultdict(stats_a)
    layer_stats_a = defaultdict(stats_a)
    overall_a = stats_a()

    mode_stats_b = defaultdict(stats_b)
    layer_stats_b = defaultdict(stats_b)
    overall_b = stats_b()

    # --- Process Exp A ---
    if data_a:
        for r in data_a["records"]:
            m_grp = "outdated" if r["mode"].startswith("outdated_") else r["mode"]
            l_grp = "4+" if r["layers"] >= 4 else str(r["layers"])
            
            def update_a(stats):
                stats["total"] += 1
                if r.get("response"):
                    # 1. 자동 채점 수행
                    pred = _normalize(r["response"])
                    gt = _normalize(r["ground_truth"]["answer"])
                    auto_correct = (pred == gt or gt in pred)
                    
                    # 2. 파일에 저장할 값 결정 (null인 경우만 자동 결과로 채움)
                    if r.get("is_correct") is None and args.save:
                        r["is_correct"] = auto_correct
                        r["grading_rationale"] = "Auto-graded"
                    elif r.get("grading_rationale") is None:
                        # 이미 is_correct가 있는데 rationale이 없는 경우 (수동 입력 등)
                        r["grading_rationale"] = "Manual Review"
                    
                    # 3. 통계용 값 결정 (수동 값이 있으면 그것을 우선시)
                    final_correct = r["is_correct"] if r.get("is_correct") is not None else auto_correct
                    if final_correct:
                        stats["exp_a_correct"] += 1

            update_a(mode_stats_a[m_grp])
            update_a(layer_stats_a[l_grp])
            update_a(overall_a)

        if args.save:
            with open(path_a, "w") as f:
                json.dump(data_a, f, indent=2)

    # --- Process Exp B ---
    if data_b:
        for r in data_b["records"]:
            m_grp = "outdated" if r["mode"].startswith("outdated_") else r["mode"]
            l_grp = "4+" if r["layers"] >= 4 else str(r["layers"])
            
            def update_b(stats):
                stats["total"] += 1
                gt_ans = _normalize(r["ground_truth"]["answer"])
                
                # No-Conflict
                nc = r.get("no_conflict", {})
                if nc.get("raw_response"):
                    ans = _normalize(parse_xml_tag(nc["raw_response"], "answer") or nc["raw_response"])
                    auto_nc_correct = (ans == gt_ans or gt_ans in ans)
                    if nc.get("is_correct") is None and args.save:
                        nc["is_correct"] = auto_nc_correct
                        nc["grading_rationale"] = "Auto-graded"
                    elif nc.get("is_correct") is not None and nc.get("grading_rationale") is None:
                        nc["grading_rationale"] = "Manual Review"
                    
                    if (nc["is_correct"] if nc.get("is_correct") is not None else auto_nc_correct):
                        stats["no_conflict_correct"] += 1
                
                # Conflict
                cf = r.get("conflict", {})
                if cf.get("raw_response"):
                    # Answer
                    ans = _normalize(parse_xml_tag(cf["raw_response"], "answer") or cf["raw_response"])
                    auto_ans_correct = (ans == gt_ans or gt_ans in ans)
                    if cf.get("is_answer_correct") is None and args.save:
                        cf["is_answer_correct"] = auto_ans_correct
                        cf["grading_rationale_answer"] = "Auto-graded"
                    
                    if (cf["is_answer_correct"] if cf.get("is_answer_correct") is not None else auto_ans_correct):
                        stats["conflict_ans_correct"] += 1
                    
                    # Evidence
                    rel = parse_xml_tag(cf["raw_response"], "relevance")
                    gt_rel = r["ground_truth"].get("correct_relevance")
                    auto_evid_correct = (rel == gt_rel)
                    if cf.get("is_evidence_correct") is None and args.save:
                        cf["is_evidence_correct"] = auto_evid_correct
                        cf["grading_rationale_evidence"] = "Auto-graded"
                    
                    if (cf["is_evidence_correct"] if cf.get("is_evidence_correct") is not None else auto_evid_correct):
                        stats["conflict_evid_correct"] += 1

            update_b(mode_stats_b[m_grp])
            update_b(layer_stats_b[l_grp])
            update_b(overall_b)

        if args.save:
            with open(path_b, "w") as f:
                json.dump(data_b, f, indent=2)

    # (이후 summary 저장 및 출력 로직은 동일)
    summary_a = {"overall": calc_metrics(overall_a), "by_mode": {m: calc_metrics(mode_stats_a[m]) for m in mode_stats_a}}
    summary_b = {"overall": calc_metrics(overall_b), "by_mode": {m: calc_metrics(mode_stats_b[m]) for m in mode_stats_b}}
    
    with open("experiments/02_pilot_closed_source/metrics/summary_exp_a.json", "w") as f: json.dump(summary_a, f, indent=2)
    with open("experiments/02_pilot_closed_source/metrics/summary_exp_b.json", "w") as f: json.dump(summary_b, f, indent=2)

    print(f"\n=== Pilot Study Evaluation Summary (Rationals Integrated) ===")
    if data_a: print(f"[Exp A] Accuracy: {summary_a['overall']['exp_a_correct']:.2%}")
    if data_b: print(f"[Exp B] Conflict Ans Accuracy: {summary_b['overall']['conflict_ans_correct']:.2%}")

if __name__ == "__main__":
    evaluate_pilot_study()
