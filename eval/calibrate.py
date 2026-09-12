import sys
import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from eval.eval import run_eval
except ImportError:
    from eval import run_eval


def calibrate(re_run_eval: bool = True):
    eval_results_file = os.path.join(os.path.dirname(__file__), 'eval_results.json')
    
    # Run evaluation if requested or if cached results don't exist
    if re_run_eval or not os.path.exists(eval_results_file):
        print("Running initial baseline evaluation for calibration sweep (threshold=0.0 to capture raw confidence)...")
        eval_data = run_eval(confidence_threshold=0.0, verbose=True)
    else:
        print(f"Loading cached baseline evaluation from {eval_results_file}...")
        with open(eval_results_file, 'r', encoding='utf-8') as f:
            eval_data = json.load(f)
            
    items = eval_data["results"]
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    ans_acc_list = []
    nic_acc_list = []
    overall_acc_list = []
    
    print("\n================================================================")
    print("                CONFIDENCE THRESHOLD CALIBRATION SWEEP           ")
    print("================================================================")
    print(f"{'Threshold':>9} | {'ANSWERED Acc':>14} | {'NOT_IN_CORPUS Acc':>18} | {'Overall Acc':>12}")
    print("-" * 62)
    
    for t in thresholds:
        ans_total = 0
        ans_correct = 0
        nic_total = 0
        nic_correct = 0
        total_correct = 0
        
        for item in items:
            expected = item["expected"]
            raw_pred = item["predicted"]
            conf = item["confidence"]
            
            # Apply threshold safety valve:
            if raw_pred == "ANSWERED" and conf < t:
                simulated_pred = "NOT_IN_CORPUS"
            else:
                simulated_pred = raw_pred
                
            if expected == "ANSWERED":
                ans_total += 1
                if simulated_pred == "ANSWERED":
                    ans_correct += 1
            elif expected == "NOT_IN_CORPUS":
                nic_total += 1
                if simulated_pred == "NOT_IN_CORPUS":
                    nic_correct += 1
                    
            if simulated_pred == expected:
                total_correct += 1
                
        ans_acc = (ans_correct / ans_total * 100) if ans_total > 0 else 0
        nic_acc = (nic_correct / nic_total * 100) if nic_total > 0 else 0
        overall_acc = (total_correct / len(items) * 100) if items else 0
        
        ans_acc_list.append(ans_acc)
        nic_acc_list.append(nic_acc)
        overall_acc_list.append(overall_acc)
        
        print(f"{t:9.2f} | {ans_acc:13.1f}% | {nic_acc:17.1f}% | {overall_acc:11.1f}%")
        
    # Determine optimal threshold (max overall accuracy, prioritizing balanced recall)
    best_idx = max(range(len(thresholds)), key=lambda i: (overall_acc_list[i], nic_acc_list[i] + ans_acc_list[i]))
    chosen_threshold = thresholds[best_idx]
    
    print("================================================================")
    print(f"Optimal Threshold: {chosen_threshold:.2f} (Overall Accuracy: {overall_acc_list[best_idx]:.1f}%)")
    print(f"Evidenced Justification: At threshold {chosen_threshold:.2f}, the system achieves the optimal trade-off")
    print(f"between answering legitimate questions ({ans_acc_list[best_idx]:.1f}%) and refusing unanswerable near-misses ({nic_acc_list[best_idx]:.1f}%).")
    print("================================================================\n")
    
    # Plot calibration curves
    plt.figure(figsize=(10, 6), dpi=150)
    plt.plot(thresholds, ans_acc_list, marker='o', color='#28a745', linewidth=2.2, label='"Should Answer" Accuracy (ANSWERED Recall)')
    plt.plot(thresholds, nic_acc_list, marker='s', color='#dc3545', linewidth=2.2, label='"Should Refuse" Accuracy (NOT_IN_CORPUS Recall)')
    plt.plot(thresholds, overall_acc_list, marker='^', color='#007bff', linestyle='--', linewidth=1.8, label='Overall State Accuracy')
    
    # Highlight chosen threshold
    plt.axvline(x=chosen_threshold, color='#6f42c1', linestyle=':', linewidth=2, label=f'Chosen Threshold ({chosen_threshold:.1f})')
    plt.scatter([chosen_threshold], [overall_acc_list[best_idx]], color='#6f42c1', s=120, zorder=5)
    
    plt.xlabel("Confidence Threshold", fontsize=12, fontweight='bold')
    plt.ylabel("Accuracy (%)", fontsize=12, fontweight='bold')
    plt.title("ClauseCheck Calibration Curve: Finding the Answer vs Refuse Boundary", fontsize=13, fontweight='bold', pad=15)
    plt.ylim(0, 105)
    plt.xlim(0.25, 0.95)
    plt.xticks(thresholds)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower left', frameon=True, shadow=True, fontsize=10)
    plt.tight_layout()
    
    eval_chart_path = os.path.join(os.path.dirname(__file__), 'calibration_chart.png')
    plt.savefig(eval_chart_path)
    print(f"Calibration chart saved to {eval_chart_path}")
    
    # Also save to frontend folder so Streamlit UI can render it
    frontend_chart = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'frontend', 'calibration_chart.png')
    plt.savefig(frontend_chart)
    print(f"Calibration chart copied to {frontend_chart}")
    
    return {
        "thresholds": thresholds,
        "chosen_threshold": chosen_threshold,
        "answered_accuracy": ans_acc_list,
        "not_in_corpus_accuracy": nic_acc_list,
        "overall_accuracy": overall_acc_list
    }


if __name__ == "__main__":
    rerun = "--rerun" in sys.argv
    calibrate(re_run_eval=rerun)
