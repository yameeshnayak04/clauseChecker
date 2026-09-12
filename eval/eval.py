import json
import sys
import os
import time

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval import retrieve
from classifier import classify_and_answer
from config import CONFIDENCE_THRESHOLD


def run_eval(confidence_threshold=None, verbose=True):
    if confidence_threshold is None:
        confidence_threshold = CONFIDENCE_THRESHOLD
        
    questions_path = os.path.join(os.path.dirname(__file__), 'questions.json')
    with open(questions_path, 'r', encoding='utf-8') as f:
        questions = json.load(f)
        
    results = []
    
    total = len(questions)
    correct = 0
    state_correct = {"ANSWERED": 0, "NOT_IN_CORPUS": 0, "CONTRADICTION": 0}
    state_total = {"ANSWERED": 0, "NOT_IN_CORPUS": 0, "CONTRADICTION": 0}
    
    confusion_matrix = {
        "ANSWERED": {"ANSWERED": 0, "NOT_IN_CORPUS": 0, "CONTRADICTION": 0},
        "NOT_IN_CORPUS": {"ANSWERED": 0, "NOT_IN_CORPUS": 0, "CONTRADICTION": 0},
        "CONTRADICTION": {"ANSWERED": 0, "NOT_IN_CORPUS": 0, "CONTRADICTION": 0}
    }

    if verbose:
        print(f"================================================================")
        print(f"  ClauseCheck Evaluation Suite")
        print(f"  Confidence Threshold: {confidence_threshold}")
        print(f"  Total Questions: {total}")
        print(f"================================================================\n")
    
    for idx, q in enumerate(questions, start=1):
        expected = q["expected_state"]
        query = q["question"]
        
        chunks = retrieve(query)
        res = classify_and_answer(query, chunks, confidence_threshold)
        predicted = res.state
        
        state_total[expected] += 1
        confusion_matrix[expected][predicted] += 1
        
        is_correct = (expected == predicted)
        if is_correct:
            correct += 1
            state_correct[expected] += 1
            status_str = "[PASS]"
        else:
            status_str = "[FAIL]"
            
        if verbose:
            print(f"[{idx:02d}/{total}] {status_str} Expected: {expected:<14} | Predicted: {predicted:<14} | Conf: {res.confidence:.2f}", flush=True)
            print(f"     Q: {query}", flush=True)
            if not is_correct:
                print(f"     Reasoning: {res.reasoning_note}", flush=True)
            print("", flush=True)
            
        results.append({
            "id": q["id"],
            "question": query,
            "expected": expected,
            "predicted": predicted,
            "correct": is_correct,
            "confidence": res.confidence,
            "raw_answer": res.answer,
            "citations": [c.model_dump() for c in res.citations],
            "reasoning_note": res.reasoning_note
        })
        
        time.sleep(0.5)  # Rate limit mitigation for Groq
        
    overall_acc = correct / total if total > 0 else 0
    
    if verbose:
        print("================================================================")
        print("                       EVALUATION SCORECARD                     ")
        print("================================================================")
        print(f"Overall State Accuracy: {overall_acc*100:.1f}% ({correct}/{total})\n")
        
        for st in ["ANSWERED", "NOT_IN_CORPUS", "CONTRADICTION"]:
            if state_total[st] > 0:
                acc = state_correct[st] / state_total[st]
                print(f"  - {st:<14} Accuracy: {acc*100:5.1f}% ({state_correct[st]:2d}/{state_total[st]:2d})")
                
        print("\n----------------------------------------------------------------")
        print("Confusion Matrix (Rows: Expected, Columns: Predicted)")
        print("----------------------------------------------------------------")
        print(f"{'Expected \\ Pred':<18} | {'ANSWERED':>10} | {'NOT_IN_CORPUS':>14} | {'CONTRADICTION':>14}")
        print("-" * 65)
        for expected_st in ["ANSWERED", "NOT_IN_CORPUS", "CONTRADICTION"]:
            row = f"{expected_st:<18} | "
            row += f"{confusion_matrix[expected_st]['ANSWERED']:>10} | "
            row += f"{confusion_matrix[expected_st]['NOT_IN_CORPUS']:>14} | "
            row += f"{confusion_matrix[expected_st]['CONTRADICTION']:>14}"
            print(row)
        print("================================================================\n")
        
    # Save full results to disk
    results_path = os.path.join(os.path.dirname(__file__), 'eval_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump({
            "confidence_threshold": confidence_threshold,
            "overall_accuracy": overall_acc,
            "state_accuracy": {st: state_correct[st] / state_total[st] if state_total[st] > 0 else 0 for st in state_correct},
            "confusion_matrix": confusion_matrix,
            "results": results
        }, f, indent=2)
        
    return {
        "overall_accuracy": overall_acc,
        "state_accuracy": {st: state_correct[st] / state_total[st] if state_total[st] > 0 else 0 for st in state_correct},
        "confusion_matrix": confusion_matrix,
        "results": results
    }


if __name__ == "__main__":
    run_eval()
