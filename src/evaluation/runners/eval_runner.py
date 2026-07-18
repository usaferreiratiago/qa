import json
from src.evaluation.metrics.hallucination import evaluate_hallucination

def run_batch_evaluation(dataset_path: str):
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    results = []
    for entry in data:
        # Evaluate using the local judge
        score = evaluate_hallucination(
            input_text=entry['input'], 
            actual_output=entry['expected_output'], # Use expected as actual for testing your setup
            retrieval_context=entry['context']
        )
        results.append({"query": entry['input'], "metrics": score})
    
    return results