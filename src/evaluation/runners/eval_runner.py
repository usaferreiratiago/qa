import json
from src.evaluation.metrics.hallucination import evaluate_hallucination

def run_batch_evaluation(dataset_path: str):
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    results = []
    for entry in data:
        # Use the correct key: 'output'
        # I've added a fallback to an empty string to prevent crashes if a key is missing
        actual_output = entry.get('output', "")
        input_text = entry.get('input', "")
        context = entry.get('context', [])
        
        # Evaluate using the local judge
        score = evaluate_hallucination(
            input_text=input_text, 
            actual_output=actual_output, 
            retrieval_context=context
        )
        results.append({"query": input_text, "metrics": score})
    
    return results