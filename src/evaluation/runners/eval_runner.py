import json
from src.evaluation.metrics.hallucination import evaluate_hallucination

def run_batch_evaluation(dataset_path: str):
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    results = []
    for entry in data:
        # Here you would call your actual Agent logic
        # result = agent.run(entry['input']) 
        
        # Mocking the evaluation
        score = evaluate_hallucination(entry['input'], entry['output'], entry['context'])
        results.append({"query": entry['input'], "metrics": score})
    
    return results