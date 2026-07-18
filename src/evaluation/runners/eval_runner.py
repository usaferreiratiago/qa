import json
import time
from src.evaluation.metrics.hallucination import evaluate_agent

def run_batch_evaluation(dataset_path: str):
    # Load your dataset
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    results = []
    for entry in data:

        time.sleep(5)
        # Extract inputs
        actual_output = entry.get('output', "")
        input_text = entry.get('input', "")
        context = entry.get('context', [])
        
        
        # Evaluate using the function defined in hallucination.py
        metrics = evaluate_agent(
            input_text=input_text, 
            actual_output=actual_output, 
            retrieval_context=context
            
        )
        
        # Log to terminal
        print(f"Query: {input_text}")
        print(f"  -> Hallucination Score: {metrics['hallucination']['score']}")
        print(f"  -> Relevance Score: {metrics['relevance']['score']}")
        
        results.append({
            "query": input_text, 
            "metrics": metrics
        })
    
    return results