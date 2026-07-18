from src.evaluation.runners.eval_runner import run_batch_evaluation
from config.settings import settings

def main():
    print(f"Starting QA Suite on {settings.MODEL}...")
    
    # Run the evaluation pipeline
    results = run_batch_evaluation("data/golden_dataset.json")
    
    # Generate a report
    for res in results:
        metrics = res['metrics']
        h_score = metrics['hallucination']['score']
        r_score = metrics['relevance']['score']
        
        print(f"Query: {res['query']}")
        print(f"  -> Hallucination Score: {h_score:.2f}")
        print(f"  -> Relevance Score: {r_score:.2f}")
        
        # Optional: Print reasons if the score is low (e.g., < 0.7)
        if h_score < 0.7:
            print(f"  -> Hallucination Reason: {metrics['hallucination']['reason']}")
        if r_score < 0.7:
            print(f"  -> Relevance Reason: {metrics['relevance']['reason']}")

if __name__ == "__main__":
    main()