from src.evaluation.runners.eval_runner import run_batch_evaluation
from config.settings import settings

def main():
    print(f"Starting QA Suite on {settings.MODEL}...")
    
    # Run the evaluation pipeline
    results = run_batch_evaluation("data/golden_dataset.json")
    
    # Generate a report
    for res in results:
        print(f"Query: {res['query']} | Score: {res['metrics']['score']}")

if __name__ == "__main__":
    main()