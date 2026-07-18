from deepeval.metrics import HallucinationMetric
from deepeval.test_case import LLMTestCase
from deepeval.models import GeminiModel
from config.settings import settings

# Initialize the model at the module level so it's available everywhere
custom_model = GeminiModel(
    model=settings.MODEL,
    api_key=settings.GOOGLE_API_KEY
)

def evaluate_hallucination(input_text: str, actual_output: str, retrieval_context: list[str]):
    # Use the pre-initialized custom_model
    print(f"DEBUG - Query: {input_text}")
    print(f"DEBUG - Agent Answer: {actual_output}")
    print(f"DEBUG - Context Snippet: {retrieval_context[:100]}") # Show start of context
    metric = HallucinationMetric(threshold=0.7, model=custom_model)
    
    test_case = LLMTestCase(
        input=input_text,
        actual_output=actual_output,
        context=retrieval_context
    )
    
    metric.measure(test_case)
    
    # Debugging output (optional but recommended for your current 0.0 score)
    if metric.score < 0.7:
        print(f"--- FAILED EVALUATION ---")
        print(f"Reason: {metric.reason}")
        
    return {"score": metric.score, "reason": metric.reason}