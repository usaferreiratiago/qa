from deepeval.metrics import HallucinationMetric
from deepeval.models import OllamaModel
from deepeval.test_case import LLMTestCase

# Initialize Ollama model
# This connects to your local instance (default: http://localhost:11434)
custom_model = OllamaModel(model="llama3")

def evaluate_hallucination(input_text: str, actual_output: str, retrieval_context: list[str]):
    # Use the Ollama model as the LLM-as-a-Judge
    metric = HallucinationMetric(threshold=0.7, model=custom_model)
    
    test_case = LLMTestCase(
        input=input_text,
        actual_output=actual_output,
        retrieval_context=retrieval_context
    )
    
    metric.measure(test_case)
    return {"score": metric.score, "reason": metric.reason}