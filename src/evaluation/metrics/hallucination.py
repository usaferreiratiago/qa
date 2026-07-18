from deepeval.metrics import HallucinationMetric
from deepeval.models import OllamaModel
from deepeval.test_case import LLMTestCase

# Initialize Ollama model
custom_model = OllamaModel(model="llama3")

def evaluate_hallucination(input_text: str, actual_output: str, retrieval_context: list[str]):
    # HallucinationMetric requires the 'context' parameter, NOT 'retrieval_context'
    metric = HallucinationMetric(threshold=0.7, model=custom_model)
    
    test_case = LLMTestCase(
        input=input_text,
        actual_output=actual_output,
        context=retrieval_context  # Pass your list of strings here as 'context'
    )
    
    metric.measure(test_case)
    return {"score": metric.score, "reason": metric.reason}