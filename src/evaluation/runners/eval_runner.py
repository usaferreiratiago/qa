from deepeval.metrics import HallucinationMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from deepeval.models import GeminiModel
from config.settings import settings

# Initialize the model at the module level
custom_model = GeminiModel(
    model=settings.MODEL,
    api_key=settings.GOOGLE_API_KEY
)

def evaluate_agent(input_text: str, actual_output: str, retrieval_context: list[str]):
    # Define the test case
    test_case = LLMTestCase(
        input=input_text,
        actual_output=actual_output,
        context=retrieval_context
    )
    
    # Define metrics using the correct library class names
    h_metric = HallucinationMetric(threshold=0.7, model=custom_model)
    r_metric = AnswerRelevancyMetric(threshold=0.7, model=custom_model)
    
    # Measure
    h_metric.measure(test_case)
    r_metric.measure(test_case)
    
    # Return results
    return {
        "hallucination": {
            "score": h_metric.score,
            "reason": h_metric.reason
        },
        "relevance": {
            "score": r_metric.score,
            "reason": r_metric.reason
        }
    }