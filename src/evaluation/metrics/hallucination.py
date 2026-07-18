from deepeval.metrics import HallucinationMetric
from deepeval.models import GeminiModel
from deepeval.test_case import LLMTestCase
from config.settings import settings

# Initialize GeminiModel using the dynamic model name from settings
custom_model = GeminiModel(
    model=settings.MODEL,  # Now pulling "gemini-1.5-flash" from .env
    api_key=settings.GOOGLE_API_KEY
)

def evaluate_hallucination(input_text: str, actual_output: str, retrieval_context: list[str]):
    metric = HallucinationMetric(threshold=0.7, model=custom_model)
    
    test_case = LLMTestCase(
        input=input_text,
        actual_output=actual_output,
        context=retrieval_context
    )
    
    metric.measure(test_case)
    return {"score": metric.score, "reason": metric.reason}