from deepeval.metrics import HallucinationMetric
from deepeval.models import Gemini
from deepeval.test_case import LLMTestCase
from config.settings import settings

# Initialize Gemini
# DeepEval automatically picks up GOOGLE_API_KEY from your environment 
# if you don't pass it explicitly, but you can pass it via 'google_api_key' argument.
custom_model = Gemini(
    model="gemini-1.5-flash",
    google_api_key=settings.GOOGLE_API_KEY
)

def evaluate_hallucination(input_text: str, actual_output: str, retrieval_context: list[str]):
    # The metric now uses Gemini instead of Ollama
    metric = HallucinationMetric(threshold=0.7, model=custom_model)
    
    test_case = LLMTestCase(
        input=input_text,
        actual_output=actual_output,
        context=retrieval_context
    )
    
    metric.measure(test_case)
    return {"score": metric.score, "reason": metric.reason}