from deepeval.metrics import HallucinationMetric, AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from deepeval.models import GeminiModel
from config.settings import settings

# Only initialize the model if we aren't in debug mode
custom_model = None
if not settings.DEBUG_MODE:
    custom_model = GeminiModel(model=settings.MODEL, api_key=settings.GOOGLE_API_KEY)

def evaluate_agent(input_text: str, actual_output: str, retrieval_context: list[str]):
    # MOCK MODE: Returns fake data instantly
    if settings.DEBUG_MODE:
        return {
            "hallucination": {"score": 0.85, "reason": "Mock score (debug mode active)"},
            "relevance": {"score": 0.90, "reason": "Mock score (debug mode active)"},
            "faithfulness": {"score": 0.95, "reason": "Mock score (debug mode active)"}
        }

    # REAL MODE: Only runs if DEBUG_MODE is False
    test_case = LLMTestCase(input=input_text, actual_output=actual_output, context=retrieval_context)
    
    h_metric = HallucinationMetric(threshold=0.7, model=custom_model)
    r_metric = AnswerRelevancyMetric(threshold=0.7, model=custom_model)
    f_metric = FaithfulnessMetric(threshold=0.7, model=custom_model)
    
    h_metric.measure(test_case)
    r_metric.measure(test_case)
    f_metric.measure(test_case)
    
    return {
        "hallucination": {"score": h_metric.score, "reason": h_metric.reason},
        "relevance": {"score": r_metric.score, "reason": r_metric.reason},
        "faithfulness": {"score": f_metric.score, "reason": f_metric.reason}
    }