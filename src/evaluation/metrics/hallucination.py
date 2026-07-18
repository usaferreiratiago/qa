from deepeval.metrics import HallucinationMetric, AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from deepeval.models import GeminiModel
from config.settings import settings

custom_model = GeminiModel(model=settings.MODEL, api_key=settings.GOOGLE_API_KEY)

def evaluate_agent(input_text: str, actual_output: str, retrieval_context: list[str]):
    test_case = LLMTestCase(
        input=input_text,
        actual_output=actual_output,
        context=retrieval_context
    )
    
    # Initialize metrics
    h_metric = HallucinationMetric(threshold=0.7, model=custom_model)
    r_metric = AnswerRelevancyMetric(threshold=0.7, model=custom_model)
    f_metric = FaithfulnessMetric(threshold=0.7, model=custom_model)
    
    # Measure
    h_metric.measure(test_case)
    r_metric.measure(test_case)
    f_metric.measure(test_case)
    
    return {
        "hallucination": {"score": h_metric.score, "reason": h_metric.reason},
        "relevance": {"score": r_metric.score, "reason": r_metric.reason},
        "faithfulness": {"score": f_metric.score, "reason": f_metric.reason}
    }