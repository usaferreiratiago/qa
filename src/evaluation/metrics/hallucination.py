from deepeval.test_case import LLMTestCase
from deepeval.metrics import HallucinationMetric

def evaluate_hallucination(input_text: str, actual_output: str, retrieval_context: list[str]):
    # Threshold 0.7: If score < 0.7, it's considered a hallucination
    metric = HallucinationMetric(threshold=0.7)
    test_case = LLMTestCase(
        input=input_text,
        actual_output=actual_output,
        retrieval_context=retrieval_context
    )
    metric.measure(test_case)
    return {"score": metric.score, "reason": metric.reason}