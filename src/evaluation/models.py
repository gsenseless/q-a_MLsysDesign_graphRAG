from pydantic import BaseModel


class MetricResult(BaseModel):
    passed: bool
    reasoning: str


class EvaluationChecklist(BaseModel):
    factually_grounded: MetricResult
    key_information_retrieved: MetricResult
    search_relevance: MetricResult
    citation_accuracy: MetricResult
    formatting_compliance: MetricResult


class QuestionsList(BaseModel):
    questions: list[str]
