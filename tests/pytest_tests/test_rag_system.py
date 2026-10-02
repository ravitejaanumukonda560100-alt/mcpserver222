from deepeval.metrics import GEval, ContextualRelevancyMetric
from deepeval.evaluate import AsyncConfig, evaluate
from deepeval.test_case import LLMTestCase, LLMTestCaseParams;
from conftest import chat, get_cart, clear_cart, rag_search
import json
import pytest


def retrieval_context(query: str, k: int = 5) -> list[str]:
    results = rag_search(query)
    contexts =[]
    for p in results[:k]:
        text = f"{p["name"]} ({p['category']}, ${p['price']:.2f}): {p.get('description', '')}"

        if p.get("options"):
            opts = "; ".join(
                f"{o['name']}: {', '.join(o['values'])}"
                for o in p['options']
            )
            text += f" Options - {opts}."
        contexts.append(text)
    return contexts


class TestRAG:
    @pytest.mark.parametrize("search_phrase", [
            "workout and fitness gear",
            "PC equipment",
        ])
    def test_rag_for_context_relevancy(self, judge, session_id, search_phrase):

        # Arrange
        actual_output = chat(search_phrase,session_id)
        retrieval_ctx = retrieval_context(search_phrase)

        contextualRelevancy = ContextualRelevancyMetric(
            include_reason=True,
            threshold=0.3,
            model=judge
        )

        test_case = LLMTestCase(
            input=search_phrase,
            actual_output=actual_output,
            retrieval_context=retrieval_ctx
        )

        # Assertion
        evaluate(test_cases =[test_case], 
                metrics = [contextualRelevancy]
        )

    @pytest.mark.parametrize("search_phrase, equipments", [
        ("build me a sports fitness bundle","sports gear"),
        ("build me a PC gaming machine","pc components"),
        ("build me a kitchen equipment", "kitchen items")
    ])
    def test_rag_for_relavancy_with_Geval(self, judge, session_id, search_phrase,equipments):

        # Arrange
        actual_output = chat(search_phrase,session_id)
        retrieval_ctx = retrieval_context(search_phrase)

        contextualRelevancyWithGeval = GEval(
            name="Contextual Relevancy with GEval",
            criteria=(
                f"the user is asking for {search_phrase}"
                f"the retrieval context should contain the relevant information about each of the equipments as {equipments}"
            ),
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.RETRIEVAL_CONTEXT
            ],
            threshold=0.3,
            model=judge
        )

        test_case = LLMTestCase(
            input=search_phrase,
            actual_output=actual_output,
            retrieval_context=retrieval_ctx
        )

        # Assertion
        evaluate(test_cases =[test_case], 
                metrics = [contextualRelevancyWithGeval]
        )

  