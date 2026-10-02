from deepeval.metrics import GEval, ConversationalGEval
from deepeval.evaluate import AsyncConfig, evaluate
from deepeval.test_case import LLMTestCase, LLMTestCaseParams, ConversationalTestCase, Turn, TurnParams
from conftest import chat
import pytest

class TestChatbotConversation:

    @pytest.mark.parametrize("general_phrases", [
        "what is the capital of France?",
        "how to write Selenium with C#.net code",
        "who is the current Chief Minister of Tamil Nadu, India?" 
    ])
    def test_Out_ofscope_graceful_handling(self, judge, session_id, general_phrases):

        # Arrange
        actual_output = chat(general_phrases,session_id)

        graceful = GEval(
            name="Out-of-scope Graceful Handling",
            criteria=(
                "the assistant is a shopping assistant and should not be expected to answer general questions"
                "it should politely redirect the user to shopping site topics or acknowledge. It cannot help with the question."
            ),
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
            threshold=0.3,
            model=judge
        )

        test_case = LLMTestCase(
            input=general_phrases,
            actual_output=actual_output
        )

        # Assertion
        evaluate(test_cases =[test_case], 
                metrics = [graceful],
                async_config=AsyncConfig(run_async=False)
        )


    def test_multi_turn_price_negotiation(self, judge, session_id):
        turn1_input = "recommend me a good pair of shoes"
        turn1_output = chat(turn1_input, session_id)
        turn2_input= "do you have anything cheaper?"
        turn2_output= chat(turn2_input, session_id)

        cheaper_option = ConversationalGEval(
            name = "Cheaper Alternative Suggestions",
            criteria=(
                "The user asked for a cheaper alternative after being shown a list of shoes"
                "the assistant should suggest one or more cheaper alternative of the same item"
            ),
            evaluation_params=[TurnParams.ROLE, TurnParams.CONTENT],
            threshold=0.4,
            model=judge
        )

        test_case = ConversationalTestCase(
            turns=[
                Turn(role="user", content=turn1_input),
                Turn(role="assistant", content=turn1_output),
                Turn(role="user", content=turn2_input),
                Turn(role="assistant", content=turn2_output)
            ],
            chatbot_role="Shopping assistant that suggests products and prices"
        )

        evaluate(
            test_cases=[test_case],
            metrics=[cheaper_option]
        )

    def test_pronoun_resolution_after_product_mention(self, judge, session_id):
        turn1_input = "Tell me about the wireless Bluetooth headphones you haves"
        turn1_output = chat(turn1_input, session_id)
        turn2_input= "What colors it has got"
        turn2_output= chat(turn2_input, session_id)

        pronoun_resolution = ConversationalGEval(
            name = "Pronoun Resolution",
            criteria=(
              "the prayer conversation was about a wireless Bluetooth headphone"
              "the follow up question what color it has got should be resolved to the headphone and return the colors the product has got"
            ),
            evaluation_params=[TurnParams.ROLE, TurnParams.CONTENT],
            threshold=0.4,
            model=judge
        )

        test_case = ConversationalTestCase(
            turns=[
                Turn(role="user", content=turn1_input),
                Turn(role="assistant", content=turn1_output),
                Turn(role="user", content=turn2_input),
                Turn(role="assistant", content=turn2_output)
            ],
            chatbot_role="Shopping assistant that suggests products and prices"
        )

        evaluate(
            test_cases=[test_case],
            metrics=[pronoun_resolution]
        )

