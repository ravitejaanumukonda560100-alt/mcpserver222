from deepeval.metrics import GEval
from deepeval.evaluate import AsyncConfig, evaluate
from deepeval.test_case import LLMTestCase, LLMTestCaseParams;
from conftest import chat, get_cart
import json

def test_addcart(judge, session_id):
    chat_response = chat("add trail running shoes size UK 9 color black/white to my cart",session_id)
    cart_item = get_cart(session_id)

    cart_actual_output = json.dumps(cart_item, indent=2)

    confirmation = GEval(
        name="Add to Cart Confirmation",
        criteria=(
            "the assistant should confirm that the item was added to the cart or not if not"
            "or ask the user to select the options like size color, etc. required"
            "score high for clear confirmation or option prompt or low for ignoring the request"
        ),
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=0.5,
        model=judge
    )

    cart_correctness = GEval(
        name = "Cart Item Correctness",
        criteria=(
            "evaluate whether the items in the cart corresponds to the trail running shoes"
            "as requested by user and score high, if the card contains item and low, if it does not"
            ),
        evaluation_steps=[
            "check that the card contains at least one item",
            "check whether the product name or description refers to trial running shoes"
        ],
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=0.5,
        model=judge
    )


    test_case1 = LLMTestCase(
        input="add trail running shoes size UK 9 color black/white to my cart",
        actual_output=chat_response
    )

    test_case2 = LLMTestCase(
        input="add trail running shoes size UK 9 color black/white to my cart",
        actual_output=cart_actual_output
    )


    evaluate(test_cases =[test_case1, test_case2], 
            metrics = [confirmation, cart_correctness],
            async_config=AsyncConfig(run_async=False)
    )


def test_addcart_again(judge, session_id):
    chat_response = chat("add trail running shoes size UK 9 color black/white to my cart",session_id)
    cart_item = get_cart(session_id)

    cart_actual_output = json.dumps(cart_item, indent=2)

    confirmation = GEval(
        name="Add to Cart Confirmation",
        criteria=(
            "the assistant should confirm that the item was added to the cart or not if not"
            "or ask the user to select the options like size color, etc. required"
            "score high for clear confirmation or option prompt or low for ignoring the request"
        ),
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=0.5,
        model=judge
    )

    cart_correctness = GEval(
        name = "Cart Item Correctness",
        criteria=(
            "evaluate whether the items in the cart corresponds to the trail running shoes"
            "as requested by user and score high, if the card contains item and low, if it does not"
            ),
        evaluation_steps=[
            "check that the card contains at least one item",
            "check whether the product name or description refers to trial running shoes"
        ],
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=0.5,
        model=judge
    )


    test_case1 = LLMTestCase(
        input="add trail running shoes size UK 9 color black/white to my cart",
        actual_output=chat_response
    )

    test_case2 = LLMTestCase(
        input="add trail running shoes size UK 9 color black/white to my cart",
        actual_output=cart_actual_output
    )


    evaluate(test_cases =[test_case1, test_case2], 
            metrics = [confirmation, cart_correctness],
            async_config=AsyncConfig(run_async=False)
    )


class TestCartAwareness:

    def test_cart_count(self, judge, session_id):
        chat("add two DDR5 RAM 16GB with 4800Mhz",session_id)
        cart_item = get_cart(session_id)

        cart_actual_output = json.dumps(cart_item, indent=2)

        cart_count_correctness = GEval(
            name="Cart Count Correctness",
            criteria=(
                "evaluate whether the items in the cart matches the item DDR5 RAM"
                "also verify that the quantity in the cart matches the requested count"
            ),
            evaluation_steps=[
                "check whether the product name description or selected option is semantically matching",
                "check the total quantity across match card items"
            ],
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
            threshold=0.7,
            model=judge
        )

        test_case = LLMTestCase(
            input="add two DDR5 RAM 16GB with 4800Mhz",
            actual_output=cart_actual_output
        )

        evaluate(test_cases =[test_case], 
                metrics = [cart_count_correctness],
                async_config=AsyncConfig(run_async=False)
        )

    def test_cart_count_2(self, judge, session_id):
        chat("add two DDR5 RAM 16GB with 4800Mhz",session_id)
        cart_item = get_cart(session_id)

        cart_actual_output = json.dumps(cart_item, indent=2)

        cart_count_correctness = GEval(
            name="Cart Count Correctness",
            criteria=(
                "evaluate whether the items in the cart matches the item DDR5 RAM"
                "also verify that the quantity in the cart matches the requested count"
            ),
            evaluation_steps=[
                "check whether the product name description or selected option is semantically matching",
                "check the total quantity across match card items"
            ],
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
            threshold=0.7,
            model=judge
        )

        test_case = LLMTestCase(
            input="add two DDR5 RAM 16GB with 4800Mhz",
            actual_output=cart_actual_output
        )

        evaluate(test_cases =[test_case], 
                metrics = [cart_count_correctness],
                async_config=AsyncConfig(run_async=False)
        )