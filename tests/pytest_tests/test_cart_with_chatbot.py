from deepeval.metrics import GEval
from deepeval.evaluate import AsyncConfig, evaluate
from deepeval.test_case import LLMTestCase, LLMTestCaseParams;
from conftest import chat, get_cart, clear_cart
import json
import pytest

class TestCartClearAwareness:

    @pytest.mark.parametrize("clear_phrase", [
        "clear my cart",
        "empty my cart",
        "remove everything from my cart" 
    ])
    def test_clear_cart_via_chat(self, judge, session_id, clear_phrase):

        # Arrange
        # Step 0 - Add an Item explicity in the cart
        chat("add two DDR5 RAM 16GB with 4800Mhz",session_id)

        # Act
        # Step 1 - Clear the cart
        chat(clear_phrase, session_id)

        cart_item = get_cart(session_id)

        cart_actual_output = json.dumps(cart_item, indent=2)

        cart_count_correctness = GEval(
            name="Cart Count Correctness",
            criteria=(
                "evaluate whether the items in the cart is empty and fully removed as requested by user"
            ),
            evaluation_steps=[
                "check the total quantity of items in the cart is zero"
            ],
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
            threshold=0.7,
            model=judge
        )

        test_case = LLMTestCase(
            input="add two DDR5 RAM 16GB with 4800Mhz",
            actual_output=cart_actual_output
        )

        # Assertion
        evaluate(test_cases =[test_case], 
                metrics = [cart_count_correctness],
                async_config=AsyncConfig(run_async=False)
        )



class TestCartUpdate:

    @pytest.mark.parametrize("item_to_add,item_detail", [
        ("add trail running shoes size UK 9 color black/white to my cart","XT Trail Running Shoes"),
        ("add two DDR4 RAM 16GB with 4800Mhz","Corsair DDR4 RAM"),
        ("add three wireless earbuds with black color", "wireless earbuds")
    ])
    @pytest.mark.parametrize("update_item",[
        "change the quantity",
        "Change quantity of ",
        "Change quantity to "
        "update the quantity",
        "make it",
        "lower to",
        "want only",
        "just"
    ])
    def test_cart_item_update(self, judge, session_id, item_to_add, item_detail, update_item):

        # Arrange
        # Step 0 - Clear Cart
        clear_cart(session_id=session_id)

        # Step 0 - Add an Item in the cart
        chat(item_to_add,session_id)

        # Act
        # Step 1 - Update the cart item
        update_prompt = f"{update_item} of {item_detail} to 1"
        chat(update_prompt, session_id)

        cart_item = get_cart(session_id)

        cart_actual_output = json.dumps(cart_item, indent=2)

        cart_count_correctness = GEval(
            name="Cart Count Correctness",
            criteria=(
                f"evaluate whether the items in the cart match {item_detail} as requested by user"
                "verify that the quantity in the cart matches the requested count exactly"
            ),
            evaluation_steps=[
                "check the total quantity of items in the cart item is equal to the requested quantity of 1"
            ],
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
            threshold=0.7,
            model=judge
        )

        test_case = LLMTestCase(
            input=update_prompt,
            actual_output=cart_actual_output
        )

        # Assertion
        evaluate(test_cases =[test_case], 
                metrics = [cart_count_correctness],
                async_config=AsyncConfig(run_async=False)
        )

