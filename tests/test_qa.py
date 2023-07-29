import random

from langchain.evaluation import load_evaluator
from tests.qa_data import qa_data
from module_rag.chatbot import ChatBot

bot = ChatBot()


def test_chatbot():

    evaluator = load_evaluator("labeled_criteria", criteria="correctness")

    for qa in qa_data:
        # We can even override the model's learned knowledge using ground truth labels
        eval_result = evaluator.evaluate_strings(
            input=qa['question'],
            prediction=bot.chat(qa['question'], user_id=random.randint(1, 1000)),
            reference=qa['answer']
        )
        print(f'With ground truth: {eval_result["score"]}')


test_chatbot()