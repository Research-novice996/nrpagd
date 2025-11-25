import logging
import random

from abc import ABC, abstractmethod
from typing import List
from core.gen_models import GenerationModel

logger = logging.getLogger(__name__)


class RespRanker(ABC):
    @abstractmethod
    def evaluate(self, context, resp_a, resp_b):
        """
        Compare two responses and return the preference.
        """
        raise NotImplementedError


class ESCEvaluator(RespRanker):
    def __init__(self, gen_model: GenerationModel):
        super().__init__()
        self.gen_model = gen_model
        self.inference_args = {
            "max_tokens": 10,
            "temperature": 0.7,
            "echo": False,
            "n": 5,
            "stop": ""
        }

    def evaluate(self, context, resp_a, resp_b):
        do_swap = False
        if random.random() < 0.5:
            do_swap = True
            resp_a, resp_b = resp_b, resp_a
        prompt = f"""
		The following is a conversation between a Therapist and a Patient in an emotional support session. The patient is seeking help for emotional distress, and the therapist aims to provide effective psychological support to alleviate the patient's concerns or improve their emotional well-being.

Conversation Context:
{context}

Which of the following therapist responses would more effectively help the patient address their emotional struggles or feel supported?

A. Therapist: {resp_a}

B. Therapist: {resp_b}

C. Hard to tell

Choose the best option (A, B, or C).
Your choice:


		""".replace('\t', '').strip()
        logger.debug(f"prompt: {prompt}")
        resps = self.gen_model.generate(prompt, **self.inference_args)
        choices, rationales = self._process_resps(resps)
        preference = self._majority_vote(choices, do_swap)
        return preference, {'choices': choices, 'rationales': rationales, 'do_swap': do_swap}

    def _process_resps(self, resps: List[dict]):
        choices = []
        rationales = []
        for resp in resps:
            gen = resp['generated_text'].strip()

            if len(gen) == 0:
                print("Empty response")
                choice = 'c'
            else:
                choice = gen[0].lower()

            if choice not in ['a', 'b', 'c']:
                print(f"Invalid choice: {choice}")
                choice = 'c'
            choices.append(choice)
            # see if there is a rationale  # just dump the entire response
            rationale = gen
            rationales.append(rationale)
        return choices, rationales

    def _majority_vote(self, resps: List[str], do_swap=False):
        # if there is a majority vote between A=0 and B=1, return the majority vote
        # otherwise, return C=2
        a_cnt = 0
        b_cnt = 0
        for resp in resps:
            if resp == 'a':
                a_cnt += 1
            elif resp == 'b':
                b_cnt += 1
        if a_cnt > b_cnt:
            return 0 if not do_swap else 1
        elif b_cnt > a_cnt:
            return 1 if not do_swap else 0
        return 2