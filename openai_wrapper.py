import openai
import os
from gpt_suite import gpt_mp_handler  # Make sure to import your concurrent handling module

class OpenAIWrapper:
    def __init__(self, max_tokens=1500, temperature=0.1):
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.client = openai.OpenAI()
        self.max_tokens = max_tokens
        self.temperature = temperature
        openai.api_key = self.api_key

    def infer(self, prompts, engine='gpt-4o-mini', max_tokens=None, num_workers=1):
        # # CHANGE THIS LATER BRO
        # num_workers = 1
        # print(f"infer input: {prompts}")
        if max_tokens is None:
            max_tokens = self.max_tokens

        if num_workers > 1:
            config = {"temperature": self.temperature, "max_tokens": max_tokens}
            handler = gpt_mp_handler.GPTMPHandler(api_key=self.api_key, gen_conf=config, num_worker=num_workers)
            batch = []
            for prompt in prompts:
                ins = {
                    'init_context': '',
                    'questions': [prompt],
                    'model_name': engine
                }
                batch.append(ins)
            handler.add_batch(batch)
            outs = handler.process()
            print("outs: ", outs)
            responses = [list(d.values())[0] for d in outs]
            # print(f"infer (multiprocess) output: {responses}")
            return responses
        else:
            responses = []
            for prompt in prompts:
                try:
                    response = self.client.chat.completions.create(
                        model=engine,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=max_tokens
                    )
                    responses.append(response.choices[0].message.content.strip())
                except Exception as e:
                    responses.append('Error: ' + str(e))
            # print(f"infer output: {responses}")
            return responses


def wrapper_test():
    model = OpenAIWrapper()
    prompts = ["The sky is", "My haircut looks really"]
    return model.infer(prompts, num_workers=2)  # Adjust num_workers as needed for concurrency

