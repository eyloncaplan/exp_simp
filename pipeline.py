import json
import re
from openai_wrapper import OpenAIWrapper
import logging
import random


DATASET_FILE = 'data/complex_dataset.json'
debug_mode = False  # Change to True to enable DEBUG messages

# Function to configure logging level
def configure_logging(debug=False):
    if debug:
        logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
    else:
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    logging.getLogger("urllib3").setLevel(logging.CRITICAL)  # Example: urllib3 logs
    logging.getLogger("requests").setLevel(logging.CRITICAL)  # Example: requests logs
    logging.getLogger("openai").setLevel(logging.CRITICAL)
    logging.getLogger("httpx").setLevel(logging.WARNING)  # Suppress httpx logs
    logging.getLogger("httpcore").setLevel(logging.WARNING)

configure_logging(debug=debug_mode)

def get_pi(problem_instance, test_index):
    # Converts problem instance to pi, excluding the test_index
    pi = problem_instance['examples']
    pi.pop(test_index)
    return pi

def get_H(pi):
    model = OpenAIWrapper()
    # Read prompt
    with open(f'prompts/gen_H.txt', 'r') as file:
        raw_prompt = file.read()
    
    # Get pi as a string
    pi_string = ""
    for example in pi:
        pi_string += f"Before: {example['before']}   ->   After: {example['after']}\n"
    
    prompt = eval(raw_prompt)
    logging.debug(f"Prompt:\n{prompt}\n\n")  # Log the prompt
    output = model.infer([prompt], num_workers=1)[0].split('\n')
    logging.debug(f"Output:\n{output}\n\n")  # Log the output
    return output

if __name__ == '__main__':
    # Step 1: Load the JSON data from the file
    with open(DATASET_FILE, 'r') as file:
        dataset = json.load(file)

    # randomly select a problem instance
    problem_instance = random.choice(dataset)
    test_index = 4
    pi = get_pi(problem_instance, test_index)
    get_H(pi)
