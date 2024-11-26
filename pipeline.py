import json
import re
from openai_wrapper import OpenAIWrapper
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level (INFO in this case)
    format='%(asctime)s - %(levelname)s - %(message)s',  # Set the logging format
    handlers=[
        logging.StreamHandler()  # Output logs to console
    ]
)

DATASET_FILE = 'data/complex_dataset.json'

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
    logging.info(f"Prompt:\n{prompt}\n\n")  # Log the prompt
    output = model.infer([prompt], num_workers=1)[0].split('\n')
    logging.info(f"Output:\n{output}\n\n")  # Log the output

if __name__ == '__main__':
    # Step 1: Load the JSON data from the file
    with open(DATASET_FILE, 'r') as file:
        dataset = json.load(file)

    for i in range(len(dataset)):
        for j in range(len(dataset)):
            if i == j:
                continue
            if dataset[i]['decision1'] == dataset[j]['decision1'] and dataset[i]['decision2'] == dataset[j]['decision2']:
                logging.error(f"Duplicate decisions found with decision1: {dataset[i]['decision1']} and decision2: {dataset[i]['decision2']}")

    exit()

    problem_instance = dataset[7]
    test_index = 4
    pi = get_pi(problem_instance, test_index)
    get_H(pi)
