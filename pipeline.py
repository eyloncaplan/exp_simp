import json
import re
from openai_wrapper import OpenAIWrapper
import logging
import random
import copy


DATASET_FILE = 'data/complex_dataset.json'
debug_mode = True  # Change to True to enable DEBUG messages

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

def get_pi(problem_instance, test_index):
    # Converts problem instance to pi, excluding the test_index
    pi = problem_instance['examples']
    pi = copy.deepcopy(pi)
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
    # logging.debug(f"Prompt:\n{prompt}\n\n")  # Log the prompt
    output = model.infer([prompt], num_workers=1)[0].split('\n')
    output = [x.strip() for x in output if x != '']
    # logging.debug(f"Output:\n{output}\n\n")  # Log the output
    return output

def get_x_prime(H, x, temperature=0.1):
    # Get the x_prime from the H and x, which is a string
    model = OpenAIWrapper(temperature=temperature)
    # Read prompt
    with open(f'prompts/gen_x_primes_from_x_and_H.txt', 'r') as file:
        raw_prompt = file.read()
    
    H_string = ""
    for h in H:
        H_string += f"{h}\n"
    
    exp = x
    prompt = eval(raw_prompt)
    # logging.debug(f"Prompt:\n{prompt}\n\n")  # Log the prompt
    output = model.infer([prompt], num_workers=1)[0].strip()
    # logging.debug(f"Output:\n{output}\n\n")  # Log the output
    return output

def get_all_x_primes(H, x, branching_factor, max_attempts_multiplier=3):
    # keeps generating x_primes until the branching factor is reached
    # discards x_primes that are the same as x or have already been generated
    # does not attempt more than max_attempts * branching_factor times
    x_primes = []
    attempts = 0
    temperature = 0.1
    max_temperature = 2
    temperature_delta = max_temperature / ((max_attempts_multiplier - 1) * branching_factor)

    logging.debug(f"H: {H}")
    logging.debug(f"x: {x}")   

    while len(x_primes) < branching_factor and attempts < max_attempts_multiplier * branching_factor:
        logging.debug(f"Attempt {attempts + 1}")
        x_prime = get_x_prime(H, x, temperature=temperature)
        if x_prime not in x_primes and x_prime != x:
            x_primes.append(x_prime)
        else:
            logging.debug(f"Discarded x_prime: {x_prime}")
        attempts += 1
        if attempts > branching_factor and temperature < max_temperature:
            temperature += temperature_delta
            logging.debug(f"Temperature increased to {temperature}")
    return x_primes
    

if __name__ == '__main__':
    configure_logging(debug=debug_mode)
    # Step 1: Load the JSON data from the file
    with open(DATASET_FILE, 'r') as file:
        dataset = json.load(file)

    # randomly select a problem instance
    problem_instance = random.choice(dataset)
    test_index = 4
    pi = get_pi(problem_instance, test_index)
    H = get_H(pi)
    x = problem_instance['examples'][test_index]['before']
    branching_factor = 10
    x_primes = get_all_x_primes(H, x, branching_factor)
    for x_prime in x_primes:
        logging.info(f"x_prime: {x_prime}")
