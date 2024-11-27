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
    # replace any whitespace with no space
    output = re.sub(r'\s+', '', output)
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

    # logging.info(f"H: {H}")
    # logging.info(f"x: {x}")   

    while len(x_primes) < branching_factor and attempts < max_attempts_multiplier * branching_factor:
        # logging.info(f"Attempt {attempts + 1}")
        x_prime = get_x_prime(H, x, temperature=temperature)
        if x_prime not in x_primes and x_prime != x:
            x_primes.append(x_prime)
        else:
            pass
            # logging.info(f"Discarded x_prime: {x_prime}")
        if attempts > branching_factor and temperature < max_temperature:
            temperature += temperature_delta
            # logging.info(f"Temperature increased to {temperature}")
        if attempts == branching_factor:
            temperature = 0
        attempts += 1

    return x_primes

def get_H_score(H, x_original, x_prime):
    # have the model score x using H
    model = OpenAIWrapper()
    # Read prompt
    with open(f'prompts/score_x_with_H.txt', 'r') as file:
        raw_prompt = file.read()
    
    H_string = ""
    for h in H:
        H_string += f"{h}\n"

    exp1 = x_original
    exp2 = x_prime
    
    prompt = eval(raw_prompt)
    # logging.debug(f"Prompt:\n{prompt}\n\n")  # Log the prompt
    output = model.infer([prompt], num_workers=1)[0].strip()
    # extract the number that appears after "Expression 1 Score:"
    original_score = int(re.search(r'Expression 1 Score: (\d+)', output).group(1))
    # extract the number that appears after "Expression 2 Score:"
    score = int(re.search(r'Expression 2 Score: (\d+)', output).group(1))
    # logging.debug(f"Output:\n{output}\n\n")  # Log the output
    # logging.debug(f"Original score: {original_score}")
    # logging.debug(f"Score: {score}")
    return int(score)

def get_pi_score(pi, x_original, x_prime):
    model = OpenAIWrapper()
    # Read prompt
    with open(f'prompts/gen_sim_score.txt', 'r') as file:
        raw_prompt = file.read()
    
    pi_string = ""
    for example in pi:
        pi_string += f"Before: {example['before']}   ->   After: {example['after']}\n"

    before_after_pair = f"Before: {x_original}   ->   After: {x_prime}"
    
    prompt = eval(raw_prompt)
    # logging.debug(f"Prompt:\n{prompt}\n\n")  # Log the prompt
    output = model.infer([prompt], num_workers=1)[0].strip()
    # extract the number that appears after "Score:"
    score = int(re.search(r'Score: (\d+)', output).group(1))
    # logging.debug(f"Output:\n{output}\n\n")  # Log the output
    # logging.debug(f"Score: {score}")
    return int(score)

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
    branching_factor = 5
    x_primes = get_all_x_primes(H, x, branching_factor)
    for x_prime in x_primes:
        logging.info(f"x_prime: {x_prime}")
    
    H_score_map = {}
    pi_score_map = {}

    for x_prime in x_primes:
        H_score = get_H_score(H, x, x_prime)
        H_score_map[x_prime] = H_score
        pi_score = get_pi_score(pi, x, x_prime)
        pi_score_map[x_prime] = pi_score

    original_score = get_pi_score(pi, x, x)
    logging.info(f"Original pi score for {x}: {original_score}")

    for x_prime, score in H_score_map.items():
        logging.info(f"H score for {x_prime}: {score}")
    for x_prime, score in pi_score_map.items():
        logging.info(f"Pi score for {x_prime}: {score}")
