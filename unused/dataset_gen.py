import json
import re
from openai_wrapper import OpenAIWrapper
import random, itertools

LEGAL_SYMBOLS = ['a', 'b', 'c', 'x', 'y', 'z', '-', '+', '*', '/', '(', ')', '^', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
DECISIONS_FILE = 'style_decisions.json'
ORIGINAL_PROMPT_FILE = 'gen_bad_example.txt'
FIXED_PROMPT_FILE = 'fix_example.txt' 
NUM_EXPRESSIONS = 3
MIN_SYMBOLS = 5

def parse_expressions(output: str):
    # Regular expression to match the pairs of expressions
    expression_pattern = re.compile(r"(.+?),\s*(.+)")
    expression_pairs = []
    
    # Split the output by newlines and match each line to the pattern
    for line in output.strip().split('\n'):
        match = expression_pattern.match(line)
        if match:
            # Strip out any < or > symbols from the expressions
            exp1, exp2 = match.groups()
            exp1 = exp1.replace('<', '').replace('>', '').strip()
            exp2 = exp2.replace('<', '').replace('>', '').strip()
            expression_pairs.append((exp1, exp2))
    
    return expression_pairs

# Step 1: Load the JSON data from the file
with open(f'data/{DECISIONS_FILE}', 'r') as file:
    decisions = json.load(file)

# Load the entire prompt as a single string
with open(f'prompts/{ORIGINAL_PROMPT_FILE}', 'r') as file:
    original_raw_prompt = file.read()

# Load the entire prompt as a single string
with open(f'prompts/{FIXED_PROMPT_FILE}', 'r') as file:
    fixed_raw_prompt = file.read()

model = OpenAIWrapper(temperature=0.5)

expressions = []

# sample some random combinations of two decisions
decision_sample = random.sample(list(itertools.combinations(decisions, 2)), NUM_EXPRESSIONS)

for item in decision_sample:
    decision_str = ""
    for decision_item in item:
        decision = decision_item['decision']
        explanation = decision_item['explanation']
        decision_str += f"{decision}: {explanation}\n"

    print(f"Decision: {decision_str}")

    # num_expressions = NUM_EXPRESSIONS
    legal_symbols = LEGAL_SYMBOLS
    min_symbols = MIN_SYMBOLS

    prompt = eval(original_raw_prompt)
    output = model.infer([prompt], num_workers=1, engine='gpt-4o-mini')[0]
    # get the string after the string "Expression:"
    original_expression = output[output.index("Expression:") + len("Expression:"):].split('\n')[0].strip()

    # print(f"Prompt:\n{prompt}")

    # print(f"Original:\n{output}")
    # print()

    prompt = eval(fixed_raw_prompt)
    output = model.infer([prompt], num_workers=1, engine='gpt-4o-mini')[0]
    fixed_expression = output[output.index("Expression:") + len("Expression:"):].split('\n')[0].strip()
    # print(f"Prompt:\n{prompt}")
    # print(f"Fixed:\n{output}")
    expressions.append((original_expression, fixed_expression))

for exp in expressions:
    print(f"Original: {exp[0]}\nFixed: {exp[1]}\n")


