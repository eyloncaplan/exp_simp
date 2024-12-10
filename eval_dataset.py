from pipeline import *
from tqdm import tqdm
import json
import queue
import random
import pickle


DATASET_FILE = 'data_stats/complex_dataset_decision_stats.json'
BRANCHING_FACTOR = 5


def eval_dataset(dataset, sample_size=3):
    random.seed(2)
    results = []

    # a sample is made up of a problem instance and a test index
    problem_instances = random.sample(range(len(dataset)), sample_size)
    test_indices = random.choices(range(5), k=sample_size)

    for i in range(len(problem_instances)):
        instance_index = problem_instances[i]
        problem_instance = list(dataset.values())[instance_index]
        pi = get_pi(problem_instance, test_indices[i])
        x = problem_instance['examples'][test_indices[i]]['expression1']

        if 'REVERSED' in list(dataset.keys())[instance_index]:
            d1 = list(dataset.values())[instance_index - len(dataset) // 2]['decision1']
            d2 = list(dataset.values())[instance_index - len(dataset) // 2]['decision2']
            big_string = f"(REVERSE) {d1}/{d2}\n{problem_instance['examples'][test_indices[i]]['expression1']} -> {problem_instance['examples'][test_indices[i]]['expression2']}"
        else:
            big_string = f"{problem_instance['decision1']}/{problem_instance['decision2']}\n{problem_instance['examples'][test_indices[i]]['expression1']} -> {problem_instance['examples'][test_indices[i]]['expression2']}"

        # Generate the tree and compute stats
        tree, scores, averages = bfs(x, pi, branching_factor=BRANCHING_FACTOR, debug=True)
        tree_stats = compute_tree_stats(tree, scores, averages, x)

        # Create a JSON representation of the results
        result = {
            "problem_instance": {
                "examples": problem_instance["examples"],
                "description": big_string
            },
            "test_index": test_indices[i],
            "branching_factor": BRANCHING_FACTOR,
            "tree": tree,
            "scores": scores,
            "averages": averages,
            "tree_stats": tree_stats
        }

        results.append(result)

    # Save results to a JSON file
    with open("tree_results.json", "w") as f:
        json.dump(results, f, indent=4)

def eval_few_shot(dataset):
    results = []
    pis = []
    xs = []

    for dp, item in tqdm(dataset.items()):
        for i in range(5):
            pi = get_pi(item, i)
            pis.append(pi)
    
    Hs = get_H_batch(pis)

    for dp, item in tqdm(dataset.items()):
        for i in range(5):
            x = item['examples'][i]['expression1']
            xs.append(x)
    


    final_xs = get_x_final_with_pi_batch(pis, xs)

    pi_scores = get_pi_score_batch(pis, xs, final_xs)
    h_scores = get_H_score_batch(Hs, xs, final_xs)

    # pickle the results in case of failure
    with open("few_shot_results/few_shot_results.pkl", "wb") as f:
        pickle.dump((pis, xs, final_xs, pi_scores, h_scores), f)

    results = []
    for i in range(len(pis)):
        pi_score = pi_scores[i]
        h_score = h_scores[i]
        avg_score = (pi_score + h_score) / 2 if pi_score is not None and h_score is not None else 0

        result = {
            "problem_instance": i,
            "test_index": i%5,
            "pi_score": pi_score,
            "h_score": h_score,
            "avg_score": avg_score
        }

        results.append(result)
    
    # Save results to a JSON file
    with open("few_shot_results/few_shot_results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    return results



def bfs(x, pi, branching_factor=5, debug=True):
    H = get_H(pi)

    q = queue.Queue()
    q.put((x, None))  # Add a tuple of (node, parent)
    tree = {}  # A dictionary to track parent-child relationships
    scores = {}  # A dictionary to track scores for each node
    averages = {}  # A dictionary to track average scores for each node
    highest_scoring_node = (x, 0)  # Track the highest scoring node (node, average_score)

    while not q.empty():
        current_node, parent = q.get()

        # Calculate scores if not already calculated
        if current_node not in scores:
            if parent is not None:
                pi_score = get_pi_score(pi, parent, current_node)
                h_score = get_H_score(H, parent, current_node)
            else:
                pi_score = 1
                h_score = 1
            scores[current_node] = (pi_score, h_score)
            averages[current_node] = (pi_score + h_score) / 2 if pi_score is not None and h_score is not None else 0

        # Update the highest scoring node
        if averages[current_node] > highest_scoring_node[1]:
            highest_scoring_node = (current_node, averages[current_node])

        # Add the current node to the tree structure
        if parent not in tree:
            tree[parent] = []
        tree[parent].append(current_node)

        # If the average score is 5, print the tree (if debug) and return the current node
        if averages[current_node] == 5:
            if debug:
                print_tree(tree, scores, averages, x)
            return tree, scores, averages

        # Stop search if the node's average score is <= its parent's
        if parent is not None and averages[current_node] <= averages[parent]:
            continue

        # Get child nodes
        x_primes = get_all_x_primes(H, current_node, branching_factor, strategy='z3')

        # Add child nodes to the queue
        for x_prime in x_primes:
            q.put((x_prime, current_node))

    # If no node with an average score of 5 is found, print the tree (if debug) and return the highest scoring node
    if debug:
        print_tree(tree, scores, averages, x)
    return tree, scores, averages

def print_tree(tree, scores, averages, root_node):
    """Prints the tree structure with scores and averages."""
    def recursive_print(node, depth=0, is_root=True):
        pi_score, h_score = scores[node]
        avg_score = averages[node]
        score_str = f" (Pi: {pi_score}, H: {h_score}, Avg: {avg_score:.2f})"
        if is_root:
            print(f"|-- {node}{score_str}")
        else:
            print(f"{'|   ' * depth}|-- {node}{score_str}")
        if node in tree:
            for child in tree[node]:
                recursive_print(child, depth + 1, is_root=False)

    recursive_print(root_node)

def compute_tree_stats(tree, scores, averages, root_node):
    """Computes statistics for a tree."""
    from collections import defaultdict
    stats = {
        "total_nodes": 0,
        "max_pi_score": float('-inf'),
        "max_h_score": float('-inf'),
        "max_avg_score": float('-inf'),
        "average_pi_score": 0,
        "average_h_score": 0,
        "average_avg_score": 0,
        "max_depth": 0,
        "nodes_per_depth": defaultdict(int),
        "average_score_per_depth": defaultdict(list),
        "average_pi_score_per_depth": defaultdict(list),
        "average_h_score_per_depth": defaultdict(list)
    }

    total_pi, total_h, total_avg = 0, 0, 0

    def traverse(node, depth=0):
        nonlocal total_pi, total_h, total_avg

        # Get scores and averages
        pi_score, h_score = scores[node]
        avg_score = averages[node]

        # Update statistics
        stats["total_nodes"] += 1
        stats["max_pi_score"] = max(stats["max_pi_score"], pi_score)
        stats["max_h_score"] = max(stats["max_h_score"], h_score)
        stats["max_avg_score"] = max(stats["max_avg_score"], avg_score)
        stats["max_depth"] = max(stats["max_depth"], depth)

        total_pi += pi_score
        total_h += h_score
        total_avg += avg_score

        # Track nodes and scores per depth
        stats["nodes_per_depth"][depth] += 1
        stats["average_score_per_depth"][depth].append(avg_score)
        stats["average_pi_score_per_depth"][depth].append(pi_score)
        stats["average_h_score_per_depth"][depth].append(h_score)

        # Traverse children
        if node in tree:
            for child in tree[node]:
                traverse(child, depth + 1)

    # Traverse the tree starting at the root
    traverse(root_node)

    # Calculate averages
    stats["average_pi_score"] = total_pi / stats["total_nodes"]
    stats["average_h_score"] = total_h / stats["total_nodes"]
    stats["average_avg_score"] = total_avg / stats["total_nodes"]

    # Calculate average scores per depth
    stats["average_score_per_depth"] = {
        depth: sum(scores) / len(scores)
        for depth, scores in stats["average_score_per_depth"].items()
    }
    stats["average_pi_score_per_depth"] = {
        depth: sum(pi_scores) / len(pi_scores)
        for depth, pi_scores in stats["average_pi_score_per_depth"].items()
    }
    stats["average_h_score_per_depth"] = {
        depth: sum(h_scores) / len(h_scores)
        for depth, h_scores in stats["average_h_score_per_depth"].items()
    }

    return stats


if __name__ == '__main__':
    dataset = json.load(open(DATASET_FILE, 'r'))

    # make a new dataset with only the decision pairs that passed all 5 tests
    new_dataset = {}
    new_dataset['decision_pairs'] = {}
    for dp, item in dataset['decision_pairs'].items():
        if item['passed'] == 5:
            new_dataset['decision_pairs'][dp] = item

    new_items = {}
    # double the dataset by making a reversed problem instance for each pair
    for dp, item in new_dataset['decision_pairs'].items():
        new_items[f'{dp} REVERSED'] = {
            'examples': [{"expression1": example['expression2'], "expression2": example['expression1']} for example in item['examples']]
        }

    new_dataset['decision_pairs'].update(new_items)


    # eval_dataset(new_dataset['decision_pairs'], sample_size=1)
    eval_few_shot(new_dataset['decision_pairs'])
    print("All tasks completed.")