from pipeline import *
import json
import queue
import argparse

DATASET_FILE = 'data_stats/complex_dataset_decision_stats.json'
BRANCHING_FACTOR = 5  # Global variable for the branching factor
OUTPUT_DIR = 'tree_results'

def eval_single_instance(dataset, instance_index, test_index):
    problem_instance = list(dataset.values())[instance_index]
    pi = get_pi(problem_instance, test_index)
    x = problem_instance['examples'][test_index]['expression1']

    if 'REVERSED' in list(dataset.keys())[instance_index]:
        d1 = list(dataset.values())[instance_index - len(dataset) // 2]['decision1']
        d2 = list(dataset.values())[instance_index - len(dataset) // 2]['decision2']
        big_string = f"(REVERSE) {d1}/{d2}\n{problem_instance['examples'][test_index]['expression1']} -> {problem_instance['examples'][test_index]['expression2']}"
    else:
        big_string = f"{problem_instance['decision1']}/{problem_instance['decision2']}\n{problem_instance['examples'][test_index]['expression1']} -> {problem_instance['examples'][test_index]['expression2']}"
    
    print(big_string)
    for i in range(len(pi)):
        print(f"Example: {pi[i]['expression1']} -> {pi[i]['expression2']}")

    # Generate the tree and compute stats
    tree, scores, averages = bfs(x, pi, branching_factor=BRANCHING_FACTOR, debug=True)
    tree_stats = compute_tree_stats(tree, scores, averages, x)

    # Create a JSON representation of the results
    result = {
        "problem_instance": {
            "examples": problem_instance["examples"],
            "description": big_string
        },
        "test_index": test_index,
        "branching_factor": BRANCHING_FACTOR,
        "tree": tree,
        "scores": scores,
        "averages": averages,
        "tree_stats": tree_stats
    }

    # Save results to a JSON file
    output_filename = f"{OUTPUT_DIR}/tree_results_instance_{instance_index}_test_{test_index}.json"
    with open(output_filename, "w") as f:
        json.dump(result, f, indent=4)

    print(f"Results saved to {output_filename}")

def bfs(x, pi, branching_factor=5, debug=True):
    H = get_H(pi)

    q = queue.Queue()
    q.put((x, None))  # Add a tuple of (node, parent)
    tree = {}  # A dictionary to track parent-child relationships
    scores = {}  # A dictionary to track scores for each node
    averages = {}  # A dictionary to track average scores for each node
    visited = set()  # Track visited nodes to prevent cycles

    while not q.empty():
        current_node, parent = q.get()

        # Skip processing if this node has already been visited
        if current_node in visited:
            continue
        visited.add(current_node)

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

        # Add the current node to the tree structure
        if parent not in tree:
            tree[parent] = []
        tree[parent].append(current_node)

        # Stop search if the node's average score is <= its parent's
        if parent is not None and averages[current_node] <= averages[parent]:
            continue

        # Get child nodes
        x_primes = get_all_x_primes(H, current_node, branching_factor, strategy='z3')

        # Add child nodes to the queue
        for x_prime in x_primes:
            q.put((x_prime, current_node))

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
    parser = argparse.ArgumentParser(description="Evaluate a single problem instance.")
    parser.add_argument("instance_index", type=int, help="The index of the problem instance to evaluate.")
    parser.add_argument("test_index", type=int, help="The index of the test to evaluate.")

    args = parser.parse_args()
    instance_index = args.instance_index
    test_index = args.test_index

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

    eval_single_instance(new_dataset['decision_pairs'], instance_index, test_index)
