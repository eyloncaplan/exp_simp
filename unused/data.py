import json
import pandas as pd

DATASET_FILE = 'data/complex_dataset.json'

def load_dataset():
    with open(DATASET_FILE, 'r') as f:
        dict_data = json.load(f)
    
    # Convert the dictionary into a dataframe
    rows = []
    for item in dict_data:
        decision1 = item["decision1"]
        decision2 = item["decision2"]
        for example in item["examples"]:
            rows.append({
                "decision1": decision1,
                "decision2": decision2,
                "before": example["before"],
                "after": example["after"]
            })

    # Create DataFrame
    df = pd.DataFrame(rows)
    return df

import random

def print_random_sample_groups(df, num_groups=3, rows_per_group=3):
    """
    Prints out random sample groups of rows from the dataframe, where each group has the same decision1 and decision2.
    
    Parameters:
        df: pandas DataFrame
            The DataFrame to print the groups from.
        num_groups: int
            The number of groups of rows to print.
        rows_per_group: int
            The number of rows per group.
    """
    # Group by 'decision1' and 'decision2'
    grouped = df.groupby(['decision1', 'decision2'])

    # Randomly select num_groups groups
    unique_pairs = list(grouped.groups.keys())
    selected_pairs = random.sample(unique_pairs, num_groups)
    
    group_count = 0

    for decision1, decision2 in selected_pairs:
        group = grouped.get_group((decision1, decision2))

        # Print decision pairs with spacing
        print(f"### Decision Pair {group_count + 1}: ###\n")
        print(f"Decision 1: {decision1}")
        print(f"Decision 2: {decision2}")
        print("-" * 100)
        
        # Print rows with extra space between columns
        print(group.head(rows_per_group).to_string(index=False, col_space=20))
        
        print("\n" + "=" * 100 + "\n" * 3)  # More spacing between groups

        group_count += 1

# Example usage within your main block:
if __name__ == "__main__":
    df = load_dataset()
    print_random_sample_groups(df)
    print("#" * 100)
    print(len(df))  # Print the total number of rows in the dataset

