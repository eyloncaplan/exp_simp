#!/usr/bin/env python3
import json
import networkx as nx
from networkx.drawing.nx_agraph import to_agraph
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import tempfile

# ─── ADJUST THESE ────────────────────────────────────────────────────────────────
HORIZONTAL_SEP = 1.5   # inches between sibling nodes
VERTICAL_SEP   = 0.65  # inches between ranks (parent/child)
# ────────────────────────────────────────────────────────────────────────────────

def build_graph_from_tree(data):
    """
    Build a DiGraph where each node is labeled with its average score.
    """
    tree     = data.get("tree", {})
    averages = data.get("averages", {})
    G = nx.DiGraph()

    # Root(s)
    for root in tree.get("null", []):
        avg   = averages.get(root, 0)
        label = f"{root}\n(avg score: {avg})"
        G.add_node(root, label=label, avg=avg)

    # Edges + other nodes
    for parent, children in tree.items():
        if parent == "null":
            continue
        if not G.has_node(parent):
            avg   = averages.get(parent, 0)
            label = f"{parent}\n(avg score: {avg})"
            G.add_node(parent, label=label, avg=avg)
        for child in children:
            if not G.has_node(child):
                avg   = averages.get(child, 0)
                label = f"{child}\n(avg score: {avg})"
                G.add_node(child, label=label, avg=avg)
            G.add_edge(parent, child)

    return G

def plot_tree(G, output_file=None):
    """
    Layout + draw via AGraph so that nodesep/ranksep actually take effect,
    shape nodes as rectangles, and color them by avg score.
    """
    # Convert to a PyGraphviz AGraph
    A = to_agraph(G)
    # Set Graphviz attributes
    A.graph_attr.update(
        nodesep=str(HORIZONTAL_SEP),
        ranksep=str(VERTICAL_SEP),
        rankdir='TB'         # top→bottom
    )
    # Make all nodes rectangles
    A.node_attr.update(shape='rectangle')

    # Extract avg scores
    avgs = [data.get('avg', 0) for _, data in G.nodes(data=True)]
    min_avg = min(avgs)
    max_avg = max(avgs)
    cmap = cm.get_cmap('RdYlGn')  # red=low, green=high

    # Color each node
    for node in G.nodes():
        avg = G.nodes[node].get('avg', 0)
        norm = 0.0 if max_avg == min_avg else (avg - min_avg) / (max_avg - min_avg)
        rgba = cmap(norm)
        hex_color = mcolors.to_hex(rgba)
        n = A.get_node(node)
        n.attr['style']     = 'filled'
        n.attr['fillcolor'] = hex_color

    if output_file:
        A.draw(output_file, prog='dot')
        print(f"Plot saved to {output_file}")
    else:
        tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        A.draw(tmp.name, prog='dot')
        img = plt.imread(tmp.name)
        plt.figure(figsize=(20, 8))
        plt.imshow(img)
        plt.axis('off')
        plt.show()

def plot_tree_from_json(json_file, output_file=None):
    """
    Reads the JSON file, builds a directed graph from the tree data,
    and plots the tree hierarchically, saving it if specified.
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    G = build_graph_from_tree(data)
    plot_tree(G, output_file=output_file)

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python plot_tree.py <json_file> [output_file.png]")
        sys.exit(1)
    json_path   = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) >= 3 else None
    plot_tree_from_json(json_path, output_path)
