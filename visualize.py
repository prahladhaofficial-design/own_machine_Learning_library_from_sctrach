import os
from pathlib import Path

default_graphviz_bin = Path(r"C:\Program Files\Graphviz\bin")
if default_graphviz_bin.exists():
    os.environ["PATH"] += os.pathsep + str(default_graphviz_bin)

import graphviz
from my_ml_lib.nn.autograd import Value
import numpy as np


def get_all_nodes_and_edges(root_node: Value):
    """
    Performs a backward traversal from the root_node
    to find all unique Value nodes and the directed edges connecting them
    in the computation graph.


    Args:
        root_node (Value): The final node in the graph (e.g., the loss Value object).


    Returns:
        tuple: (nodes, edges)
               nodes (set): A set containing all Value objects found during traversal.
               edges (set): A set of tuples (parent_Value, child_Value) representing
                            the directed edges: parent -> child.
    """
    # --- Step 1 - Initialize Sets ---
    nodes = set()
    edges = set()
    visited = set()


    # --- Step 2 - DFS Traversal Function (`build_graph`) ---
    def build_graph(v):
        # a) Check if `v` has already been visited. If yes, return.
        if v in visited:
            return
        # b) Add `v` to the `visited` set.
        visited.add(v)
        # c) Add `v` to the `nodes` set.
        nodes.add(v)
        # d) Iterate through the parents of `v`
        for parent in v._prev:
            # i) Add the edge `(parent, v)` to the `edges` set.
            edges.add((parent, v))
            # ii) Recursively call the helper function on the `parent`.
            build_graph(parent)


    # --- Step 3 - Start Traversal ---
    # Call your recursive helper function starting from the `root_node`.
    build_graph(root_node)


    # --- Step 4 - Return Results ---
    # Return the populated `nodes` and `edges` sets.
    return nodes, edges
# ---  Graph Drawing Function ---
def draw_dot(root_node: Value, format='png', rankdir='LR'):
    """
    Generates a visualization of the computation graph using graphviz.
    Requires the `get_all_nodes_and_edges` function to be implemented correctly.


    Args:
        root_node (Value): The final node of the graph to visualize (e.g., loss).
        format (str): Output format ('svg', 'png', etc.). Default 'svg'.
        rankdir (str): Graph layout direction ('LR' or 'TB'). Default 'LR'.


    Returns:
        graphviz.Digraph: The graph object ready for rendering.
    """
    assert rankdir in ['LR', 'TB']
    nodes, edges = get_all_nodes_and_edges(root_node)


    dot = graphviz.Digraph(format=format, graph_attr={'rankdir': rankdir})


    for n in nodes:
        uid = str(id(n))


        data_str = f"shape={n.data.shape}" if hasattr(n, 'data') and isinstance(n.data, np.ndarray) and n.data.ndim > 0 else f"{getattr(n, 'data', '?'):.4f}"
        grad_str = f"shape={n.grad.shape}" if hasattr(n, 'grad') and isinstance(n.grad, np.ndarray) and n.grad.ndim > 0 else f"{getattr(n, 'grad', '?'):.4f}"
        label_str = f" | {getattr(n, 'label', '')}" if getattr(n, 'label', '') else ""
        node_label = f"{{ data {data_str} | grad {grad_str}{label_str} }}"
        dot.node(name=uid, label=node_label, shape='record')


        op = getattr(n, '_op', '')
        if op:
            op_uid = uid + op
            dot.node(name=op_uid, label=op)
            dot.edge(op_uid, uid)


    for n1, n2 in edges:
        parent_uid = str(id(n1))
        child_op = getattr(n2, '_op', '')
        if child_op:
            child_op_uid = str(id(n2)) + child_op
            dot.edge(parent_uid, child_op_uid)


    return dot


# --- Example Usage ---
if __name__ == '__main__':
    print("\n================= MLP Visualization Example =================")

    try:
        from my_ml_lib.nn.modules.linear import Linear
        from my_ml_lib.nn.modules.activations import ReLU
        from my_ml_lib.nn.modules.containers import Sequential

        # Build tiny model: Linear(2  3) + ReLU
        model = Sequential(Linear(2, 3), ReLU())

        # Input Value
        x = Value(np.array([[1.0, 2.0]]), label='x')

        # Forward pass
        out = model(x)
        out.label = 'output'

        # Backpropagation
        try:
            # We need to sum the output to get a scalar for backward()
            # This is a common pattern for testing.
            scalar_out = out.sum()
            scalar_out.backward()
        except Exception as err:
            print(f"Backward pass failed: {err}. Visualization will still work.")

        # Draw and save computation graph, starting from the final output `out`
        dot_graph = draw_dot(out)
        if dot_graph:
            try:
                output_filename = 'mlp_computation_graph'
                dot_graph.render(output_filename, view=False)
                print(f"Saved MLP graph as {output_filename}.svg/.png")
                print("Please include this generated MLP graph in your report for Problem 4.")
            except graphviz.backend.execute.ExecutableNotFound:
                print("\n--- Graphviz Error ---")
                print("Graphviz executable not found. Visualization not saved.")
                print("Please install Graphviz (from www.graphviz.org)")
                print("and ensure the 'dot' command is available in your system's PATH.")
                print("----------------------\n")
            except Exception as err:
                print(f"An error occurred during graph rendering: {err}")

    except ImportError as err:
        print(f"Could not import neural-network modules: {err}")
        print("Ensure that my_ml_lib.nn.modules.linear, activations, and containers exist.")

    print("\n============================================================\n")
