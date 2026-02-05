import networkx as nx
import pickle
import plotly.graph_objects as go
from collections import deque, defaultdict

# ----------------------------
# LEGALITY & CANONICAL FUNCTIONS
# ----------------------------
def is_legal(vec):
    n = len(vec)
    for i in range(n):
        if (vec[i] == 1 and vec[(i + 1) % n] == -1) or (vec[i] == -1 and vec[(i + 1) % n] == 1):
            return False
    return True

def find_longest_circular_runs(vec):
    n = len(vec)
    doubled = vec + vec
    max_run = 0
    current_run = 0
    temp_start = 0
    start_indices = []

    for i, v in enumerate(doubled):
        if v == 1:
            if current_run == 0:
                temp_start = i
            current_run += 1
            if i < n:
                if current_run > max_run:
                    max_run = current_run
                    start_indices = [temp_start % n]
                elif current_run == max_run:
                    start_indices.append(temp_start % n)
        else:
            current_run = 0

    return max_run, list(set(start_indices))

def canonicalize_circular(vec):
    vec = [int(x) for x in vec]
    if vec.count(1) < vec.count(-1):
        vec = [-x for x in vec]

    max_run, start_indices = find_longest_circular_runs(vec)
    if max_run == 0:
        return tuple(vec)

    n = len(vec)
    best = None
    for start in start_indices:
        rotation = tuple(vec[start:] + vec[:start])
        if best is None or rotation > best:
            best = rotation
    return best

def legal_moves(board):
    for i, val in enumerate(board):
        if val == 0:
            for new_val in [1, -1]:
                new_board = list(board)
                new_board[i] = new_val
                if is_legal(new_board):
                    yield tuple(new_board)

# ----------------------------
# GENERATE GRAPH
# ----------------------------
def generate_game_graph(n):
    empty_board = tuple([0] * n)
    start_board = canonicalize_circular(empty_board)

    nodes = {}  # key: board tuple, value: attributes dict
    edges = defaultdict(list)  # store tuples: (dst, (index, value))

    queue = deque([start_board])

    while queue:
        board = queue.popleft()
        if board in nodes:
            continue

        # Add node with initial turn attributes all False
        nodes[board] = {
            "Previous": False,
            "Current": False,
            "Next": False,
            "layer": sum(1 for x in board if x != 0)  # number of moves
        }

        for i, val in enumerate(board):
            if val == 0:
                for new_val in [1, -1]:
                    new_board = list(board)
                    new_board[i] = new_val
                    if is_legal(new_board):
                        move_canonical = canonicalize_circular(new_board)
                        edges[board].append((move_canonical, (i, new_val)))  # store both index and value
                        if move_canonical not in nodes:
                            queue.append(move_canonical)

    return nodes, edges


# ----------------------------
# Backward propogation of winning strategies
# ----------------------------

def compute_winning_strategies(nodes, edges, save_path=None):
    """
    Compute winning strategies for all nodes.
    Optionally save the graph to a .gpickle file for later loading.

    Args:
        nodes: dict[node] -> {"Previous": bool, "Current": bool, "Next": bool, "layer": int, ...}
        edges: dict[node] -> iterable of child nodes
        save_path: str or None. If provided, save as NetworkX gpickle.
    """

    # Group nodes by layer
    layers = defaultdict(list)
    for node, data in nodes.items():
        layer = data["layer"]
        layers[layer].append(node)

    max_layer = max(layers.keys())

    # Process layers from last to first
    for layer in range(max_layer, -1, -1):
        for node in layers[layer]:
            children = [dst for dst, move in edges.get(node, [])]

            # ---- Sink nodes ----
            if len(children) == 0:
                nodes[node]["Previous"] = False
                nodes[node]["Current"] = True
                nodes[node]["Next"] = True
                continue

            child_prevs = [nodes[c]["Previous"] for c in children]
            child_currs = [nodes[c]["Current"] for c in children]
            child_nexts = [nodes[c]["Next"] for c in children]

            # ---- Current ----
            nodes[node]["Current"] = any(prev for prev in child_prevs)

            # ---- Next ----
            nodes[node]["Next"] = all(curr for curr in child_currs)

            # ---- Previous ----
            nodes[node]["Previous"] = all(nxt for nxt in child_nexts)

    # Optional: save graph as pickle
    if save_path:
        G = nx.DiGraph()
        for node, attrs in nodes.items():
            G.add_node(node, **attrs)
        for src, dsts in edges.items():
            for dst in dsts:
                if isinstance(dst, tuple) and len(dst) == 2:
                    dst, move = dst
                    G.add_edge(src, dst, move=move)
                else:
                    G.add_edge(src, dst, move=None)

        with open(save_path, "wb") as f:
            pickle.dump(G, f)
        print(f"Graph saved to {save_path}")


# ----------------------------
# Node coloring helper function
# ----------------------------

def winning_color(attrs):
    p = attrs["Previous"]
    c = attrs["Current"]
    n = attrs["Next"]

    if c and not p and not n:
        return "blue"
    if c and p and not n:
        return "green"
    if p and not c and not n:
        return "yellow"
    if p and not c and n:
        return "orange"
    if n and not p and not c:
        return "red"
    if c and n and not p:
        return "purple"

    # Safety net (should not happen)
    if p and c and n:
        return "black"   # logically suspicious
    return "gray"        # no one wins (also suspicious)


# ----------------------------
# INTERACTIVE PLOT
# ----------------------------

def assign_grid_positions(nodes):
    """
    Assigns (x, y) positions so that:
    - x = layer
    - y = index within that layer
    """

    layers = defaultdict(list)

    # Preserve insertion order
    for node, attrs in nodes.items():
        layers[attrs["layer"]].append(node)

    for layer, layer_nodes in layers.items():
        for i, node in enumerate(layer_nodes):
            nodes[node]["pos"] = (layer, -i)


def plot_graph_interactive(nodes, edges):
    """
    Interactive Plot:
    - Hover over a node to highlight all outgoing winning moves.
    - Edge hover disabled; only node hover triggers display.
    """

    G = nx.DiGraph()

    # ----------------------------
    # Add nodes and edges
    # ----------------------------
    for node, attrs in nodes.items():
        G.add_node(node, **attrs)

    for src, dsts in edges.items():
        for dst, move_info in dsts:
            G.add_edge(src, dst, move=move_info)

    # ----------------------------
    # Node positions (grid)
    # ----------------------------
    pos = {node: attrs["pos"] for node, attrs in nodes.items()}

    ordered_nodes = sorted(
        nodes,
        key=lambda n: (nodes[n]["layer"], pos[n][1])
    )

    # ----------------------------
    # EDGES
    # ----------------------------
    edge_x_normal, edge_y_normal, edge_text_normal = [], [], []
    edge_x_win, edge_y_win, edge_text_win = [], [], []

    for src, dst, data in G.edges(data=True):
        x0, y0 = pos[src]
        x1, y1 = pos[dst]

        move_info = data.get("move")
        label = ""
        if move_info is not None:
            move_index, move_value = move_info
            label = f"Move: index {move_index}, value {move_value}"

        # Determine winning move for Current player
        if nodes[src]["Current"] and nodes[dst]["Previous"]:
            edge_x_win += [x0, x1, None]
            edge_y_win += [y0, y1, None]
            edge_text_win += [label, label, None]
        else:
            edge_x_normal += [x0, x1, None]
            edge_y_normal += [y0, y1, None]
            edge_text_normal += [label, label, None]

    # Normal edges
    edge_trace_normal = go.Scatter(
        x=edge_x_normal,
        y=edge_y_normal,
        mode="lines",
        line=dict(width=1, color="#000"),
        showlegend=False
    )

    # Winning edges: thin visible green line
    edge_trace_win = go.Scatter(
        x=edge_x_win,
        y=edge_y_win,
        mode="lines",
        line=dict(width=3, color="green"),
        showlegend=False
    )


    # ----------------------------
    # Nodes
    # ----------------------------
    node_x = [pos[n][0] for n in ordered_nodes]
    node_y = [pos[n][1] for n in ordered_nodes]
    node_colors = [winning_color(nodes[n]) for n in ordered_nodes]

    node_text = []
    for n in ordered_nodes:
        # List all outgoing moves and their resulting node
        moves_info = []
        for dst, move_info in edges.get(n, []):
            idx, val = move_info
            moves_info.append(f"Move {idx}: {val} → {list(dst)}")
        move_text = "<br>".join(moves_info) if moves_info else "No moves"
        node_text.append(
            f"{list(n)}<br>"
            f"Previous: {nodes[n]['Previous']}<br>"
            f"Current: {nodes[n]['Current']}<br>"
            f"Next: {nodes[n]['Next']}<br>"
            f"{move_text}"
        )

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        hoverinfo="text",
        text=node_text,
        marker=dict(
            size=16,
            color=node_colors,
            line=dict(width=1, color="black")
        ),
        showlegend=False
    )

    # ----------------------------
    # Legend
    # ----------------------------
    legend_items = [
        ("Current only", "blue"),
        ("Current + Previous", "green"),
        ("Previous only", "yellow"),
        ("Previous + Next", "orange"),
        ("Next only", "red"),
        ("Current + Next", "purple"),
    ]

    legend_traces = [
        go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(size=14, color=color),
            name=label,
            showlegend=True
        )
        for label, color in legend_items
    ]

    # ----------------------------
    # Figure
    # ----------------------------
    fig = go.Figure(
        data=[edge_trace_normal, edge_trace_win, node_trace] + legend_traces,
        layout=go.Layout(
            title=dict(
                text="Game Graph — Winning Strategy Coalitions",
                font=dict(size=20)
            ),
            showlegend=True,
            hovermode="closest",
            margin=dict(l=10, r=10, t=50, b=10),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
    )

    fig.show()


# -------------------------------------
# Looking Locally
# -------------------------------------

def extract_subgraph(nodes, edges, center_node, layers=2):
    """
    Extract a subgraph centered at `center_node`.
    - layers=2 includes two moves forward from the center node.
    """
    sub_nodes = {center_node: nodes[center_node]}
    sub_edges = defaultdict(set)

    frontier = [center_node]
    for _ in range(layers):
        next_frontier = []
        for node in frontier:
            for child in edges.get(node, []):
                # If the edge is stored with move info
                if isinstance(child, tuple) and len(child) == 2:
                    dst, move = child
                else:
                    dst, move = child, None

                sub_nodes[dst] = nodes[dst]
                sub_edges[node].add((dst, move))
                next_frontier.append((dst, move))
        frontier = [dst for dst, _ in next_frontier]

    return sub_nodes, sub_edges


def plot_zoomed_node(nodes, edges, center_node, layers=2):
    """
    Extracts a local neighborhood and plots it.
    """
    sub_nodes, sub_edges = extract_subgraph(nodes, edges, center_node, layers)

    # Assign grid positions for this subgraph
    assign_grid_positions(sub_nodes)

    plot_graph_interactive(sub_nodes, sub_edges)



# ----------------------------
# EXAMPLE USAGE
# ----------------------------

import time
import tracemalloc

if __name__ == "__main__":
    n = 20  # set your board size here

    for n in range(17,21):
        print(f"Generating game graph for n = {n}...")

        # ----------------------------
        # Start timing and memory tracking
        # ----------------------------
        start_time = time.time()
        tracemalloc.start()

        # ----------------------------
        # Generate nodes and edges
        # ----------------------------
        nodes, edges = generate_game_graph(n)
        compute_winning_strategies(nodes, edges, save_path=f"game_graph_n{n}.pkl")
        assign_grid_positions(nodes)  # needed for plotting if n < 12

        # ----------------------------
        # Stop memory tracking
        # ----------------------------
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        elapsed_time = time.time() - start_time

        # ----------------------------
        # Report general info
        # ----------------------------
        print(f"Total nodes: {len(nodes)}")
        print(f"Time elapsed: {elapsed_time:.3f} seconds")
        print(f"Current memory usage: {current_mem / 1024**2:.3f} MB")
        print(f"Peak memory usage: {peak_mem / 1024**2:.3f} MB")

        if n < 12:
            # Plot interactive figure with winning moves highlighted
            plot_graph_interactive(nodes, edges)
        else:
            # For large n, just report base node
            start_board = tuple([0] * n)
            start_canon = canonicalize_circular(start_board)
            attrs = nodes[start_canon]
            print(f"Starting board: {list(start_canon)}")
            print(f"Winning strategy at start:")
            print(f"  Current:  {attrs['Current']}")
            print(f"  Next:     {attrs['Next']}")
            print(f"  Previous: {attrs['Previous']}")


    # ---------------------------------------
    # For zooming in to a specific node:
    # ---------------------------------------

"""
    n = 8
    nodes, edges = generate_game_graph(n)
    compute_winning_strategies(nodes, edges)

    # User chooses a node to zoom into (can also be start_board)
    start_board = canonicalize_circular(tuple([0] * n))

    # Plot a neighborhood of 2 layers around the chosen node
    plot_zoomed_node(nodes, edges, center_node=start_board, layers=2)
"""