import networkx as nx
import plotly.graph_objects as go
from collections import deque, defaultdict
import itertools as it

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
    edges = defaultdict(set)
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

        for move in legal_moves(board):
            move_canonical = canonicalize_circular(move)
            edges[board].add(move_canonical)
            if move_canonical not in nodes:
                queue.append(move_canonical)

    return nodes, edges

# ----------------------------
# Backward propogation of winning strategies
# ----------------------------

def compute_winning_strategies(nodes, edges):
    """
    nodes: dict[node] -> {"Previous": bool, "Current": bool, "Next": bool, "layer": int}
    edges: dict[node] -> iterable of child nodes
    """

    # Group nodes by layer
    layers = {}
    for node, data in nodes.items():
        layer = data["layer"]
        layers.setdefault(layer, []).append(node)

    max_layer = max(layers.keys())

    # Process layers from last to first
    for layer in range(max_layer, -1, -1):
        for node in layers[layer]:
            children = edges.get(node, [])

            # ---- Sink nodes ----
            if len(children) == 0:
                # Player who made the last move (Previous) loses
                nodes[node]["Previous"] = False
                nodes[node]["Current"] = True
                nodes[node]["Next"] = True
                continue

            # Collect child attributes
            child_prevs = [nodes[c]["Previous"] for c in children]
            child_currs = [nodes[c]["Current"] for c in children]
            child_nexts = [nodes[c]["Next"] for c in children]

            # ---- Current ----
            # If there exists a move such that the resulting Previous wins,
            # then Current can force a win
            nodes[node]["Current"] = any(prev for prev in child_prevs)

            # ---- Next ----
            # If in all child positions, Current wins,
            # then Next is guaranteed to win
            nodes[node]["Next"] = all(curr for curr in child_currs)

            # ---- Previous ----
            # If in all child positions, Next wins,
            # then Previous is guaranteed to win
            nodes[node]["Previous"] = all(nxt for nxt in child_nexts)


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
    nodes: dict[node] -> {
        'Previous': bool,
        'Current': bool,
        'Next': bool,
        'layer': int,
        'pos': (x, y)
    }

    edges: dict[src] -> iterable of dst (or (dst, move))
    """

    G = nx.DiGraph()

    # ----------------------------
    # ADD NODES + EDGES
    # ----------------------------
    for node, attrs in nodes.items():
        G.add_node(node, **attrs)

    for src, dsts in edges.items():
        for dst in dsts:
            if isinstance(dst, tuple) and len(dst) == 2:
                dst, move = dst
                G.add_edge(src, dst, move=move)
            else:
                G.add_edge(src, dst, move=None)

    # ----------------------------
    # NODE POSITIONS (GRID)
    # ----------------------------
    pos = {node: attrs["pos"] for node, attrs in nodes.items()}

    ordered_nodes = sorted(
        nodes,
        key=lambda n: (nodes[n]["layer"], pos[n][1])
    )

    # ----------------------------
    # EDGES (NORMAL vs WINNING)
    # ----------------------------
    edge_x_normal, edge_y_normal = [], []
    edge_x_win, edge_y_win = [], []

    for src, dst in G.edges():
        x0, y0 = pos[src]
        x1, y1 = pos[dst]

        if nodes[src]["Current"] and nodes[dst]["Previous"]:
            # Winning move for Current
            edge_x_win += [x0, x1, None]
            edge_y_win += [y0, y1, None]
        else:
            edge_x_normal += [x0, x1, None]
            edge_y_normal += [y0, y1, None]

    edge_trace_normal = go.Scatter(
        x=edge_x_normal,
        y=edge_y_normal,
        mode="lines",
        line=dict(width=1, color="#000"),
        hoverinfo="none",
        showlegend=False
    )

    edge_trace_win = go.Scatter(
        x=edge_x_win,
        y=edge_y_win,
        mode="lines",
        line=dict(width=3, color="green"),
        hoverinfo="none",
        showlegend=False
    )

    # ----------------------------
    # NODES
    # ----------------------------
    node_x = [pos[n][0] for n in ordered_nodes]
    node_y = [pos[n][1] for n in ordered_nodes]
    node_colors = [winning_color(nodes[n]) for n in ordered_nodes]

    node_text = [
        f"{list(n)}<br>"
        f"Previous: {nodes[n]['Previous']}<br>"
        f"Current: {nodes[n]['Current']}<br>"
        f"Next: {nodes[n]['Next']}"
        for n in ordered_nodes
    ]

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
    # LEGEND (FAKE TRACES)
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
    # FIGURE
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


# ----------------------------
# EXAMPLE USAGE
# ----------------------------
if __name__ == "__main__":
    n = 6
    nodes, edges = generate_game_graph(n)
    compute_winning_strategies(nodes, edges)

    assign_grid_positions(nodes)

    print(f"Total nodes: {len(nodes)}")
    plot_graph_interactive(nodes, edges)