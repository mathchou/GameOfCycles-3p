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
            "Current": False,
            "Next": False,
            "Previous": False,
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
    # Initialize all nodes as unknown
    for node in nodes:
        nodes[node]["Current"] = False
        nodes[node]["Previous"] = False
        nodes[node]["Next"] = False

    changed = True
    while changed:
        changed = False
        for node, node_data in nodes.items():
            children = edges[node]

            if len(children) == 0:
                # Sink: current player loses
                new_current = False
                new_previous = True
                new_next = True
            else:
                # If any child leads to a node where Current loses, we can win
                child_currents = [nodes[child]["Current"] for child in children]

                if any(not c for c in child_currents):
                    new_current = True  # we can force a win
                    new_previous = False
                    new_next = False
                else:
                    new_current = False  # all moves bad → lose
                    new_previous = True
                    new_next = True

            # Update if changed
            if (node_data["Current"] != new_current or
                node_data["Previous"] != new_previous or
                node_data["Next"] != new_next):
                node_data["Current"] = new_current
                node_data["Previous"] = new_previous
                node_data["Next"] = new_next
                changed = True




# ----------------------------
# INTERACTIVE PLOT
# ----------------------------
def plot_graph_interactive(nodes, edges):
    G = nx.DiGraph()

    # Add nodes and edges
    for node, attr in nodes.items():
        G.add_node(node, **attr)
    for src, dsts in edges.items():
        for dst in dsts:
            G.add_edge(src, dst)

    # Sort nodes by number of non-zero entries for grid-like layout
    sorted_nodes = sorted(G.nodes(), key=lambda n: G.nodes[n]["layer"])
    layers = defaultdict(list)
    for node in sorted_nodes:
        layers[G.nodes[node]["layer"]].append(node)

    # Assign x/y positions for grid layout: columns = layer, rows = nodes within layer
    pos = {}
    max_len = max(len(nodes) for nodes in layers.values())
    for x, layer in enumerate(sorted(layers.keys())):
        col = layers[layer]
        for y, node in enumerate(col):
            # spread rows evenly
            pos[node] = (x, -y)  # negative y to have top-down ordering

    # Prepare Plotly edge traces
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    # Prepare Plotly node traces
    node_x = []
    node_y = []
    node_text = []
    node_labels = []
    for i, node in enumerate(sorted_nodes):
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        attr = nodes[node]
        node_text.append(
            f"{list(node)}<br>Current: {attr['Current']}, Next: {attr['Next']}, Previous: {attr['Previous']}"
        )
        node_labels.append(str(i))

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_labels,
        hovertext=node_text,
        hoverinfo='text',
        marker=dict(
            size=15,
            line_width=2
        )
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=dict(
                            text='Game Graph',
                            font=dict(size=20)
                        ),
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    )
                    )
    fig.show()

# ----------------------------
# EXAMPLE USAGE
# ----------------------------
if __name__ == "__main__":
    n = 6  # keep small for interactivity
    nodes, edges = generate_game_graph(n)
    compute_winning_strategies(nodes, edges)
    print(f"Total nodes: {len(nodes)}")
    plot_graph_interactive(nodes, edges)
