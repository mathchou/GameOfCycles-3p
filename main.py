from collections import deque, defaultdict

# ----------------------------
# LEGALITY CHECK
# ----------------------------
def is_legal(vec):
    """Return True if no 1 is adjacent to -1 (circular)."""
    n = len(vec)
    for i in range(n):
        if (vec[i] == 1 and vec[(i + 1) % n] == -1) or (vec[i] == -1 and vec[(i + 1) % n] == 1):
            return False
    return True

# ----------------------------
# CIRCULAR CANONICAL FORM
# ----------------------------
def find_longest_circular_runs(vec):
    """Return the length and start indices of all longest runs of 1's (circular)."""
    n = len(vec)
    doubled = vec + vec  # duplicate for circularity
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

    return max_run, list(set(start_indices))  # remove duplicates

def canonicalize_circular(vec):
    """
    Strong canonical form:
    1. Flip if -1's outnumber 1's
    2. Rotate so the longest run of 1's is first (circular)
    3. Pick the lexicographically largest rotation
    """
    vec = [int(x) for x in vec]  # ensure Python ints

    # Step 1: flip if needed
    if vec.count(1) < vec.count(-1):
        vec = [-x for x in vec]

    # Step 2: find longest circular runs
    max_run, start_indices = find_longest_circular_runs(vec)
    if max_run == 0:
        return tuple(vec)  # no 1's

    # Step 3: rotations starting at longest runs
    rotations = [vec[start:] + vec[:start] for start in start_indices]

    # Step 4: pick lexicographically largest
    return tuple(max(rotations))

# ----------------------------
# LEGAL MOVES
# ----------------------------
def legal_moves(board):
    """Return all legal boards after one move (change a 0 to 1 or -1)."""
    moves = []
    for i, val in enumerate(board):
        if val == 0:
            for new_val in [1, -1]:
                new_board = list(board)
                new_board[i] = new_val
                if is_legal(new_board):
                    moves.append(new_board)
    return moves

# ----------------------------
# GENERATE GAME GRAPH
# ----------------------------
def generate_game_graph(n):
    """
    Generate all legal canonical boards and map moves.
    Returns:
        nodes: set of canonical boards
        edges: dict mapping canonical board -> list of canonical boards reachable in one move
    """
    empty_board = tuple([0] * n)
    start_board = canonicalize_circular(empty_board)

    nodes = set()
    edges = defaultdict(list)
    queue = deque([start_board])

    while queue:
        board = queue.popleft()
        if board in nodes:
            continue
        nodes.add(board)

        for move in legal_moves(board):
            move_canonical = canonicalize_circular(move)
            if move_canonical not in edges[board]:
                edges[board].append(move_canonical)
            if move_canonical not in nodes:
                queue.append(move_canonical)

    return nodes, edges

# ----------------------------
# PRETTY PRINT
# ----------------------------
def print_graph(edges, n=5):
    """Print a sample of the graph nicely."""
    for b, moves in list(edges.items())[:n]:
        print(f"{list(b)} -> {[list(m) for m in moves]}")

# ----------------------------
# EXAMPLE USAGE
# ----------------------------
if __name__ == "__main__":
    n = 10
    nodes, edges = generate_game_graph(n)
    print(f"Total legal canonical boards for n={n}: {len(nodes)}")
    print("Sample edges:")
    print_graph(edges, n)
