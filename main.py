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

    # Step 1: flip if needed
    if vec.count(1) < vec.count(-1):
        vec = [-x for x in vec]

    # Step 2: find longest circular runs
    max_run, start_indices = find_longest_circular_runs(vec)
    if max_run == 0:
        return tuple(vec)  # no 1's

    # Step 3: generator to avoid storing all rotations
    n = len(vec)
    best = None
    for start in start_indices:
        rotation = tuple(vec[start:] + vec[:start])
        if best is None or rotation > best:
            best = rotation
    return best

# ----------------------------
# LEGAL MOVES
# ----------------------------
def legal_moves(board):
    """Generate all legal moves (0 -> 1 or -1)."""
    for i, val in enumerate(board):
        if val == 0:
            for new_val in [1, -1]:
                new_board = list(board)
                new_board[i] = new_val
                if is_legal(new_board):
                    yield tuple(new_board)  # yield instead of list

# ----------------------------
# GENERATE GAME GRAPH
# ----------------------------
def generate_game_graph(n):
    empty_board = tuple([0] * n)
    start_board = canonicalize_circular(empty_board)

    nodes = set()
    edges = defaultdict(set)
    queue = deque([start_board])

    while queue:
        board = queue.popleft()
        if board in nodes:
            continue
        nodes.add(board)

        for move in legal_moves(board):
            move_canonical = canonicalize_circular(move)
            edges[board].add(move_canonical)
            if move_canonical not in nodes:
                queue.append(move_canonical)

    return nodes, edges

# ----------------------------
# PRETTY PRINT
# ----------------------------
def print_graph(edges, n=5):
    for b, moves in list(edges.items())[:n]:
        print(f"{list(b)} -> {[list(m) for m in moves]}")

# ----------------------------
# EXAMPLE USAGE
# ----------------------------
if __name__ == "__main__":
    n = 15  # can now go higher without memory bloat
    nodes, edges = generate_game_graph(n)
    print(f"Total legal canonical boards for n={n}: {len(nodes)}")
    print("Sample edges:")
    print_graph(edges, 5)
