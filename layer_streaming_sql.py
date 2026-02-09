import sqlite3
from main import canonicalize_circular


def boards_at_depth(n, depth):
    boards = set()
    board = [0] * n

    # canonical anchor
    if depth > 0:
        board[0] = 1
        start_pos = 1
        remaining = depth - 1
    else:
        return {tuple(board)}

    def fill_intelligently(pos, remaining):
        # impossible to finish
        if remaining > n - pos:
            return

        if pos == n:
            if remaining == 0 and is_legal(board):
                boards.add(tuple(canonicalize_circular(board)))
                #boards.add(tuple(board)) #first just add the board so we can see what's happening
            return

        # option 1: fill in zero, doesn't change remaining non-zeros
        if n - pos > remaining: # only run this if there are more positions left than remaining +-1's to add
            board[pos] = 0
            fill_intelligently(pos + 1, remaining)
            # so, the first boards it will look at will be filling 0's until the last spots are force to be non-zero

        # option 2: fill in +1 or -1, but don't make it negative the previous (illegal board)
        if remaining > 0:
            for v in (1, -1):
                if board[pos-1] == 0 or board[pos-1] == v: # only continues if we aren't creating a 1/-1 or -1/1 sequence
                    board[pos] = v
                    if is_partial_legal(board, pos): # this is now only here for a sanity check, our rules should already guarantee this
                        fill_intelligently(pos + 1, remaining - 1)

    fill_intelligently(start_pos, remaining)
    return boards

def generate_labeled_edges(source):
    """
    Generate all legal moves from source.
    Returns:
        dict: {canonical_dest: set((pos, val))}
    """
    n = len(source)
    labeled_dests = defaultdict(set)

    for i in range(n):
        if source[i] != 0:
            continue

        left = source[(i - 1) % n]
        right = source[(i + 1) % n]

        for val in (1, -1):
            # local legality check
            if left == -val or right == -val:
                continue

            # now it's guaranteed legal
            new_board = list(source)
            new_board[i] = val

            dest = tuple(canonicalize_circular(new_board))
            labeled_dests[dest].add((i, val))

    return labeled_dests

def generate_edges_from_layer(layer):
    """
    Generate all labeled edges from a layer of canonical board states.

    For each source board in the given layer, this function applies all legal
    single-move placements (+1 or -1 in a zero position), canonicalizes the
    resulting board, and records the move(s) that produce each destination.

    Returns:
        dict:
            A nested dictionary of the form

                edges[source][dest] = set((pos, val))

            where:
              - source is a canonical board in the input layer
              - dest is a canonical board reachable from source in one move
              - (pos, val) indicates that placing val ∈ {+1, -1} at index pos
                produces dest (after canonicalization)

            Multiple distinct moves may map to the same destination due to
            rotational or sign symmetries.
    """
    edges = defaultdict(lambda: defaultdict(set))

    for source in layer:
        move_map = generate_labeled_edges(source)

        for dest, moves in move_map.items():
            edges[source][dest].update(moves)

    return {
        source: dict(dest_map)
        for source, dest_map in edges.items()
    }



# -------------------------------------------------------------
# 1. Initialize database
# -------------------------------------------------------------
def init_db(db_path="game_graph.db"):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Table for edges: source -> dest, move info, depth
    cur.execute("""
        CREATE TABLE IF NOT EXISTS edges (
            source TEXT,
            dest TEXT,
            pos INTEGER,
            val INTEGER,
            depth INTEGER
        )
    """)

    # Indexes for fast lookups
    cur.execute("CREATE INDEX IF NOT EXISTS idx_source ON edges(source)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_dest ON edges(dest)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_depth ON edges(depth)")

    conn.commit()
    return conn

# -------------------------------------------------------------
# 2. Generate graph into SQLite
# -------------------------------------------------------------
def generate_game_graph_sqlite(n, db_path="game_graph.db"):
    conn = init_db(db_path)
    cur = conn.cursor()

    prev_layer = set()

    for depth in reversed(range(n + 1)):
        print(f"Processing depth {depth}...")

        # Generate canonical boards at this depth
        raw_layer = boards_at_depth(n, depth)
        current_layer = set(canonicalize_circular(board) for board in raw_layer)

        # Generate labeled edges
        edges_dict = generate_edges_from_layer(current_layer)

        rows = []
        for source, dest_map in edges_dict.items():
            # Only include moves to previous layer (lower depth)
            for dest, moves in dest_map.items():
                if dest not in prev_layer:
                    continue
                for (pos, val) in moves:
                    rows.append((str(source), str(dest), pos, val, depth))

        # Bulk insert edges
        if rows:
            cur.executemany("""
                INSERT INTO edges (source, dest, pos, val, depth)
                VALUES (?, ?, ?, ?, ?)
            """, rows)
            conn.commit()

        prev_layer = current_layer  # next iteration

    conn.close()
    print(f"Graph generation complete for n={n}. Saved to {db_path}")

# -------------------------------------------------------------
# 3. Utility functions for querying SQLite
# -------------------------------------------------------------
def get_moves(conn, board):
    """Return all moves from a board: list of (dest, pos, val)."""
    cur = conn.cursor()
    cur.execute("""
        SELECT dest, pos, val
        FROM edges
        WHERE source = ?
    """, (str(board),))
    return [(eval(dest), pos, val) for dest, pos, val in cur.fetchall()]

def successors(conn, board):
    """Return distinct destinations from a board."""
    cur = conn.cursor()
    cur.execute("""
        SELECT DISTINCT dest
        FROM edges
        WHERE source = ?
    """, (str(board),))
    return [eval(row[0]) for row in cur.fetchall()]

def boards_at_depth_sqlite(conn, depth):
    """Return all canonical boards at a specific depth."""
    cur = conn.cursor()
    cur.execute("""
        SELECT DISTINCT source
        FROM edges
        WHERE depth = ?
    """, (depth,))
    return [eval(row[0]) for row in cur.fetchall()]

# -------------------------------------------------------------
# 4. Example usage
# -------------------------------------------------------------
if __name__ == "__main__":
    import time, tracemalloc

    n = 15  # board size

    start_time = time.time()
    tracemalloc.start()

    generate_game_graph_sqlite(n, db_path=f"game_graph_n{n}.db")

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    elapsed_time = time.time() - start_time

    print(f"Memory usage: {current_mem/1024**2:.3f} MB, Peak: {peak_mem/1024**2:.3f} MB")
    print(f"Time elapsed: {elapsed_time:.3f} seconds")
