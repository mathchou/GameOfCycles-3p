import sqlite3
from collections import defaultdict
from main import canonicalize_circular, is_legal
import time, tracemalloc

def is_partial_legal(board, i):
    # no adjacent different nonzeros
    if i > 0 and board[i] != 0 and board[i] + board[i - 1] == 0:
        return False
    return True

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
# 2a. Generate graph into SQLite
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
# 2b. Generate graph with strategies into SQLite
# -------------------------------------------------------------

def generate_game_graph_and_strategies_sqlite(n, db_path=None):
    """
    Memory-efficient, streaming game graph generation with winning strategies.
    Stores everything in SQLite. Measures memory and time.

    Returns:
        elapsed_time: float
        peak_memory_MB: float
    """
    if db_path is None:
        db_path = f"game_graph_n{n}.db"

    start_time = time.time()
    tracemalloc.start()

    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # -----------------------------
    # Create tables
    # -----------------------------
    c.execute("DROP TABLE IF EXISTS edges")
    c.execute("DROP TABLE IF EXISTS nodes")

    c.execute("""
              CREATE TABLE edges
              (
                  source TEXT,
                  dest   TEXT,
                  move   TEXT
              )
              """)
    c.execute("""
              CREATE TABLE nodes
              (
                  board    TEXT PRIMARY KEY,
                  layer    INT,
                  Previous INT,
                  Current  INT,
                  Next     INT
              )
              """)
    conn.commit()

    # -----------------------------
    # Streaming generation: from deepest layer to shallowest
    # -----------------------------
    prev_layer = set()

    for depth in reversed(range(n + 1)):
        print(f"Processing layer {depth}...")
        raw_layer = boards_at_depth(n, depth)
        current_layer = set(canonicalize_circular(board) for board in raw_layer)

        # Insert nodes
        rows_nodes = [(str(board), depth, 0, 0, 0) for board in current_layer]
        c.executemany(
            "INSERT OR IGNORE INTO nodes (board, layer, Previous, Current, Next) VALUES (?, ?, ?, ?, ?)",
            rows_nodes
        )
        conn.commit()

        # Generate edges and bulk insert
        edge_rows = []
        for source in current_layer:
            labeled_edges = generate_labeled_edges(source)
            for dest, moves in labeled_edges.items():
                move_str = str(list(moves))
                edge_rows.append((str(source), str(dest), move_str))
        if edge_rows:
            c.executemany("INSERT INTO edges (source, dest, move) VALUES (?, ?, ?)", edge_rows)
            conn.commit()

        # Compute winning strategies for current layer
        for node in current_layer:
            c.execute("SELECT dest FROM edges WHERE source=?", (str(node),))
            children = [eval(row[0]) for row in c.fetchall()]

            if not children:
                # Sink node
                c.execute(
                    "UPDATE nodes SET Previous=0, Current=1, Next=1 WHERE board=?",
                    (str(node),)
                )
                continue

            # Fetch child statuses
            child_prevs, child_currs, child_nexts = [], [], []
            for child in children:
                c.execute("SELECT Previous, Current, Next FROM nodes WHERE board=?", (str(child),))
                row = c.fetchone()
                if row:
                    child_prevs.append(row[0])
                    child_currs.append(row[1])
                    child_nexts.append(row[2])
                else:
                    child_prevs.append(0)
                    child_currs.append(0)
                    child_nexts.append(0)

            Previous = int(all(child_nexts))
            Current = int(any(child_prevs))
            Next = int(all(child_currs))

            c.execute(
                "UPDATE nodes SET Previous=?, Current=?, Next=? WHERE board=?",
                (Previous, Current, Next, str(node))
            )

        conn.commit()
        prev_layer = current_layer

    conn.close()

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    elapsed_time = time.time() - start_time
    print("Graph and strategies written to SQLite successfully.")
    print(f"n={n} completed: time={elapsed_time:.2f}s, peak memory={peak_mem / 1024 ** 2:.2f} MB")
    return elapsed_time, peak_mem / 1024 ** 2


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
# 4. Building plot
# -------------------------------------------------------------

import plotly.graph_objects as go

## ----------------------------
## Winning color helper
## ----------------------------
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

    if p and c and n:
        return "black"
    return "gray"

## ----------------------------
## Streaming incremental plot with progressive HTML saving
## ----------------------------
def plot_graph_streaming_html(db_path, n, html_path="game_graph_stream.html", save_every_layer=1):
    """
    Streams layers from the SQLite database and plots incrementally.
    Saves the figure progressively to an HTML file.

    Args:
        db_path: str, path to SQLite DB
        n: int, number of positions (max depth)
        html_path: str, output HTML path
        save_every_layer: int, save HTML every N layers
    """
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    fig = go.Figure()

    # Add a little padding on the x-axis so the first node (layer 0) is visible
    fig.update_layout(
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[-1, n + 1]  # <-- left padding for layer 0, right padding for last layer
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False
        ),
    )

    layer_prev = {}

    # ----------------------------
    # Load first layer
    # ----------------------------
    c.execute("SELECT board, Previous, Current, Next FROM nodes WHERE layer = 0")
    for idx, (board_str, prev, curr, nxt) in enumerate(c.fetchall()):
        board = tuple(eval(board_str))
        layer_prev[board] = {
            "layer": 0, "Previous": prev, "Current": curr, "Next": nxt, "pos": (0, -idx)
        }
        for board, attrs in layer_prev.items():
            x, y = attrs["pos"]
            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode="markers",
                marker=dict(size=16, color=winning_color(attrs), line=dict(width=1, color="black")),
                text=f"{list(board)}<br>Prev:{attrs['Previous']} Curr:{attrs['Current']} Next:{attrs['Next']}",
                hoverinfo="text",
                showlegend=False
            ))
    # ----------------------------
    # Stream layers 1..n
    # ----------------------------
    for layer in range(1, n + 1):
        print(f"Processing layers {layer - 1} and {layer}...")

        # Load current layer
        layer_curr = {}
        c.execute("SELECT board, Previous, Current, Next FROM nodes WHERE layer = ?", (layer,))
        for idx, (board_str, prev, curr, nxt) in enumerate(c.fetchall()):
            board = tuple(eval(board_str))
            layer_curr[board] = {
                "layer": layer, "Previous": prev, "Current": curr, "Next": nxt, "pos": (layer, -idx)
            }

        # Load edges for prev layer
        if layer_prev:
            placeholders = ",".join("?" for _ in layer_prev)
            c.execute(f"SELECT source, dest, move FROM edges WHERE source IN ({placeholders})",
                      [str(b) for b in layer_prev])
            edges_rows = c.fetchall()

            edges_layer = defaultdict(list)
            for src_str, dest_str, move_str in edges_rows:
                src = tuple(eval(src_str))
                dest = tuple(eval(dest_str))
                move_info = eval(move_str)
                for move in move_info:
                    edges_layer[src].append((dest, move))

            # Add edges to figure
            for src, dsts in edges_layer.items():
                x0, y0 = layer_prev[src]["pos"]
                for dst, move_info in dsts:
                    x1, y1 = layer_curr[dst]["pos"] if dst in layer_curr else (layer, 0)
                    color = "green" if layer_prev[src]["Current"] and layer_prev[src]["Previous"] else "black"
                    fig.add_trace(go.Scatter(
                        x=[x0, x1], y=[y0, y1],
                        mode="lines",
                        line=dict(color=color, width=2),
                        hoverinfo="none",
                        showlegend=False
                    ))

        # Add nodes for current layer
        for board, attrs in layer_curr.items():
            x, y = attrs["pos"]
            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode="markers",
                marker=dict(size=16, color=winning_color(attrs), line=dict(width=1, color="black")),
                text=f"{list(board)}<br>Prev:{attrs['Previous']} Curr:{attrs['Current']} Next:{attrs['Next']}",
                hoverinfo="text",
                showlegend=False
            ))

        # Progressive HTML save
        if layer % save_every_layer == 4 or layer == n:
            fig.write_html(html_path)
            print(f"Saved progress to {html_path} at layer {layer}")

        # Move current to previous for next iteration
        del layer_prev
        layer_prev = layer_curr
        layer_curr = {}

    conn.close()
    print("Streaming plot completed.")

## -------------------------------------------------------------
## Example usage
## -------------------------------------------------------------
"""
if __name__ == "__main__":
    import time, tracemalloc

    n = 15  # board size

    start_time = time.time()
    tracemalloc.start()

    generate_game_graph_and_strategies_sqlite(n, db_path=f"game_graph_n{n}.db")

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    elapsed_time = time.time() - start_time

    print(f"Memory usage: {current_mem/1024**2:.3f} MB, Peak: {peak_mem/1024**2:.3f} MB")
    print(f"Time elapsed: {elapsed_time:.3f} seconds")

    if n < 10:
        db_path = f"game_graph_n{n}.db"  # your SQLite database with nodes + edges
        html_path = f"game_graph_n{n}_stream.html"
        plot_graph_streaming_html(db_path, n, html_path=html_path, save_every_layer=1)
"""


# ----------------------------------------------------------------------
# 5. Loop to generate graphs for increasing n until memory limit is hit
# ----------------------------------------------------------------------

def generate_until_memory_limit(start_n=1, max_n=None, max_memory_MB=800, db_prefix="game_graph_n"):
    """
    Iteratively generate game graphs for boards of size n = start_n, start_n+1, ..., max_n.
    Stops if the peak memory of a generation exceeds max_memory_MB.

    Args:
        start_n (int): starting board size
        max_n (int or None): maximum board size to generate
        max_memory_MB (float): memory threshold in MB
        db_prefix (str): prefix for SQLite DB filenames
    """
    if max_n is None:
        max_n = float('inf')  # effectively no limit except memory

    for n in range(start_n, max_n + 1):
        db_path = f"{db_prefix}{n}.db"
        print(f"\n=== Generating game graph for n={n} ===")

        import time, tracemalloc
        tracemalloc.start()
        start_time = time.time()

        try:
            generate_game_graph_and_strategies_sqlite(n, db_path=db_path)
        except MemoryError:
            print(f"MemoryError at n={n}, stopping generation.")
            break

        elapsed_time = time.time() - start_time
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        current_MB = current_mem / 1024 ** 2
        peak_MB = peak_mem / 1024 ** 2

        print(f"Finished n={n}: elapsed={elapsed_time:.2f}s, current={current_MB:.2f}MB, peak={peak_MB:.2f}MB")

        if peak_MB > max_memory_MB:
            print(f"Peak memory {peak_MB:.2f} MB exceeded limit {max_memory_MB} MB, stopping generation.")
            break
## -------------------------------------------------------------
## Example usage
## -------------------------------------------------------------
if __name__ == "__main__":
    generate_until_memory_limit(start_n=18, max_n=24, max_memory_MB=8000)

