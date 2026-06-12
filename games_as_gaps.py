from dataclasses import dataclass
from typing import List, Tuple
from itertools import product, groupby
import sqlite3
import os
import json
import time
import tracemalloc

# --------------------------
# GAP CLASS
# --------------------------
class Gap:
    def __init__(self, size: int, orientation: int):
        self.size = size
        self.orientation = orientation # +1 or -1

    def is_dead(self) -> bool:
        return self.orientation == -1 and self.size == 1

    def legal_moves(self) -> List[Tuple["Gap", ...]]:
        moves = []
        if self.size <= 0 or self.is_dead():
            return moves

        # SHRINK
        new_size = self.size - 1
        if new_size > 0:
            moves.append((Gap(new_size, self.orientation),))
        else:
            moves.append(tuple())  # gap disappears

        # SPLIT
        if self.size >= 3:
            for a in range(1, self.size - 1):
                b = self.size - 1 - a
                # Positive gaps split into two same-orientation children.
                # Negative gaps split asymmetrically: one child of each orientation.
                type_pairs = [(1,1), (-1,-1)] if self.orientation == 1 else [(1,-1), (-1,1)]
                for tL, tR in type_pairs:
                    moves.append((Gap(a, tL), Gap(b, tR)))

        return moves

    def __repr__(self) -> str:
        sign = '+' if self.orientation == 1 else '-'
        return f"{sign}{self.size}"


# --------------------------
# GAMESTATE CLASS
# --------------------------
INEVITABLE_MOVES = {
    (1, 1): 1,  # +1: 1 move
    (2, 1): 2,  # +2: 2 moves
    (2, -1): 1,  # -2: 1 move
    (3, -1): 2,  # -3: 2 moves
    (4, -1): 3,  # -4: 3 moves
}


class GameState:
    def __init__(self, gaps: List[Gap], turn: int = 0, inevitable_moves: int = 0):
        gaps_list = [g if isinstance(g, Gap) else Gap(*g) for g in gaps]

        # Separate inevitable and live gaps
        live_gaps = []
        total_inevitable = inevitable_moves
        for g in gaps_list:
            if g.is_dead():
                pass  # silently drop -1 gaps
            elif (g.size, g.orientation) in INEVITABLE_MOVES:
                total_inevitable += INEVITABLE_MOVES[(g.size, g.orientation)]
            else:
                live_gaps.append(g)

        self.gaps = sorted(live_gaps, key=lambda g: (-g.size, -g.orientation))
        self.inevitable_moves = total_inevitable % 3
        self.turn = turn % 3

    def canonical(self) -> Tuple[Tuple[int, int], ...]:
        return tuple((g.size, g.orientation) for g in self.gaps)

    def legal_moves(self) -> List["GameState"]:
        next_states_set = set()
        next_states_list = []

        for i, gap in enumerate(self.gaps):
            for move in gap.legal_moves():
                new_gaps = self.gaps[:i] + list(move) + self.gaps[i + 1:]
                new_state = GameState(new_gaps, self.turn + 1, self.inevitable_moves)
                key = (new_state.canonical(), new_state.turn)
                if key not in next_states_set:
                    next_states_set.add(key)
                    next_states_list.append(new_state)

        return next_states_list

    def is_terminal(self) -> bool:
        return all(not gap.legal_moves() for gap in self.gaps)

    def __repr__(self) -> str:
        gaps_str = ', '.join(str(g) for g in self.gaps)
        return f"[{gaps_str}] inevitable={self.inevitable_moves} (Player {self.turn + 1}'s turn)"

def layer_of(state: GameState) -> int:
    return sum(g.size for g in state.gaps) + state.inevitable_moves

# --------------------------
# CANONICAL STRING FOR SQL
# --------------------------
def canonical_str(state: GameState) -> str:
    gaps_part = ",".join(f"{'+' if g.orientation == 1 else '-'}{g.size}" for g in state.gaps)
    return f"{gaps_part}|{state.inevitable_moves}"

def parse_gaps(canonical: str) -> Tuple[List[Gap], int]:
    """
    converting canonical_str back into gamestate
    """
    if '|' in canonical:
        gaps_part, inevitable_part = canonical.rsplit('|', 1)
        inevitable_moves = int(inevitable_part)
    else:
        gaps_part = canonical
        inevitable_moves = 0  # backwards compatibility with old DBs

    if not gaps_part:
        return [], inevitable_moves

    gaps = []
    for g in gaps_part.split(','):
        if not g:
            continue
        orientation = 1 if g[0] == '+' else -1
        size = int(g[1:])
        gaps.append(Gap(size, orientation))

    return gaps, inevitable_moves

# ----------------------------------------------------
# Generating all game-states at a particular layer
# ----------------------------------------------------

def partitions(n):
    """setting up all partitions of a number n"""
    def helper(n, max_val):
        if n == 0:
            yield ()
            return
        for i in range(min(n, max_val), 0, -1):
            for rest in helper(n - i, i):
                yield (i,) + rest
    return list(helper(n, n))

# we can save time by doing this (from Claude):
# 1. Generate only canonical orientations directly
# Instead of generating all orientations and deduplicating, only generate orientations that are already in canonical
# order for a given partition. For a partition like (3, 3, 2), the two size-3 parts are interchangeable, so you only
# need orientations where the first + comes before the first - among equal-sized parts.
# This eliminates duplicates at the source rather than filtering them out after.
#
# How to do this?
# The insight is that we only need to generate non-increasing orientation sequences within groups of equal parts, this
# eliminates duplicates at the source

def canonical_orientations(partition):
    groups = [(val, sum(1 for _ in grp)) for val, grp in groupby(partition)]
    def group_options(m):
        for j in range(m + 1):
            yield (1,) * (m - j) + (-1,) * j
    for combo in product(*[group_options(m) for _, m in groups]):
        orientations = sum(combo, ())
        if orientations.count(-1) % 2 == 0:
            yield orientations

def signed_partitions(n):
    result = []
    for partition in partitions(n):
        for orientations in canonical_orientations(partition):
            gaps = [Gap(size, ori) for size, ori in zip(partition, orientations)]
            result.append(GameState(gaps))
    return result

def signed_partitions_gen(n):
    for partition in partitions(n):
        for orientations in canonical_orientations(partition):
            gaps = [Gap(size, ori) for size, ori in zip(partition, orientations)]
            yield GameState(gaps)

# --------------------------
# DATABASE SETUP
# --------------------------


def get_db_connection(n: int):
    db_name = os.path.join(os.getcwd(), f"gamestates_n{n}.db")
    conn = sqlite3.connect(db_name)
    cur = conn.cursor()

    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-64000")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS gamestates (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        canonical TEXT NOT NULL,
        turn INTEGER NOT NULL,
        layer INTEGER,
        winner TEXT,
        UNIQUE(canonical, turn)
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS edges (
        parent_canonical TEXT NOT NULL,
        parent_turn INTEGER NOT NULL,
        child_canonical TEXT NOT NULL,
        child_turn INTEGER NOT NULL,
        PRIMARY KEY (parent_canonical, parent_turn, child_canonical, child_turn)
    )
    """)

    cur.execute("CREATE INDEX IF NOT EXISTS idx_edges_parent ON edges (parent_canonical, parent_turn)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_edges_child ON edges (child_canonical, child_turn)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_gamestates_layer ON gamestates (layer)")

    conn.commit()
    return conn, cur


def reconstruct_seen(cur):
    cur.execute("SELECT canonical, turn FROM gamestates")
    return set(cur.fetchall())


def insert_layer(states: List[GameState], layer_num: int, cur, conn):
    state_rows = []
    edge_rows = []
    children = []
    seen_children = set()

    for state in states:
        canonical = canonical_str(state)
        state_rows.append((canonical, state.turn, layer_num))
        for child in state.legal_moves():
            child_canonical = canonical_str(child)
            edge_rows.append((canonical, state.turn, child_canonical, child.turn))
            key = (child_canonical, child.turn)
            if key not in seen_children:
                seen_children.add(key)
                children.append(child)

    cur.executemany(
        "INSERT OR IGNORE INTO gamestates (canonical, turn, layer) VALUES (?, ?, ?)",
        state_rows
    )
    cur.executemany(
        "INSERT OR IGNORE INTO edges (parent_canonical, parent_turn, child_canonical, child_turn) VALUES (?, ?, ?, ?)",
        edge_rows
    )
    conn.commit()
    return children


# ------------------------------------------------------------------------------
# BUILD ALL POSSIBLE GAME STATES (full game tree, not just from one starting node)
# ------------------------------------------------------------------------------

def build_game_tree(n: int, chunk_size: int = 50000):
    import time
    conn, cur = get_db_connection(n)
    total_start = time.time()

    # Layer 0 is now variable — group initial states by their layer
    print(f"Generating initial states...", flush=True)
    from collections import defaultdict
    layers = defaultdict(list)
    for state in signed_partitions_gen(n):
        layers[layer_of(state)].append(state)

    # Process layers from highest to lowest
    max_layer = max(layers.keys())
    print(f"Max layer: {max_layer}", flush=True)

    seen = set()
    for layer_num in range(max_layer, -1, -1):
        current_layer = layers[layer_num]
        if not current_layer:
            continue

        layer_start = time.time()
        print(f"Layer {layer_num}: {len(current_layer)} states", end="", flush=True)

        state_rows = []
        edge_rows = []

        for state in current_layer:
            canonical = canonical_str(state)
            key = (canonical, state.turn)
            if key in seen:
                continue
            seen.add(key)
            state_rows.append((canonical, state.turn, layer_num))

            for child in state.legal_moves():
                child_canonical = canonical_str(child)
                edge_rows.append((canonical, state.turn, child_canonical, child.turn))
                child_key = (child_canonical, child.turn)
                if child_key not in seen:
                    layers[layer_of(child)].append(child)

            if len(edge_rows) >= chunk_size * 10:
                cur.executemany(
                    "INSERT OR IGNORE INTO gamestates (canonical, turn, layer) VALUES (?, ?, ?)",
                    state_rows
                )
                cur.executemany(
                    "INSERT OR IGNORE INTO edges (parent_canonical, parent_turn, child_canonical, child_turn) VALUES (?, ?, ?, ?)",
                    edge_rows
                )
                conn.commit()
                state_rows = []
                edge_rows = []

        if state_rows:
            cur.executemany(
                "INSERT OR IGNORE INTO gamestates (canonical, turn, layer) VALUES (?, ?, ?)",
                state_rows
            )
        if edge_rows:
            cur.executemany(
                "INSERT OR IGNORE INTO edges (parent_canonical, parent_turn, child_canonical, child_turn) VALUES (?, ?, ?, ?)",
                edge_rows
            )
        conn.commit()

        layer_time = time.time() - layer_start
        total_time = time.time() - total_start
        print(f" | layer: {layer_time:.2f}s | total: {total_time:.2f}s")

    print(f"Done. {time.time() - total_start:.2f}s total.")
    conn.close()

def migrate_db(n: int):
    import time
    old_db = f"gamestates_n{n}.db"
    new_db = f"gamestates_n{n}_reduced.db"

    old_conn = sqlite3.connect(old_db)
    old_cur = old_conn.cursor()

    new_conn = sqlite3.connect(new_db)
    new_cur = new_conn.cursor()

    # Set up new DB
    new_cur.execute("""
        CREATE TABLE IF NOT EXISTS gamestates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            canonical TEXT NOT NULL,
            turn INTEGER NOT NULL,
            layer INTEGER,
            winner TEXT,
            UNIQUE(canonical, turn)
        )
    """)
    new_cur.execute("""
        CREATE TABLE IF NOT EXISTS edges (
            parent_canonical TEXT NOT NULL,
            parent_turn INTEGER NOT NULL,
            child_canonical TEXT NOT NULL,
            child_turn INTEGER NOT NULL,
            PRIMARY KEY (parent_canonical, parent_turn, child_canonical, child_turn)
        )
    """)
    new_cur.execute("CREATE INDEX IF NOT EXISTS idx_edges_parent ON edges (parent_canonical, parent_turn)")
    new_cur.execute("CREATE INDEX IF NOT EXISTS idx_edges_child ON edges (child_canonical, child_turn)")
    new_cur.execute("CREATE INDEX IF NOT EXISTS idx_gamestates_layer ON gamestates (layer)")
    new_conn.commit()

    # Helper to reduce an old canonical string to new format
    def reduce_canonical(canonical: str, turn: int) -> Tuple[str, int, int]:
        """Returns (new_canonical, new_turn, new_layer)"""
        # handle old format without inevitable moves
        if '|' in canonical:
            gaps_part, inevitable_part = canonical.rsplit('|', 1)
            inevitable_moves = int(inevitable_part)
        else:
            gaps_part = canonical
            inevitable_moves = 0

        gaps = []
        for g in gaps_part.split(','):
            if not g:
                continue
            orientation = 1 if g[0] == '+' else -1
            size = int(g[1:])
            gaps.append(Gap(size, orientation))

        # Apply reduction
        state = GameState(gaps, turn, inevitable_moves)
        new_canonical = canonical_str(state)
        new_layer = layer_of(state)
        return new_canonical, state.turn, new_layer

    # Migrate states
    print("Migrating states...", flush=True)
    old_cur.execute("SELECT COUNT(*) FROM gamestates")
    total = old_cur.fetchone()[0]
    print(f"  {total:,} states to migrate", flush=True)

    old_cur.execute("SELECT canonical, turn, layer, winner FROM gamestates")
    state_rows = []
    count = 0
    start = time.time()

    while True:
        rows = old_cur.fetchmany(50000)
        if not rows:
            break
        for canonical, turn, layer, winner in rows:
            new_canonical, new_turn, new_layer = reduce_canonical(canonical, turn)
            state_rows.append((new_canonical, new_turn, new_layer, winner))
            count += 1

        new_cur.executemany(
            "INSERT OR IGNORE INTO gamestates (canonical, turn, layer, winner) VALUES (?, ?, ?, ?)",
            state_rows
        )
        new_conn.commit()
        state_rows = []
        print(f"  {count:,} / {total:,} states migrated ({time.time()-start:.1f}s)", flush=True)

    # Migrate edges
    print("Migrating edges...", flush=True)
    old_cur.execute("SELECT COUNT(*) FROM edges")
    total_edges = old_cur.fetchone()[0]
    print(f"  {total_edges:,} edges to migrate", flush=True)

    old_cur.execute("SELECT parent_canonical, parent_turn, child_canonical, child_turn FROM edges")
    edge_rows = []
    count = 0

    while True:
        rows = old_cur.fetchmany(50000)
        if not rows:
            break
        for parent_canonical, parent_turn, child_canonical, child_turn in rows:
            new_parent, new_parent_turn, _ = reduce_canonical(parent_canonical, parent_turn)
            new_child, new_child_turn, _ = reduce_canonical(child_canonical, child_turn)
            # Skip self-loops that arise from merging
            if (new_parent, new_parent_turn) != (new_child, new_child_turn):
                edge_rows.append((new_parent, new_parent_turn, new_child, new_child_turn))
            count += 1

        new_cur.executemany(
            "INSERT OR IGNORE INTO edges (parent_canonical, parent_turn, child_canonical, child_turn) VALUES (?, ?, ?, ?)",
            edge_rows
        )
        new_conn.commit()
        edge_rows = []
        print(f"  {count:,} / {total_edges:,} edges migrated ({time.time()-start:.1f}s)", flush=True)

    old_conn.close()
    new_conn.close()
    print(f"Done. New DB: {new_db}")


# ------------------------------------------------------------------------------
# (OLD) BFS LAYER-BY-LAYER WITH PAUSE/RESUME - creating game tree from single node
# ------------------------------------------------------------------------------
def bfs_store_sql_resume(root_state: GameState, cur, conn):
    seen = reconstruct_seen(cur)

    cur.execute("SELECT MAX(layer) FROM gamestates")
    result = cur.fetchone()
    max_layer = result[0] if result[0] is not None else -1

    if max_layer == -1:
        # Fresh start
        current_layer = [root_state]
        layer_num = 0
    else:
        # Check if max_layer was fully expanded by seeing if max_layer+1 exists
        cur.execute("SELECT COUNT(*) FROM gamestates WHERE layer=?", (max_layer + 1,))
        next_layer_exists = cur.fetchone()[0] > 0

        if next_layer_exists:
            # max_layer was fully expanded; resume from max_layer+1
            resume_layer = max_layer + 1
        else:
            # max_layer may be incomplete; roll back and re-expand from max_layer-1
            cur.execute("DELETE FROM gamestates WHERE layer=?", (max_layer,))
            conn.commit()
            seen = reconstruct_seen(cur)  # rebuild seen after deletion
            resume_layer = max_layer  # will re-expand layer max_layer-1 into it

        cur.execute("SELECT canonical, turn FROM gamestates WHERE layer=?", (resume_layer - 1,))
        rows = cur.fetchall()
        current_layer = []
        for canonical, turn in rows:
            gaps = []
            for g in (canonical or "").split(','):
                if not g:
                    continue
                orientation = 1 if g[0] == '+' else -1
                size = int(g[1:])
                gaps.append(Gap(size, orientation))
            current_layer.append(GameState(gaps, turn))
        layer_num = resume_layer

    # BFS loop
    while current_layer:
        print(f"Processing layer {layer_num}, {len(current_layer)} states")
        insert_layer(current_layer, layer_num, cur, conn)

        next_layer = []
        for state in current_layer:
            for next_state in state.legal_moves():
                key = (next_state.canonical(), next_state.turn)
                if key not in seen:
                    seen.add(key)
                    next_layer.append(next_state)

        current_layer = next_layer
        layer_num += 1

    print(f"Finished BFS. Total layers: {layer_num}")


# --------------------------
# DETERMINING WINNER FROM LAST LAYER
# --------------------------
def compute_misere_winners(n: int):
    """
    Computes 3-player misère winners [curr, next, prev] for all GameStates
    in gamestates_n{n}.db and updates the 'winner' column.

    Misère rule: last move loses.
    """
    db_name = f"gamestates_n{n}_reduced.db"
    conn = sqlite3.connect(db_name)
    cur = conn.cursor()

    cur.execute("SELECT MAX(layer) FROM gamestates")
    max_layer = cur.fetchone()[0]

    print(f"Computing misère winners from layer {max_layer} → 0...")

    for layer in range(max_layer + 1):  # process 0, 1, 2, ... max_layer
        cur.execute("SELECT canonical, turn FROM gamestates WHERE layer=?", (layer,))
        rows = cur.fetchall()
        print(f"Processing layer {layer}, {len(rows)} states")

        for canonical, turn in rows:
            gaps, inevitable_moves = parse_gaps(canonical)
            state = GameState(gaps, turn, inevitable_moves)

            if state.is_terminal():
                # Terminal state: last move loses
                winner = [1, 1, 0]
            else:
                # Fetch all children via edges table
                # Non-terminal: get child winners
                cur.execute(
                    """SELECT child_canonical, child_turn FROM edges
                       WHERE parent_canonical=? AND parent_turn=?""",
                    (canonical, turn)
                )
                child_rows = cur.fetchall()

                child_statuses = []
                for child_canonical, child_turn in child_rows:
                    cur.execute(
                        "SELECT winner FROM gamestates WHERE canonical=? AND turn=?",
                        (child_canonical, child_turn)
                    )
                    row = cur.fetchone()
                    if row and row[0]:
                        child_statuses.append(json.loads(row[0]))
                    else:
                        child_statuses.append([0, 0, 0])

                # Apply 3-player misère propagation
                # prev = all children next
                # curr = any children prev
                # next = all children curr
                prev_val = int(all(c[1] for c in child_statuses))
                curr_val = int(any(c[2] for c in child_statuses))
                next_val = int(all(c[0] for c in child_statuses))
                winner = [curr_val, next_val, prev_val]

            cur.execute(
                "UPDATE gamestates SET winner=? WHERE canonical=? AND turn=?",
                (json.dumps(winner), canonical, turn)
            )

        conn.commit()

    print("Finished computing misère winners.")
    conn.close()



def Get_Gamegraph_and_Strategy(n):
    conn, cur = get_db_connection(n)

    # Confirm table exists
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    print("Tables in DB:", cur.fetchall())

    root_state = GameState([Gap(n, 1)], 0)
    bfs_store_sql_resume(root_state, cur, conn)

    # Count total states
    cur.execute("SELECT COUNT(*) FROM gamestates")
    total_states = cur.fetchone()[0]
    print(f"Total unique game states stored in gamestates_n{n}.db: {total_states}")

    # Compute winning strategy
    compute_misere_winners(n)

    # Example query: print root state winner
    conn = sqlite3.connect(f"gamestates_n{n}.db")
    cur = conn.cursor()
    cur.execute("SELECT winner FROM gamestates WHERE layer=0")
    print("Root state winner:", cur.fetchone()[0])
    conn.close()



def filter_nodes_with_target_child(n: int, layer: int, target_winner: list = [0, 0, 0]) -> List[GameState]:
    """
    For each node in `layer`, check if any child is terminal with the given winner value.
    Returns the list of parent nodes where such a child exists.
    """
    conn, cur = get_db_connection(n)
    target_str = json.dumps(target_winner)

    # Load all nodes in the given layer
    cur.execute("SELECT canonical, turn FROM gamestates WHERE layer=?", (layer,))
    rows = cur.fetchall()

    matching_parents = []

    for canonical, turn in rows:
        # Reconstruct parent state
        gaps = []
        for g in (canonical or "").split(','):
            if not g:
                continue
            orientation = 1 if g[0] == '+' else -1
            size = int(g[1:])
            gaps.append(Gap(size, orientation))
        state = GameState(gaps, turn)

        # Check each child
        for child in state.legal_moves():
            child_canonical = canonical_str(child)
            cur.execute(
                "SELECT winner, layer FROM gamestates WHERE canonical=? AND turn=?",
                (child_canonical, child.turn)
            )
            row = cur.fetchone()
            if row is None:
                continue
            child_winner, child_layer = row

            # Check: terminal (layer+1 has no children, i.e. child is in max layer or is_terminal)
            # We use child.is_terminal() since we have the object already
            if child_winner == target_str:
                matching_parents.append(state)
                break  # no need to check further children

    conn.close()
    print(f"Layer {layer}: {len(matching_parents)} / {len(rows)} nodes have a terminal child with winner {target_winner}")
    return matching_parents

    # Example usage:
    # for k in range(6):
    #     filter_nodes_with_target_child(n=43,layer=k)
    ## Layer 0: 1 / 1 nodes have a terminal child with winner [0, 0, 0]
    ## Layer 1: 43 / 43 nodes have a terminal child with winner [0, 0, 0]
    ## Layer 2: 581 / 581 nodes have a terminal child with winner [0, 0, 0]
    ## Layer 3: 3980 / 3980 nodes have a terminal child with winner [0, 0, 0]
    ## Layer 4: 16469 / 16469 nodes have a terminal child with winner [0, 0, 0]
    ## Layer 5: 46814 / 46829 nodes have a terminal child with winner [0, 0, 0]


# ------------------------------------------------------------------------------
# Querying nodes with grandchildren [0,x,x] fully in SQL to save memory/time
# ------------------------------------------------------------------------------

def check_grandchildren_winners_sql(n: int, layer: int):
    conn, cur = get_db_connection(n)
    grandchild_layer = layer + 2

    print("Step 1: finding green grandchildren...", flush=True)
    cur.execute("DROP TABLE IF EXISTS temp_green_2")
    cur.execute("""
        CREATE TEMPORARY TABLE temp_green_2 AS
        SELECT canonical, turn FROM gamestates
        WHERE layer = ? AND json_extract(winner, '$[0]') = 0
    """, (grandchild_layer,))
    cur.execute("SELECT COUNT(*) FROM temp_green_2")
    print(f"  {cur.fetchone()[0]:,} green grandchildren", flush=True)

    print("Step 2: finding green middle...", flush=True)
    cur.execute("DROP TABLE IF EXISTS temp_green_1")
    cur.execute("""
        CREATE TEMPORARY TABLE temp_green_1 AS
        SELECT DISTINCT e.parent_canonical, e.parent_turn
        FROM edges e
        JOIN temp_green_2 g ON g.canonical = e.child_canonical AND g.turn = e.child_turn
    """)
    cur.execute("SELECT COUNT(*) FROM temp_green_1")
    print(f"  {cur.fetchone()[0]:,} green middle states", flush=True)

    print("Step 3: finding green top...", flush=True)
    cur.execute("DROP TABLE IF EXISTS temp_green_0")
    cur.execute("""
        CREATE TEMPORARY TABLE temp_green_0 AS
        SELECT DISTINCT e.parent_canonical, e.parent_turn
        FROM edges e
        JOIN gamestates g ON g.canonical = e.parent_canonical AND g.turn = e.parent_turn
        JOIN temp_green_1 g1 ON g1.parent_canonical = e.child_canonical AND g1.parent_turn = e.child_turn
        WHERE g.layer = ?
    """, (layer,))
    cur.execute("SELECT COUNT(*) FROM temp_green_0")
    print(f"  {cur.fetchone()[0]:,} green top states", flush=True)

    print("Step 4: finding missing...", flush=True)
    cur.execute("""
        SELECT g.canonical, g.turn, g.winner FROM gamestates g
        LEFT JOIN temp_green_0 t ON t.parent_canonical = g.canonical AND t.parent_turn = g.turn
        WHERE g.layer = ? AND t.parent_canonical IS NULL
    """, (layer,))
    missing = cur.fetchall()

    cur.execute("SELECT COUNT(*) FROM gamestates WHERE layer=?", (layer,))
    total = cur.fetchone()[0]

    # cleanup
    cur.execute("DROP TABLE IF EXISTS temp_green_2")
    cur.execute("DROP TABLE IF EXISTS temp_green_1")
    cur.execute("DROP TABLE IF EXISTS temp_green_0")
    conn.commit()
    conn.close()

    print(f"Layer {layer}: {total - len(missing):,}/{total:,} states have a grandchild with winner[0]==0")
    print(f"  {len(missing):,} states do NOT", flush=True)

    # Write missing states to a new SQLite DB
    out_db = f"missing_grandchildren_n{n}_layer{layer}.db"
    out_conn = sqlite3.connect(out_db)
    out_cur = out_conn.cursor()
    out_cur.execute("""
        CREATE TABLE IF NOT EXISTS missing_states (
            canonical TEXT NOT NULL,
            turn INTEGER NOT NULL,
            winner TEXT,
            PRIMARY KEY (canonical, turn)
        )
    """)
    out_cur.executemany(
        "INSERT OR IGNORE INTO missing_states (canonical, turn, winner) VALUES (?, ?, ?)",
        missing
    )
    out_conn.commit()
    out_conn.close()
    print(f"  saved to {out_db}")

    return missing

def longest_gap_in_missing(n: int, layer: int):
    out_db = f"missing_grandchildren_n{n}_layer{layer}.db"
    conn = sqlite3.connect(out_db)
    cur = conn.cursor()

    cur.execute("SELECT canonical FROM missing_states")
    rows = cur.fetchall()
    conn.close()

    max_size = 0
    max_state = None

    for (canonical,) in rows:
        gaps, inevitable_moves = parse_gaps(canonical)
        first_gap = gaps[0]
        if first_gap.size > max_size:
            max_size = first_gap.size
            max_state = canonical

    print(f"Longest gap size: {max_size}")
    print(f"In state: {max_state}")
    return max_size, max_state


# --------------------------
# MAIN
# --------------------------
if __name__ == "__main__":
    """
    import time
    import tracemalloc
    import sqlite3

    n = 1  # starting n
    MAX_HOURS = 24
    MAX_GB = 16
    output_file = "progress_log.txt"

    # Initialize log file
    with open(output_file, "w") as f:
        f.write("n,elapsed_time_hours,peak_memory_GB,root_winner\n")

    while True:
        start_time = time.time()
        tracemalloc.start()

        try:
            # --------------------------
            # 1. BFS store in DB
            # --------------------------
            root_state = GameState([Gap(n, 1)], 0)
            conn, cur = get_db_connection(n)
            bfs_store_sql_resume(root_state, cur, conn)

            # --------------------------
            # 2. Compute misère winners
            # --------------------------
            compute_misere_winners(n)

            # --------------------------
            # 3. Get root state winner
            # --------------------------
            root_canonical = canonical_str(root_state)
            cur.execute(
                "SELECT winner FROM gamestates WHERE canonical=? AND turn=?",
                (root_canonical, root_state.turn)
            )
            row = cur.fetchone()
            root_winner = row[0] if row else "UNKNOWN"

            conn.close()

        except MemoryError:
            msg = f"MemoryError: n={n} too large"
            print(msg)
            with open(output_file, "a") as f:
                f.write(msg + "\n")
            break
        except Exception as e:
            msg = f"Exception at n={n}: {e}"
            print(msg)
            with open(output_file, "a") as f:
                f.write(msg + "\n")
            break

        # --------------------------
        # 4. Measure time & memory
        # --------------------------
        elapsed_time = time.time() - start_time
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_gb = peak_mem / (1024 ** 3)

        # --------------------------
        # 5. Log output
        # --------------------------
        msg = f"n={n} done: time={elapsed_time / 3600:.2f} hours, peak memory={peak_gb:.2f} GB, root_winner={root_winner}"
        print(msg)
        with open(output_file, "a") as f:
            f.write(f"{n},{elapsed_time / 3600:.4f},{peak_gb:.4f},{root_winner}\n")

        # --------------------------
        # 6. Stop criteria
        # --------------------------
        if elapsed_time > MAX_HOURS * 3600:
            stop_msg = "Stopping: computation exceeded 1 hour"
            print(stop_msg)
            with open(output_file, "a") as f:
                f.write(stop_msg + "\n")
            break
        if peak_gb > MAX_GB:
            stop_msg = "Stopping: memory exceeded 8 GB"
            print(stop_msg)
            with open(output_file, "a") as f:
                f.write(stop_msg + "\n")
            break

        n += 1
        """
