from typing import List, Tuple
from itertools import product, groupby
import sqlite3
import os
import json
import time


# --------------------------
# GAP CLASS
# --------------------------
class Gap:
    def __init__(self, size: int, orientation: int):
        self.size = size
        self.orientation = orientation  # +1 or -1

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
            moves.append(tuple())

        # SPLIT
        if self.size >= 3:
            for a in range(1, self.size - 1):
                b = self.size - 1 - a
                type_pairs = [(1, 1), (-1, -1)] if self.orientation == 1 else [(1, -1), (-1, 1)]
                for tL, tR in type_pairs:
                    moves.append((Gap(a, tL), Gap(b, tR)))

        return moves

    def __repr__(self) -> str:
        sign = '+' if self.orientation == 1 else '-'
        return f"{sign}{self.size}"


# --------------------------
# INEVITABLE MOVES TABLE
# --------------------------
INEVITABLE_MOVES = {
    (1, -1): 0,  # -1: dead gap, 0 moves
    (1, 1): 1,   # +1: 1 move
    (2, 1): 2,   # +2: 2 moves
    (2, -1): 1,  # -2: 1 move
    (3, -1): 2,  # -3: 2 moves
    (4, -1): 3,  # -4: 3 moves
}


# --------------------------
# GAMESTATE CLASS
# --------------------------
class GameState:
    def __init__(self, gaps: List[Gap], turn: int = 0, inevitable_moves: int = 0):
        gaps_list = [g if isinstance(g, Gap) else Gap(*g) for g in gaps]
        # Keep ALL gaps — dead and inevitable gaps handled uniformly via INEVITABLE_MOVES
        self.gaps = sorted(
            gaps_list,
            key=lambda g: (-g.size, -g.orientation)
        )
        self.inevitable_moves = inevitable_moves
        self.turn = turn % 3

        # Precompute inevitable vs live split once at init time
        self._live_gaps = [g for g in self.gaps if (g.size, g.orientation) not in INEVITABLE_MOVES]
        self._inevitable_contribution = sum(
            INEVITABLE_MOVES[(g.size, g.orientation)]
            for g in self.gaps
            if (g.size, g.orientation) in INEVITABLE_MOVES
        )

    def reduced_canonical(self) -> str:
        """
        Returns 'live_gaps|inevitable_moves', computed cheaply from
        precomputed live/inevitable split.
        """
        gaps_str = ','.join(
            f"{'+' if g.orientation == 1 else '-'}{g.size}"
            for g in self._live_gaps
        )
        return f"{gaps_str}|{self.inevitable_moves + self._inevitable_contribution}"

    def live_gap_size(self) -> int:
        return sum(g.size for g in self._live_gaps)

    def is_terminal(self) -> bool:
        return len(self.gaps) == 0 and self.inevitable_moves == 0

    def legal_moves(self) -> List["GameState"]:
        """
        Act on live gaps only. Inevitable gaps are kept in child self.gaps
        and only absorbed when computing reduced_canonical().
        """
        next_states_set = set()
        next_states_list = []

        for i, gap in enumerate(self.gaps):
            for move in gap.legal_moves():
                new_gaps = self.gaps[:i] + list(move) + self.gaps[i + 1:]
                new_state = GameState(new_gaps, self.turn + 1, self.inevitable_moves)
                key = (new_state.reduced_canonical(), new_state.turn)
                if key not in next_states_set:
                    next_states_set.add(key)
                    next_states_list.append(new_state)

        return next_states_list

    def __repr__(self) -> str:
        gaps_str = ', '.join(str(g) for g in self.gaps)
        return f"[{gaps_str}] inevitable={self.inevitable_moves} (Player {self.turn + 1}'s turn)"


# --------------------------
# PARTITION GENERATION
# --------------------------
def partitions(n):
    def helper(n, max_val):
        if n == 0:
            yield ()
            return
        for i in range(min(n, max_val), 0, -1):
            for rest in helper(n - i, i):
                yield (i,) + rest
    return list(helper(n, n))


def canonical_orientations(partition):
    groups = [(val, sum(1 for _ in grp)) for val, grp in groupby(partition)]
    def group_options(m):
        for j in range(m + 1):
            yield (1,) * (m - j) + (-1,) * j
    for combo in product(*[group_options(m) for _, m in groups]):
        orientations = sum(combo, ())
        if orientations.count(-1) % 2 == 0:
            yield orientations


def signed_partitions_gen(n):
    for partition in partitions(n):
        for orientations in canonical_orientations(partition):
            gaps = [Gap(size, ori) for size, ori in zip(partition, orientations)]
            yield GameState(gaps)


# --------------------------
# DATABASE SETUP
# --------------------------
def get_db_connection(n: int):
    db_name = os.path.join(os.getcwd(), f"reduced_gamestates_n{n}.db")
    conn = sqlite3.connect(db_name)

    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-64000")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS gamestates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            reduced_canonical TEXT NOT NULL,
            layer INTEGER NOT NULL,
            turn INTEGER NOT NULL,
            winner TEXT,
            UNIQUE(reduced_canonical, layer, turn)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS edges (
            parent_reduced TEXT NOT NULL,
            parent_layer INTEGER NOT NULL,
            parent_turn INTEGER NOT NULL,
            child_reduced TEXT NOT NULL,
            child_layer INTEGER NOT NULL,
            child_turn INTEGER NOT NULL,
            PRIMARY KEY (parent_reduced, parent_layer, parent_turn,
                         child_reduced, child_layer, child_turn)
        )
    """)
    conn.commit()
    return conn


def build_indices(conn):
    """Create indices after bulk insert is complete for much faster build."""
    print("Building indices...", flush=True)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_parent ON edges (parent_reduced, parent_layer, parent_turn)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_child ON edges (child_reduced, child_layer, child_turn)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_parent_layer ON edges (parent_layer)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_child_layer ON edges (child_layer)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_gamestates_layer ON gamestates (layer)")
    conn.commit()
    print("Indices built.", flush=True)


def add_layer_indices(n: int):
    db_path = os.path.join(os.getcwd(), f"reduced_gamestates_n{n}.db")
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA cache_size = -524288")  # 512 MB
    conn.execute("PRAGMA temp_store = MEMORY")
    print(f"Adding layer indices to reduced_gamestates_n{n}.db...")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_parent_layer ON edges (parent_layer)")
    print("  parent_layer index done")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_child_layer ON edges (child_layer)")
    print("  child_layer index done")
    conn.commit()
    conn.close()
    print("Done.")

# --------------------------
# BUILD REDUCED GAME TREE
# --------------------------
def build_reduced_game_tree(n: int, chunk_size: int = 50000):
    """
    Builds the reduced canonical game tree for game size n.

    Nodes are (reduced_canonical, layer, turn), where layer starts at n
    and decrements by 1 with each move.

    For each unique (reduced_canonical, layer, turn), only the representative
    GameState with the smallest live gap size is kept for expansion.

    `seen` is flushed at the end of each layer to save memory.
    Indices are deferred until after the full tree is built.
    """
    conn = get_db_connection(n)
    total_start = time.time()

    # Layer n: deduplicated reduced canonical forms of all signed partitions
    print(f"Layer {n}: generating signed partitions...", flush=True)
    layer_start = time.time()

    # frontier: (reduced_canonical, layer, turn) -> shortest representative GameState
    frontier = {}

    for state in signed_partitions_gen(n):
        rc = state.reduced_canonical()
        key = (rc, n, state.turn)
        if key not in frontier or state.live_gap_size() < frontier[key].live_gap_size():
            frontier[key] = state

    print(f"  {len(frontier):,} unique nodes at layer {n}", flush=True)

    conn.executemany(
        "INSERT OR IGNORE INTO gamestates (reduced_canonical, layer, turn) VALUES (?, ?, ?)",
        [(rc, layer, turn) for (rc, layer, turn) in frontier]
    )
    conn.commit()

    layer_time = time.time() - layer_start
    print(f"  Layer {n} done | {layer_time:.2f}s", flush=True)

    # Expand layer by layer
    current_layer = n
    while frontier:
        layer_start = time.time()
        state_rows = []
        edge_rows = []
        next_frontier = {}
        seen_this_layer = set()  # flushed at end of each layer

        for (rc, layer, turn), state in frontier.items():
            for child in state.legal_moves():
                child_rc = child.reduced_canonical()
                child_layer = layer - 1
                child_key = (child_rc, child_layer, child.turn)

                edge_rows.append((rc, layer, turn, child_rc, child_layer, child.turn))

                if child_key not in seen_this_layer:
                    seen_this_layer.add(child_key)
                    state_rows.append((child_rc, child_layer, child.turn))
                    next_frontier[child_key] = child
                elif child.live_gap_size() < next_frontier[child_key].live_gap_size():
                    next_frontier[child_key] = child

            if len(edge_rows) >= chunk_size * 10:
                conn.executemany(
                    "INSERT OR IGNORE INTO edges VALUES (?, ?, ?, ?, ?, ?)",
                    edge_rows
                )
                conn.commit()
                edge_rows = []

        if state_rows:
            conn.executemany(
                "INSERT OR IGNORE INTO gamestates (reduced_canonical, layer, turn) VALUES (?, ?, ?)",
                state_rows
            )
        if edge_rows:
            conn.executemany(
                "INSERT OR IGNORE INTO edges VALUES (?, ?, ?, ?, ?, ?)",
                edge_rows
            )
        conn.commit()

        current_layer -= 1
        layer_time = time.time() - layer_start
        total_time = time.time() - total_start
        print(f"Layer {current_layer}: {len(next_frontier):,} new nodes | "
              f"layer: {layer_time:.2f}s | total: {total_time:.2f}s", flush=True)

        # Flush seen — no longer needed after this layer is committed
        seen_this_layer.clear()
        frontier = next_frontier

        if current_layer < 0:
            break

    build_indices(conn)
    print(f"Done. {time.time() - total_start:.2f}s total.", flush=True)
    conn.close()


# --------------------------
# COMPUTE MISERE WINNERS
# --------------------------
def compute_misere_winners_reduced(n: int):
    """
    Computes 3-player misère winners [curr, next, prev] for all nodes in
    reduced_gamestates_n{n}.db. Processes layer 0 first (terminals), up to layer n.
    """
    db_name = os.path.join(os.getcwd(), f"reduced_gamestates_n{n}.db")
    conn = sqlite3.connect(db_name)
    cur = conn.cursor()

    cur.execute("SELECT MAX(layer) FROM gamestates")
    max_layer = cur.fetchone()[0]

    print(f"Computing misère winners from layer 0 → {max_layer}...", flush=True)

    for layer in range(max_layer + 1):
        cur.execute(
            "SELECT reduced_canonical, turn FROM gamestates WHERE layer=?", (layer,)
        )
        rows = cur.fetchall()
        print(f"Processing layer {layer}, {len(rows):,} states", flush=True)

        for rc, turn in rows:
            cur.execute(
                """SELECT COUNT(*) FROM edges
                   WHERE parent_reduced=? AND parent_layer=? AND parent_turn=?""",
                (rc, layer, turn)
            )
            child_count = cur.fetchone()[0]

            if child_count == 0:
                # Terminal: misère last-move-loses, outcome depends on inevitable mod 3
                inevitable = int(rc.split('|')[1])
                m = inevitable % 3
                if m == 0:
                    winner = [1, 1, 0]
                elif m == 1:
                    winner = [0, 1, 1]
                else:
                    winner = [1, 0, 1]
            else:
                cur.execute(
                    """SELECT gc.winner FROM edges e
                       JOIN gamestates gc
                         ON gc.reduced_canonical = e.child_reduced
                        AND gc.layer = e.child_layer
                        AND gc.turn = e.child_turn
                       WHERE e.parent_reduced=? AND e.parent_layer=? AND e.parent_turn=?""",
                    (rc, layer, turn)
                )
                child_rows = cur.fetchall()
                child_statuses = [
                    json.loads(r[0]) for r in child_rows if r[0] is not None
                ]

                if not child_statuses:
                    continue

                prev_val = int(all(c[1] for c in child_statuses))
                curr_val = int(any(c[2] for c in child_statuses))
                next_val = int(all(c[0] for c in child_statuses))
                winner = [curr_val, next_val, prev_val]

            cur.execute(
                """UPDATE gamestates SET winner=?
                   WHERE reduced_canonical=? AND layer=? AND turn=?""",
                (json.dumps(winner), rc, layer, turn)
            )

        conn.commit()

    print("Finished computing misère winners.", flush=True)
    conn.close()


if __name__ == "__main__":
    """
    import sys
    if len(sys.argv) < 2:
        print("Usage: python reduced_game_tree.py <n>")
        sys.exit(1)
    n = int(sys.argv[1])
    build_reduced_game_tree(n)
    compute_misere_winners_reduced(n)
    """