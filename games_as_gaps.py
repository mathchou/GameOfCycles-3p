from dataclasses import dataclass
from typing import List, Tuple
import sqlite3
import os

# --------------------------
# GAP CLASS
# --------------------------
@dataclass
class Gap:
    size: int
    orientation: int   # +1 or -1

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
class GameState:
    def __init__(self, gaps: List[Gap], turn: int = 0):
        gaps_list = [g if isinstance(g, Gap) else Gap(*g) for g in gaps]
        self.gaps = sorted(gaps_list, key=lambda g: (-g.size, -g.orientation))
        self.turn = turn % 3

    def canonical(self) -> Tuple[Tuple[int,int], ...]:
        return tuple((g.size, g.orientation) for g in self.gaps)

    def legal_moves(self) -> List["GameState"]:
        next_states_set = set()
        next_states_list = []

        for i, gap in enumerate(self.gaps):
            for move in gap.legal_moves():
                new_gaps = self.gaps[:i] + list(move) + self.gaps[i+1:]
                new_state = GameState(new_gaps, self.turn + 1)
                key = (new_state.canonical(), new_state.turn)
                if key not in next_states_set:
                    next_states_set.add(key)
                    next_states_list.append(new_state)

        return next_states_list

    def is_terminal(self) -> bool:
        return all(not gap.legal_moves() for gap in self.gaps)

    def __repr__(self) -> str:
        gaps_str = ', '.join(str(g) for g in self.gaps)
        return f"[{gaps_str}] (Player {self.turn + 1}'s turn)"


# --------------------------
# CANONICAL STRING FOR SQL
# --------------------------
def canonical_str(state: GameState) -> str:
    if not state.gaps:
        return ''  # empty state
    return ",".join(f"{'+' if g.orientation == 1 else '-'}{g.size}" for g in state.gaps)


# --------------------------
# DATABASE SETUP
# --------------------------
def get_db_connection(n: int):
    db_name = os.path.join(os.getcwd(), f"gamestates_n{n}.db")
    conn = sqlite3.connect(db_name)
    cur = conn.cursor()

    # Ensure table exists

    cur.execute("""
    CREATE TABLE IF NOT EXISTS gamestates (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        canonical TEXT NOT NULL,
        turn INTEGER NOT NULL,
        layer INTEGER,
        winner TEXT,   -- stores [curr, next, prev] as string
        UNIQUE(canonical, turn)
    )
    """)
    conn.commit()
    return conn, cur


def reconstruct_seen(cur):
    cur.execute("SELECT canonical, turn FROM gamestates")
    return set(cur.fetchall())


def insert_layer(states: List[GameState], layer_num: int, cur, conn):
    for state in states:
        canonical = canonical_str(state)
        try:
            cur.execute(
                "INSERT OR IGNORE INTO gamestates (canonical, turn, layer) VALUES (?, ?, ?)",
                (canonical, state.turn, layer_num)
            )
        except sqlite3.IntegrityError:
            pass
    conn.commit()


# --------------------------
# BFS LAYER-BY-LAYER WITH PAUSE/RESUME
# --------------------------
def bfs_store_sql_resume(root_state: GameState, cur, conn):
    seen = reconstruct_seen(cur)

    # Find last processed layer
    cur.execute("SELECT MAX(layer) FROM gamestates")
    result = cur.fetchone()
    last_layer_num = result[0] if result[0] is not None else -1

    # Initialize current layer
    if last_layer_num == -1:
        current_layer = [root_state]
        layer_num = 0
    else:
        # Load last layer from DB
        cur.execute("SELECT canonical, turn FROM gamestates WHERE layer=?", (last_layer_num,))
        rows = cur.fetchall()
        current_layer = []
        for r in rows:
            if not r[0]:
                gaps = []
            else:
                gaps = []
                for g in r[0].split(','):
                    if not g:
                        continue
                    orientation = 1 if g[0] == '+' else -1
                    size = int(g[1:])
                    gaps.append(Gap(size, orientation))
            current_layer.append(GameState(gaps, r[1]))
        layer_num = last_layer_num + 1

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
    import sqlite3

    # Connect to DB
    db_name = f"gamestates_n{n}.db"
    conn = sqlite3.connect(db_name)
    cur = conn.cursor()

    # Get max layer to work backwards
    cur.execute("SELECT MAX(layer) FROM gamestates")
    max_layer = cur.fetchone()[0]

    print(f"Computing misère winners from layer {max_layer} → 0...")

    # Helper to parse canonical string back to gaps
    def parse_gaps(canonical: str) -> List[Gap]:
        if not canonical:
            return []
        gaps = []
        for g in canonical.split(','):
            if not g:
                continue
            orientation = 1 if g[0] == '+' else -1
            size = int(g[1:])
            gaps.append(Gap(size, orientation))
        return gaps

    # Process layers from deepest to root
    for layer in reversed(range(max_layer + 1)):
        cur.execute("SELECT canonical, turn FROM gamestates WHERE layer=?", (layer,))
        rows = cur.fetchall()
        print(f"Processing layer {layer}, {len(rows)} states")

        for canonical, turn in rows:
            gaps = parse_gaps(canonical)
            state = GameState(gaps, turn)

            if state.is_terminal():
                # Terminal state: last move loses
                curr = 1          # current player cannot move
                nextp = 1         # next player wins
                prev = 0          # previous player cannot move next
                winner = [curr, nextp, prev]
            else:
                # Non-terminal: get child winners
                children = state.legal_moves()
                child_statuses = []

                for child in children:
                    key = (canonical_str(child), child.turn)
                    cur.execute(
                        "SELECT winner FROM gamestates WHERE canonical=? AND turn=?",
                        (key[0], key[1])
                    )
                    row = cur.fetchone()
                    if row and row[0]:
                        # Convert string back to list
                        child_statuses.append(eval(row[0]))
                    else:
                        # If child not yet computed, assume neutral [0,0,0]
                        child_statuses.append([0,0,0])

                # Apply 3-player misère propagation
                # prev = all children next
                # curr = any children prev
                # next = all children curr
                prev_val = int(all(c[1] for c in child_statuses))
                curr_val = int(any(c[2] for c in child_statuses))
                next_val = int(all(c[0] for c in child_statuses))
                winner = [curr_val, next_val, prev_val]

            # Store in DB as string
            cur.execute(
                "UPDATE gamestates SET winner=? WHERE canonical=? AND turn=?",
                (str(winner), canonical, turn)
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

# --------------------------
# MAIN
# --------------------------
if __name__ == "__main__":
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
