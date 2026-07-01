import sqlite3
import json
import os


INEVITABLE_MOVES = {
    (1, 1): 1,   # +1: 1 move
    (2, 1): 2,   # +2: 2 moves
    (2, -1): 1,  # -2: 1 move
    (3, -1): 2,  # -3: 2 moves
    (4, -1): 3,  # -4: 3 moves
}


def compute_reduced_canonical(canonical: str) -> str:
    """
    Strips inevitable and dead gaps from a canonical string, sums their
    move counts (not reduced mod 3), and returns:
        'live_gaps|inevitable_moves'
    e.g. '+5,+3|3'
    """
    if not canonical:
        return '|0'

    live_parts = []
    total_inevitable = 0

    for g in canonical.split(','):
        if not g:
            continue
        orientation = 1 if g[0] == '+' else -1
        size = int(g[1:])

        if orientation == -1 and size == 1:
            pass  # dead gap, drop silently
        elif (size, orientation) in INEVITABLE_MOVES:
            total_inevitable += INEVITABLE_MOVES[(size, orientation)]
        else:
            live_parts.append(g)

    return '|'.join([','.join(live_parts), str(total_inevitable)])


def _find_parents(conn, base_canonicals: list, tmp_name: str):
    """Find distinct parents of base_canonicals. Leaves base_canonicals in
    a temp table named tmp_name. Returns list of parent canonicals."""
    _create_temp_table(conn, tmp_name, base_canonicals)
    cur = conn.execute(f"""
        SELECT DISTINCT e.parent_canonical
        FROM edges e
        JOIN {tmp_name} t ON t.canonical = e.child_canonical
    """)
    return [r[0] for r in cur.fetchall()]

def filter_avoidable_nodes(conn, target_canonicals: list, k: int):
    """
    Given a subset of nodes at layer k, find all grandparents (layer k+2)
    and classify them by whether they can avoid the target set — i.e. whether
    there exists a grandchild (via ANY child, unfiltered) with winner[0]=0.

    Returns (can_escape, cant_escape) lists of grandparent canonicals.
    """
    _create_temp_table(conn, "tmp_target", target_canonicals)

    cur = conn.execute("""
        SELECT DISTINCT e1.parent_canonical
        FROM edges e1
        JOIN edges e2      ON e2.parent_canonical = e1.child_canonical
        JOIN tmp_target t  ON t.canonical = e2.child_canonical
    """)
    grandparent_canonicals = [r[0] for r in cur.fetchall()]

    if not grandparent_canonicals:
        print(f"  No grandparents found for {len(target_canonicals):,} target nodes.")
        return [], []

    print(f"  Found {len(grandparent_canonicals):,} distinct grandparents at layer {k+2}")

    _create_temp_table(conn, "tmp_gp", grandparent_canonicals)

    cur = conn.execute("""
        SELECT
            e1.parent_canonical AS grandparent,
            MIN(json_extract(gc.winner, '$[0]')) AS min_gc_p1
        FROM edges e1
        JOIN tmp_gp tg     ON tg.canonical = e1.parent_canonical
        JOIN edges e2      ON e2.parent_canonical = e1.child_canonical
        JOIN gamestates gc ON gc.canonical = e2.child_canonical
                           AND gc.winner IS NOT NULL
        GROUP BY e1.parent_canonical
    """)
    can_escape = []
    cant_escape = []
    for row in cur.fetchall():
        if row[1] == 0:
            can_escape.append(row[0])
        else:
            cant_escape.append(row[0])

    print(f"  Can escape (have a [0,x,x] grandchild):  {len(can_escape):,}")
    print(f"  Can't escape (no [0,x,x] grandchild):    {len(cant_escape):,}")

    return can_escape, cant_escape


def check_avoidable(n: int, k: int, max_rounds: int = 10):
    """
    Starting from all nodes at layer k with winner[0]=1, repeatedly:
      1. Find grandparents (layer+2) that can't escape the target set
      2. Get all parents of those grandparents (layer+3) — the new target set

    Repeats until no cant_escape grandparents remain, or max_rounds is reached.
    Writes results to check_avoidable_n{n}_layer{k}.json.
    """
    db_path = os.path.join(os.getcwd(), f"gamestates_n{n}.db")
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA cache_size = -65536")
    conn.execute("PRAGMA temp_store = MEMORY")

    try:
        print(f"\n=== Initializing: layer {k} nodes with winner[0]=1 ===")
        cur = conn.execute("""
            SELECT canonical
            FROM gamestates
            WHERE layer = ?
              AND winner IS NOT NULL
              AND json_extract(winner, '$[0]') = 1
        """, (k,))
        initial_target = [r[0] for r in cur.fetchall()]
        print(f"  Found {len(initial_target):,} initial target nodes")

        if not initial_target:
            print("Nothing to propagate.")
            return []

        target = initial_target
        current_layer = k
        results = []

        for round_num in range(1, max_rounds + 1):
            print(f"\n=== Round {round_num}: target set size={len(target):,} at layer {current_layer} ===")

            can_escape, cant_escape = filter_avoidable_nodes(conn, target, current_layer)

            if not cant_escape:
                print(f"  All grandparents can escape. Stopping at round {round_num}.")
                break

            new_target = _find_parents(conn, cant_escape, f"tmp_prop_parents_r{round_num}")
            print(f"  New target set: {len(new_target):,} nodes at layer {current_layer + 3}")

            results.append({
                "round": round_num,
                "target_layer": current_layer,
                "cant_escape_layer": current_layer + 2,
                "cant_escape_reduced": sorted(set(compute_reduced_canonical(c) for c in cant_escape)),
            })

            target = new_target
            current_layer += 3

        filename = f"check_avoidable_n{n}_layer{k}.json"
        path = os.path.join(os.getcwd(), filename)
        with open(path, "w") as f:
            json.dump({
                "n": n,
                "layer_k": k,
                "rounds_completed": len(results),
                "results": results,
            }, f, indent=2)
        print(f"\nWrote results to {filename}")

        return results

    finally:
        conn.close()


# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------

def _create_temp_table(conn: sqlite3.Connection, name: str, values: list):
    """Create (or replace) an in-memory temp table for fast IN-list joins."""
    conn.execute(f"DROP TABLE IF EXISTS {name}")
    conn.execute(f"CREATE TEMP TABLE {name} (canonical TEXT PRIMARY KEY)")
    conn.executemany(f"INSERT OR IGNORE INTO {name} VALUES (?)",
                     ((v,) for v in values))


def _write_json(n: int, k: int, failing_nodes: list, cant_escape_sets: dict, unique_reduced: list):
    filename = f"forced_p1_wins_n{n}_layer{k}.json"
    path = os.path.join(os.getcwd(), filename)
    output = {
        "n": n,
        "layer_k": k,
        "failing_layer_k_count": len(failing_nodes),
        "cant_escape_set_counts": {key: len(v) for key, v in cant_escape_sets.items()},
        "unique_reduced_canonical_forms": unique_reduced,
        "failing_layer_k_nodes": failing_nodes,
        "cant_escape_sets": cant_escape_sets,
    }
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nWrote results to {filename}")


if __name__ == "__main__":
    """
    import sys
    if len(sys.argv) != 3:
        print("Usage: python old_analyze_forced_wins.py <n> <layer>")
        sys.exit(1)
    analyze_forced_p1_wins(int(sys.argv[1]), int(sys.argv[2]))
    """