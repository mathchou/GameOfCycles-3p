import sqlite3
import json
import os


def _create_temp_table(conn, name, values, columns="canonical TEXT", pk=None):
    """Create (or replace) an in-memory temp table for fast IN-list joins."""
    conn.execute(f"DROP TABLE IF EXISTS {name}")
    pk_clause = f", PRIMARY KEY ({pk})" if pk else ""
    conn.execute(f"CREATE TEMP TABLE {name} ({columns}{pk_clause})")
    if not values:
        return
    placeholders = ",".join("?" * len(values[0])) if isinstance(values[0], tuple) else "?"
    conn.executemany(
        f"INSERT OR IGNORE INTO {name} VALUES ({placeholders})",
        (v if isinstance(v, tuple) else (v,) for v in values)
    )


def filter_avoidable_nodes(conn, target: list, k: int):
    """
    Given a list of (reduced_canonical, layer) tuples at layer k, find all
    grandparents (layer k+2) and classify them as:
      - can_escape:  grandparent.winner[2] = 0  (has a grandchild with winner[0]=0)
      - cant_escape: grandparent.winner[2] = 1  (all grandchildren have winner[0]=1)

    This works because by misère propagation:
      grandparent.prev = all(child.next) = all(all(grandchild.curr))
    So prev=1 iff ALL grandchildren have curr=1, i.e. no escape exists.

    Returns (can_escape, cant_escape) lists of (reduced_canonical, layer) tuples.
    """
    _create_temp_table(conn, "tmp_target", target,
                       columns="reduced_canonical TEXT, layer INTEGER",
                       pk="reduced_canonical, layer")

    # Find grandparents (2 hops up from target) and check their winner[2] directly
    cur = conn.execute("""
        SELECT DISTINCT
            g.reduced_canonical,
            g.layer,
            json_extract(g.winner, '$[2]') AS prev
        FROM edges e1
        JOIN edges e2      ON e2.parent_reduced = e1.child_reduced
                           AND e2.parent_layer = e1.child_layer
                           AND e2.parent_turn = e1.child_turn
        JOIN tmp_target t  ON t.reduced_canonical = e2.child_reduced
                           AND t.layer = e2.child_layer
        JOIN gamestates g  ON g.reduced_canonical = e1.parent_reduced
                           AND g.layer = e1.parent_layer
                           AND g.turn = e1.parent_turn
        WHERE g.winner IS NOT NULL
    """)
    rows = cur.fetchall()

    if not rows:
        print(f"  No grandparents found for {len(target):,} target nodes.")
        return [], []

    print(f"  Found {len(rows):,} distinct grandparents at layer {k+2}")

    can_escape = []
    cant_escape = []
    for rc, layer, prev in rows:
        if prev == 1:
            cant_escape.append((rc, layer))
        else:
            can_escape.append((rc, layer))

    print(f"  Can escape  (winner[2]=0): {len(can_escape):,}")
    print(f"  Cant escape (winner[2]=1): {len(cant_escape):,}")

    return can_escape, cant_escape


def _find_parents(conn, canonicals: list, tmp_name: str):
    """
    Given a list of (reduced_canonical, layer) tuples, find all distinct
    parent (reduced_canonical, layer) tuples.
    """
    _create_temp_table(conn, tmp_name, canonicals,
                       columns="reduced_canonical TEXT, layer INTEGER",
                       pk="reduced_canonical, layer")
    cur = conn.execute(f"""
        SELECT DISTINCT e.parent_reduced, e.parent_layer
        FROM edges e
        JOIN {tmp_name} t ON t.reduced_canonical = e.child_reduced
                          AND t.layer = e.child_layer
    """)
    return cur.fetchall()


def check_avoidable(n: int, k: int, max_rounds: int = 10):
    """
    Starting from all nodes at layer k with winner[0]=1, repeatedly:
      1. Find grandparents (layer+2) that cant escape (winner[2]=1)
      2. Get all parents of those grandparents (layer+3) — the new target set

    Repeats until no cant_escape grandparents remain, or max_rounds is reached.
    Writes results to check_avoidable_n{n}_layer{k}.json.
    """
    db_path = os.path.join(os.getcwd(), f"reduced_gamestates_n{n}.db")
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA cache_size = -65536")
    conn.execute("PRAGMA temp_store = MEMORY")

    try:
        print(f"\n=== Initializing: layer {k} nodes with winner[0]=1 ===")
        cur = conn.execute("""
            SELECT reduced_canonical
            FROM gamestates
            WHERE layer = ?
              AND winner IS NOT NULL
              AND json_extract(winner, '$[0]') = 1
        """, (k,))
        initial_target = [(r[0], k) for r in cur.fetchall()]
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
                "target_reduced": sorted(set(rc for rc, _ in target)),
                "cant_escape_layer": current_layer + 2,
                "cant_escape_reduced": sorted(set(rc for rc, _ in cant_escape)),
                "new_target_layer": current_layer + 3,
                "new_target_reduced": sorted(set(rc for rc, _ in new_target)),
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