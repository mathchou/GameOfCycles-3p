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


# -------------------------------------------------------------------------------------------
# Here, we want to have players A and B keep the parity of the remaining moves to be 1 mod 3
# for player C's turn. However, this is only possible if the game board is "big enough".
# For now, our notion of "big enough" will be "there exist at least two gaps of size >= +3
# -------------------------------------------------------------------------------------------


def total_moves_from_reduced(rc: str) -> int:
    """
    Computes total remaining moves from a reduced canonical string.
      +m gap  -> m moves
      -i gap  -> i-1 moves
      inevitable counter -> j moves
    """
    gaps_str, inevitable_str = rc.split('|')
    total = int(inevitable_str)
    if gaps_str:
        for g in gaps_str.split(','):
            orientation = 1 if g[0] == '+' else -1
            size = int(g[1:])
            total += size if orientation == 1 else size - 1
    return total


def has_two_plus_gaps(rc: str) -> bool:
    """Returns True if the reduced canonical has >= 2 positive gaps of size >= 3."""
    gaps_str = rc.split('|')[0]
    if not gaps_str:
        return False
    count = sum(
        1 for g in gaps_str.split(',')
        if g[0] == '+' and int(g[1:]) >= 3
    )
    return count >= 2


def filter_new_target(new_target: list) -> list:
    """
    From the new target set (list of (reduced_canonical, layer) tuples),
    remove nodes where A and B can apply the 'keep moves 1 mod 3' strategy:
      - has >= 2 positive gaps of size >= 3, AND
      - total remaining moves is not 1 mod 3

    Returns the filtered list (nodes that CANNOT be removed, i.e. still unavoidable).
    """
    filtered = []
    removed = 0
    for rc, layer in new_target:
        if has_two_plus_gaps(rc) and total_moves_from_reduced(rc) % 3 != 1:
            removed += 1
        else:
            filtered.append((rc, layer))
    print(f"  Strategy filter: removed {removed:,}, kept {len(filtered):,}")
    return filtered

def dedup_by_inevitable_mod3(new_target: list) -> list:
    """
    Among nodes that share the same (live_gaps, inevitable mod 3, layer),
    keep only the one with the smallest inevitable count.
    """
    # key: (live_gaps_str, inevitable mod 3, layer) -> (rc, layer, inevitable)
    best = {}
    for rc, layer in new_target:
        gaps_str, inev_str = rc.split('|')
        inev = int(inev_str)
        key = (gaps_str, inev % 3, layer)
        if key not in best or inev < best[key][2]:
            best[key] = (rc, layer, inev)

    filtered = [(rc, layer) for rc, layer, _ in best.values()]
    removed = len(new_target) - len(filtered)
    print(f"  Mod-3 dedup: removed {removed:,}, kept {len(filtered):,}")
    return filtered


# --------------------------------------
# Backward Grandparent Analysis
# --------------------------------------
def get_layer_nodes_by_moves(n: int, layer: int) -> list:
    """
    Returns all (reduced_canonical, layer) nodes at the given layer
    with winner[0]=1, sorted in descending order of total remaining moves.
    """
    db_path = os.path.join(os.getcwd(), f"reduced_gamestates_n{n}.db")
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute("""
            SELECT reduced_canonical FROM gamestates
            WHERE layer = ?
              AND winner IS NOT NULL
              AND json_extract(winner, '$[0]') = 1
        """, (layer,)).fetchall()
    finally:
        conn.close()

    nodes = [(rc, layer) for (rc,) in rows]
    return sorted(nodes, key=lambda x: total_moves_from_reduced(x[0]), reverse=True)

def mark_avoidable(n: int, start_rc: str, start_layer: int, max_rounds: int = 10):
    """
    Starting from a single (reduced_canonical, layer) node, runs iterative
    grandparent escape analysis. If the chain terminates with all grandparents
    able to escape, marks the starting node as avoidable=1 in the DB.
    Writes per-round analysis to mark_avoidable_n{n}_{start_rc}.json.
    """
    db_path = os.path.join(os.getcwd(), f"reduced_gamestates_n{n}.db")
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA cache_size = -65536")
    conn.execute("PRAGMA temp_store = MEMORY")

    try:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(gamestates)").fetchall()]
        if 'avoidable' not in cols:
            conn.execute("ALTER TABLE gamestates ADD COLUMN avoidable INTEGER DEFAULT NULL")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_avoidable ON gamestates (avoidable)")
            conn.commit()
            print("Added avoidable column.")

        target = [(start_rc, start_layer)]
        current_layer = start_layer
        proven_avoidable = False
        results = []

        for round_num in range(1, max_rounds + 1):
            print(f"\n=== Round {round_num}: {len(target):,} nodes at layer {current_layer} ===")

            can_escape, cant_escape = filter_avoidable_nodes(conn, target, current_layer)

            if not cant_escape:
                print(f"  All grandparents can escape — {start_rc} is avoidable.")
                proven_avoidable = True
                results.append({
                    "round": round_num,
                    "target_layer": current_layer,
                    "target": sorted(rc for rc, _ in target),
                    "cant_escape_layer": current_layer + 2,
                    "cant_escape": [],
                    "new_target_layer": current_layer + 3,
                    "new_target": [],
                    "outcome": "all grandparents can escape",
                })
                break

            cant_escape = dedup_by_inevitable_mod3(cant_escape)

            _create_temp_table(conn, "tmp_cant_escape_check", cant_escape,
                                columns="reduced_canonical TEXT, layer INTEGER",
                                pk="reduced_canonical, layer")
            cant_escape = conn.execute("""
                SELECT t.reduced_canonical, t.layer
                FROM tmp_cant_escape_check t
                JOIN gamestates g ON g.reduced_canonical = t.reduced_canonical
                WHERE g.avoidable IS NULL
            """).fetchall()
            print(f"  After removing already-avoidable: {len(cant_escape):,} nodes remain")

            if not cant_escape:
                print(f"  All cant-escape nodes already proven avoidable — {start_rc} is avoidable.")
                proven_avoidable = True
                results.append({
                    "round": round_num,
                    "target_layer": current_layer,
                    "target": sorted(rc for rc, _ in target),
                    "cant_escape_layer": current_layer + 2,
                    "cant_escape": [],
                    "new_target_layer": current_layer + 3,
                    "new_target": [],
                    "outcome": "all cant-escape nodes already avoidable",
                })
                break

            new_target = conn.execute("""
                SELECT DISTINCT e.parent_reduced, e.parent_layer
                FROM edges e
                JOIN tmp_cant_escape_check t ON t.reduced_canonical = e.child_reduced
                                             AND t.layer = e.child_layer
            """).fetchall()
            new_target = filter_new_target(new_target)
            new_target = dedup_by_inevitable_mod3(new_target)

            results.append({
                "round": round_num,
                "target_layer": current_layer,
                "target": sorted(rc for rc, _ in target),
                "cant_escape_layer": current_layer + 2,
                "cant_escape": sorted(rc for rc, _ in cant_escape),
                "new_target_layer": current_layer + 3,
                "new_target": sorted(rc for rc, _ in new_target),
                "outcome": "continuing" if new_target else "new target empty after filtering",
            })

            if not new_target:
                print(f"  New target set is empty after filtering — {start_rc} is avoidable.")
                proven_avoidable = True
                break

            print(f"  New target set: {len(new_target):,} nodes at layer {current_layer + 3}")
            target = new_target
            current_layer += 3

        if proven_avoidable:
            conn.execute("""
                UPDATE gamestates SET avoidable = 1
                WHERE reduced_canonical = ? AND avoidable IS NULL
            """, (start_rc,))
            conn.commit()
            print(f"  Marked {start_rc} as avoidable in DB.")
        else:
            print(f"  Could not prove {start_rc} avoidable within {max_rounds} rounds.")

        # Write JSON output
        safe_rc = start_rc.replace('|', '_').replace(',', '-').replace('+', 'p').replace(' ', '')
        filename = f"mark_avoidable_n{n}_{safe_rc}_layer{start_layer}.json"
        path = os.path.join(os.getcwd(), filename)
        with open(path, "w") as f:
            json.dump({
                "n": n,
                "start_rc": start_rc,
                "start_layer": start_layer,
                "proven_avoidable": proven_avoidable,
                "rounds_completed": len(results),
                "results": results,
            }, f, indent=2)
        print(f"\nWrote results to {filename}")

        return proven_avoidable

    finally:
        conn.close()

def old_check_avoidable(n: int, k: int, max_rounds: int = 10, version: str = 1): # I believe this is not the correct way to look at it, we should evaluate each node one at a time
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

            cant_escape = dedup_by_inevitable_mod3(cant_escape) # deduplicate inevitable moves mod 3

            new_target = _find_parents(conn, cant_escape, f"tmp_prop_parents_r{round_num}")
            new_target = filter_new_target(new_target)  # apply A+B strategy filter
            new_target = dedup_by_inevitable_mod3(new_target) # deduplicate inevitable moves mod 3
            print(f"  New target set after filter: {len(new_target):,} nodes at layer {current_layer + 3}")

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

        filename = f"check_avoidable_n{n}_layer{k}_v{version}.json"
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