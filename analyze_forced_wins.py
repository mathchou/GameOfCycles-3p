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

def reduced_canonical_mod3(rc: str) -> str:
    """Returns the reduced canonical with inevitable moves taken mod 3."""
    gaps_str, inev_str = rc.split('|')
    return f"{gaps_str}|{int(inev_str) % 3}"


def _ensure_tables(conn):
    """Ensure avoidable column and avoidable_mod3 table exist."""
    cols = [r[1] for r in conn.execute("PRAGMA table_info(gamestates)").fetchall()]
    if 'avoidable' not in cols:
        conn.execute("ALTER TABLE gamestates ADD COLUMN avoidable INTEGER DEFAULT NULL")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_avoidable ON gamestates (avoidable)")
        print("Added avoidable column.")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS avoidable_mod3 (
            reduced_canonical_mod3 TEXT PRIMARY KEY
        )
    """)
    conn.commit()


def _create_temp_table_with_mod3(conn, name, canonicals_with_layers):
    """Create temp table with mod3 column for fast avoidable filtering."""
    conn.execute(f"DROP TABLE IF EXISTS {name}")
    conn.execute(f"""
        CREATE TEMP TABLE {name} (
            reduced_canonical TEXT,
            layer INTEGER,
            reduced_canonical_mod3 TEXT,
            PRIMARY KEY (reduced_canonical, layer)
        )
    """)
    conn.executemany(f"INSERT OR IGNORE INTO {name} VALUES (?, ?, ?)", [
        (rc, layer, reduced_canonical_mod3(rc))
        for rc, layer in canonicals_with_layers
    ])


def filter_avoidable_nodes(conn, target: list, k: int):
    """
    Given a list of (reduced_canonical, layer) tuples at layer k, find all
    grandparents (layer k+2) and classify them by winner[2].
    Uses tight layer constraints (k+1, k+2) for fast index lookups.

    Can escape:  grandparent.winner[2] = 0  (has a grandchild with winner[0]=0)
    Cant escape: grandparent.winner[2] = 1  (all grandchildren have winner[0]=1)

    Returns (can_escape, cant_escape, found_any) lists of (reduced_canonical, layer) tuples.
    found_any=False means no grandparents were found at all (computational limit reached).
    """
    _create_temp_table(conn, "tmp_target", target,
                       columns="reduced_canonical TEXT, layer INTEGER",
                       pk="reduced_canonical, layer")

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
        WHERE e2.child_layer = ?
          AND e2.parent_layer = ?
          AND e1.child_layer = ?
          AND e1.parent_layer = ?
          AND g.winner IS NOT NULL
    """, (k, k + 1, k + 1, k + 2))
    rows = cur.fetchall()

    if not rows:
        print(f"  No grandparents found for {len(target):,} target nodes — computational limit reached.")
        return [], [], False

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

    return can_escape, cant_escape, True


def _find_parents(conn, canonicals: list, tmp_name: str):
    """
    Given a list of (reduced_canonical, layer) tuples, find all distinct
    parent (reduced_canonical, layer) tuples.
    Uses layer+1 constraint for fast index lookup.
    """
    _create_temp_table(conn, tmp_name, canonicals,
                       columns="reduced_canonical TEXT, layer INTEGER",
                       pk="reduced_canonical, layer")
    cur = conn.execute(f"""
        SELECT DISTINCT e.parent_reduced, e.parent_layer
        FROM edges e
        JOIN {tmp_name} t ON t.reduced_canonical = e.child_reduced
                          AND t.layer = e.child_layer
                          AND e.parent_layer = t.layer + 1
    """)
    return cur.fetchall()

def _has_children(history: list) -> bool:
    """Check if the history tree has any child nodes."""
    if not history:
        return False
    for entry in history:
        if entry.get("children"):
            return True
    return False


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

def _propagate_node(conn, node_rc, node_layer, origin_rc_mod3, max_rounds, current_round, live_moves_fn):
    """
    Recursively propagate a single node until it's cleared or max_rounds is reached.
    Returns (cleared, history) where history is a list of round dicts.
    Mod3-identical nodes are kept in history as leaves but not recursed into.
    """
    if current_round > max_rounds:
        return False, []

    node_rc_mod3 = reduced_canonical_mod3(node_rc)
    print(f"  {'  ' * (current_round - 1)}[Round {current_round}] Analyzing {node_rc} at layer {node_layer}")

    can_escape, cant_escape, found_any = filter_avoidable_nodes(
        conn, [(node_rc, node_layer)], node_layer
    )

    if not found_any:
        return False, [{"node": node_rc, "round": current_round, "outcome": "computational limit"}]

    if not cant_escape:
        return True, [{"node": node_rc, "round": current_round, "outcome": "cleared - all grandparents escape"}]

    cant_escape = dedup_by_inevitable_mod3(cant_escape)

    _create_temp_table_with_mod3(conn, "tmp_cant_escape_check", cant_escape)

    cant_escape = conn.execute("""
        SELECT DISTINCT t.reduced_canonical, t.layer
        FROM tmp_cant_escape_check t
        WHERE NOT EXISTS (
            SELECT 1 FROM avoidable_mod3 a
            WHERE a.reduced_canonical_mod3 = t.reduced_canonical_mod3
        )
    """).fetchall()

    if not cant_escape:
        return True, [{"node": node_rc, "round": current_round, "outcome": "cleared - all cant-escape already avoidable"}]

    cant_escape_layer = node_layer + 2

    new_nodes = conn.execute("""
        SELECT DISTINCT e.parent_reduced, e.parent_layer
        FROM edges e
        JOIN tmp_cant_escape_check t ON t.reduced_canonical = e.child_reduced
                                     AND t.layer = e.child_layer
        WHERE e.child_layer = ?
          AND e.parent_layer = ?
    """, (cant_escape_layer, cant_escape_layer + 1)).fetchall()

    new_nodes = filter_new_target(new_nodes)
    new_nodes = dedup_by_inevitable_mod3(new_nodes)

    # Separate loop nodes (mod3-identical) — keep in history but don't recurse
    loop_nodes = [
        (rc, layer) for rc, layer in new_nodes
        if reduced_canonical_mod3(rc) == node_rc_mod3
        or reduced_canonical_mod3(rc) == origin_rc_mod3
    ]
    new_nodes = [
        (rc, layer) for rc, layer in new_nodes
        if (rc, layer) not in set(loop_nodes)
    ]

    if not new_nodes and not loop_nodes:
        return True, [{"node": node_rc, "round": current_round, "outcome": "cleared - new target empty after filtering"}]

    # Sort by live_moves descending
    new_nodes = sorted(new_nodes, key=lambda x: live_moves_fn(x[0]), reverse=True)

    # Build children list — loop nodes as leaves, others recursed
    children = []

    # Add loop nodes as non-recursive leaves
    for child_rc, child_layer in loop_nodes:
        children.append({
            "child": child_rc,
            "cleared": False,
            "history": [{
                "node": child_rc,
                "round": current_round + 1,
                "outcome": "loop - mod3 identical to origin",
            }]
        })

    # Recurse into non-loop nodes
    all_cleared = True
    for child_rc, child_layer in new_nodes:
        child_cleared, child_history = _propagate_node(
            conn, child_rc, child_layer, origin_rc_mod3,
            max_rounds, current_round + 1, live_moves_fn
        )
        children.append({
            "child": child_rc,
            "cleared": child_cleared,
            "history": child_history,
        })
        if not child_cleared:
            all_cleared = False

    # Loop nodes don't affect cleared status — they're informational only
    outcome = "cleared" if all_cleared else "failed"
    history = [{
        "node": node_rc,
        "round": current_round,
        "new_nodes": [rc for rc, _ in (loop_nodes + new_nodes)],
        "children": children,
        "outcome": outcome,
    }]

    return all_cleared, history


def get_layer_nodes_by_moves(n: int, layer: int) -> list:
    """
    Returns all (reduced_canonical, layer) nodes at the given layer
    with winner[0]=1, sorted in descending order of non-inevitable moves
    (i.e. sum of live gap moves only, excluding the inevitable counter).
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

    def live_moves(rc: str) -> int:
        gaps_str = rc.split('|')[0]
        if not gaps_str:
            return 0
        total = 0
        for g in gaps_str.split(','):
            orientation = 1 if g[0] == '+' else -1
            size = int(g[1:])
            total += size if orientation == 1 else size - 1
        return total

    nodes = [(rc, layer) for (rc,) in rows]
    return sorted(nodes, key=lambda x: live_moves(x[0]), reverse=True)


def get_layer_nodes_by_moves_include_all(n: int, layer: int) -> list:
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

def _has_children(history: list) -> bool:
    """Check if the history tree has any child nodes."""
    if not history:
        return False
    for entry in history:
        if entry.get("children"):
            return True
    return False


def mark_avoidable(n: int, start_rc: str, start_layer: int, max_rounds: int = 10):
    """
    Starting from a single (reduced_canonical, layer) node, recursively
    propagates each node's chain independently. Cleared only if all
    sub-chains are cleared.
    """
    def live_moves(rc: str) -> int:
        gaps_str = rc.split('|')[0]
        if not gaps_str:
            return 0
        total = 0
        for g in gaps_str.split(','):
            orientation = 1 if g[0] == '+' else -1
            size = int(g[1:])
            total += size if orientation == 1 else size - 1
        return total

    db_path = os.path.join(os.getcwd(), f"reduced_gamestates_n{n}.db")
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA cache_size = -4194304")  # 4 GB
    conn.execute("PRAGMA temp_store = MEMORY")

    try:
        _ensure_tables(conn)

        origin_rc_mod3 = reduced_canonical_mod3(start_rc)

        print(f"\n=== Analyzing {start_rc} at layer {start_layer} ===")
        proven_avoidable, history = _propagate_node(
            conn, start_rc, start_layer, origin_rc_mod3,
            max_rounds, 1, live_moves
        )

        if proven_avoidable:
            conn.execute("""
                UPDATE gamestates SET avoidable = 1
                WHERE reduced_canonical = ? AND avoidable IS NULL
            """, (start_rc,))
            conn.execute("""
                INSERT OR IGNORE INTO avoidable_mod3 VALUES (?)
            """, (origin_rc_mod3,))
            conn.commit()
            print(f"  Marked {start_rc} (mod3: {origin_rc_mod3}) as avoidable in DB.")
        else:
            conn.execute("""
                UPDATE gamestates SET avoidable = ?
                WHERE reduced_canonical = ?
                AND (avoidable IS NULL OR (avoidable < 0 AND avoidable > ?))
            """, (-max_rounds, start_rc, -max_rounds))
            conn.commit()
            print(f"  Marked {start_rc} as not proven avoidable at depth {max_rounds}.")

        if _has_children(history):
            safe_rc = start_rc.replace('|', '_').replace(',', '-').replace('+', 'p').replace(' ', '')
            subdir = os.path.join(os.getcwd(), f"avoidable_n{n}_layer{start_layer}")
            os.makedirs(subdir, exist_ok=True)
            if proven_avoidable:
                filename = f"avoidable_{safe_rc}_maxiter{max_rounds}.json"
            else:
                filename = f"failed_{safe_rc}_maxiter{max_rounds}.json"
            path = os.path.join(subdir, filename)
            with open(path, "w") as f:
                json.dump({
                    "n": n,
                    "start_rc": start_rc,
                    "start_rc_mod3": origin_rc_mod3,
                    "start_layer": start_layer,
                    "proven_avoidable": proven_avoidable,
                    "max_rounds": max_rounds,
                    "history": history,
                }, f, indent=2)
            print(f"\nWrote results to {subdir}\\{filename}")

        return proven_avoidable

    finally:
        conn.close()


def resume_mark_avoidable(n: int, k: int, max_rounds: int = 10):
    db_path = os.path.join(os.getcwd(), f"reduced_gamestates_n{n}.db")
    conn = sqlite3.connect(db_path)

    try:
        try:
            already_done = set(
                r[0] for r in conn.execute(
                    "SELECT reduced_canonical_mod3 FROM avoidable_mod3"
                ).fetchall()
            )
            print(f"Found {len(already_done):,} already-proven avoidable mod3 forms in DB.")
        except sqlite3.OperationalError:
            already_done = set()
            print("No avoidable_mod3 table found — starting fresh.")

        # Also skip nodes already analyzed at this depth or deeper
        try:
            already_analyzed = set(
                r[0] for r in conn.execute("""
                    SELECT reduced_canonical FROM gamestates
                    WHERE avoidable <= ?
                """, (-max_rounds,)).fetchall()
            )
            print(f"Found {len(already_analyzed):,} nodes already analyzed at depth >= {max_rounds}.")
        except sqlite3.OperationalError:
            already_analyzed = set()

    finally:
        conn.close()

    nodes = get_layer_nodes_by_moves(n, k)
    print(f"Found {len(nodes):,} total nodes at layer {k} with winner[0]=1.")

    remaining = [
        (rc, layer) for rc, layer in nodes
        if reduced_canonical_mod3(rc) not in already_done
        and rc not in already_analyzed
    ]
    print(f"Resuming from {len(remaining):,} remaining nodes.")

    if not remaining:
        print("All nodes already processed.")
        return

    for i, (rc, layer) in enumerate(remaining):
        print(f"\n{'='*60}")
        print(f"Node {i+1}/{len(remaining)}: {rc} at layer {layer}")
        print(f"{'='*60}")
        mark_avoidable(n, rc, layer, max_rounds=max_rounds)



def clear_avoidable(n: int, max_layer: int = None):
    """
    Clears all avoidable labels from the gamestates table and empties
    the avoidable_mod3 table. If max_layer is specified, only clears
    nodes at layers <= max_layer.
    """
    db_path = os.path.join(os.getcwd(), f"reduced_gamestates_n{n}.db")
    conn = sqlite3.connect(db_path)
    try:
        if max_layer is not None:
            conn.execute("""
                UPDATE gamestates SET avoidable = NULL
                WHERE layer <= ? AND avoidable IS NOT NULL
            """, (max_layer,))
            print(f"Cleared avoidable labels for n={n}, layers 0-{max_layer}.")
        else:
            conn.execute("UPDATE gamestates SET avoidable = NULL")
            print(f"Cleared all avoidable labels for n={n}.")

        conn.execute("DELETE FROM avoidable_mod3")
        conn.commit()
    finally:
        conn.close()


# ----------------------------------------------------------------------
# Some analysis tools
# ----------------------------------------------------------------------

def get_longest_chains(n: int, layer: int, top_k: int = 20):
    """
    For all nodes at the given layer with winner[0]=1, finds the maximum
    depth of their avoidability proof chain from the JSON files in the
    avoidable_n{n}_layer{layer} subdirectory.

    Returns a sorted list of (rc, max_round, proven_avoidable) tuples,
    longest chains first.
    """
    import glob

    subdir = os.path.join(os.getcwd(), f"avoidable_n{n}_layer{layer}")
    if not os.path.exists(subdir):
        print(f"No directory found: {subdir}")
        return []

    def max_depth(history):
        """Recursively find the maximum round number in a history tree."""
        if not history:
            return 0
        best = 0
        for entry in history:
            if not entry:
                continue
            best = max(best, entry.get("round", 0))
            if entry.get("children"):
                for child in entry["children"]:
                    if child.get("history"):
                        best = max(best, max_depth(child["history"]))
        return best

    results = []
    for path in glob.glob(os.path.join(subdir, "*.json")):
        with open(path) as f:
            try:
                data = json.load(f)
            except Exception:
                continue
        rc = data.get("start_rc", "")
        proven = data.get("proven_avoidable", False)
        depth = max_depth(data.get("history", []))
        results.append((rc, depth, proven))

    # Sort by depth descending
    results.sort(key=lambda x: x[1], reverse=True)

    print(f"\n=== Longest chains at layer {layer} (n={n}) ===")
    print(f"{'Rank':<6} {'Max Round':<12} {'Proven':<10} {'Reduced Canonical'}")
    print("-" * 60)
    for i, (rc, depth, proven) in enumerate(results[:top_k], 1):
        print(f"{i:<6} {depth:<12} {str(proven):<10} {rc}")

    # Also report nodes with winner[0]=1 that have NO json file (not yet proven)
    db_path = os.path.join(os.getcwd(), f"reduced_gamestates_n{n}.db")
    conn = sqlite3.connect(db_path)
    try:
        all_nodes = set(
            r[0] for r in conn.execute("""
                SELECT reduced_canonical FROM gamestates
                WHERE layer = ?
                  AND winner IS NOT NULL
                  AND json_extract(winner, '$[0]') = 1
            """, (layer,)).fetchall()
        )
        analyzed = set(rc for rc, _, _ in results)
        unanalyzed = all_nodes - analyzed
        print(f"\n  Total nodes with winner[0]=1: {len(all_nodes):,}")
        print(f"  With JSON files (analyzed):   {len(analyzed):,}")
        print(f"  Without JSON files:           {len(unanalyzed):,}")
    finally:
        conn.close()

    return results
