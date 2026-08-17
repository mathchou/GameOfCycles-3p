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

# -------------------------------------------------------------------------------------------
# Here, we want to have players A and B keep the parity of the remaining moves to be 1 mod 3
# for player C's turn. However, this is only possible if the game board is "big enough".
# For now, our notion of "big enough" will be "there exist at least two gaps of size >= +3
# -------------------------------------------------------------------------------------------


def _live_moves(rc: str) -> int:
    """
    Moves available from the live gaps only, excluding the inevitable counter.
      +m gap -> m moves
      -i gap -> i-1 moves
    """
    gaps_str = rc.split('|')[0]
    if not gaps_str:
        return 0
    total = 0
    for g in gaps_str.split(','):
        orientation = 1 if g[0] == '+' else -1
        size = int(g[1:])
        total += size if orientation == 1 else size - 1
    return total


def total_moves_from_reduced(rc: str) -> int:
    """
    Total remaining moves: live gap moves plus the inevitable counter.
    Distinct from compute_total, which counts -i as i and adds 1 for an odd
    number of negative live gaps.
    """
    return _live_moves(rc) + int(rc.split('|')[1])


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
def _mark_node_avoidable(conn, rc: str, state: dict = None):
    """
    Mark a reduced canonical as avoidable in gamestates and avoidable_mod3.
    Bumps state['generation'] only when a genuinely new mod3 form is inserted,
    since that is the only event that can turn a cached failure into a success.
    """
    conn.execute("""
        UPDATE gamestates SET avoidable = 1
        WHERE reduced_canonical = ? AND avoidable IS NULL
    """, (rc,))
    rc_mod3 = reduced_canonical_mod3(rc)
    cur = conn.execute("INSERT OR IGNORE INTO avoidable_mod3 VALUES (?)", (rc_mod3,))
    if state is not None and cur.rowcount > 0:
        state["generation"] += 1
        state["avoidable_set"].add(rc_mod3)


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

    nodes = [(rc, layer) for (rc,) in rows]
    return sorted(nodes, key=lambda x: _live_moves(x[0]), reverse=True)


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


def clear_avoidable(n: int):
    """
    Clears all avoidability results for n. Nodes and edges are untouched.
    """
    db_path = os.path.join(os.getcwd(), f"reduced_gamestates_n{n}.db")
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("UPDATE gamestates SET avoidable = NULL WHERE avoidable IS NOT NULL")
        conn.execute("DELETE FROM avoidable_mod3")
        conn.commit()
        print(f"Cleared all avoidable labels for n={n}.")
    finally:
        conn.close()


# --------------------------------------------------------------------------------------------
# Main Backwards Analysis
# --------------------------------------------------------------------------------------------


def filter_avoidable_nodes(conn, target: list, k: int, shift_cache: dict = None,
                           max_layer: int = None):
    """
    Given a list of (reduced_canonical, layer) tuples, find all grandparents
    and classify them by winner[2].

    Applies the shift check at each hop: if compute_total(rc)[0] == layer for a
    node, shift it to (rc, layer+2) before looking up its parents.

    A shift that would land above max_layer means the tree simply does not
    extend far enough to decide this node, so it is reported as the
    computational limit rather than raised. A shift target that is missing
    while BELOW the ceiling is still an error, since that indicates a genuine
    gap in how the tree was built.

    shift_cache: {(rc, layer): shifted_layer}, with None meaning "out of tree".

    Returns (can_escape, cant_escape, found_any) lists of (reduced_canonical, layer).
    found_any=False means no grandparents were found (computational limit reached).
    """
    if shift_cache is None:
        shift_cache = {}
    if max_layer is None:
        max_layer = conn.execute("SELECT MAX(layer) FROM gamestates").fetchone()[0]

    def apply_shift(nodes, label):
        """
        Shift any node whose compute_total equals its layer. Cached.
        Returns None if any shift would run past the top of the tree.
        """
        out = []
        for rc, layer in nodes:
            key = (rc, layer)
            if key in shift_cache:
                shifted = shift_cache[key]
                if shifted is None:
                    return None
                out.append((rc, shifted))
                continue
            if compute_total(rc)[0] == layer:
                shifted_layer = layer + 2
                if shifted_layer > max_layer:
                    shift_cache[key] = None
                    print(f"  → shift of {label} ({rc}, {layer}) needs layer "
                          f"{shifted_layer} > max {max_layer} — out of tree")
                    return None
                exists = conn.execute("""
                    SELECT 1 FROM gamestates
                    WHERE reduced_canonical = ? AND layer = ? LIMIT 1
                """, (rc, shifted_layer)).fetchone()
                if not exists:
                    raise ValueError(
                        f"Shifted {label} missing below the ceiling: rc='{rc}' at "
                        f"layer {shifted_layer} (max {max_layer}) — real gap in the tree"
                    )
                shift_cache[key] = shifted_layer
                out.append((rc, shifted_layer))
            else:
                shift_cache[key] = layer
                out.append((rc, layer))
        return out

    def parents_of(nodes, tmp_name):
        """
        Find parents of nodes, grouped by layer so each query can use literal
        layer parameters (index-friendly) rather than a computed expression.
        """
        by_layer = {}
        for rc, layer in nodes:
            by_layer.setdefault(layer, []).append((rc, layer))

        found = []
        for layer, group in by_layer.items():
            _create_temp_table(conn, tmp_name, group,
                               columns="reduced_canonical TEXT, layer INTEGER",
                               pk="reduced_canonical, layer")
            found.extend(conn.execute(f"""
                SELECT DISTINCT e.parent_reduced, e.parent_layer
                FROM edges e
                JOIN {tmp_name} t ON t.reduced_canonical = e.child_reduced
                                  AND t.layer = e.child_layer
                WHERE e.child_layer = ?
                  AND e.parent_layer = ?
            """, (layer, layer + 1)).fetchall())
        return list(set(found))

    # Hop 0: shift the targets themselves
    shifted_target = apply_shift(target, "target")
    if shifted_target is None:
        return [], [], False

    # Hop 1: targets -> parents
    parents = parents_of(shifted_target, "tmp_target")
    if not parents:
        print(f"  No parents found for {len(shifted_target):,} target nodes — computational limit reached.")
        return [], [], False

    # Hop 1.5: shift the parents before going up again
    shifted_parents = apply_shift(parents, "parent")
    if shifted_parents is None:
        return [], [], False

    # Hop 2: parents -> grandparents, reading winner[2] directly
    by_layer = {}
    for rc, layer in shifted_parents:
        by_layer.setdefault(layer, []).append((rc, layer))

    all_rows = []
    for layer, group in by_layer.items():
        _create_temp_table(conn, "tmp_parents", group,
                           columns="reduced_canonical TEXT, layer INTEGER",
                           pk="reduced_canonical, layer")
        all_rows.extend(conn.execute("""
            SELECT DISTINCT
                g.reduced_canonical,
                g.layer,
                json_extract(g.winner, '$[2]') AS prev
            FROM edges e
            JOIN tmp_parents p ON p.reduced_canonical = e.child_reduced
                               AND p.layer = e.child_layer
            JOIN gamestates g  ON g.reduced_canonical = e.parent_reduced
                               AND g.layer = e.parent_layer
                               AND g.turn = e.parent_turn
            WHERE e.child_layer = ?
              AND e.parent_layer = ?
              AND g.winner IS NOT NULL
        """, (layer, layer + 1)).fetchall())

    if not all_rows:
        print(f"  No grandparents found — computational limit reached.")
        return [], [], False

    print(f"  Found {len(all_rows):,} distinct grandparents")

    can_escape, cant_escape = [], []
    for rc, layer, prev in all_rows:
        (cant_escape if prev == 1 else can_escape).append((rc, layer))

    print(f"  Can escape  (winner[2]=0): {len(can_escape):,}")
    print(f"  Cant escape (winner[2]=1): {len(cant_escape):,}")

    return can_escape, cant_escape, True


def _new_state(conn=None, avoidable_set=None):
    """
    Build the shared run state.

    avoidable_set: mod-3 forms already proven avoidable. Pass the caller's own
                   set to share it by reference rather than copying, so nodes
                   marked mid-recursion are visible to the caller too.
                   If omitted and conn is given, it is loaded from the DB.
    max_layer:     cached once so filter_avoidable_nodes need not re-query it.
    """
    if avoidable_set is None:
        if conn is not None:
            avoidable_set = set(
                r[0] for r in conn.execute(
                    "SELECT reduced_canonical_mod3 FROM avoidable_mod3"
                ).fetchall()
            )
        else:
            avoidable_set = set()

    max_layer = None
    if conn is not None:
        max_layer = conn.execute("SELECT MAX(layer) FROM gamestates").fetchone()[0]

    return {
        "generation": 0,
        "memo_hits": 0,
        "memo_stale_gen": 0,
        "memo_stale_budget": 0,
        "comp_limit": 0,
        "avoidable_hits": 0,
        "avoidable_set": avoidable_set,
        "max_layer": max_layer,
    }


def _memo_lookup(memo, key, remaining, state):
    """
    Return (hit, cleared, history). A miss returns (False, None, None).

    Successes are stored with budget=None and generation=None: avoidability is
    a property of the node and the tree, so once proven it holds regardless of
    how many rounds remain or what has been proven since.

    Failures mean "not proven within the budget available", so they are scoped
    both ways:
      budget     — a failure with budget B holds only for budgets <= B.
      generation — a failure can become a success once new mod-3 forms are
                   added to avoidable_mod3, so it is valid only while the
                   generation counter is unchanged.

    The exception is 'no grandparents exist in the DB', stored with both fields
    None, since that depends only on DB contents and never changes in a run.
    """
    entry = memo.get(key)
    if entry is None:
        return False, None, None

    if not entry["cleared"]:
        if entry["budget"] is not None and remaining > entry["budget"]:
            state["memo_stale_budget"] += 1
            return False, None, None
        if entry["generation"] is not None and entry["generation"] != state["generation"]:
            state["memo_stale_gen"] += 1
            return False, None, None

    state["memo_hits"] += 1
    return True, entry["cleared"], entry["history"]

def _memo_store(memo, key, cleared, history, remaining, state,
                budget_independent=False):
    memo[key] = {
        "cleared": cleared,
        "history": history,
        "generation": None if (cleared or budget_independent) else state["generation"],
        "budget": None if (cleared or budget_independent) else remaining,
    }


def _propagate_node(conn, node_rc, node_layer, origin_rc_mod3, max_rounds,
                    current_round, live_moves_fn, memo=None, shift_cache=None,
                    state=None):
    """
    Recursively propagate a single node until it clears or max_rounds is reached.

    memo:  {(rc_mod3, layer): entry} — keyed on the mod-3 form because
           dedup_by_inevitable_mod3 guarantees only one representative per
           mod-3 class is ever propagated.
    state: shared across the whole run; see _new_state().
    """
    if memo is None:
        memo = {}
    if shift_cache is None:
        shift_cache = {}
    if state is None:
        state = _new_state()

    if current_round > max_rounds:
        return False, []

    remaining = max_rounds - current_round
    node_rc_mod3 = reduced_canonical_mod3(node_rc)
    memo_key = (node_rc_mod3, node_layer)

    # Layer-independent: avoidability is a property of the position, so a
    # mod-3 form proven anywhere is proven here. Checked before the memo
    # because the memo is layer-keyed and would miss the same form at a
    # different layer.
    if node_rc_mod3 in state["avoidable_set"]:
        state["avoidable_hits"] += 1
        print(f"  {'  ' * (current_round - 1)}[Round {current_round}] "
              f"{node_rc} @ layer {node_layer} — already avoidable ({node_rc_mod3})")
        return True, [{"node": node_rc, "round": current_round,
                       "outcome": "already proven avoidable"}]

    hit, cleared, cached_history = _memo_lookup(memo, memo_key, remaining, state)
    if hit:
        tag = "cleared" if cleared else "failed"
        print(f"  {'  ' * (current_round - 1)}[Round {current_round}] "
              f"{node_rc} @ layer {node_layer} — cached ({tag})")
        return cleared, cached_history

    print(f"  {'  ' * (current_round - 1)}[Round {current_round}] "
          f"Analyzing {node_rc} at layer {node_layer}")

    can_escape, cant_escape, found_any = filter_avoidable_nodes(
        conn, [(node_rc, node_layer)], node_layer,
        shift_cache=shift_cache, max_layer=state["max_layer"]
    )

    if not found_any:
        # No grandparents exist in the DB for this node. This depends only on
        # the DB contents, so it can never change within a run — cache it
        # permanently regardless of budget or generation.
        state["comp_limit"] += 1
        hist = [{"node": node_rc, "round": current_round,
                 "outcome": "computational limit"}]
        _memo_store(memo, memo_key, False, hist, remaining, state,
                    budget_independent=True)
        return False, hist

    if not cant_escape:
        _mark_node_avoidable(conn, node_rc, state)
        hist = [{"node": node_rc, "round": current_round,
                 "outcome": "cleared - all grandparents escape"}]
        _memo_store(memo, memo_key, True, hist, remaining, state)
        return True, hist

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
        _mark_node_avoidable(conn, node_rc, state)
        hist = [{"node": node_rc, "round": current_round,
                 "outcome": "cleared - all cant-escape already avoidable"}]
        _memo_store(memo, memo_key, True, hist, remaining, state)
        return True, hist

    # Parents of cant_escape, grouped by layer so each query can use literal
    # layer parameters. Layers may differ because filter_avoidable_nodes can
    # shift mid-traversal.
    by_layer = {}
    for rc, layer in cant_escape:
        by_layer.setdefault(layer, []).append((rc, layer))

    new_nodes = []
    for layer, group in by_layer.items():
        _create_temp_table_with_mod3(conn, "tmp_ce_layer", group)
        new_nodes.extend(conn.execute("""
            SELECT DISTINCT e.parent_reduced, e.parent_layer
            FROM edges e
            JOIN tmp_ce_layer t ON t.reduced_canonical = e.child_reduced
                                AND t.layer = e.child_layer
            WHERE e.child_layer = ?
              AND e.parent_layer = ?
        """, (layer, layer + 1)).fetchall())
    new_nodes = list(set(new_nodes))

    new_nodes = filter_new_target(new_nodes)
    new_nodes = dedup_by_inevitable_mod3(new_nodes)

    loop_set = {
        (rc, layer) for rc, layer in new_nodes
        if reduced_canonical_mod3(rc) == node_rc_mod3
           or reduced_canonical_mod3(rc) == origin_rc_mod3
    }
    loop_nodes = sorted(loop_set)
    new_nodes = [nl for nl in new_nodes if nl not in loop_set]

    if not new_nodes and not loop_nodes:
        _mark_node_avoidable(conn, node_rc, state)
        hist = [{"node": node_rc, "round": current_round,
                 "outcome": "cleared - new target empty after filtering"}]
        _memo_store(memo, memo_key, True, hist, remaining, state)
        return True, hist

    new_nodes = sorted(new_nodes, key=lambda x: live_moves_fn(x[0]), reverse=True)

    children = []
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

    all_cleared = True
    for child_rc, child_layer in new_nodes:
        child_cleared, child_history = _propagate_node(
            conn, child_rc, child_layer, origin_rc_mod3,
            max_rounds, current_round + 1, live_moves_fn,
            memo=memo, shift_cache=shift_cache, state=state
        )
        children.append({
            "child": child_rc,
            "cleared": child_cleared,
            "history": child_history,
        })
        if not child_cleared:
            all_cleared = False

    if all_cleared:
        _mark_node_avoidable(conn, node_rc, state)

    outcome = "cleared" if all_cleared else "failed"
    history = [{
        "node": node_rc,
        "round": current_round,
        "new_nodes": [rc for rc, _ in (loop_nodes + new_nodes)],
        "children": children,
        "outcome": outcome,
    }]

    _memo_store(memo, memo_key, all_cleared, history, remaining, state)
    return all_cleared, history


def mark_avoidable(n: int, start_rc: str, start_layer: int, max_rounds: int = 10,
                   conn=None, memo=None, shift_cache=None, state=None):
    """
    Analyze a single (reduced_canonical, layer) node.

    conn/memo/shift_cache/state may be supplied by resume_mark_avoidable so a
    whole layer shares one connection, one memo, and one generation counter.
    """
    owns_conn = conn is None
    if owns_conn:
        db_path = os.path.join(os.getcwd(), f"reduced_gamestates_n{n}.db")
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA cache_size = -4194304")  # 4 GB
        conn.execute("PRAGMA temp_store = MEMORY")
        _ensure_tables(conn)

    if memo is None:
        memo = {}
    if shift_cache is None:
        shift_cache = {}
    if state is None:
        # Standalone call: seed from the DB so prior runs still prune.
        state = _new_state(conn)

    try:
        origin_rc_mod3 = reduced_canonical_mod3(start_rc)

        print(f"\n=== Analyzing {start_rc} at layer {start_layer} ===")
        proven_avoidable, history = _propagate_node(
            conn, start_rc, start_layer, origin_rc_mod3,
            max_rounds, 1, _live_moves,
            memo=memo, shift_cache=shift_cache, state=state
        )

        if proven_avoidable:
            # _propagate_node already called _mark_node_avoidable on this node
            # when it cleared, which wrote both tables and updated
            # state['avoidable_set']. Only the commit is needed here.
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
            prefix = "avoidable" if proven_avoidable else "failed"
            filename = f"{prefix}_{safe_rc}_maxiter{max_rounds}.json"
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
        if owns_conn:
            conn.close()


def resume_mark_avoidable(n: int, k: int, max_rounds: int = 10):
    """
    Resume avoidability analysis for all layer-k nodes with winner[0]=1,
    sharing one connection, one memo, and one generation counter across
    every node in the layer.
    """
    import time

    db_path = os.path.join(os.getcwd(), f"reduced_gamestates_n{n}.db")
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA cache_size = -4194304")  # 4 GB
    conn.execute("PRAGMA temp_store = MEMORY")

    try:
        _ensure_tables(conn)

        already_done = set(
            r[0] for r in conn.execute(
                "SELECT reduced_canonical_mod3 FROM avoidable_mod3"
            ).fetchall()
        )
        print(f"Found {len(already_done):,} already-proven avoidable mod3 forms in DB.")

        already_analyzed = set(
            r[0] for r in conn.execute("""
                SELECT reduced_canonical FROM gamestates
                WHERE avoidable <= ?
            """, (-max_rounds,)).fetchall()
        )
        print(f"Found {len(already_analyzed):,} nodes already analyzed at depth >= {max_rounds}.")

        rows = conn.execute("""
            SELECT reduced_canonical FROM gamestates
            WHERE layer = ?
              AND winner IS NOT NULL
              AND json_extract(winner, '$[0]') = 1
        """, (k,)).fetchall()
        nodes = sorted(
            [(rc, k) for (rc,) in rows],
            key=lambda x: _live_moves(x[0]), reverse=True
        )
        print(f"Found {len(nodes):,} total nodes at layer {k} with winner[0]=1.")

        remaining_nodes = [
            (rc, layer) for rc, layer in nodes
            if reduced_canonical_mod3(rc) not in already_done
               and rc not in already_analyzed
        ]
        print(f"Resuming from {len(remaining_nodes):,} remaining nodes.")

        if not remaining_nodes:
            print("All nodes already processed.")
            return

        memo = {}
        shift_cache = {}
        # Share already_done by reference, not a copy, so forms marked
        # mid-recursion are visible to the top-level skip check below.
        state = _new_state(conn, avoidable_set=already_done)

        total_start = time.time()
        for i, (rc, layer) in enumerate(remaining_nodes):
            if reduced_canonical_mod3(rc) in already_done:
                continue
            node_start = time.time()
            print(f"\n{'=' * 60}")
            print(f"Node {i + 1}/{len(remaining_nodes)}: {rc} at layer {layer}")
            print(f"{'=' * 60}")
            mark_avoidable(n, rc, layer, max_rounds=max_rounds,
                           conn=conn, memo=memo, shift_cache=shift_cache,
                           state=state)
            print(f"  [node {time.time() - node_start:.2f}s | "
                  f"total {time.time() - total_start:.2f}s]")
            print(f"  [memo {len(memo):,} entries, {state['memo_hits']:,} hits | "
                  f"avoidable-set {len(already_done):,} forms, "
                  f"{state['avoidable_hits']:,} hits | "
                  f"stale gen {state['memo_stale_gen']:,}, "
                  f"budget {state['memo_stale_budget']:,} | "
                  f"comp-limit {state['comp_limit']:,} | "
                  f"gen {state['generation']:,}]")

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
            except (json.JSONDecodeError, ValueError):
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



# Compute layer vs gap+inevitable with extra +1 if odd # of negative gaps for each instance
# These seem to be
def compute_total(rc):
    gaps_str, inev_str = rc.split('|')
    inevitable = int(inev_str)
    live_gap_total = 0
    num_negs = 0
    if gaps_str:
        for g in gaps_str.split(','):
            orientation = 1 if g[0] == '+' else -1
            size = int(g[1:])
            live_gap_total += size
            if orientation == -1:
                num_negs += 1
    total = live_gap_total + inevitable + (1 if num_negs % 2 != 0 else 0)
    return total, live_gap_total, inevitable, num_negs

# The following checks which reduced canonicals differ in their parent sets across layers
# It seems like it's only the ones where the reduced canonical does not "reduce" any move numbers
def check_parents_layer_independent(n: int, check_layer: int, compare_layers: range = None, sample_size: int = 50, show: bool = False):
    """
    For a sample of reduced canonicals that appear in check_layer AND in
    compare_layers, checks whether the set of parent reduced canonicals is
    the same across all those layers.

    Ignores layer n (the top layer) since no parents are computed there.

    Args:
        n:              game size
        check_layer:    the primary layer to check
        compare_layers: range of layers to compare against, e.g. range(10, 16)
                        defaults to all other layers below n
        sample_size:    max number of nodes to check
        show:           if True, print all results; if False, only print
                        violations where total moves != layer
    """
    db_path = os.path.join(os.getcwd(), f"reduced_gamestates_n{n}.db")
    conn = sqlite3.connect(db_path)

    try:
        compare_list = []
        if compare_layers is None:
            layer_filter = f"g2.layer != {check_layer} AND g2.layer < {n}"
            layer_desc = f"all layers except {check_layer} and {n}"
        else:
            compare_list = [l for l in compare_layers if l != check_layer and l < n]
            if not compare_list:
                print("No valid comparison layers.")
                return []
            placeholders = ','.join('?' * len(compare_list))
            layer_filter = f"g2.layer IN ({placeholders})"
            layer_desc = f"layers {min(compare_list)}–{max(compare_list)}"

        print(f"Checking layer {check_layer} vs {layer_desc}...")

        if compare_layers is None:
            rows = conn.execute(f"""
                SELECT g1.reduced_canonical, COUNT(DISTINCT g2.layer) as layer_count
                FROM gamestates g1
                JOIN gamestates g2 ON g2.reduced_canonical = g1.reduced_canonical
                                   AND {layer_filter}
                WHERE g1.layer = ?
                GROUP BY g1.reduced_canonical
                LIMIT ?
            """, (check_layer, sample_size)).fetchall()
        else:
            rows = conn.execute(f"""
                SELECT g1.reduced_canonical, COUNT(DISTINCT g2.layer) as layer_count
                FROM gamestates g1
                JOIN gamestates g2 ON g2.reduced_canonical = g1.reduced_canonical
                                   AND {layer_filter}
                WHERE g1.layer = ?
                GROUP BY g1.reduced_canonical
                LIMIT ?
            """, (*compare_list, check_layer, sample_size)).fetchall()

        print(f"Found {len(rows)} reduced canonicals in layer {check_layer} "
              f"that also appear in {layer_desc}...")

        if compare_layers is None:
            instance_filter = f"layer < {n}"
            instance_params = lambda rc: (rc, n)
        else:
            all_layers = sorted(set([check_layer] + compare_list))
            placeholders = ','.join('?' * len(all_layers))
            instance_filter = f"layer IN ({placeholders})"
            instance_params = lambda rc: (rc, *all_layers)

        violations = []
        for (rc, layer_count) in rows:
            instances = conn.execute(f"""
                SELECT layer, turn FROM gamestates
                WHERE reduced_canonical = ? AND {instance_filter}
                ORDER BY layer
            """, instance_params(rc)).fetchall()

            parent_sets = {}
            for layer, turn in instances:
                parents = conn.execute("""
                    SELECT DISTINCT e.parent_reduced
                    FROM edges e
                    WHERE e.child_reduced = ?
                      AND e.child_layer = ?
                      AND e.child_turn = ?
                """, (rc, layer, turn)).fetchall()
                parent_sets[(layer, turn)] = frozenset(r[0] for r in parents)



            # Check if any instance has total != layer
            layer_mismatches = []
            for (layer, turn), pset in sorted(parent_sets.items()):
                if layer != check_layer:
                    continue
                total, live_gap_total, inevitable, num_negs = compute_total(rc)
                if total != layer:
                    layer_mismatches.append((layer, turn, total, live_gap_total, inevitable))

            unique_sets = set(parent_sets.values())
            has_parent_violation = len(unique_sets) > 1
            has_layer_mismatch = len(layer_mismatches) > 0

            # Decide whether to print this node
            if show or has_layer_mismatch:
                if has_parent_violation:
                    print(f"\n  VIOLATION: {rc} (appears in {len(instances)} layers)")
                    for (layer, turn), pset in sorted(parent_sets.items()):
                        print(f"    Layer {layer}, turn {turn} ({len(pset)} parents): {sorted(pset)}")

                    print(f"    Differences:")
                    items = sorted(parent_sets.items())
                    for i in range(len(items)):
                        for j in range(i+1, len(items)):
                            (l1, t1), s1 = items[i]
                            (l2, t2), s2 = items[j]
                            only_in_1 = sorted(s1 - s2)
                            only_in_2 = sorted(s2 - s1)
                            if only_in_1 or only_in_2:
                                print(f"      Layer {l1} vs Layer {l2}:")
                                if only_in_1:
                                    print(f"        Only in L{l1}: {only_in_1}")
                                if only_in_2:
                                    print(f"        Only in L{l2}: {only_in_2}")
                else:
                    if show:
                        print(f"  OK: {rc} — same {len(next(iter(unique_sets)))} parents across {len(instances)} layers")

            if has_parent_violation:
                violations.append({
                    "rc": rc,
                    "layer_count": len(instances),
                    "parent_sets": {str(k): sorted(v) for k, v in parent_sets.items()}, #(layer, turn): parents set
                    "layer_mismatches": layer_mismatches,
                })

        print(f"\n{'='*50}")
        print(f"Parent set violations: {len(violations)} / {len(rows)}")
        return violations

    finally:
        conn.close()