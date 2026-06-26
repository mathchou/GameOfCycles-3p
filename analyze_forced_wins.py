import sqlite3
import json
import os
from collections import defaultdict
from games_as_gaps import compute_reduced_canonical


def analyze_forced_p1_wins(n: int, k: int):
    """
    Backward reachability analysis for Game of Cycles.

    Starting from layer k, finds nodes with winner[0]=1, propagates upward
    two layers, and identifies which layer-k nodes are part of a forced-win
    chain that cannot be escaped even at the grandparent level.

    Outputs:
      - Console: counts at each stage + unique reduced_canonical forms
      - JSON file: failing layer-k nodes + cant-escape grandparents
    """
    db_path = os.path.join(os.getcwd(), f"gamestates_n{n}.db")
    print(f"Connecting to {db_path}")
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA cache_size = -65536")  # 64 MB page cache
    conn.execute("PRAGMA temp_store = MEMORY")

    try:
        # ----------------------------------------------------------------
        # STEP 1: Layer-k nodes with winner[0]=1
        # ----------------------------------------------------------------
        print(f"\n=== STEP 1: Layer {k} nodes with winner[0]=1 ===")
        cur = conn.execute("""
            SELECT canonical, winner
            FROM gamestates
            WHERE layer = ?
              AND winner IS NOT NULL
              AND json_extract(winner, '$[0]') = 1
        """, (k,))
        rows = cur.fetchall()
        layer_k_nodes = {r[0]: r[1] for r in rows}  # canonical -> winner str
        print(f"  Found {len(layer_k_nodes):,} nodes")

        if not layer_k_nodes:
            print("Nothing to analyze.")
            return

        # ----------------------------------------------------------------
        # STEP 2: Parents of those nodes; classify by whether any child
        #         has winner[0]=0.
        #
        # GROUP BY + MIN(json_extract): min=0 iff at least one child has
        # winner[0]=0, which is exactly "can escape".
        # ----------------------------------------------------------------
        print(f"\n=== STEP 2: Parents (layer {k + 1}) ===")
        _create_temp_table(conn, "tmp_layer_k", list(layer_k_nodes.keys()))

        cur = conn.execute("""
            SELECT
                e.parent_canonical,
                MIN(json_extract(gc.winner, '$[0]')) AS min_p1
            FROM edges e
            JOIN tmp_layer_k t ON t.canonical = e.child_canonical
            JOIN gamestates gc ON gc.canonical = e.child_canonical
                               AND gc.winner IS NOT NULL
            GROUP BY e.parent_canonical
        """)
        rows = cur.fetchall()
        print(f"  Found {len(rows):,} distinct parents")
        if not rows:
            print("No parents found.")
            _write_json(n, k, [], [], [])
            return

        cant_escape_parents = []
        can_escape_count = 0
        for row in rows:
            if row[1] == 0:
                can_escape_count += 1
            else:
                cant_escape_parents.append(row[0])

        print(f"  Can escape:    {can_escape_count:,}")
        print(f"  Cannot escape: {len(cant_escape_parents):,}")

        if not cant_escape_parents:
            print("All parents can escape. No propagation needed.")
            _write_json(n, k, [], [], [])
            return

        # ----------------------------------------------------------------
        # STEP 3: Grandparents of cant_escape_parents
        # ----------------------------------------------------------------
        print(f"\n=== STEP 3: Grandparents (layer {k + 2}) ===")
        _create_temp_table(conn, "tmp_cant_escape_parents", cant_escape_parents)

        cur = conn.execute("""
            SELECT DISTINCT e.parent_canonical
            FROM edges e
            JOIN tmp_cant_escape_parents t ON t.canonical = e.child_canonical
        """)
        grandparent_canonicals = [r[0] for r in cur.fetchall()]
        print(f"  Found {len(grandparent_canonicals):,} distinct grandparents")

        if not grandparent_canonicals:
            print("No grandparents found.")
            _write_json(n, k, [], [], [])
            return

        # ----------------------------------------------------------------
        # STEP 4: For each grandparent, does ANY grandchild have winner[0]=0?
        #         Two-hop join: grandparent -> child -> grandchild
        # ----------------------------------------------------------------
        print(f"\n=== STEP 4: Checking grandchildren of grandparents ===")
        _create_temp_table(conn, "tmp_grandparents", grandparent_canonicals)

        cur = conn.execute("""
            SELECT
                e1.parent_canonical AS grandparent,
                MIN(json_extract(gc.winner, '$[0]')) AS min_gc_p1
            FROM edges e1
            JOIN tmp_grandparents tg ON tg.canonical = e1.parent_canonical
            JOIN edges e2            ON e2.parent_canonical = e1.child_canonical
            JOIN gamestates gc       ON gc.canonical = e2.child_canonical
                                     AND gc.winner IS NOT NULL
            GROUP BY e1.parent_canonical
        """)
        cant_escape_grandparents = []
        can_escape_gp_count = 0
        for row in cur.fetchall():
            if row[1] == 0:
                can_escape_gp_count += 1
            else:
                cant_escape_grandparents.append(row[0])

        print(f"  Can escape:    {can_escape_gp_count:,}")
        print(f"  Cannot escape: {len(cant_escape_grandparents):,}")

        # ----------------------------------------------------------------
        # STEP 5: Trace back — which layer-k nodes are downstream of a
        #         cant_escape_grandparent?
        # Path: cant_escape_grandparent -> cant_escape_parent -> layer-k node
        # ----------------------------------------------------------------
        print(f"\n=== STEP 5: Tracing failing layer-k nodes ===")
        if not cant_escape_grandparents:
            print("  No cant-escape grandparents — no layer-k nodes fail the full chain.")
            _write_json(n, k, [], [], [])
            return

        _create_temp_table(conn, "tmp_cant_escape_gp", cant_escape_grandparents)

        cur = conn.execute("""
            SELECT DISTINCT e2.child_canonical AS layer_k_canonical
            FROM edges e1
            JOIN tmp_cant_escape_gp tcg      ON tcg.canonical = e1.parent_canonical
            JOIN tmp_cant_escape_parents tcp  ON tcp.canonical = e1.child_canonical
            JOIN edges e2                     ON e2.parent_canonical = e1.child_canonical
            JOIN tmp_layer_k tlk              ON tlk.canonical = e2.child_canonical
        """)
        failing_k_canonicals = [r[0] for r in cur.fetchall()]
        print(f"  Layer-{k} nodes that fail the full chain: {len(failing_k_canonicals):,}")

        failing_k_nodes = [
            {
                "canonical": c,
                "winner": json.loads(layer_k_nodes[c]),
                "reduced_canonical": compute_reduced_canonical(c),
            }
            for c in failing_k_canonicals
            if c in layer_k_nodes
        ]

        unique_reduced = sorted(set(node["reduced_canonical"] for node in failing_k_nodes))

        # ----------------------------------------------------------------
        # SUMMARY + OUTPUT
        # ----------------------------------------------------------------
        print(f"\n=== SUMMARY ===")
        print(f"  Layer {k}   — nodes with winner[0]=1:          {len(layer_k_nodes):,}")
        print(f"  Layer {k + 1} — parents that cannot escape:       {len(cant_escape_parents):,}")
        print(f"  Layer {k + 2} — grandparents that cannot escape:  {len(cant_escape_grandparents):,}")
        print(f"  Layer {k}   — nodes failing full chain:         {len(failing_k_nodes):,}")
        print(f"\n  Unique reduced_canonical forms ({len(unique_reduced)}):")
        for rc in unique_reduced:
            print(f"    {rc}")

        _write_json(n, k, failing_k_nodes, cant_escape_grandparents, unique_reduced)

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


def _write_json(n: int, k: int, failing_nodes: list, cant_escape_gps: list, unique_reduced: list):
    filename = f"forced_p1_wins_n{n}_layer{k}.json"
    path = os.path.join(os.getcwd(), filename)
    output = {
        "n": n,
        "layer_k": k,
        "failing_layer_k_count": len(failing_nodes),
        "cant_escape_grandparent_count": len(cant_escape_gps),
        "unique_reduced_canonical_forms": unique_reduced,
        "failing_layer_k_nodes": failing_nodes,
        "cant_escape_grandparents": cant_escape_gps,
    }
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nWrote results to {filename}")


if __name__ == "__main__":
    """
    print("Welcome")
    """