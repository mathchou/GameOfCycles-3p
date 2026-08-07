"""
reduced_game_tree_server.py
Flask server for the reduced canonical game tree viewer.

Usage:
    python reduced_game_tree_server.py --n 50 --port 5000
"""

import sqlite3
import os
import json
import argparse
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

DB_PATH = None
N = None
MAX_NODES = 2000  # hard cap on nodes per request

# Single shared read-only connection
_conn = None

def get_conn():
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True,
                                check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.execute("PRAGMA cache_size = -32768")   # 32MB
        _conn.execute("PRAGMA temp_store = MEMORY")
    return _conn


@app.route('/info')
def info():
    conn = get_conn()
    row = conn.execute(
        "SELECT MIN(layer) as min_l, MAX(layer) as max_l, COUNT(*) as total FROM gamestates"
    ).fetchone()
    has_avoidable = 'avoidable' in [
        r[1] for r in conn.execute("PRAGMA table_info(gamestates)").fetchall()
    ]
    return jsonify({
        "n": N,
        "min_layer": row["min_l"],
        "max_layer": row["max_l"],
        "total_nodes": row["total"],
        "has_avoidable": has_avoidable,
        "max_nodes_per_request": MAX_NODES,
    })


@app.route('/layers')
def layers():
    conn = get_conn()
    rows = conn.execute("""
        SELECT layer, COUNT(*) as count
        FROM gamestates
        GROUP BY layer
        ORDER BY layer DESC
    """).fetchall()
    return jsonify([{"layer": r["layer"], "count": r["count"]} for r in rows])


@app.route('/nodes')
def nodes():
    layer_min = int(request.args.get('layer_min', 0))
    layer_max = int(request.args.get('layer_max', 10))
    conn = get_conn()

    has_avoidable = 'avoidable' in [
        r[1] for r in conn.execute("PRAGMA table_info(gamestates)").fetchall()
    ]
    avoidable_col = ", avoidable" if has_avoidable else ""

    # Check count first
    count = conn.execute("""
        SELECT COUNT(*) FROM gamestates WHERE layer BETWEEN ? AND ?
    """, (layer_min, layer_max)).fetchone()[0]

    if count > MAX_NODES:
        return jsonify({
            "error": f"Too many nodes ({count:,}) — narrow your layer range. Max is {MAX_NODES:,}.",
            "count": count,
        }), 400

    node_rows = conn.execute(f"""
        SELECT reduced_canonical, layer, turn, winner {avoidable_col}
        FROM gamestates
        WHERE layer BETWEEN ? AND ?
    """, (layer_min, layer_max)).fetchall()

    nodes_out = []
    node_ids = set()
    for r in node_rows:
        nid = f"{r['reduced_canonical']}|{r['layer']}|{r['turn']}"
        node_ids.add(nid)
        winner = json.loads(r['winner']) if r['winner'] else None
        nodes_out.append({
            "id": nid,
            "rc": r['reduced_canonical'],
            "layer": r['layer'],
            "turn": r['turn'],
            "winner": winner,
            "avoidable": r['avoidable'] if has_avoidable else None,
        })

    edge_rows = conn.execute("""
        SELECT parent_reduced, parent_layer, parent_turn,
               child_reduced, child_layer, child_turn
        FROM edges
        WHERE parent_layer BETWEEN ? AND ?
          AND child_layer BETWEEN ? AND ?
    """, (layer_min, layer_max, layer_min, layer_max)).fetchall()

    edges_out = []
    for r in edge_rows:
        src = f"{r['parent_reduced']}|{r['parent_layer']}|{r['parent_turn']}"
        tgt = f"{r['child_reduced']}|{r['child_layer']}|{r['child_turn']}"
        if src in node_ids and tgt in node_ids:
            edges_out.append({"source": src, "target": tgt})

    return jsonify({
        "nodes": nodes_out,
        "edges": edges_out,
        "truncated": False,
        "count": len(nodes_out),
    })


@app.route('/search')
def search():
    q = request.args.get('q', '').strip()
    limit = min(int(request.args.get('limit', 20)), 50)
    if not q:
        return jsonify([])
    conn = get_conn()
    rows = conn.execute("""
        SELECT reduced_canonical, layer, turn, winner
        FROM gamestates
        WHERE reduced_canonical LIKE ?
        ORDER BY layer DESC
        LIMIT ?
    """, (f"{q}%", limit)).fetchall()
    return jsonify([{
        "id": f"{r['reduced_canonical']}|{r['layer']}|{r['turn']}",
        "rc": r['reduced_canonical'],
        "layer": r['layer'],
        "turn": r['turn'],
        "winner": json.loads(r['winner']) if r['winner'] else None,
    } for r in rows])


@app.route('/node')
def node_detail():
    rc = request.args.get('rc', '')
    layer = int(request.args.get('layer', 0))
    turn = int(request.args.get('turn', 0))
    conn = get_conn()

    has_avoidable = 'avoidable' in [
        r[1] for r in conn.execute("PRAGMA table_info(gamestates)").fetchall()
    ]
    avoidable_col = ", avoidable" if has_avoidable else ""

    row = conn.execute(f"""
        SELECT reduced_canonical, layer, turn, winner {avoidable_col}
        FROM gamestates
        WHERE reduced_canonical=? AND layer=? AND turn=?
    """, (rc, layer, turn)).fetchone()

    if not row:
        return jsonify({"error": "Node not found"}), 404

    children = conn.execute("""
        SELECT g.reduced_canonical, g.layer, g.turn, g.winner
        FROM edges e
        JOIN gamestates g ON g.reduced_canonical=e.child_reduced
                          AND g.layer=e.child_layer
                          AND g.turn=e.child_turn
        WHERE e.parent_reduced=? AND e.parent_layer=? AND e.parent_turn=?
        LIMIT 100
    """, (rc, layer, turn)).fetchall()

    parents = conn.execute("""
        SELECT g.reduced_canonical, g.layer, g.turn, g.winner
        FROM edges e
        JOIN gamestates g ON g.reduced_canonical=e.parent_reduced
                          AND g.layer=e.parent_layer
                          AND g.turn=e.parent_turn
        WHERE e.child_reduced=? AND e.child_layer=? AND e.child_turn=?
        LIMIT 100
    """, (rc, layer, turn)).fetchall()

    def fmt(r):
        return {
            "id": f"{r['reduced_canonical']}|{r['layer']}|{r['turn']}",
            "rc": r['reduced_canonical'],
            "layer": r['layer'],
            "turn": r['turn'],
            "winner": json.loads(r['winner']) if r['winner'] else None,
        }

    return jsonify({
        "node": fmt(row),
        "avoidable": row['avoidable'] if has_avoidable else None,
        "children": [fmt(r) for r in children],
        "parents": [fmt(r) for r in parents],
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, required=True)
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--db-dir', type=str, default=os.getcwd())
    parser.add_argument('--max-nodes', type=int, default=2000)
    args = parser.parse_args()

    N = args.n
    MAX_NODES = args.max_nodes
    DB_PATH = os.path.join(args.db_dir, f"reduced_gamestates_n{N}.db")
    if not os.path.exists(DB_PATH):
        print(f"ERROR: DB not found at {DB_PATH}")
        exit(1)

    print(f"Serving reduced game tree for n={N}")
    print(f"DB: {DB_PATH}")
    print(f"Max nodes per request: {MAX_NODES}")
    print(f"Open reduced_game_tree_viewer.html in your browser")
    app.run(port=args.port, debug=False)