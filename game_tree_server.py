"""
game_tree_server.py
Run this from your GameOfCycles-3p directory:
    python game_tree_server.py --n 26 --version v2 --max-layer 26

Then open http://localhost:8765 in your browser.
"""

import argparse
import json
import os
import sqlite3
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

def get_db_path(n, version):
    return os.path.join(os.getcwd(), f"gamestates_n{n}_{version}.db")

def get_tree_data(n, version, min_layer, max_layer):
    db_path = get_db_path(n, version)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Fetch nodes in layer range
    cur.execute("""
        SELECT canonical, turn, layer, winner
        FROM gamestates
        WHERE layer >= ? AND layer <= ?
        ORDER BY layer, canonical, turn
    """, (min_layer, max_layer))
    node_rows = cur.fetchall()

    nodes = []
    node_ids = {}
    for i, (canonical, turn, layer, winner) in enumerate(node_rows):
        node_id = f"{canonical}|t{turn}"
        node_ids[(canonical, turn)] = i
        nodes.append({
            "id": i,
            "canonical": canonical,
            "turn": turn,
            "layer": layer,
            "winner": json.loads(winner) if winner else None,
            "label": f"{canonical} P{turn+1}"
        })

    # Fetch edges within layer range
    canonical_turn_set = set((r[0], r[1]) for r in node_rows)
    cur.execute("""
        SELECT e.parent_canonical, e.parent_turn, e.child_canonical, e.child_turn
        FROM edges e
        JOIN gamestates gp ON gp.canonical = e.parent_canonical AND gp.turn = e.parent_turn
        JOIN gamestates gc ON gc.canonical = e.child_canonical AND gc.turn = e.child_turn
        WHERE gp.layer >= ? AND gp.layer <= ?
          AND gc.layer >= ? AND gc.layer <= ?
    """, (min_layer, max_layer, min_layer, max_layer))
    edge_rows = cur.fetchall()

    edges = []
    for parent_canonical, parent_turn, child_canonical, child_turn in edge_rows:
        if (parent_canonical, parent_turn) in node_ids and (child_canonical, child_turn) in node_ids:
            edges.append({
                "source": node_ids[(parent_canonical, parent_turn)],
                "target": node_ids[(child_canonical, child_turn)]
            })

    # Layer stats
    cur.execute("""
        SELECT layer, COUNT(*) FROM gamestates
        GROUP BY layer ORDER BY layer
    """)
    layer_stats = cur.fetchall()

    cur.execute("SELECT MIN(layer), MAX(layer) FROM gamestates")
    min_l, max_l = cur.fetchone()

    conn.close()
    return {
        "nodes": nodes,
        "edges": edges,
        "layer_stats": layer_stats,
        "min_layer": min_l,
        "max_layer": max_l,
        "n": n,
        "version": version
    }


class Handler(BaseHTTPRequestHandler):
    n = 26
    version = "v2"

    def log_message(self, format, *args):
        pass  # suppress request logging

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/":
            self.serve_file("game_tree_viewer.html", "text/html")

        elif parsed.path == "/api/tree":
            params = parse_qs(parsed.query)
            min_layer = int(params.get("min_layer", [0])[0])
            max_layer = int(params.get("max_layer", [5])[0])
            data = get_tree_data(self.n, self.version, min_layer, max_layer)
            self.send_json(data)

        elif parsed.path == "/api/node":
            params = parse_qs(parsed.query)
            canonical = params.get("canonical", [""])[0]
            turn = int(params.get("turn", [0])[0])
            db_path = get_db_path(self.n, self.version)
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            cur.execute("SELECT canonical, turn, layer, winner FROM gamestates WHERE canonical=? AND turn=?",
                        (canonical, turn))
            row = cur.fetchone()
            cur.execute("SELECT child_canonical, child_turn FROM edges WHERE parent_canonical=? AND parent_turn=?",
                        (canonical, turn))
            children = cur.fetchall()
            cur.execute("SELECT parent_canonical, parent_turn FROM edges WHERE child_canonical=? AND child_turn=?",
                        (canonical, turn))
            parents = cur.fetchall()
            conn.close()
            if row:
                self.send_json({
                    "canonical": row[0],
                    "turn": row[1],
                    "layer": row[2],
                    "winner": json.loads(row[3]) if row[3] else None,
                    "children": [{"canonical": c, "turn": t} for c, t in children],
                    "parents": [{"canonical": c, "turn": t} for c, t in parents]
                })
            else:
                self.send_json({"error": "not found"})

        elif parsed.path == "/api/search":
            params = parse_qs(parsed.query)
            query = params.get("q", [""])[0]
            db_path = get_db_path(self.n, self.version)
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            cur.execute("""
                SELECT canonical, turn, layer, winner 
                FROM gamestates 
                WHERE canonical LIKE ?
                ORDER BY layer, canonical, turn
                LIMIT 50
            """, (f"%{query}%",))
            rows = cur.fetchall()
            conn.close()
            self.send_json([{
                "canonical": r[0],
                "turn": r[1],
                "layer": r[2],
                "winner": json.loads(r[3]) if r[3] else None
            } for r in rows])

        else:
            self.send_response(404)
            self.end_headers()

    def serve_file(self, filename, content_type):
        path = os.path.join(os.path.dirname(__file__), filename)
        try:
            with open(path, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.end_headers()
            self.wfile.write(content)
        except FileNotFoundError:
            self.send_response(404)
            self.end_headers()

    def send_json(self, data):
        content = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=26)
    parser.add_argument("--version", type=str, default="v2")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    Handler.n = args.n
    Handler.version = args.version

    print(f"Serving game tree for n={args.n} version={args.version}")
    print(f"Open http://localhost:{args.port} in your browser")
    HTTPServer(("", args.port), Handler).serve_forever()
