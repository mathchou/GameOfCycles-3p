from games_as_gaps import *
import tkinter as tk
import sqlite3
import json
import pandas as pd
import os

DB_DIR = "S:\\GameOfCycles-3p"



class CLI_GameExplorer:
    def __init__(self, n):
        self.db = f"gamestates_n{n}_reduced.db"
        self.conn = sqlite3.connect(self.db)
        self.cur = self.conn.cursor()
        self.history = []

    def load_state(self, canonical, turn):
        self.cur.execute(
            "SELECT winner, layer FROM gamestates WHERE canonical=? AND turn=?",
            (canonical, turn)
        )
        row = self.cur.fetchone()
        if not row:
            print("State not found in DB.")
            return None
        winner, layer = row
        return {"canonical": canonical, "turn": turn, "winner": winner, "layer": layer}

    def get_children(self, canonical, turn):
        self.cur.execute(
            "SELECT child_canonical, child_turn FROM edges WHERE parent_canonical=? AND parent_turn=?",
            (canonical, turn)
        )
        child_rows = self.cur.fetchall()
        children = []
        for child_canonical, child_turn in child_rows:
            self.cur.execute(
                "SELECT winner, layer FROM gamestates WHERE canonical=? AND turn=?",
                (child_canonical, child_turn)
            )
            row = self.cur.fetchone()
            if row:
                children.append({
                    "canonical": child_canonical,
                    "turn": child_turn,
                    "winner": row[0],
                    "layer": row[1]
                })
        return children

    def explore(self, start_canonical, start_turn=0):
        current = self.load_state(start_canonical, start_turn)
        if not current:
            return

        while True:
            print("\nCurrent state:")
            print(f"Canonical: {current['canonical']}")
            print(f"Turn: Player {current['turn'] + 1}")
            print(f"Winner tuple: {current['winner']}")
            print(f"Layer: {current['layer']}")

            children = self.get_children(current["canonical"], current["turn"])

            if not children:
                print("Terminal state.")
            else:
                print("\nMoves:")
                for i, child in enumerate(children):
                    print(f"{i}: {child['canonical']} (P{child['turn']+1}) Winner={child['winner']}")

            print("\nCommands: number = move, b = back, q = quit")
            cmd = input("> ")

            if cmd == 'q':
                break
            elif cmd == 'b':
                if self.history:
                    current = self.history.pop()
                else:
                    print("No previous state.")
            elif cmd.isdigit():
                idx = int(cmd)
                if 0 <= idx < len(children):
                    self.history.append(current)
                    current = children[idx]
                else:
                    print("Invalid move.")


BLOCK_SIZE = 25
BLOCK_HEIGHT = 40
BLOCK_SPACING = 5


class VisualGameExplorer:
    def __init__(self, n):
        self.root = tk.Tk()
        self.root.title(f"Game Explorer n={n}")
        self.root.geometry("1200x1800")
        self.root.state("zoomed")
        self.highlight_enabled = tk.BooleanVar(self.root, value=True)
        self.n = n
        self.db = os.path.join(DB_DIR, f"gamestates_n{n}_v2.db")
        self.conn = sqlite3.connect(self.db)
        self.cur = self.conn.cursor()

        self.history = []

        self.canvas = tk.Canvas(
            self.root,
            height=BLOCK_HEIGHT + 20,
            bg="white"
        )
        self.canvas.pack(fill="x", expand=False, pady=10, anchor="n")

        self.state_label = tk.Label(self.root, font=("Arial", 12))
        self.state_label.pack()

        self.info_label = tk.Label(self.root, font=("Arial", 11))
        self.info_label.pack()

        self.toggle = tk.Checkbutton(
            self.root,
            text="Highlight Winning Moves",
            variable=self.highlight_enabled,
            command=self.refresh_state
        )
        self.toggle.pack()

        self.moves_frame = tk.Frame(self.root)
        self.moves_frame.pack(pady=10)

        self.back_button = tk.Button(self.root, text="Back", command=self.go_back)
        self.back_button.pack(pady=5)

    def draw_gaps(self, canonical):
        self.canvas.delete("all")
        x = 10

        if not canonical:
            self.canvas.create_text(150, 30, text="Terminal State", font=("Arial", 12))
            return

        # Split off inevitable moves
        if '|' in canonical:
            gaps_part, inevitable_part = canonical.rsplit('|', 1)
            inevitable_moves = int(inevitable_part)
        else:
            gaps_part = canonical
            inevitable_moves = 0

        for g in gaps_part.split(','):
            if not g:
                continue
            orientation = 1 if g[0] == '+' else -1
            size = int(g[1:])
            color = "steelblue" if orientation == 1 else "indianred"
            for _ in range(size):
                self.canvas.create_rectangle(
                    x, 10,
                    x + BLOCK_SIZE, 10 + BLOCK_HEIGHT,
                    fill=color
                )
                x += BLOCK_SIZE
            x += BLOCK_SPACING

        # Draw inevitable moves as a grey number at the end
        if inevitable_moves > -1:
            self.canvas.create_text(
                x + 60, 10 + BLOCK_HEIGHT // 2,
                text=f"+{inevitable_moves} inevitable",
                font=("Arial", 10),
                fill="grey"
            )

    def load_state(self, canonical, turn):
        self.cur.execute(
            "SELECT winner, layer FROM gamestates WHERE canonical=? AND turn=?",
            (canonical, turn)
        )
        row = self.cur.fetchone()
        if not row:
            return None
        return {
            "canonical": canonical,
            "turn": turn,
            "winner": row[0],
            "layer": row[1]
        }

    def get_children(self, canonical, turn):
        self.cur.execute(
            "SELECT child_canonical, child_turn FROM edges WHERE parent_canonical=? AND parent_turn=?",
            (canonical, turn)
        )
        child_rows = self.cur.fetchall()
        children = []
        for child_canonical, child_turn in child_rows:
            self.cur.execute(
                "SELECT winner, layer FROM gamestates WHERE canonical=? AND turn=?",
                (child_canonical, child_turn)
            )
            row = self.cur.fetchone()
            if row:
                children.append({
                    "canonical": child_canonical,
                    "turn": child_turn,
                    "winner": row[0],
                    "layer": row[1]
                })
        return children

    def display_state(self, state):
        self.current_state = state
        self.draw_gaps(state["canonical"])
        self.state_label.config(text=f"Player {state['turn']+1}'s turn")
        self.info_label.config(
            text=f"Winner tuple: {state['winner']}   |   Layer: {state['layer']}"
        )

        for widget in self.moves_frame.winfo_children():
            widget.destroy()

        children = self.get_children(state["canonical"], state["turn"])

        if not children:
            tk.Label(self.moves_frame, text="Terminal state").pack()
            return

        current_winner = json.loads(state["winner"]) if state["winner"] else None
        highlight = self.highlight_enabled.get()

        button_placement = 0
        MAX_ROWS = max(self.n // 2, 1)

        for child in children:
            btn = tk.Button(
                self.moves_frame,
                text=f"{child['canonical']} (P{child['turn']+1}) → {child['winner']}",
                command=lambda c=child: self.make_move(c),
                width=50
            )

            if highlight and child["winner"]:
                child_winner = json.loads(child["winner"])
                if child_winner[2] == 1:
                    btn.config(bg="lightgreen")

            row = button_placement % MAX_ROWS
            col = button_placement // MAX_ROWS
            button_placement += 1
            btn.grid(row=row, column=col, padx=5, pady=5, sticky="ew")

    def refresh_state(self):
        self.display_state(self.current_state)

    def make_move(self, child_state):
        self.history.append(self.current_state)
        self.display_state(child_state)

    def go_back(self):
        if self.history:
            previous = self.history.pop()
            self.display_state(previous)

    def start(self, size: int, orientation: int = 1, start_turn: int = 0):
        canonical = find_start_canonical(self.cur, size, orientation)
        if not canonical:
            return
        print(f"Starting at canonical: {canonical}")
        state = self.load_state(canonical, start_turn)
        if state:
            self.display_state(state)
            self.root.mainloop()
        else:
            print("Start state not found.")

def find_start_canonical(cur, size: int, orientation: int = 1):
    """Find the canonical string for a single gap of given size and orientation."""
    cur.execute(
        "SELECT canonical, turn, layer FROM gamestates WHERE canonical LIKE ? AND turn=0",
        (f"{'+' if orientation == 1 else '-'}{size}%",)
    )
    rows = cur.fetchall()
    if not rows:
        print(f"No state found for gap {'+' if orientation==1 else '-'}{size}")
        return None
    if len(rows) > 1:
        print(f"Multiple matches found:")
        for row in rows:
            print(f"  {row}")
        return None
    return rows[0][0]

def get_winner_and_moves(db_path, gaps):
    gaps_list = [g if isinstance(g, Gap) else Gap(*g) for g in gaps]
    state = GameState(gaps_list)
    canonical = canonical_str(state)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("SELECT winner FROM gamestates WHERE canonical=?", (canonical,))
    row = cur.fetchone()
    if not row:
        conn.close()
        raise ValueError(f"Gamestate {canonical} not found in DB")

    winner = json.loads(row[0])

    cur.execute(
        "SELECT child_canonical, child_turn FROM edges WHERE parent_canonical=? AND parent_turn=?",
        (canonical, state.turn)
    )
    winning_moves = []
    for child_canonical, child_turn in cur.fetchall():
        cur.execute("SELECT winner FROM gamestates WHERE canonical=? AND turn=?",
                    (child_canonical, child_turn))
        move_row = cur.fetchone()
        if move_row and move_row[0]:
            move_winner = json.loads(move_row[0])
            if move_winner[2] == 1:
                winning_moves.append(child_canonical)

    conn.close()
    return winner, winning_moves


def table_positive_gaps(db_path, max_k):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    print(f"{'Gap':>8} | {'Winner (curr,next,prev)':>25} | Winning Moves")
    print("-" * 80)

    for k in range(1, max_k + 1):
        gaps = [Gap(k, 1)]
        state = GameState(gaps)
        canonical = canonical_str(state)

        cur.execute("SELECT winner FROM gamestates WHERE canonical=?", (canonical,))
        row = cur.fetchone()
        if not row:
            print(f"{'+' + str(k):>8} | {'-GameState-not-found-':>25} | {'N/A'}")
            continue

        winner = json.loads(row[0])

        cur.execute(
            "SELECT child_canonical, child_turn FROM edges WHERE parent_canonical=? AND parent_turn=?",
            (canonical, state.turn)
        )
        winning_moves = []
        for child_canonical, child_turn in cur.fetchall():
            cur.execute("SELECT winner FROM gamestates WHERE canonical=? AND turn=?",
                        (child_canonical, child_turn))
            move_row = cur.fetchone()
            if move_row and move_row[0]:
                move_winner = json.loads(move_row[0])
                if move_winner[2] == 1:
                    winning_moves.append(child_canonical)

        print(f"{'+' + str(k):>8} | {str(winner):>25} | {winning_moves}")

    conn.close()


def table_negative_gaps(db_path, max_k):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    print(f"{'Gap':>8} | {'Winner (curr,next,prev)':>25} | Winning Moves")
    print("-" * 80)

    for k in range(1, max_k + 1):
        gaps = [Gap(k, -1), Gap(1, -1)]
        state = GameState(gaps)
        canonical = canonical_str(state)

        cur.execute("SELECT winner FROM gamestates WHERE canonical=?", (canonical,))
        row = cur.fetchone()
        if not row:
            print(f"{'-' + str(k) + ', -1':>8} | {'-GameState-not-found-':>25} | {'N/A'}")
            continue

        winner = json.loads(row[0])

        cur.execute(
            "SELECT child_canonical, child_turn FROM edges WHERE parent_canonical=? AND parent_turn=?",
            (canonical, state.turn)
        )
        winning_moves = []
        for child_canonical, child_turn in cur.fetchall():
            cur.execute("SELECT winner FROM gamestates WHERE canonical=? AND turn=?",
                        (child_canonical, child_turn))
            move_row = cur.fetchone()
            if move_row and move_row[0]:
                move_winner = json.loads(move_row[0])
                if move_winner[2] == 1:
                    winning_moves.append(child_canonical)

        print(f"{'-' + str(k) + ', -1':>8} | {str(winner):>25} | {winning_moves}")

    conn.close()


def table_double_positive_gaps(db_path, max_k, return_df=True):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    rows = []
    print(f"{'Gap':>12} | {'Individual Gap Winners':>30} | {'Winner (curr,next,prev)':>25} | Winning Moves")
    print("-" * 160)

    for k in range(1, max_k + 1):
        gap_1 = [Gap(k, 1)]
        state_1 = GameState(gap_1)
        canonical_1 = canonical_str(state_1)
        cur.execute("SELECT winner FROM gamestates WHERE canonical=?", (canonical_1,))
        gap1_row = cur.fetchone()
        if not gap1_row:
            continue
        gap1_winner = json.loads(gap1_row[0])

        for m in range(1, k + 1):
            gap_2 = [Gap(m, 1)]
            state_2 = GameState(gap_2)
            canonical_2 = canonical_str(state_2)
            cur.execute("SELECT winner FROM gamestates WHERE canonical=?", (canonical_2,))
            gap2_row = cur.fetchone()
            if not gap2_row:
                continue
            gap2_winner = json.loads(gap2_row[0])

            gaps = [Gap(k, 1), Gap(m, 1)]
            state = GameState(gaps)
            canonical = canonical_str(state)

            cur.execute("SELECT winner FROM gamestates WHERE canonical=?", (canonical,))
            row = cur.fetchone()
            if not row:
                print(f"{'+' + str(k) + ', +' + str(m):>12} | {'':>30} | {'-GameState-not-found-':>25} | {'N/A'}")
                continue

            winner = json.loads(row[0])

            cur.execute(
                "SELECT child_canonical, child_turn FROM edges WHERE parent_canonical=? AND parent_turn=?",
                (canonical, state.turn)
            )
            winning_moves = []
            for child_canonical, child_turn in cur.fetchall():
                cur.execute("SELECT winner FROM gamestates WHERE canonical=? AND turn=?",
                            (child_canonical, child_turn))
                move_row = cur.fetchone()
                if move_row and move_row[0]:
                    move_winner = json.loads(move_row[0])
                    if move_winner[2] == 1:
                        winning_moves.append(child_canonical)

            gap_label = f"+{k}, +{m}"
            individual_label = f"{gap1_winner}, {gap2_winner}"
            print(f"{gap_label:>12} | {individual_label:>30} | {str(winner):>25} | {winning_moves}")

            rows.append({
                "Gap": gap_label,
                "Gap_k": k,
                "Gap_m": m,
                "Gap_k_winner": gap1_winner,
                "Gap_m_winner": gap2_winner,
                "Combined_winner": winner,
                "Winning_moves": winning_moves
            })

    conn.close()

    if return_df:
        return pd.DataFrame(rows)


if __name__ == "__main__":
    explorer = VisualGameExplorer(n=26)
    explorer.start(26)  # looks up +26 automatically