from games_as_gaps import *
import tkinter as tk
import sqlite3

def parse_gaps(canonical: str):
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



class CLI_GameExplorer:
    def __init__(self, n):
        self.db = f"gamestates_n{n}.db"
        self.conn = sqlite3.connect(self.db)
        self.cur = self.conn.cursor()
        self.history = []  # for navigating backward

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
        gaps = parse_gaps(canonical)
        state = GameState(gaps, turn)
        children = []

        for child in state.legal_moves():
            child_key = (canonical_str(child), child.turn)
            self.cur.execute(
                "SELECT winner, layer FROM gamestates WHERE canonical=? AND turn=?",
                child_key
            )
            row = self.cur.fetchone()
            if row:
                children.append({
                    "canonical": child_key[0],
                    "turn": child_key[1],
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
        self.root.title(f"Game Explorer n={n}")   # CREATE ROOT FIRST
        self.root.geometry("1200x1800")
        self.root.state("zoomed")
        self.highlight_enabled = tk.BooleanVar(self.root, value=True)
        self.n = n

        self.db = f"gamestates_n{n}.db"
        self.conn = sqlite3.connect(self.db)
        self.cur = self.conn.cursor()

        self.history = []

        # Canvas for drawing gaps
        self.canvas = tk.Canvas(
            self.root,
            height=BLOCK_HEIGHT + 20,  # fixed height
            bg="white"
        )

        self.canvas.pack(
            fill="x",  # expand horizontally only
            expand=False,  # do NOT expand vertically
            pady=10,
            anchor="n"  # keep it at the top
        )

        self.state_label = tk.Label(self.root, font=("Arial", 12))
        self.state_label.pack()

        self.info_label = tk.Label(self.root, font=("Arial", 11))
        self.info_label.pack()

        # Highlight toggle
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

    # -----------------------------
    # Drawing gaps visually
    # -----------------------------
    def draw_gaps(self, canonical):
        self.canvas.delete("all")
        x = 10

        if not canonical:
            self.canvas.create_text(150, 30, text="Terminal State", font=("Arial", 12))
            return

        for g in canonical.split(','):
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

    # -----------------------------
    # DB Helpers
    # -----------------------------
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
        gaps = parse_gaps(canonical)
        state = GameState(gaps, turn)

        children = []
        for child in state.legal_moves():
            key = (canonical_str(child), child.turn)
            self.cur.execute(
                "SELECT winner, layer FROM gamestates WHERE canonical=? AND turn=?",
                key
            )
            row = self.cur.fetchone()
            if row:
                children.append({
                    "canonical": key[0],
                    "turn": key[1],
                    "winner": row[0],
                    "layer": row[1]
                })
        return children

    # -----------------------------
    # UI Logic
    # -----------------------------
    def display_state(self, state):
        self.current_state = state

        self.draw_gaps(state["canonical"])

        self.state_label.config(
            text=f"Player {state['turn']+1}'s turn"
        )

        self.info_label.config(
            text=f"Winner tuple: {state['winner']}   |   Layer: {state['layer']}"
        )

        for widget in self.moves_frame.winfo_children():
            widget.destroy()

        children = self.get_children(state["canonical"], state["turn"])

        if not children:
            tk.Label(self.moves_frame, text="Terminal state").pack()
            return

        # Parse current state's winner
        current_winner = eval(state["winner"])
        highlight = self.highlight_enabled.get()

        button_placement = 0
        MAX_ROWS = self.n // 2
        for child in children:
            btn = tk.Button(
                self.moves_frame,
                text=f"{child['canonical']} (P{child['turn']+1}) → {child['winner']}",
                command=lambda c=child: self.make_move(c),
                width=50
            )

            if highlight:
                child_winner = eval(child["winner"])
                # Winning move condition
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

    def start(self, start_canonical, start_turn=0):
        state = self.load_state(start_canonical, start_turn)
        if state:
            self.display_state(state)
            self.root.mainloop()
        else:
            print("Start state not found.")


# --------------------------------
# Query a specific position
# --------------------------------


def get_winner_and_moves(db_path, gaps):
    """
    Query the DB for a specific gap layout and list winning moves.

    Args:
        db_path: Path to the gamestates DB
        gaps: List of Gap objects or (size, orientation) tuples
    """
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

    winner = eval(row[0])

    winning_moves = []
    for move_state in state.legal_moves():
        move_canonical = canonical_str(move_state)
        cur.execute("SELECT winner FROM gamestates WHERE canonical=?", (move_canonical,))
        move_row = cur.fetchone()
        if move_row:
            move_winner = eval(move_row[0])
            if move_winner[2] == 1:
                winning_moves.append(move_canonical)

    conn.close()
    return winner, winning_moves

def table_positive_gaps(db_path, max_k):
    """
    Generate a table of positive gaps (k, 1) up to max_k.
    Prints winner tuple and winning moves.
    """

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    print(f"{'Gap':>8} | {'Winner (curr,next,prev)':>25} | Winning Moves")
    print("-" * 80)

    for k in range(1, max_k + 1):
        gaps = [Gap(k, 1)]
        state = GameState(gaps)
        canonical = canonical_str(state)

        # Query ignoring turn
        cur.execute("SELECT winner FROM gamestates WHERE canonical=?", (canonical,))
        row = cur.fetchone()
        if not row:
            # No such gamestate exists
            print(f"{'+' + str(k):>8} | {'-GameState-not-found-':>25} | {'N/A'}")
            continue

        winner = eval(row[0])

        # Find winning moves
        winning_moves = []
        for move_state in state.legal_moves():
            move_canonical = canonical_str(move_state)
            cur.execute("SELECT winner FROM gamestates WHERE canonical=?", (move_canonical,))
            move_row = cur.fetchone()
            if move_row:
                move_winner = eval(move_row[0])
                if move_winner[2] == 1:  # previous player wins → current player can force a win
                    winning_moves.append(move_canonical)

        print(f"{'+' + str(k):>8} | {str(winner):>25} | {winning_moves}")

    conn.close()

def table_negative_gaps(db_path, max_k):
    """
    Generate a table of negative gaps (-k, -1) up to max_k.
    Prints winner tuple and winning moves.
    """

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    print(f"{'Gap':>8} | {'Winner (curr,next,prev)':>25} | Winning Moves")
    print("-" * 80)

    for k in range(1, max_k + 1):
        gaps = [Gap(k, -1), Gap(1, -1)]
        state = GameState(gaps)
        canonical = canonical_str(state)

        # Query ignoring turn
        cur.execute("SELECT winner FROM gamestates WHERE canonical=?", (canonical,))
        row = cur.fetchone()
        if not row:
            # No such gamestate exists
            print(f"{'-' + str(k) + ', -1':>8} | {'-GameState-not-found-':>25} | {'N/A'}")
            continue

        winner = eval(row[0])

        # Find winning moves
        winning_moves = []
        for move_state in state.legal_moves():
            move_canonical = canonical_str(move_state)
            cur.execute("SELECT winner FROM gamestates WHERE canonical=?", (move_canonical,))
            move_row = cur.fetchone()
            if move_row:
                move_winner = eval(move_row[0])
                if move_winner[2] == 1:  # previous player wins → current player can force a win
                    winning_moves.append(move_canonical)

        print(f"{'-' + str(k) + ', -1':>8} | {str(winner):>25} | {winning_moves}")

    conn.close()


def table_double_positive_gaps(db_path, max_k, max_m):
    """
    Generate a table of positive gaps (k, 1), (m, 1) up to max_k and max_m.
    Prints winner tuple and winning moves.
    """

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    print(f"{'Gap':>8} | {'Individual Gap Winners':>30} | {'Winner (curr,next,prev)':>25} | Winning Moves")
    print("-" * 150)

    for k in range(1, max_k + 1):
        gap_1 = [Gap(k, 1)]
        state_1 = GameState(gap_1)
        canonical_1 = canonical_str(state_1)
        # Query getting individual gap data
        cur.execute("SELECT winner FROM gamestates WHERE canonical=?", (canonical_1,))
        gap1_winner = eval(cur.fetchone()[0])
        for m in range(1, k + 1):
            gap_2 = [Gap(m, 1)]
            state_2 = GameState(gap_2)
            canonical_2 = canonical_str(state_2)
            # Query getting individual gap data
            cur.execute("SELECT winner FROM gamestates WHERE canonical=?", (canonical_2,))
            gap2_winner = eval(cur.fetchone()[0])

            gaps = [Gap(k, 1), Gap(m, 1)]
            state = GameState(gaps)
            canonical = canonical_str(state)

            # Query ignoring turn
            cur.execute("SELECT winner FROM gamestates WHERE canonical=?", (canonical,))
            row = cur.fetchone()
            if not row:
                # No such gamestate exists
                print(f"{'-' + str(k) + ', -1':>8} | {'':>30} | {'-GameState-not-found-':>25} | {'N/A'}")
                continue

            winner = eval(row[0])

            # Find winning moves
            winning_moves = []
            for move_state in state.legal_moves():
                move_canonical = canonical_str(move_state)
                cur.execute("SELECT winner FROM gamestates WHERE canonical=?", (move_canonical,))
                move_row = cur.fetchone()
                if move_row:
                    move_winner = eval(move_row[0])
                    if move_winner[2] == 1:  # previous player wins → current player can force a win
                        winning_moves.append(move_canonical)

            print(f"{'+' + str(k) + ', +' + str(m) :>8} | {str(gap1_winner) + ', ' + str(gap2_winner):>30} |  {str(winner):>25} | {winning_moves}")

    conn.close()

if __name__ == "__main__":
    #explorer = CLI_GameExplorer(n=20)
    #explorer.explore("+20", 0)
    #explorer = VisualGameExplorer(n=20)
    #explorer.start("+20", 0)


    db_path = "gamestates_n39.db"

    """
    # Single +k gap
    winner, moves = get_winner_and_moves(db_path, [(5, 1)], turn=0)
    print("Gap +5:", winner)
    print("Winning moves:", moves)

    # Single -k gap
    winner, moves = get_winner_and_moves(db_path, [(3, -1)], turn=0)
    print("Gap -3:", winner)
    print("Winning moves:", moves)

    # Multi-gap layout
    winner, moves = get_winner_and_moves(db_path, [(3, 1), (2, -1)], turn=0)
    print("Layout +3,-2:", winner)
    print("Winning moves:", moves)

    for k in range(2,19):
        winner, moves = get_winner_and_moves(db_path, [(k, -1),(1,-1)], turn=0)
        print(f"Gap -{k}:", winner)
        print("Winning moves:", moves)
    """

    # table_positive_gaps(db_path, max_k=30)

    # table_negative_gaps(db_path, max_k=37)

    table_double_positive_gaps(db_path, max_k = 10, max_m = 10)